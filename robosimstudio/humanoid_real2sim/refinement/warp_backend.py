"""GPU batched dynamics using a model compiled by RSS/Newton.

Hypothetical rollouts use MuJoCo Warp; committed steps use MuJoCo CPU with
the identical RSS-compiled model, providing a deterministic replay oracle.
The only state copies are at the start of a hypothetical rollout. A committed
trajectory advances continuously; neither the free root nor object is driven.
"""

import numpy as np
import warp as wp


@wp.kernel
def set_control(actions: wp.array3d(dtype=float), index: wp.array(dtype=int), ctrl: wp.array2d(dtype=float)):
    w, j = wp.tid()
    ctrl[w, j] = actions[w, index[0], j]


@wp.kernel
def advance_index(index: wp.array(dtype=int)):
    index[0] += 1


@wp.kernel
def score(
    q: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float),
    weights: wp.array(dtype=float),
    quat_starts: wp.array(dtype=int),
    index: wp.array(dtype=int),
    costs: wp.array(dtype=float),
):
    w = wp.tid()
    t = index[0] + 1
    value = float(0.0)
    for j in range(q.shape[1]):
        delta = q[w, j] - target[t, j]
        value += weights[j] * delta * delta
    for k in range(quat_starts.shape[0]):
        a = quat_starts[k]
        dot = float(0.0)
        norm = float(0.0)
        for j in range(4):
            dot += q[w, a + j] * target[t, a + j]
            norm += q[w, a + j] * q[w, a + j]
        value += 5.0 * (1.0 - dot * dot / wp.max(norm, 1.0e-8))
    for j in range(v.shape[1]):
        value += 0.00002 * v[w, j] * v[w, j]
    costs[w] += value


@wp.kernel
def body_score(
    x: wp.array2d(dtype=wp.vec3),
    quat: wp.array2d(dtype=wp.quat),
    target_x: wp.array2d(dtype=wp.vec3),
    target_quat: wp.array2d(dtype=wp.quat),
    weights: wp.array2d(dtype=float),
    relative_ids: wp.array(dtype=int),
    object_body: int,
    index: wp.array(dtype=int),
    costs: wp.array(dtype=float),
):
    w = wp.tid()
    t = index[0] + 1
    value = float(0.0)
    for b in range(x.shape[1]):
        delta = x[w, b] - target_x[t, b]
        dot = wp.dot(quat[w, b], target_quat[t, b])
        value += weights[b, 0] * wp.dot(delta, delta) + weights[b, 1] * (1.0 - dot * dot)
    if object_body >= 0:
        for k in range(relative_ids.shape[0]):
            b = relative_ids[k]
            delta = (x[w, b] - x[w, object_body]) - (target_x[t, b] - target_x[t, object_body])
            value += 100.0 * wp.dot(delta, delta)
    costs[w] += value


@wp.kernel
def unwanted_contact(
    nacon: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    world: wp.array(dtype=int),
    dist: wp.array(dtype=float),
    kind: wp.array(dtype=int),
    costs: wp.array(dtype=float),
):
    c = wp.tid()
    if c < nacon[0] and dist[c] < 0.0:
        pair = geom[c]
        a = kind[pair[0]]
        b = kind[pair[1]]
        if (a == 1 and b == 2) or (a == 2 and b == 1):
            wp.atomic_add(costs, world[c], 5.0)


class WarpBackend:
    def __init__(
        self,
        model,
        qpos,
        qvel,
        targets,
        weights,
        quat_starts,
        *,
        samples,
        horizon,
        substeps=10,
        body_weights=None,
        relative_body_ids=(),
        object_body=-1,
        contact_geom_kind=None,
    ):
        import mujoco
        import mujoco_warp as mjw

        if not wp.is_cuda_available():
            raise RuntimeError("CUDA is required for the GPU refinement backend")
        if model.na or model.nmocap:
            raise ValueError("this backend requires stateless actuators and no mocap bodies")
        if np.any(model.body_gravcomp != 0):
            raise ValueError("gravity compensation is not allowed in an unassisted refinement")
        if targets.ndim != 2 or targets.shape[1] != model.nq or len(targets) < 2:
            raise ValueError("targets must be a frame-by-qpos array with at least two frames")
        if not np.isfinite(targets).all() or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("initial state and targets must be finite")
        if substeps < 1 or horizon < 1 or samples < 1:
            raise ValueError("substeps, horizon and samples must be positive")
        self.model = model
        self.lower = model.actuator_ctrlrange[:, 0]
        self.upper = model.actuator_ctrlrange[:, 1]
        self.samples, self.horizon, self.substeps = samples, horizon, substeps
        self.targets = targets
        self.object_body = object_body
        self.use_body_objective = body_weights is not None
        if self.use_body_objective:
            body_weights = np.asarray(body_weights)
            if (
                body_weights.shape != (model.nbody, 2)
                or not np.isfinite(body_weights).all()
                or np.any(body_weights < 0)
            ):
                raise ValueError("body weights must be a nonnegative finite nbody-by-2 matrix")
            if contact_geom_kind is None:
                contact_geom_kind = np.zeros(model.ngeom, dtype=np.int32)
            if np.asarray(contact_geom_kind).shape != (model.ngeom,):
                raise ValueError("contact classification must have one entry per geom")
            if any(b < 0 or b >= model.nbody for b in relative_body_ids) or object_body >= model.nbody:
                raise ValueError("relative body id outside model")
        if self.use_body_objective:
            fk = mujoco.MjData(model)
            self.body_targets_x, self.body_targets_q = [], []
            for pose in targets:
                fk.qpos[:] = pose
                mujoco.mj_forward(model, fk)
                self.body_targets_x.append(fk.xpos.copy())
                self.body_targets_q.append(fk.xquat.copy())
            self.body_targets_x = np.asarray(self.body_targets_x, dtype=np.float32)
            self.body_targets_q = np.asarray(self.body_targets_q, dtype=np.float32)
        self.qpos, self.qvel = qpos.copy(), qvel.copy()
        self.history_q, self.history_v = [qpos.copy()], [qvel.copy()]
        self.warm = np.zeros(model.nv)
        self.execution = mujoco.MjData(model)
        self.execution.qpos[:] = qpos
        self.execution.qvel[:] = qvel
        self.overflow_bits = 0
        self.device = "cuda:0"
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(model)
            self.m.opt.warn_overflow = False  # collect bitmask in the report instead of per-world spam
            d = mujoco.MjData(model)
            self.d = mjw.put_data(model, d, nworld=samples, nconmax=512, njmax=1024)
            self.actions = wp.zeros((samples, horizon, model.nu), dtype=float)
            self.target = wp.zeros((horizon + 1, model.nq), dtype=float)
            self.weights = wp.array(weights.astype(np.float32))
            self.quats = wp.array(np.array(quat_starts, dtype=np.int32))
            self.index = wp.zeros(1, dtype=int)
            self.costs = wp.zeros(samples, dtype=float)
            if self.use_body_objective:
                self.body_weights = wp.array(np.asarray(body_weights, dtype=np.float32))
                self.relative_ids = wp.array(np.asarray(relative_body_ids, dtype=np.int32))
                self.body_target_x = wp.zeros((horizon + 1, model.nbody), dtype=wp.vec3)
                self.body_target_q = wp.zeros((horizon + 1, model.nbody), dtype=wp.quat)
                self.contact_kind = wp.array(np.asarray(contact_geom_kind, dtype=np.int32))
            # Compile outside capture. State is reset before any scored rollout.
            self._step()
            with wp.ScopedCapture() as capture:
                self._step()
            self.graph = capture.graph
            with wp.ScopedCapture() as horizon_capture:
                for _ in range(horizon):
                    self._step()
                    wp.launch(advance_index, 1, [self.index])
            self.horizon_graph = horizon_capture.graph

    def _step(self):
        import mujoco_warp as mjw

        wp.launch(set_control, (self.samples, self.model.nu), [self.actions, self.index, self.d.ctrl])
        for _ in range(self.substeps):
            mjw.step(self.m, self.d)
        wp.launch(
            score,
            self.samples,
            [self.d.qpos, self.d.qvel, self.target, self.weights, self.quats, self.index, self.costs],
        )

        if self.use_body_objective:
            wp.launch(
                body_score,
                self.samples,
                [
                    self.d.xpos,
                    self.d.xquat,
                    self.body_target_x,
                    self.body_target_q,
                    self.body_weights,
                    self.relative_ids,
                    self.object_body,
                    self.index,
                    self.costs,
                ],
            )
            wp.launch(
                unwanted_contact,
                self.d.contact.dist.shape[0],
                [
                    self.d.nacon,
                    self.d.contact.geom,
                    self.d.contact.worldid,
                    self.d.contact.dist,
                    self.contact_kind,
                    self.costs,
                ],
            )

    def _reset(self):
        for name, state in [("qpos", self.qpos), ("qvel", self.qvel), ("qacc_warmstart", self.warm)]:
            getattr(self.d, name).assign(np.tile(state, (self.samples, 1)).astype(np.float32))
        self.d.time.zero_()
        self.d.qfrc_applied.zero_()
        self.d.xfrc_applied.zero_()
        self.costs.zero_()
        self.d.overflow.zero_()

    def evaluate(self, actions, start):
        h = actions.shape[1]
        buffer = np.zeros(self.actions.shape, np.float32)
        buffer[:, :h] = actions
        target = self.targets[np.minimum(start + np.arange(self.horizon + 1), len(self.targets) - 1)]
        self.actions.assign(buffer)
        self.target.assign(target.astype(np.float32))
        if self.use_body_objective:
            indices = np.minimum(start + np.arange(self.horizon + 1), len(self.targets) - 1)
            self.body_target_x.assign(self.body_targets_x[indices])
            self.body_target_q.assign(self.body_targets_q[indices])
        self._reset()
        if h == self.horizon:
            self.index.zero_()
            wp.capture_launch(self.horizon_graph)
        else:
            for t in range(h):
                self.index.assign(np.array([t], np.int32))
                wp.capture_launch(self.graph)
        overflow = self.d.overflow.numpy()
        self.overflow_bits |= int(np.bitwise_or.reduce(overflow))
        costs = self.costs.numpy().astype(float) / h
        # Penalize target chatter, measured in radians at control frequency.
        costs += 0.02 * np.mean(np.diff(actions, axis=1) ** 2, axis=(1, 2)) if h > 1 else 0
        costs[(overflow & ~1024) != 0] = np.inf  # reject capacity/CCD failures; report line-search exhaustion
        return costs

    def reset_execution(self, qpos, qvel):
        import mujoco

        mujoco.mj_resetData(self.model, self.execution)
        self.execution.qpos[:] = qpos
        self.execution.qvel[:] = qvel
        self.qpos, self.qvel = qpos.copy(), qvel.copy()
        self.warm = np.zeros(self.model.nv)
        self.history_q, self.history_v = [qpos.copy()], [qvel.copy()]

    def commit(self, actions):
        import mujoco

        # This data object is never touched by sampling, nor reset between
        # windows. Save the real executed state for the next planning window.
        for action in actions:
            self.execution.ctrl[:] = action
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.execution)
            self.qpos = self.execution.qpos.copy()
            self.qvel = self.execution.qvel.copy()
            if not np.isfinite(self.qpos).all() or not np.isfinite(self.qvel).all():
                raise RuntimeError("nonfinite committed state")
            self.history_q.append(self.qpos.copy())
            self.history_v.append(self.qvel.copy())
        self.warm = self.execution.qacc_warmstart.copy()
