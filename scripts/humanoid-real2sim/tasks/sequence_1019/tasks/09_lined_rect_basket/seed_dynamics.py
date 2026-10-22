"""Generate an optional BFM control warm start in an offline run's CPU model.

No policy training or Stage 4 changes. The feedback tracker supplies a starting
control sequence only; the sampling optimizer remains independent.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.refinement.export import unique_suffix
from robosimstudio.humanoid_real2sim.retarget.motion_fields import sha256
from robosimstudio.humanoid_real2sim.track.launcher import (
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA,
    DEFAULT_ORDER,
    SCALEBRIDGE,
)
from robosimstudio.humanoid_real2sim.track.timeline import FUTURE_OFFSETS
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference-arms", action="store_true")
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()
    out = args.run
    model = mujoco.MjModel.from_binary_path(str(out / "scene.mjb"))
    manifest = json.loads((out / "manifest.json").read_text())
    meta = json.loads(DEFAULT_METADATA.read_text())
    order = json.loads(DEFAULT_ORDER.read_text())
    with np.load(Path(manifest["candidate"]) / "motion.npz") as motion:
        body_indices = [order["robot_body_names"].index(n) for n in meta["selected_body_names"]]
        positions = motion["body_pos_w"][:, body_indices].copy()
        quaternions = motion["body_quat_w"][:, body_indices].copy()
    reference = np.load(out / "reference.npz")
    initial = np.load(out / "initial_state.npz") if (out / "initial_state.npz").exists() else None
    data = mujoco.MjData(model)
    data.qpos[:] = initial["qpos"] if initial is not None else reference["qpos"][0]
    data.qvel[:] = initial["qvel"] if initial is not None else reference["initial_qvel"]
    joints = [unique_suffix(model, "joint", n) for n in meta["joint_names"]]
    bodies = [unique_suffix(model, "body", n) for n in meta["selected_body_names"]]
    actuator_indices = [int(np.flatnonzero(model.actuator_trnid[:, 0] == j)[0]) for j in joints]
    history = deque(maxlen=meta.get("history_buffer_size", 3))
    actions = deque(maxlen=history.maxlen)
    qs, vs, targets = [data.qpos.copy()], [data.qvel.copy()], []
    (out / "tracker_seed_provenance.json").write_text(
        json.dumps(
            {
                "checkpoint": str(DEFAULT_CHECKPOINT),
                "checkpoint_sha256": sha256(DEFAULT_CHECKPOINT),
                "metadata": str(DEFAULT_METADATA),
                "metadata_sha256": sha256(DEFAULT_METADATA),
                "model_sha256": sha256(out / "scene.mjb"),
                "reference_arms": args.reference_arms,
                "policy_training": False,
            },
            indent=2,
        )
    )
    tracker = BFMTracker(
        python=SCALEBRIDGE / ".venv/bin/python",
        scalebridge=SCALEBRIDGE,
        checkpoint=DEFAULT_CHECKPOINT,
        metadata=DEFAULT_METADATA,
    )
    try:
        count = len(reference["qpos"]) if args.frames is None else min(args.frames, len(reference["qpos"]))
        if count < 2:
            raise ValueError("seed requires at least two frames")
        for frame in range(count - 1):
            mujoco.mj_forward(model, data)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bodies[0], velocity, 1)
            current = {
                "root_quat": data.xquat[bodies[0]].copy(),
                "base_ang_vel": velocity[:3].copy(),
                "dof_pos": data.qpos[model.jnt_qposadr[joints]].copy(),
                "dof_vel": data.qvel[model.jnt_dofadr[joints]].copy(),
            }
            if not history:
                for _ in range(history.maxlen):
                    history.append(current)
                    actions.append(np.zeros(len(joints)))
            history.append(current)
            future = np.minimum(frame + np.array(FUTURE_OFFSETS), len(positions) - 1)
            observation = {f"{key}_buffer": np.stack([h[key] for h in history])[None] for key in current}
            observation.update(
                root_pos=data.xpos[bodies[0]][None],
                actions_buffer=np.stack(actions)[None],
                robot_body_pos_w=data.xpos[bodies][None],
                robot_body_quat_w=data.xquat[bodies][None],
                body_pos_w_future=positions[future][None],
                body_quat_w_future=quaternions[future][None],
                future_time_offsets=np.array(FUTURE_OFFSETS)[None, :, None],
            )
            target, action = tracker.infer(observation, 7)
            ctrl = reference["controls"][frame].copy()
            ctrl[actuator_indices] = target
            if args.reference_arms:
                arm_ids = [
                    actuator_indices[i]
                    for i, n in enumerate(meta["joint_names"])
                    if any(part in n for part in ("shoulder", "elbow", "wrist"))
                ]
                ctrl[arm_ids] = reference["controls"][frame, arm_ids]
            ctrl = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = ctrl
            for _ in range(10):
                mujoco.mj_step(model, data)
            targets.append(ctrl)
            qs.append(data.qpos.copy())
            vs.append(data.qvel.copy())
            actions.append(action)
            if frame % 50 == 0:
                print(frame, "root_z", data.qpos[2], flush=True)
        np.savez(
            out / ("tracker_seed_reference_arms.npz" if args.reference_arms else "tracker_seed.npz"),
            controls=np.array(targets),
            qpos=np.array(qs),
            qvel=np.array(vs),
            actuator_joint_names=np.array([model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]),
        )
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
