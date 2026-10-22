"""Drive a G1 in RSS from BFM and cache each achieved physical state.

The policy supplies joint targets only. ``sim.step_teleop`` advances the RSS
solver; neither the worker nor Isaac Sim owns physics. Cached playback loads
the saved robot and object state without advancing physics.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.root_joint import root_joint_index
from robosimstudio.scene.entity import label_index

from .timeline import FUTURE_OFFSETS, ReferenceMotion, RepairTimeline


def _wxyz(xyzw: np.ndarray) -> np.ndarray:
    return np.asarray(xyzw)[..., [3, 0, 1, 2]]


def _angular_velocity_body(xyzw: np.ndarray, angular_world: np.ndarray) -> np.ndarray:
    """MuJoCo free-joint qvel uses the root body frame; Newton body_qd is world frame."""
    x, y, z, w = np.asarray(xyzw, dtype=np.float64)
    vector = np.asarray(angular_world, dtype=np.float64)
    qvec = np.array((x, y, z))
    # Rotate by the conjugate quaternion, without a SciPy dependency.
    t = -2.0 * np.cross(qvec, vector)
    return (vector + w * t - np.cross(qvec, t)).astype(np.float32)


@dataclass
class RecordedStep:
    source_frame: int
    phase: str
    mode: int
    target_q: np.ndarray
    action: np.ndarray
    joint_q: np.ndarray
    joint_qd: np.ndarray
    body_q: np.ndarray
    body_qd: np.ndarray
    particle_q: np.ndarray
    particle_qd: np.ndarray
    hand_override_after: dict[str, float | None]
    hand_recovery_after: dict[str, dict[str, float]] | None
    timeline_after: RepairTimeline
    step_count: int = 0
    recovery_joint_pos: np.ndarray | None = None
    recovery_root_pos: np.ndarray | None = None
    recovery_root_quat: np.ndarray | None = None
    recovery_alpha: float | None = None


class RepairRuntime:
    def __init__(
        self,
        sim: Any,
        motion: ReferenceMotion,
        tracker: Any,
        *,
        hand_open_frame: int | None = None,
        hand_close_frame: int | None = None,
        max_edit_m: float = 0.05,
        original_model: str | Path | None = None,
        motion_order: str | Path | None = None,
        reference_pid_wrists: tuple[str, ...] = (),
        reference_pid_right_arm: bool = False,
    ) -> None:
        if abs(float(sim.pipeline.control_hz) - motion.fps) > 1e-6:
            raise ValueError("RSS control rate must match motion fps (50 Hz)")
        self.sim = sim
        self.motion = motion
        self.tracker = tracker
        self.original_model = Path(original_model) if original_model is not None else None
        self.motion_order = Path(motion_order) if motion_order is not None else None
        self.timeline = RepairTimeline(motion, max_edit_m=max_edit_m)
        meta = tracker.metadata
        if tuple(meta["joint_names"]) != motion.joint_names or tuple(meta["selected_body_names"]) != motion.body_names:
            raise ValueError("tracker metadata and motion orders differ")
        if tuple(meta["action_names"]) != motion.joint_names:
            raise ValueError("tracker action order must match controlled RSS joint order")
        labels = list(sim.model.joint_label)
        q_start = sim.model.joint_q_start.numpy()
        qd_start = sim.model.joint_qd_start.numpy()
        root_index = root_joint_index(sim.model, motion.body_names[0])
        self.root_coord = int(q_start[root_index])
        self.root_dof = int(qd_start[root_index])
        self._scene_start_joint_q = np.asarray(sim._start_joint_q, dtype=np.float32).copy()
        self.anchor_source_frame = 0
        self.joint_coords = np.array(
            [q_start[label_index(labels, name)] for name in motion.joint_names], dtype=np.int32
        )
        self.joint_dofs = np.array([qd_start[label_index(labels, name)] for name in motion.joint_names], dtype=np.int32)
        if any(side not in ("left", "right") for side in reference_pid_wrists):
            raise ValueError("reference PID wrists must be left or right")
        self.reference_pid_joint_names = tuple(
            f"{side}_wrist_{axis}_joint"
            for side in dict.fromkeys(reference_pid_wrists)
            for axis in ("roll", "pitch", "yaw")
        )
        if reference_pid_right_arm:
            self.reference_pid_joint_names += (
                *(f"right_shoulder_{axis}_joint" for axis in ("pitch", "roll", "yaw")),
                "right_elbow_joint",
            )
        missing = set(self.reference_pid_joint_names) - set(motion.joint_names)
        if missing:
            raise ValueError(f"reference PID wrist joints missing from motion: {sorted(missing)}")
        self.reference_pid_joint_indices = np.array(
            [motion.joint_names.index(name) for name in self.reference_pid_joint_names], dtype=np.int32
        )
        self.reference_pid_recovery_indices = (
            np.array([motion.all_joint_names.index(name) for name in self.reference_pid_joint_names], dtype=np.int32)
            if self.reference_pid_joint_names
            else np.empty(0, dtype=np.int32)
        )
        body_labels = list(sim.model.body_label)
        self.body_indices = np.array([label_index(body_labels, name) for name in motion.body_names], dtype=np.int32)
        self.root_body = int(self.body_indices[0])
        self.hand_coords = {
            side: {
                name: int(q_start[label_index(labels, name)])
                for name in motion.all_joint_names
                if name.startswith(f"{side}_hand_")
            }
            for side in ("left", "right")
        }
        self.hand_profiles: dict[str, dict[str, dict[str, float]]] = {}
        if hand_open_frame is not None and hand_close_frame is not None:
            if motion.all_joint_pos is None:
                raise ValueError("hand profiles require full joint motion")
            for frame in (hand_open_frame, hand_close_frame):
                if not 0 <= frame < motion.frames:
                    raise ValueError(f"hand profile frame {frame} outside motion")
            for side in ("left", "right"):
                self.hand_profiles[side] = {
                    "open": {
                        name: float(motion.all_joint_pos[hand_open_frame, motion.all_joint_names.index(name)])
                        for name in self.hand_coords[side]
                    },
                    "close": {
                        name: float(motion.all_joint_pos[hand_close_frame, motion.all_joint_names.index(name)])
                        for name in self.hand_coords[side]
                    },
                }
            # The source clip only closes the right hand, so its sampled left
            # pose cannot serve as a manual close command. Give both sliders
            # a firm near-limit target in the flexion direction.
            lower = sim.model.joint_limit_lower.numpy()
            upper = sim.model.joint_limit_upper.numpy()
            for side in ("left", "right"):
                for name in self.hand_coords[side]:
                    dof = int(qd_start[label_index(labels, name)])
                    lo, hi = float(lower[dof]), float(upper[dof])
                    self.hand_profiles[side]["open"][name] = float(np.clip(0.0, lo, hi))
                    if "thumb_0" in name:
                        closed = 0.35 if side == "right" else -0.35
                    elif "thumb_1" in name:
                        closed = lo + 0.05 * (hi - lo)
                    elif abs(lo) < abs(hi):
                        closed = hi - 0.05 * (hi - lo)
                    else:
                        closed = lo + 0.05 * (hi - lo)
                    self.hand_profiles[side]["close"][name] = float(np.clip(closed, lo, hi))
        else:
            # Dex3's source clip can hold a fixed grasp for every frame. In
            # that case derive a useful one-dimensional open/close profile
            # from the finger joint limits rather than its zero motion range.
            lower = sim.model.joint_limit_lower.numpy()
            upper = sim.model.joint_limit_upper.numpy()
            for side in ("left", "right"):
                profiles = {"open": {}, "close": {}}
                for name, coord in self.hand_coords[side].items():
                    dof = int(qd_start[label_index(labels, name)])
                    lo, hi = float(lower[dof]), float(upper[dof])
                    neutral = float(self._scene_start_joint_q[coord])
                    if "thumb_0" in name:
                        profiles["open"][name] = neutral
                        profiles["close"][name] = neutral
                    else:
                        profiles["open"][name] = float(np.clip(0.0, lo, hi))
                        profiles["close"][name] = (
                            lo
                            if side == "left" and "thumb_1" in name
                            else (
                                hi
                                if side == "left" and "thumb_2" in name
                                else lo
                                if side == "left"
                                else hi
                                if "middle_" in name or "index_" in name
                                else lo
                            )
                        )
                self.hand_profiles[side] = profiles
        self.hand_override: dict[str, float | None] = {"left": None, "right": None}
        self._hand_recovery_start: dict[str, dict[str, float]] | None = None
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=int(meta.get("history_buffer_size", 3)))
        self.action_history: deque[np.ndarray] = deque(maxlen=self.history.maxlen)
        self.records: list[RecordedStep] = []
        self.cursor = 0
        self.branches: list[list[RecordedStep]] = []
        self._pending: tuple[int, str, int, np.ndarray, np.ndarray, tuple | None] | None = None
        self._seed_reference_velocity(0)
        self._initial_state = self._snapshot()
        self._initial_physics = self._physics_snapshot()
        self._initial_target = self.sim.pipeline.snapshot.joint_target_q.copy()
        self._initial_step_count = self.sim.pipeline.step_count
        self._seed_history()

    def _physics_snapshot(self) -> dict[str, np.ndarray]:
        snap = self.sim.pipeline.snapshot
        return {
            name: getattr(snap, name).copy()
            for name in (
                "joint_q",
                "joint_qd",
                "body_q",
                "body_qd",
                "particle_q",
                "particle_qd",
            )
        }

    def _restore_physics(self, state: dict[str, np.ndarray], target: np.ndarray, step_count: int) -> None:
        pipeline = self.sim.pipeline
        for buffer in (pipeline.state_0, pipeline.state_1):
            for name, value in state.items():
                field = getattr(buffer, name)
                if field is not None:
                    field.assign(value)
        self.sim.control.joint_target_q.assign(target)
        pipeline.step_count = step_count
        pipeline.mark_dirty()

    def seek_cached(self, slot: int) -> None:
        """Display a contiguous cached state without advancing the physics solver."""
        if self._pending is not None:
            raise RuntimeError("cannot load a state between command and physics step")
        if not 0 <= slot <= len(self.records):
            raise ValueError("slot is not in the active cache")
        if slot:
            record = self.records[slot - 1]
            state = {name: getattr(record, name) for name in self._initial_physics}
            self._restore_physics(state, record.target_q, record.step_count)
            self.timeline = copy.deepcopy(record.timeline_after)
            self.hand_override = dict(record.hand_override_after)
            self._hand_recovery_start = copy.deepcopy(record.hand_recovery_after)
        else:
            self._restore_physics(self._initial_physics, self._initial_target, self._initial_step_count)
            self.timeline = RepairTimeline(self.motion, max_edit_m=self.timeline.max_edit_m)
            self.timeline.source_frame = self.anchor_source_frame
            self.hand_override = {"left": None, "right": None}
            self._hand_recovery_start = None
        self.cursor = slot
        self._seed_history()
        for record in self.records[:slot][-self.history.maxlen :]:
            self.history.append(self._state_from_record(record))
            self.action_history.append(record.action.copy())

    def branch_from_here(self) -> None:
        """Invalidate only the cache suffix that no longer follows this state."""
        if self.cursor < len(self.records):
            self.branches.append(self.records.copy())
            del self.records[self.cursor :]
        # The default MuJoCo reset also restores model qpos. A branch must
        # preserve the cached physical state and clear only solver internals.
        self.sim.pipeline.reset_solvers(flags=0)

    def _seed_reference_velocity(self, frame: int) -> None:
        """Restore the mid-clip velocity after Sim.reset() clears all joint rates."""
        if (
            self.motion.all_joint_vel is None
            or self.motion.root_lin_vel_w is None
            or self.motion.root_ang_vel_w is None
        ):
            return
        import newton
        import warp as wp

        qd = self.sim.pipeline.snapshot.joint_qd.astype(np.float32).copy()
        qd[self.root_dof : self.root_dof + 3] = self.motion.root_lin_vel_w[frame]
        qd[self.root_dof + 3 : self.root_dof + 6] = self.motion.root_ang_vel_w[frame]
        for name, dof in zip(self.motion.joint_names, self.joint_dofs, strict=True):
            qd[dof] = self.motion.all_joint_vel[frame, self.motion.all_joint_names.index(name)]
        for side in ("left", "right"):
            for name in self.hand_coords[side]:
                dof = int(self.sim.model.joint_qd_start.numpy()[label_index(list(self.sim.model.joint_label), name)])
                qd[dof] = self.motion.all_joint_vel[frame, self.motion.all_joint_names.index(name)]
        pipeline = self.sim.pipeline
        pipeline.state_0.joint_qd.assign(qd)
        pipeline.state_1.joint_qd.assign(qd)
        newton.eval_fk(self.sim.model, pipeline.state_0.joint_q, pipeline.state_0.joint_qd, pipeline.state_0)
        wp.copy(pipeline.state_1.body_qd, pipeline.state_0.body_qd)
        pipeline.mark_dirty()

    def seek_reference(self, frame: int) -> None:
        """Pose the robot at a source frame and start a new physics branch there.

        The scene's objects return to their authored initial state. Solver and
        tracker history are reset, so Play will infer from the displayed pose.
        Recorded physics is replayed through ``rewind`` instead of this path.
        """
        if self._pending is not None:
            raise RuntimeError("cannot seek between tracker command and physics step")
        if not 0 <= frame < self.motion.frames:
            raise ValueError(f"reference frame {frame} outside [0, {self.motion.frames - 1}]")
        q = self._scene_start_joint_q.copy()
        root = self.root_coord
        q[root : root + 3] = self.motion.body_pos_w[frame, 0]
        q[root + 3 : root + 7] = self.motion.body_quat_w[frame, 0, [1, 2, 3, 0]]
        q[self.joint_coords] = self.motion.joint_pos[frame]
        if self.motion.all_joint_pos is not None:
            for side in ("left", "right"):
                for name, coord in self.hand_coords[side].items():
                    q[coord] = self.motion.all_joint_pos[frame, self.motion.all_joint_names.index(name)]
        self.sim._start_joint_q = q
        self.sim.reset()
        self._seed_reference_velocity(frame)
        targets = self.sim.pipeline.snapshot.joint_target_q.astype(np.float32).copy()
        targets[self.joint_coords] = self.motion.joint_pos[frame]
        for side in ("left", "right"):
            for coord in self.hand_coords[side].values():
                targets[coord] = q[coord]
        self.sim.control.joint_target_q.assign(targets)
        self.anchor_source_frame = frame
        self.timeline = RepairTimeline(self.motion, max_edit_m=self.timeline.max_edit_m)
        self.timeline.source_frame = frame
        self.hand_override = {"left": None, "right": None}
        self._hand_recovery_start = None
        self.records.clear()
        self.cursor = 0
        self.branches.clear()
        self._initial_state = self._snapshot()
        self._initial_physics = self._physics_snapshot()
        self._initial_target = self.sim.pipeline.snapshot.joint_target_q.copy()
        self._initial_step_count = self.sim.pipeline.step_count
        self._seed_history()

    def _snapshot(self) -> dict[str, np.ndarray]:
        snap = self.sim.pipeline.snapshot
        root = snap.body_q[self.root_body]
        return {
            "root_pos": root[:3].astype(np.float32).copy(),
            "root_quat": _wxyz(root[3:]).astype(np.float32).copy(),
            "base_ang_vel": _angular_velocity_body(root[3:], snap.body_qd[self.root_body, 3:6]),
            "dof_pos": snap.joint_q[self.joint_coords].astype(np.float32).copy(),
            "dof_vel": snap.joint_qd[self.joint_dofs].astype(np.float32).copy(),
            "body_pos": snap.body_q[self.body_indices, :3].astype(np.float32).copy(),
            "body_quat": _wxyz(snap.body_q[self.body_indices, 3:]).astype(np.float32).copy(),
        }

    def _seed_history(self) -> None:
        self.history.clear()
        self.action_history.clear()
        for _ in range(self.history.maxlen):
            self.history.append({k: v.copy() for k, v in self._initial_state.items()})
            self.action_history.append(np.zeros(len(self.motion.joint_names), dtype=np.float32))

    def _state_from_record(self, record: RecordedStep) -> dict[str, np.ndarray]:
        root = record.body_q[self.root_body]
        return {
            "root_pos": root[:3].astype(np.float32).copy(),
            "root_quat": _wxyz(root[3:]).astype(np.float32).copy(),
            "base_ang_vel": _angular_velocity_body(root[3:], record.body_qd[self.root_body, 3:6]),
            "dof_pos": record.joint_q[self.joint_coords].astype(np.float32).copy(),
            "dof_vel": record.joint_qd[self.joint_dofs].astype(np.float32).copy(),
            "body_pos": record.body_q[self.body_indices, :3].astype(np.float32).copy(),
            "body_quat": _wxyz(record.body_q[self.body_indices, 3:]).astype(np.float32).copy(),
        }

    def _observation(self) -> dict[str, np.ndarray]:
        current = self._snapshot()
        self.history.append(current)
        target_pos, target_quat = self.timeline.targets(
            achieved_pos=current["body_pos"], achieved_quat_wxyz=current["body_quat"]
        )
        return {
            "root_pos": current["root_pos"][None],
            "root_quat_buffer": np.stack([h["root_quat"] for h in self.history])[None],
            "base_ang_vel_buffer": np.stack([h["base_ang_vel"] for h in self.history])[None],
            "dof_pos_buffer": np.stack([h["dof_pos"] for h in self.history])[None],
            "dof_vel_buffer": np.stack([h["dof_vel"] for h in self.history])[None],
            "actions_buffer": np.stack(tuple(self.action_history))[None],
            "robot_body_pos_w": current["body_pos"][None],
            "robot_body_quat_w": current["body_quat"][None],
            "body_pos_w_future": target_pos[None].astype(np.float32),
            "body_quat_w_future": target_quat[None].astype(np.float32),
            "future_time_offsets": np.asarray(FUTURE_OFFSETS, dtype=np.float32)[None, :, None],
        }

    def set_hand_amount(self, side: str, amount: float) -> None:
        if side not in self.hand_profiles or not np.isfinite(amount) or not 0 <= amount <= 1:
            raise ValueError("hand closure must be within [0, 1]")
        self.hand_override[side] = float(amount)

    def set_hand(self, side: str, state: str) -> None:
        if state not in ("open", "close"):
            raise ValueError("hand state must be open or close")
        self.set_hand_amount(side, 0.0 if state == "open" else 1.0)

    def begin_takeover(self) -> None:
        state = self._snapshot()
        self.timeline.take_over(state["body_pos"], state["body_quat"])

    def begin_recovery(self, reference_frame: int, duration_frames: int = 50) -> None:
        """Track an FK-consistent full-body interpolation, then resume source time."""
        if not 0 <= duration_frames <= 500:
            raise ValueError("recovery duration must be between 0 and 10 seconds")
        if self.motion.all_joint_pos is None or not self.motion.all_joint_names or self.original_model is None:
            raise RuntimeError("full-body recovery needs the original MJCF and joint motion")
        import mujoco
        from scipy.spatial.transform import Rotation, Slerp

        state = self._snapshot()
        model = mujoco.MjModel.from_xml_path(str(self.original_model))
        data = mujoco.MjData(model)
        snap = self.sim.pipeline.snapshot
        labels = list(self.sim.model.joint_label)
        starts = self.sim.model.joint_q_start.numpy()
        live_joint = np.asarray(
            [snap.joint_q[int(starts[label_index(labels, name)])] for name in self.motion.all_joint_names],
            dtype=np.float64,
        )
        target_joint = self.motion.all_joint_pos[reference_frame]
        live_root = state["root_pos"].astype(np.float64)
        target_root = self.motion.body_pos_w[reference_frame, 0]
        live_quat = state["root_quat"].astype(np.float64)
        target_quat = self.motion.body_quat_w[reference_frame, 0]
        if np.dot(live_quat, target_quat) < 0:
            target_quat = -target_quat
        fraction = np.array([1.0]) if duration_frames == 0 else np.linspace(0.0, 1.0, duration_frames + 1)
        eased = fraction * fraction * (3.0 - 2.0 * fraction)
        joint_path = live_joint[None] + eased[:, None] * (target_joint - live_joint)[None]
        held_hand_targets: dict[str, dict[str, float]] = {}
        current_targets = self.sim.pipeline.snapshot.joint_target_q
        for side, coords in self.hand_coords.items():
            held_hand_targets[side] = {}
            for name, coord in coords.items():
                selected = self.hand_override[side]
                if selected is None:
                    held = float(current_targets[coord])
                else:
                    opened = self.hand_profiles[side]["open"][name]
                    closed = self.hand_profiles[side]["close"][name]
                    held = float((1 - selected) * opened + selected * closed)
                held_hand_targets[side][name] = held
                column = self.motion.all_joint_names.index(name)
                joint_path[1:, column] = held
        root_path = live_root[None] + eased[:, None] * (target_root - live_root)[None]
        rotation = Rotation.from_quat(np.stack((live_quat[[1, 2, 3, 0]], target_quat[[1, 2, 3, 0]])))
        quat_xyzw = Slerp([0.0, 1.0], rotation)(eased).as_quat()
        quat_path = quat_xyzw[:, [3, 0, 1, 2]]
        joint_addresses = [int(model.joint(name).qposadr[0]) for name in self.motion.all_joint_names]
        body_ids = [int(model.body(name).id) for name in self.motion.body_names]
        body_pos_path = np.empty((len(eased), len(body_ids), 3), dtype=np.float64)
        body_quat_path = np.empty((len(eased), len(body_ids), 4), dtype=np.float64)
        for index in range(len(eased)):
            data.qpos[:3] = root_path[index]
            data.qpos[3:7] = quat_path[index]
            data.qpos[joint_addresses] = joint_path[index]
            mujoco.mj_forward(model, data)
            body_pos_path[index] = data.xpos[body_ids]
            body_quat_path[index] = data.xquat[body_ids]
        if duration_frames:
            body_pos_path[0] = state["body_pos"]
            body_quat_path[0] = state["body_quat"]
        self.timeline.begin_recovery(
            state["body_pos"],
            state["body_quat"],
            reference_frame,
            body_pos_path=body_pos_path,
            body_quat_path=body_quat_path,
            joint_pos_path=joint_path,
            root_pos_path=root_path,
            root_quat_path=quat_path,
        )
        self._hand_recovery_start = held_hand_targets if self.timeline.phase == "recovery" else None

    def prepare(self) -> None:
        if self._pending is not None:
            raise RuntimeError("previous RSS physics step has not been recorded")
        if self.cursor != len(self.records):
            raise RuntimeError("cached future exists; load it or begin a new repair")
        phase = self.timeline.phase
        source = self.timeline.source_frame
        recovery_pose = self.timeline.recovery_full_pose() if phase == "recovery" else None
        mode = 4 if phase == "takeover" else 7
        target, action = self.tracker.infer(self._observation(), mode)
        current_targets = self.sim.pipeline.snapshot.joint_target_q.astype(np.float32).copy()
        if phase in ("takeover", "recovery"):
            # Mode 4 can disagree sharply with the mode-7 joint command even
            # when the requested link speed is zero. Limit the command change
            # at the mode boundary; the physics state itself is never set.
            previous = current_targets[self.joint_coords]
            transition_step = self.timeline.takeover_steps if phase == "takeover" else self.timeline._recovery_step
            max_step = 0.04 + 0.08 * min(1.0, transition_step / 10.0)
            target = previous + np.clip(target - previous, -max_step, max_step)
        current_targets[self.joint_coords] = target
        if self.reference_pid_joint_indices.size:
            reference = (
                recovery_pose[0][self.reference_pid_recovery_indices]
                if recovery_pose is not None
                else (self.motion.joint_pos[source, self.reference_pid_joint_indices])
            )
            for index, name, value in zip(
                self.reference_pid_joint_indices, self.reference_pid_joint_names, reference, strict=True
            ):
                side = name.split("_", 1)[0]
                wrist_link = f"{side}_wrist_yaw_link"
                if (
                    phase == "takeover"
                    and wrist_link in self.timeline.link_modes
                    and (self.timeline.link_modes[wrist_link] != "reference")
                ):
                    # A repaired wrist's orientation must remain under VR-5.
                    continue
                current_targets[self.joint_coords[index]] = value
        for side, selected in self.hand_override.items():
            if phase == "recovery" and self._hand_recovery_start is not None:
                for name, coord in self.hand_coords[side].items():
                    current_targets[coord] = self._hand_recovery_start[side][name]
                continue
            if selected is None:
                if self.motion.all_joint_pos is None:
                    continue
                for name, coord in self.hand_coords[side].items():
                    current_targets[coord] = self.motion.all_joint_pos[source, self.motion.all_joint_names.index(name)]
            else:
                for name, opened in self.hand_profiles[side]["open"].items():
                    closed = self.hand_profiles[side]["close"][name]
                    coord = self.hand_coords[side][name]
                    desired = (1 - selected) * opened + selected * closed
                    previous = current_targets[coord]
                    current_targets[coord] = previous + np.clip(desired - previous, -0.08, 0.08)
        self.sim.control.joint_target_q.assign(current_targets)
        self.action_history.append(action.copy())
        self._pending = (source, phase, mode, current_targets, action, recovery_pose)

    def record_after_step(self) -> None:
        if self._pending is None:
            raise RuntimeError("RSS advanced without a tracker command")
        source, phase, mode, target, action, recovery_pose = self._pending
        self._pending = None
        snap = self.sim.pipeline.snapshot
        self.timeline.advance()
        if phase == "recovery" and self.timeline.phase == "reference":
            self._hand_recovery_start = None
        self.records.append(
            RecordedStep(
                source,
                phase,
                mode,
                target.copy(),
                action.copy(),
                snap.joint_q.copy(),
                snap.joint_qd.copy(),
                snap.body_q.copy(),
                snap.body_qd.copy(),
                snap.particle_q.copy(),
                snap.particle_qd.copy(),
                dict(self.hand_override),
                copy.deepcopy(self._hand_recovery_start),
                copy.deepcopy(self.timeline),
                self.sim.pipeline.step_count,
                None if recovery_pose is None else recovery_pose[0],
                None if recovery_pose is None else recovery_pose[1],
                None if recovery_pose is None else recovery_pose[2],
                None if recovery_pose is None else recovery_pose[3],
            )
        )
        self.cursor = len(self.records)

    def rewind(self, keep_steps: int) -> float:
        """Load a cached slot without stepping physics; retain its future."""
        self.seek_cached(keep_steps)
        return 0.0

    def restore_previous_branch(self) -> float:
        if not self.branches:
            raise ValueError("no previous branch to restore")
        branch = self.branches.pop()
        self.records = branch
        self.seek_cached(len(branch))
        return 0.0

    def load_saved_rollout(self, path: str | Path, *, repaired_link: str | None = None) -> int:
        """Restore a saved cache into a fresh editor without re-simulating it."""
        if self.records:
            raise RuntimeError("load a saved rollout only into a fresh runtime")
        from .timeline import FIVE_POINTS

        if repaired_link is not None and repaired_link not in FIVE_POINTS:
            raise ValueError(f"unknown repaired link {repaired_link!r}")
        with np.load(path, allow_pickle=False) as saved:
            saved_pid_joints = (
                tuple(saved["reference_pid_joint_names"].tolist()) if "reference_pid_joint_names" in saved else ()
            )
            if saved_pid_joints != self.reference_pid_joint_names:
                raise ValueError("saved rollout uses a different wrist reference PID configuration")
            count = len(saved["source_frame"])
            if count == 0:
                raise ValueError("saved rollout contains no physical states")
            if saved["achieved_joint_q"].shape[1:] != self._initial_physics["joint_q"].shape:
                raise ValueError("saved joint state does not match this RSS scene")
            if saved["achieved_body_q"].shape[1:] != self._initial_physics["body_q"].shape:
                raise ValueError("saved body state does not match this RSS scene")
            phases = saved["phase"]
            if any(phase not in ("reference", "takeover") for phase in phases):
                raise ValueError("resume currently requires a saved reference/takeover rollout")
            timeline = RepairTimeline(self.motion, max_edit_m=self.timeline.max_edit_m)
            for index in range(count):
                phase = str(phases[index])
                source = int(saved["source_frame"][index])
                if phase == "takeover" and timeline.phase != "takeover":
                    previous = self.records[-1].body_q if self.records else self._initial_physics["body_q"]
                    selected = previous[self.body_indices]
                    timeline.source_frame = source
                    timeline.take_over(selected[:, :3], _wxyz(selected[:, 3:]))
                    if repaired_link is not None:
                        link = self.motion.body_names.index(repaired_link)
                        timeline.hold_pose(repaired_link, selected[link, :3], _wxyz(selected[link, 3:]))
                timeline.source_frame = min(source + 1, self.motion.frames - 1)
                if phase == "takeover":
                    timeline.takeover_steps += 1
                    if repaired_link is not None and timeline._target_pos is not None:
                        link = self.motion.body_names.index(repaired_link)
                        achieved = saved["achieved_body_q"][index, self.body_indices[link]]
                        timeline.hold_pose(repaired_link, achieved[:3], _wxyz(achieved[3:]))
                self.records.append(
                    RecordedStep(
                        source,
                        phase,
                        int(saved["mode"][index]),
                        saved["commanded_joint_target_q"][index].copy(),
                        saved["tracker_action"][index].copy(),
                        saved["achieved_joint_q"][index].copy(),
                        saved["achieved_joint_qd"][index].copy(),
                        saved["achieved_body_q"][index].copy(),
                        saved["achieved_body_qd"][index].copy(),
                        saved["achieved_particle_q"][index].copy(),
                        saved["achieved_particle_qd"][index].copy(),
                        {"left": None, "right": None},
                        None,
                        copy.deepcopy(timeline),
                        self._initial_step_count + index + 1,
                    )
                )
        self.seek_cached(len(self.records))
        self.sim.pipeline.reset_solvers(flags=0)
        return len(self.records)

    def save(self, directory: str | Path) -> Path:
        """Save original-source map, commands and achieved motion as a candidate.

        This is deliberately not an approved ScaleTrack seven-field NPZ. A
        source/achieved splice requires seam optimization and replay review.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        if not self.records:
            raise ValueError("nothing has been simulated")
        np.savez_compressed(
            out / "rollout.npz",
            source_frame=np.asarray([r.source_frame for r in self.records], dtype=np.int32),
            phase=np.asarray([r.phase for r in self.records]),
            mode=np.asarray([r.mode for r in self.records], dtype=np.int8),
            reference_pid_joint_names=np.asarray(self.reference_pid_joint_names),
            commanded_joint_target_q=np.stack([r.target_q for r in self.records]),
            tracker_action=np.stack([r.action for r in self.records]),
            achieved_joint_q=np.stack([r.joint_q for r in self.records]),
            achieved_joint_qd=np.stack([r.joint_qd for r in self.records]),
            achieved_body_q=np.stack([r.body_q for r in self.records]),
            achieved_body_qd=np.stack([r.body_qd for r in self.records]),
            achieved_particle_q=np.stack([r.particle_q for r in self.records]),
            achieved_particle_qd=np.stack([r.particle_qd for r in self.records]),
            recovery_joint_pos=np.stack(
                [
                    np.full(len(self.motion.all_joint_names), np.nan)
                    if r.recovery_joint_pos is None
                    else r.recovery_joint_pos
                    for r in self.records
                ]
            ),
            recovery_root_pos=np.stack(
                [np.full(3, np.nan) if r.recovery_root_pos is None else r.recovery_root_pos for r in self.records]
            ),
            recovery_root_quat=np.stack(
                [np.full(4, np.nan) if r.recovery_root_quat is None else r.recovery_root_quat for r in self.records]
            ),
            recovery_alpha=np.asarray(
                [np.nan if r.recovery_alpha is None else r.recovery_alpha for r in self.records], dtype=np.float32
            ),
        )
        candidate_status = "not_exported"
        candidate_error = "requires original MJCF, 43-joint motion, and completed recovery"
        if self.original_model is not None and self.motion_order is not None:
            from .export import export_candidate

            try:
                labels = list(self.sim.model.joint_label)
                starts = self.sim.model.joint_q_start.numpy()
                all_coords = {name: int(starts[label_index(labels, name)]) for name in self.motion.all_joint_names}
                export_candidate(
                    self.records,
                    self.motion,
                    model_path=self.original_model,
                    order_path=self.motion_order,
                    joint_coord_by_name=all_coords,
                    root_body=self.root_body,
                    out=out / "candidate_motion.npz",
                )
                candidate_status = "candidate_unreviewed"
                candidate_error = ""
            except (ValueError, KeyError) as exc:
                candidate_error = str(exc)
        segments = []
        seen_takeover = False
        for index, record in enumerate(self.records):
            if record.phase == "takeover":
                seen_takeover = True
            kind = (
                "reference_before"
                if record.phase == "reference" and not seen_takeover
                else "reference_after"
                if record.phase == "reference"
                else "repair_live"
                if record.phase == "takeover"
                else "recovery_interpolation"
            )
            if segments and segments[-1]["kind"] == kind:
                segments[-1]["stop_output_frame_exclusive"] = index + 1
            else:
                segments.append({"kind": kind, "start_output_frame": index, "stop_output_frame_exclusive": index + 1})
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "candidate_unreviewed"
                    if candidate_status == "candidate_unreviewed"
                    else "rollout_unreviewed",
                    "physics": "robosimstudio",
                    "source_fps": self.motion.fps,
                    "anchor_source_frame": self.anchor_source_frame,
                    "output_frames": len(self.records),
                    "segments": segments,
                    "recovery_endpoint_source_frame": self.timeline._recovery_anchor
                    if self.timeline._recovery_root_pos is not None
                    else None,
                    "recovery_duration_frames": self.timeline.recovery_frames
                    if self.timeline._recovery_root_pos is not None
                    else None,
                    "joint_names": self.motion.joint_names,
                    "selected_body_names": self.motion.body_names,
                    "reference_pid_joint_names": self.reference_pid_joint_names,
                    "candidate_motion_status": candidate_status,
                    "candidate_motion_error": candidate_error,
                    "note": "achieved full-body repair samples and original reference are separate; no automatic training-data promotion",
                },
                indent=2,
            )
            + "\n"
        )
        return out
