"""A source-time clock and reversible five-point edits, independent of physics.

The editor advances this clock only after an RSS physics step. A stopped edit
therefore freezes both the reference target and the physical state. The original
motion remains immutable; recorded full-body states form a separate candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIVE_POINTS = (
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
FUTURE_OFFSETS = (0, 1, 2, 3, 4, 32)


def _name(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class ReferenceMotion:
    joint_pos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    fps: int = 50
    all_joint_pos: np.ndarray | None = None
    all_joint_names: tuple[str, ...] = ()
    all_joint_vel: np.ndarray | None = None
    root_lin_vel_w: np.ndarray | None = None
    root_ang_vel_w: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        motion_order: str | Path | None = None,
    ) -> ReferenceMotion:
        with np.load(path, allow_pickle=False) as data:
            fps = int(np.asarray(data["fps"]).item())
            if fps != 50:
                raise ValueError(f"tracker requires 50 Hz motion, got {fps}")
            order = json.loads(Path(motion_order).read_text()) if motion_order is not None else {}
            joints = (
                tuple(_name(x) for x in data["robot_joint_names"])
                if "robot_joint_names" in data
                else tuple(order.get("robot_joint_names", joint_names))
            )
            bodies = (
                tuple(_name(x) for x in data["robot_body_names"])
                if "robot_body_names" in data
                else tuple(order.get("robot_body_names", body_names))
            )
            # The supplied ScaleTrack order lists 43/50 Dex3 entries. The BFM
            # cartwheel sample is the same model without Dex3 (first 29/30).
            if "robot_joint_names" not in data and len(joints) == 43 and data["joint_pos"].shape[1] == 29:
                joints = joints[:29]
            if "robot_body_names" not in data and len(bodies) == 50 and data["body_pos_w"].shape[1] == 30:
                bodies = bodies[:30]
            if len(joints) != data["joint_pos"].shape[1] or len(bodies) != data["body_pos_w"].shape[1]:
                raise ValueError("unnamed motion order does not match dimensions; supply motion_order.json")
            missing_joints = set(joint_names) - set(joints)
            missing_bodies = set(body_names) - set(bodies)
            if missing_joints or missing_bodies:
                raise ValueError(
                    f"motion is missing joints {sorted(missing_joints)} or bodies {sorted(missing_bodies)}"
                )
            all_q = np.asarray(data["joint_pos"], dtype=np.float64).copy()
            all_qd = np.asarray(data["joint_vel"], dtype=np.float64).copy() if "joint_vel" in data else None
            q = all_q[:, [joints.index(n) for n in joint_names]].copy()
            p = np.asarray(data["body_pos_w"], dtype=np.float64)[:, [bodies.index(n) for n in body_names]].copy()
            r = np.asarray(data["body_quat_w"], dtype=np.float64)[:, [bodies.index(n) for n in body_names]].copy()
            root_v = (
                np.asarray(data["body_lin_vel_w"], dtype=np.float64)[:, bodies.index(body_names[0])].copy()
                if "body_lin_vel_w" in data
                else None
            )
            root_w = (
                np.asarray(data["body_ang_vel_w"], dtype=np.float64)[:, bodies.index(body_names[0])].copy()
                if "body_ang_vel_w" in data
                else None
            )
        if q.ndim != 2 or p.shape != (len(q), len(body_names), 3) or r.shape != (len(q), len(body_names), 4):
            raise ValueError("inconsistent joint/body motion shapes")
        if not (np.isfinite(q).all() and np.isfinite(p).all() and np.isfinite(r).all()):
            raise ValueError("motion contains non-finite values")
        return cls(q, p, r, joint_names, body_names, fps, all_q, joints, all_qd, root_v, root_w)

    @property
    def frames(self) -> int:
        return len(self.joint_pos)


class RepairTimeline:
    """Reference targets and the source-to-simulation frame map.

    ``velocity`` is a world-frame direction/speed in m/s for one of five body
    targets. The 5 cm budget is measured from the takeover pose, not added anew
    on each direction change. A recovery blends measured body poses into the
    original future reference while source time resumes.
    """

    def __init__(self, motion: ReferenceMotion, *, max_edit_m: float = 0.05, recovery_frames: int = 25):
        self.motion = motion
        self.max_edit_m = float(max_edit_m)
        self.recovery_frames = int(recovery_frames)
        if self.max_edit_m <= 0 or self.recovery_frames < 2:
            raise ValueError("max_edit_m must be positive and recovery_frames >= 2")
        self.source_frame = 0
        self.phase = "reference"
        self.velocity = np.zeros(3, dtype=np.float64)
        self.point = FIVE_POINTS[2]
        self.link_modes = dict.fromkeys(FIVE_POINTS, "reference")
        self.link_velocities = {name: np.zeros(3, dtype=np.float64) for name in FIVE_POINTS}
        self.link_speeds = dict.fromkeys(FIVE_POINTS, 0.30)
        self.link_angular_speeds = dict.fromkeys(FIVE_POINTS, 1.0)
        self.goal_controlled: set[str] = set()
        self.ever_repaired: set[str] = set()
        self._base_pos: np.ndarray | None = None
        self._target_pos: np.ndarray | None = None
        self._target_quat: np.ndarray | None = None
        self._goal_pos: np.ndarray | None = None
        self._goal_quat: np.ndarray | None = None
        self._goal_next_pos: dict[str, np.ndarray] = {}
        self._goal_next_quat: dict[str, np.ndarray] = {}
        self._takeover_frame = 0
        self.takeover_steps = 0
        self._recovery_start: np.ndarray | None = None
        self._recovery_quat: np.ndarray | None = None
        self._recovery_step = 0
        self._recovery_anchor = 0
        self._recovery_body_pos: np.ndarray | None = None
        self._recovery_body_quat: np.ndarray | None = None
        self._recovery_joint_pos: np.ndarray | None = None
        self._recovery_root_pos: np.ndarray | None = None
        self._recovery_root_quat: np.ndarray | None = None

    def take_over(self, achieved_pos: np.ndarray, achieved_quat_wxyz: np.ndarray) -> None:
        if self.phase != "reference":
            raise RuntimeError("takeover requires reference playback")
        self.phase = "takeover"
        self._takeover_frame = self.source_frame
        self.takeover_steps = 0
        self._base_pos = np.asarray(achieved_pos, dtype=np.float64).copy()
        self._target_quat = np.asarray(achieved_quat_wxyz, dtype=np.float64).copy()
        if (
            self._base_pos.shape != self.motion.body_pos_w[0].shape
            or self._target_quat.shape != self.motion.body_quat_w[0].shape
        ):
            raise ValueError("achieved body poses must match selected body order")
        self._target_pos = self._base_pos.copy()
        self._goal_pos = self._target_pos.copy()
        self._goal_quat = self._target_quat.copy()
        self._goal_next_pos.clear()
        self._goal_next_quat.clear()
        self.velocity[:] = 0
        self.link_modes = dict.fromkeys(FIVE_POINTS, "reference")
        self.link_velocities = {name: np.zeros(3, dtype=np.float64) for name in FIVE_POINTS}
        self.goal_controlled.clear()
        self.ever_repaired.clear()

    def set_goal(
        self, point: str, position: np.ndarray, quat_wxyz: np.ndarray, speed_mps: float, angular_speed_rps: float
    ) -> None:
        """Set a fixed world-frame 6D goal; each observation supplies its direction."""
        if self.phase != "takeover" or point not in FIVE_POINTS:
            raise RuntimeError("6D goal requires five-point takeover")
        pos = np.asarray(position, dtype=np.float64)
        quat = np.asarray(quat_wxyz, dtype=np.float64)
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all():
            raise ValueError("goal needs finite xyz and wxyz quaternion")
        if abs(np.linalg.norm(quat) - 1) > 1e-3:
            raise ValueError("goal quaternion must be normalized")
        if not 0 <= speed_mps <= 0.6 or not 0 <= angular_speed_rps <= 3.0:
            raise ValueError("goal speeds are outside supported ranges")
        assert self._base_pos is not None and self._goal_pos is not None and self._goal_quat is not None
        j = self.motion.body_names.index(point)
        delta = pos - self._base_pos[j]
        length = np.linalg.norm(delta)
        if length > self.max_edit_m:
            pos = self._base_pos[j] + delta * self.max_edit_m / length
        self._goal_pos[j] = pos
        self._goal_quat[j] = quat
        self.link_speeds[point] = float(speed_mps)
        self.link_angular_speeds[point] = float(angular_speed_rps)
        self.link_modes[point] = "move"
        self.goal_controlled.add(point)
        self.ever_repaired.add(point)

    def hold_pose(self, point: str, position: np.ndarray, quat_wxyz: np.ndarray) -> None:
        if self.phase != "takeover" or point not in FIVE_POINTS:
            raise RuntimeError("hold requires five-point takeover")
        assert self._target_pos is not None and self._target_quat is not None
        assert self._goal_pos is not None and self._goal_quat is not None
        j = self.motion.body_names.index(point)
        self._target_pos[j] = np.asarray(position, dtype=np.float64)
        self._target_quat[j] = np.asarray(quat_wxyz, dtype=np.float64)
        self._goal_pos[j] = self._target_pos[j]
        self._goal_quat[j] = self._target_quat[j]
        self.link_modes[point] = "hold"
        self.goal_controlled.add(point)
        self.ever_repaired.add(point)

    def set_velocity(self, point: str, direction: np.ndarray, speed_mps: float) -> None:
        if self.phase != "takeover":
            raise RuntimeError("direction control requires takeover")
        if point not in FIVE_POINTS:
            raise ValueError(f"unknown five-point target: {point}")
        vec = np.asarray(direction, dtype=np.float64)
        if vec.shape != (3,) or not np.isfinite(vec).all():
            raise ValueError("direction must be a finite xyz vector")
        speed = float(speed_mps)
        if not np.isfinite(speed) or speed < 0 or speed > 0.6:
            raise ValueError("speed must be in [0, 0.6] m/s")
        norm = float(np.linalg.norm(vec))
        self.point = point
        self.velocity = vec * (speed / norm) if norm > 1e-9 else np.zeros(3)
        self.link_velocities[point] = self.velocity.copy()
        self.link_modes[point] = "move" if speed > 0 and norm > 1e-9 else "hold"
        self.ever_repaired.add(point)

    def set_link_mode(self, point: str, mode: str) -> None:
        if self.phase != "takeover" or point not in FIVE_POINTS:
            raise RuntimeError("link mode requires five-point takeover")
        if mode not in ("reference", "hold", "move"):
            raise ValueError("link mode must be reference, hold or move")
        if mode == "reference" and point in self.ever_repaired:
            raise ValueError("a previously repaired link must hold or continue moving")
        self.link_modes[point] = mode
        if mode == "hold":
            self.link_velocities[point][:] = 0

    def begin_recovery(
        self,
        achieved_pos: np.ndarray,
        achieved_quat_wxyz: np.ndarray,
        reference_frame: int,
        *,
        body_pos_path: np.ndarray,
        body_quat_path: np.ndarray,
        joint_pos_path: np.ndarray,
        root_pos_path: np.ndarray,
        root_quat_path: np.ndarray,
    ) -> None:
        if self.phase != "takeover":
            raise RuntimeError("recovery requires takeover")
        p = np.asarray(achieved_pos, dtype=np.float64)
        q = np.asarray(achieved_quat_wxyz, dtype=np.float64)
        expected = (len(self.motion.body_names),)
        if p.shape != (*expected, 3) or q.shape != (*expected, 4):
            raise ValueError("achieved body poses must match selected body order")
        if not 0 <= reference_frame < self.motion.frames:
            raise ValueError("reference frame is outside the motion")
        count = len(body_pos_path) - 1
        if count < 0 or body_pos_path.shape != (count + 1, *p.shape) or body_quat_path.shape != (count + 1, *q.shape):
            raise ValueError("recovery body paths must include the live start and selected endpoint")
        if joint_pos_path.shape != (count + 1, len(self.motion.all_joint_names)):
            raise ValueError("recovery joint path has the wrong shape")
        if root_pos_path.shape != (count + 1, 3) or root_quat_path.shape != (count + 1, 4):
            raise ValueError("recovery root path has the wrong shape")
        self.recovery_frames = count
        self.phase = "recovery" if count else "reference"
        self.source_frame = reference_frame
        self._recovery_anchor = reference_frame
        self._recovery_start = p.copy()
        self._recovery_quat = q.copy()
        self._recovery_body_pos = np.asarray(body_pos_path, dtype=np.float64).copy()
        self._recovery_body_quat = np.asarray(body_quat_path, dtype=np.float64).copy()
        self._recovery_joint_pos = np.asarray(joint_pos_path, dtype=np.float64).copy()
        self._recovery_root_pos = np.asarray(root_pos_path, dtype=np.float64).copy()
        self._recovery_root_quat = np.asarray(root_quat_path, dtype=np.float64).copy()
        self._recovery_step = 0
        self.velocity[:] = 0

    def recovery_full_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Full-body reference commanded on the next recovery physics step."""
        if self.phase != "recovery":
            raise RuntimeError("full recovery pose requires recovery phase")
        assert self._recovery_joint_pos is not None
        assert self._recovery_root_pos is not None and self._recovery_root_quat is not None
        index = self._recovery_step + 1
        return (
            self._recovery_joint_pos[index].copy(),
            self._recovery_root_pos[index].copy(),
            self._recovery_root_quat[index].copy(),
            index / self.recovery_frames,
        )

    def targets(
        self,
        offsets: tuple[int, ...] = FUTURE_OFFSETS,
        achieved_pos: np.ndarray | None = None,
        achieved_quat_wxyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy.spatial.transform import Rotation

        offset_array = np.asarray(offsets)
        index = np.clip(self.source_frame + offset_array, 0, self.motion.frames - 1)
        pos = self.motion.body_pos_w[index].copy()
        quat = self.motion.body_quat_w[index].copy()
        if self.phase == "takeover":
            assert self._base_pos is not None and self._target_pos is not None and self._target_quat is not None
            for name in FIVE_POINTS:
                if self.link_modes[name] == "reference":
                    continue
                j = self.motion.body_names.index(name)
                if name in self.goal_controlled and self.link_modes[name] == "move":
                    assert self._goal_pos is not None and self._goal_quat is not None
                    current = self._target_pos[j] if achieved_pos is None else np.asarray(achieved_pos[j])
                    goal = self._goal_pos[j]
                    delta = goal - current
                    distance = float(np.linalg.norm(delta))
                    if distance <= 0.005 or self.link_speeds[name] == 0:
                        pos[:, j] = goal if distance <= 0.005 else self._target_pos[j]
                        self._goal_next_pos[name] = pos[0, j].copy()
                    else:
                        # Keep the commanded setpoint advancing even when the
                        # physical link lags. Reanchoring slot 0 to achieved
                        # every frame erases the error needed by VR-5.
                        direction = delta / distance
                        command = self._target_pos[j]
                        travel = offset_array * self.link_speeds[name] / self.motion.fps
                        future = command + travel[:, None] * direction
                        remaining = np.dot(goal - command, direction)
                        future[travel >= remaining] = goal
                        pos[:, j] = future
                        step = min(self.link_speeds[name] / self.motion.fps, max(remaining, 0.0))
                        self._goal_next_pos[name] = command + step * direction
                    current_q = (
                        self._target_quat[j] if achieved_quat_wxyz is None else np.asarray(achieved_quat_wxyz[j])
                    )
                    goal_q = self._goal_quat[j]
                    current_r = Rotation.from_quat(current_q[[1, 2, 3, 0]])
                    goal_r = Rotation.from_quat(goal_q[[1, 2, 3, 0]])
                    relative = (goal_r * current_r.inv()).as_rotvec()
                    angle = float(np.linalg.norm(relative))
                    if angle <= np.deg2rad(2) or self.link_angular_speeds[name] == 0:
                        quat[:, j] = goal_q if angle <= np.deg2rad(2) else self._target_quat[j]
                        self._goal_next_quat[name] = quat[0, j].copy()
                    else:
                        command_r = Rotation.from_quat(self._target_quat[j][[1, 2, 3, 0]])
                        command_delta = (goal_r * command_r.inv()).as_rotvec()
                        command_angle = float(np.linalg.norm(command_delta))
                        travel = np.minimum(
                            command_angle, offset_array * self.link_angular_speeds[name] / self.motion.fps
                        )
                        rotations = (
                            Rotation.from_rotvec(travel[:, None] * command_delta[None] / max(command_angle, 1e-9))
                            * command_r
                        )
                        quat[:, j] = rotations.as_quat()[:, [3, 0, 1, 2]]
                        one_step = min(command_angle, self.link_angular_speeds[name] / self.motion.fps)
                        next_rotation = (
                            Rotation.from_rotvec(one_step * command_delta / max(command_angle, 1e-9)) * command_r
                        )
                        self._goal_next_quat[name] = next_rotation.as_quat()[[3, 0, 1, 2]]
                    continue
                if self.link_modes[name] == "move":
                    # Expose the requested velocity to VR-5 on its first
                    # inference. A constant future target only reveals the
                    # direction after a physical step has already elapsed.
                    future = self._target_pos[j] + (
                        offset_array[:, None] * self.link_velocities[name][None] / self.motion.fps
                    )
                    delta = future - self._base_pos[j]
                    length = np.linalg.norm(delta, axis=1)
                    outside = length > self.max_edit_m
                    if np.any(outside):
                        future[outside] = self._base_pos[j] + (
                            delta[outside] * (self.max_edit_m / length[outside])[:, None]
                        )
                    pos[:, j] = future
                else:
                    pos[:, j] = self._target_pos[j]
                quat[:, j] = self._target_quat[j]
        elif self.phase == "recovery":
            assert self._recovery_body_pos is not None and self._recovery_body_quat is not None
            for k, offset in enumerate(offsets):
                step = self._recovery_step + offset + 1
                if step <= self.recovery_frames:
                    pos[k] = self._recovery_body_pos[step]
                    quat[k] = self._recovery_body_quat[step]
                else:
                    frame = min(self._recovery_anchor + step - self.recovery_frames, self.motion.frames - 1)
                    pos[k] = self.motion.body_pos_w[frame]
                    quat[k] = self.motion.body_quat_w[frame]
        return pos, quat

    def advance(self, *, converged: bool = True) -> None:
        if self.phase == "takeover":
            assert self._base_pos is not None and self._target_pos is not None
            for name in FIVE_POINTS:
                if self.link_modes[name] != "move":
                    continue
                j = self.motion.body_names.index(name)
                if name in self.goal_controlled:
                    if name in self._goal_next_pos:
                        self._target_pos[j] = self._goal_next_pos[name]
                    if name in self._goal_next_quat:
                        self._target_quat[j] = self._goal_next_quat[name]
                    continue
                proposal = self._target_pos[j] + self.link_velocities[name] / self.motion.fps
                delta = proposal - self._base_pos[j]
                length = float(np.linalg.norm(delta))
                if length > self.max_edit_m:
                    proposal = self._base_pos[j] + delta * (self.max_edit_m / length)
                self._target_pos[j] = proposal
            self.takeover_steps += 1
            self.source_frame = min(self.source_frame + 1, self.motion.frames - 1)
            return
        if self.phase == "recovery":
            self._recovery_step += 1
            if self._recovery_step >= self.recovery_frames:
                self.phase = "reference"
                self.source_frame = min(self._recovery_anchor + 1, self.motion.frames - 1)
            return
        self.source_frame = min(self.source_frame + 1, self.motion.frames - 1)
