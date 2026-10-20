"""Seven-field ScaleTrack motion IO and MuJoCo forward-kinematics refresh."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion


@dataclass(frozen=True)
class MotionFields:
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    fps: int

    @classmethod
    def load(cls, path: str | Path, order_path: str | Path) -> MotionFields:
        order = json.loads(Path(order_path).read_text())
        joint_names = tuple(order["robot_joint_names"])
        body_names = tuple(order["robot_body_names"])
        with np.load(path, allow_pickle=False) as data:
            required = {
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "fps",
            }
            if set(data.files) != required:
                raise ValueError(f"expected the seven-field motion contract, got {data.files}")
            values = {name: np.asarray(data[name]).copy() for name in required}
        fps = int(values.pop("fps"))
        frames = len(values["joint_pos"])
        expected = {
            "joint_pos": (frames, len(joint_names)),
            "joint_vel": (frames, len(joint_names)),
            "body_pos_w": (frames, len(body_names), 3),
            "body_quat_w": (frames, len(body_names), 4),
            "body_lin_vel_w": (frames, len(body_names), 3),
            "body_ang_vel_w": (frames, len(body_names), 3),
        }
        for name, shape in expected.items():
            if values[name].shape != shape or not np.isfinite(values[name]).all():
                raise ValueError(f"invalid {name}: expected {shape}, got {values[name].shape}")
        if fps <= 0 or frames < 2:
            raise ValueError("motion needs at least two frames and a positive fps")
        norm_error = np.max(np.abs(np.linalg.norm(values["body_quat_w"], axis=-1) - 1.0))
        if norm_error > 2e-5:
            raise ValueError(f"body quaternions are not normalized (max error {norm_error})")
        return cls(joint_names=joint_names, body_names=body_names, fps=fps, **values)

    def reference(self, *, joint_names: Sequence[str], body_names: Sequence[str]) -> ReferenceMotion:
        joint_names = tuple(joint_names)
        body_names = tuple(body_names)
        missing_joints = set(joint_names) - set(self.joint_names)
        missing_bodies = set(body_names) - set(self.body_names)
        if missing_joints or missing_bodies:
            raise ValueError(f"motion lacks tracker joints {sorted(missing_joints)} or bodies {sorted(missing_bodies)}")
        ji = [self.joint_names.index(name) for name in joint_names]
        bi = [self.body_names.index(name) for name in body_names]
        root = bi[0]
        return ReferenceMotion(
            joint_pos=self.joint_pos[:, ji].astype(np.float64),
            body_pos_w=self.body_pos_w[:, bi].astype(np.float64),
            body_quat_w=self.body_quat_w[:, bi].astype(np.float64),
            joint_names=joint_names,
            body_names=body_names,
            fps=self.fps,
            all_joint_pos=self.joint_pos.astype(np.float64),
            all_joint_names=self.joint_names,
            all_joint_vel=self.joint_vel.astype(np.float64),
            root_lin_vel_w=self.body_lin_vel_w[:, root].astype(np.float64),
            root_ang_vel_w=self.body_ang_vel_w[:, root].astype(np.float64),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            joint_pos=self.joint_pos.astype(np.float32),
            joint_vel=self.joint_vel.astype(np.float32),
            body_pos_w=self.body_pos_w.astype(np.float32),
            body_quat_w=self.body_quat_w.astype(np.float32),
            body_lin_vel_w=self.body_lin_vel_w.astype(np.float32),
            body_ang_vel_w=self.body_ang_vel_w.astype(np.float32),
            fps=np.asarray(self.fps, dtype=np.int64),
        )
        return output


def refresh_fk(motion: MotionFields, model_path: str | Path) -> MotionFields:
    """Recompute all body poses and velocities after joint/root edits."""
    from dataclasses import replace

    import mujoco
    from scipy.spatial.transform import Rotation

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joints = np.asarray([int(model.joint(name).qposadr[0]) for name in motion.joint_names])
    bodies = np.asarray([int(model.body(name).id) for name in motion.body_names])
    frames = len(motion.joint_pos)
    position = np.empty((frames, len(bodies), 3), dtype=np.float64)
    quaternion = np.empty((frames, len(bodies), 4), dtype=np.float64)
    for frame in range(frames):
        data.qpos[:3] = motion.body_pos_w[frame, 0]
        data.qpos[3:7] = motion.body_quat_w[frame, 0]
        data.qpos[joints] = motion.joint_pos[frame]
        mujoco.mj_forward(model, data)
        position[frame] = data.xpos[bodies]
        quaternion[frame] = data.xquat[bodies]
    dt = 1.0 / motion.fps
    edge = 2 if frames >= 3 else 1
    joint_velocity = np.gradient(motion.joint_pos, dt, axis=0, edge_order=edge)
    linear_velocity = np.gradient(position, dt, axis=0, edge_order=edge)
    rotations = Rotation.from_quat(quaternion[..., [1, 2, 3, 0]])
    angular_velocity = np.zeros_like(position)
    for frame in range(frames - 1):
        angular_velocity[frame] = (rotations[frame + 1] * rotations[frame].inv()).as_rotvec() / dt
    angular_velocity[-1] = angular_velocity[-2]
    return replace(
        motion,
        joint_vel=joint_velocity,
        body_pos_w=position,
        body_quat_w=quaternion,
        body_lin_vel_w=linear_velocity,
        body_ang_vel_w=angular_velocity,
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
