"""Build a review candidate from reference, achieved repair and recovery targets.

The seven-field NPZ uses MuJoCo for forward kinematics only. Physical contact
is never stepped there: the recorded RSS rollout is the dynamics evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .runtime import RecordedStep
    from .timeline import ReferenceMotion


def export_candidate(
    records: list[RecordedStep],
    motion: ReferenceMotion,
    *,
    model_path: str | Path,
    order_path: str | Path,
    joint_coord_by_name: dict[str, int],
    root_body: int,
    out: str | Path,
) -> Path:
    import mujoco
    from scipy.spatial.transform import Rotation

    if not records or motion.all_joint_pos is None:
        raise ValueError("full-body candidate needs recorded RSS frames and original joint motion")
    order = json.loads(Path(order_path).read_text())
    names = tuple(order["robot_joint_names"])
    bodies = tuple(order["robot_body_names"])
    if motion.all_joint_names != names or len(bodies) != 50 or len(names) != 43:
        raise ValueError("seven-field candidate requires the named 43-joint/50-body ScaleTrack order")
    if records[-1].timeline_after.phase != "reference":
        raise ValueError("finish recovery before exporting a complete candidate")
    if records[-1].timeline_after.source_frame < motion.frames - 1:
        raise ValueError("play the recovered reference to its last frame before exporting")
    if records[-1].source_frame < motion.frames - 1:
        raise ValueError("the final source frame has not been physically tracked yet")
    if not any(r.phase == "takeover" for r in records):
        raise ValueError("no motion repair has been recorded")
    missing = [name for name in names if name not in joint_coord_by_name]
    if missing:
        raise ValueError(f"RSS model lacks source joints: {missing}")

    source = np.asarray([r.source_frame for r in records], dtype=np.int32)
    phases = np.asarray([r.phase for r in records])
    joint = motion.all_joint_pos[source].copy()
    root_pos = motion.body_pos_w[source, 0].copy()
    root_quat = motion.body_quat_w[source, 0].copy()
    live_joint = np.stack([np.asarray([r.joint_q[joint_coord_by_name[name]] for name in names]) for r in records])
    live_root_pos = np.stack([r.body_q[root_body, :3] for r in records])
    live_root_quat = np.stack([r.body_q[root_body, [6, 3, 4, 5]] for r in records])
    takeover = phases == "takeover"
    recovery = phases == "recovery"
    joint[takeover] = live_joint[takeover]
    root_pos[takeover] = live_root_pos[takeover]
    root_quat[takeover] = live_root_quat[takeover]
    for index in np.flatnonzero(recovery):
        record = records[index]
        if record.recovery_joint_pos is None or record.recovery_root_pos is None or record.recovery_root_quat is None:
            raise ValueError(f"recovery step {index} lacks its interpolated full-body reference")
        joint[index] = record.recovery_joint_pos
        root_pos[index] = record.recovery_root_pos
        root_quat[index] = record.recovery_root_quat
    # Hand closure is held by RSS position control through recovery. Export
    # the achieved fingers, not the full-body interpolation's old source hand.
    for side in ("left", "right"):
        columns = [i for i, name in enumerate(names) if name.startswith(f"{side}_hand_")]
        held = np.asarray(
            [record.phase == "recovery" or record.hand_override_after[side] is not None for record in records]
        )
        rows = np.flatnonzero(held)
        joint[np.ix_(rows, columns)] = live_joint[np.ix_(rows, columns)]
    for index in range(1, len(root_quat)):
        if np.dot(root_quat[index - 1], root_quat[index]) < 0:
            root_quat[index] *= -1

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = [int(model.joint(name).qposadr[0]) for name in names]
    body_ids = [int(model.body(name).id) for name in bodies]
    pos = np.empty((len(records), len(bodies), 3), dtype=np.float32)
    quat = np.empty((len(records), len(bodies), 4), dtype=np.float32)
    for frame in range(len(records)):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[addresses] = joint[frame]
        mujoco.mj_forward(model, data)
        pos[frame] = data.xpos[body_ids]
        quat[frame] = data.xquat[body_ids]
    dt = 1.0 / motion.fps
    edge = 2 if len(records) >= 3 else 1
    joint_vel = np.gradient(joint, dt, axis=0, edge_order=edge)
    linear_vel = np.gradient(pos.astype(np.float64), dt, axis=0, edge_order=edge)
    angular_vel = np.zeros_like(pos, dtype=np.float64)
    if len(records) > 1:
        for frame in range(len(records) - 1):
            now = Rotation.from_quat(quat[frame, :, [1, 2, 3, 0]].T)
            nxt = Rotation.from_quat(quat[frame + 1, :, [1, 2, 3, 0]].T)
            angular_vel[frame] = (nxt * now.inv()).as_rotvec() / dt
        angular_vel[-1] = angular_vel[-2]

    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        joint_pos=joint.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=linear_vel.astype(np.float32),
        body_ang_vel_w=angular_vel.astype(np.float32),
        fps=np.asarray(motion.fps, dtype=np.int64),
    )
    np.savez_compressed(
        output.with_name("candidate_mapping.npz"),
        source_frame=source,
        phase=phases,
        recovery_alpha=np.asarray(
            [np.nan if r.recovery_alpha is None else r.recovery_alpha for r in records], dtype=np.float32
        ),
    )
    return output
