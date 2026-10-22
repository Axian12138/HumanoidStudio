"""Measure a retreat candidate against its seed and body-prior palm orientation."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion", "seed", "reference", "order", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--window", type=float, nargs=2, default=(120.38, 121.82))
    parser.add_argument("--seams", type=float, nargs="+", default=(120.4, 121.8))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
        "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    data = mujoco.MjData(model)
    motion, seed, reference = [MotionFields.load(p, args.order) for p in (args.motion, args.seed, args.reference)]
    addresses = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    palm = model.geom("left_hand_palm_collision").id
    arrays = []
    for item in (motion, reference):
        positions, rotations = [], []
        for k in range(len(item.joint_pos)):
            data.qpos[:3] = item.body_pos_w[k, 0]
            data.qpos[3:7] = item.body_quat_w[k, 0]
            data.qpos[addresses] = item.joint_pos[k]
            mujoco.mj_forward(model, data)
            positions.append(data.geom_xpos[palm].copy())
            rotations.append(data.geom_xmat[palm].reshape(3, 3).copy())
        arrays.append((np.array(positions), np.array(rotations)))
    positions, rotations = arrays[0]
    ref_positions, ref_rotations = arrays[1]
    angle = np.rad2deg(Rotation.from_matrix(rotations @ ref_rotations.transpose(0, 2, 1)).magnitude())
    omega = Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).magnitude() * motion.fps
    changed = abs(motion.joint_pos - seed.joint_pos) > 1e-7
    rows = np.flatnonzero(changed.any(axis=1))
    times = 108 + np.arange(len(positions)) / motion.fps
    window = (times >= args.window[0]) & (times <= args.window[1])
    seams = []
    left_columns = np.array(
        [
            i
            for i, name in enumerate(motion.joint_names)
            if name.startswith("left_")
            and ("hand_" in name or any(part in name for part in ("shoulder_", "elbow_", "wrist_")))
        ]
    )
    for time in args.seams:
        k = round((time - 108) * motion.fps)
        incoming = (motion.joint_pos[k] - motion.joint_pos[k - 1]) * motion.fps
        outgoing = (motion.joint_pos[k + 1] - motion.joint_pos[k]) * motion.fps
        jump = outgoing - incoming
        for scope, columns in (("all_joints", np.arange(len(motion.joint_names))), ("left_arm_hand", left_columns)):
            index = columns[np.argmax(abs(jump[columns]))]
            seams.append(
                {
                    "source_s": time,
                    "scope": scope,
                    "worst_joint": motion.joint_names[index],
                    "incoming_rad_s": float(incoming[index]),
                    "outgoing_rad_s": float(outgoing[index]),
                    "velocity_jump_rad_s": float(jump[index]),
                    "adjacent_acceleration_rad_s2": float(jump[index] * motion.fps),
                }
            )
    acceleration = np.diff(motion.joint_pos[:, left_columns], n=2, axis=0) * motion.fps**2
    acceleration_indices = np.flatnonzero(window[1:-1])
    ranked = np.argsort(abs(acceleration[acceleration_indices]).ravel())[-10:][::-1]
    worst_acceleration = []
    for index in ranked:
        row, column = np.unravel_index(index, (len(acceleration_indices), len(left_columns)))
        k = acceleration_indices[row]
        worst_acceleration.append(
            {
                "source_s": float(times[k + 1]),
                "joint": motion.joint_names[left_columns[column]],
                "acceleration_rad_s2": float(acceleration[k, column]),
            }
        )
    report = {
        "status": "Diagnostic only; matching prior orientation does not establish source similarity.",
        "motion_sha256": sha256(args.motion),
        "seed_sha256": sha256(args.seed),
        "changed_source_range_s": [float(times[rows.min()]), float(times[rows.max()])] if len(rows) else [],
        "changed_joints": [
            n for n, changed_column in zip(motion.joint_names, changed.any(axis=0), strict=True) if changed_column
        ],
        "root_position_max_delta_m": float(abs(motion.body_pos_w[:, 0] - seed.body_pos_w[:, 0]).max()),
        "root_quaternion_max_delta": float(abs(motion.body_quat_w[:, 0] - seed.body_quat_w[:, 0]).max()),
        "seams": seams,
        "window_s": args.window,
        "left_arm_hand_acceleration": {
            "method": "Adjacent interval velocity difference at source sample, using 50 Hz positions; not stored derivatives.",
            "max_abs_rad_s2": float(abs(acceleration[acceleration_indices]).max()),
            "p99_abs_rad_s2": float(np.percentile(abs(acceleration[acceleration_indices]), 99)),
            "largest": worst_acceleration,
        },
        "records": [
            {
                "source_s": float(times[k]),
                "prior_palm_angle_deg": float(angle[k]),
                "prior_palm_position_error_m": float(np.linalg.norm(positions[k] - ref_positions[k])),
                "next_interval_palm_angular_speed_rad_s": float(omega[k]),
            }
            for k in np.flatnonzero(window)
        ],
    }
    with args.out.open("x") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
