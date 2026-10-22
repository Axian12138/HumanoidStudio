"""Measure world-pose preservation by fresh FK between exported motion frames."""

import argparse
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256


def main():
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=float, nargs=2, required=True)
    parser.add_argument("--bodies", nargs="+", required=True)
    parser.add_argument("--source-start-s", type=float, default=27)
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
        ),
    )
    args = parser.parse_args()
    times = np.arange(round(args.window[0] * args.rate), round(args.window[1] * args.rate) + 1) / args.rate
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    body_ids = [model.body(n).id for n in args.bodies]
    results, hashes = {}, {str(args.model): sha256(args.model)}
    for label, folder in (("baseline", args.baseline), ("candidate", args.candidate)):
        motion = MotionFields.load(folder / "motion.npz", folder / "motion_order.json")
        source = args.source_start_s + np.arange(len(motion.joint_pos)) / motion.fps
        if times[0] < source[0] - 1e-8 or times[-1] > source[-1] + 1e-8:
            raise ValueError("Requested interval is outside motion")
        addresses = [int(model.joint(n).qposadr[0]) for n in motion.joint_names]
        coordinates = np.column_stack([np.interp(times, source, motion.joint_pos[:, j]) for j in range(len(addresses))])
        roots = np.column_stack([np.interp(times, source, motion.body_pos_w[:, 0, j]) for j in range(3)])
        orientations = Slerp(source, Rotation.from_quat(motion.body_quat_w[:, 0, [1, 2, 3, 0]]))(times).as_quat()
        positions, quaternions = [], []
        for i in range(len(times)):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = roots[i]
            data.qpos[3:7] = orientations[i, [3, 0, 1, 2]]
            data.qpos[addresses] = coordinates[i]
            mujoco.mj_forward(model, data)
            positions.append(data.xpos[body_ids].copy())
            quaternions.append(data.xquat[body_ids].copy())
        results[label] = (np.array(positions), np.array(quaternions))
        hashes.update({str(folder / n): sha256(folder / n) for n in ("motion.npz", "motion_order.json")})
    position_error = np.linalg.norm(results["candidate"][0] - results["baseline"][0], axis=2)
    dots = abs(np.einsum("tbi,tbi->tb", results["candidate"][1], results["baseline"][1]))
    angular_error = 2 * np.arccos(np.clip(dots, 0, 1))
    rows = {}
    for j, name in enumerate(args.bodies):
        k = int(np.argmax(position_error[:, j]))
        a = int(np.argmax(angular_error[:, j]))
        rows[name] = {
            "max_position_difference_m": float(position_error[k, j]),
            "position_worst_source_s": float(times[k]),
            "max_orientation_difference_rad": float(angular_error[a, j]),
            "orientation_worst_source_s": float(times[a]),
            "position_difference_m": position_error[:, j].tolist(),
            "orientation_difference_rad": angular_error[:, j].tolist(),
        }
    report = {
        "scope": "Fresh MuJoCo FK after linear joint/root-position interpolation and quaternion SLERP, compared to interpolated baseline. Not source fit or force/stability verification.",
        "samples": len(times),
        "rate_hz": args.rate,
        "times_s": times.tolist(),
        "bodies": rows,
        "input_sha256": hashes,
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({name: row["max_position_difference_m"] for name, row in rows.items()})


if __name__ == "__main__":
    main()
