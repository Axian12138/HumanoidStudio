"""Audit named joint limits and adjacent-frame speeds without hiding splice jumps."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("order", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-start-s", type=float, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
        ),
    )
    parser.add_argument(
        "--velocity-urdf",
        type=Path,
        default=Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/body.urdf"),
    )
    args = parser.parse_args()
    motion = MotionFields.load(args.motion, args.order)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    urdf_limits = {
        joint.get("name"): float(joint.find("limit").get("velocity", "20"))
        for joint in ET.parse(args.velocity_urdf).findall("joint")
        if joint.find("limit") is not None
    }
    q = motion.joint_pos.astype(float)
    speed = np.diff(q, axis=0) * motion.fps
    acceleration = np.diff(speed, axis=0) * motion.fps
    times = args.source_start_s + np.arange(len(q)) / motion.fps
    rows = []
    for column, name in enumerate(motion.joint_names):
        joint = model.joint(name)
        lower, upper = joint.range
        excess = np.maximum(lower - q[:, column], q[:, column] - upper)
        position_bad = np.flatnonzero(excess > 1e-5) if joint.limited else np.array([], dtype=int)
        # This reproduces retarget3.robot.MJCasadiRobot's existing contract.
        # The fallback is a planning bound, not an identified hardware rating.
        limit = urdf_limits.get(name, 3.0)
        speed_bad = np.flatnonzero(abs(speed[:, column]) > limit + 1e-5)
        peak = int(np.argmax(abs(speed[:, column])))
        rows.append(
            {
                "joint": name,
                "position_limit_rad": [float(lower), float(upper)] if joint.limited else None,
                "position_violation_source_s": times[position_bad].tolist(),
                "position_excess_max_rad": float(max(0, excess.max())) if joint.limited else 0,
                "speed_limit_rad_s": limit,
                "speed_limit_source": "body.urdf"
                if name in urdf_limits
                else "retarget3 3rad/s planning fallback; hardware rating unidentified",
                "speed_violation_interval_start_s": times[speed_bad].tolist(),
                "peak_adjacent_speed_rad_s": float(abs(speed[peak, column])),
                "peak_speed_interval_start_s": float(times[peak]),
                "peak_stored_speed_rad_s": float(abs(motion.joint_vel[:, column]).max()),
                "peak_adjacent_acceleration_rad_s2": float(abs(acceleration[:, column]).max()),
            }
        )
    position_count = sum(len(row["position_violation_source_s"]) for row in rows)
    speed_count = sum(len(row["speed_violation_interval_start_s"]) for row in rows)
    report = {
        "status": "passed_joint_contract" if position_count == speed_count == 0 else "failed_joint_contract",
        "scope": "Position bounds and adjacent-state speeds only; not collision, support, visual review or hardware certification.",
        "source_interval_s": [float(times[0]), float(times[-1])],
        "frames": len(q),
        "fps": motion.fps,
        "input_sha256": {str(p): sha256(p) for p in (args.motion, args.order, args.model, args.velocity_urdf)},
        "position_violation_samples": position_count,
        "speed_violation_intervals": speed_count,
        "joints": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "frames", "position_violation_samples", "speed_violation_intervals")
            }
        )
    )


if __name__ == "__main__":
    main()
