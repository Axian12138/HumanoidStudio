"""Build a diagnostic transition between accepted push and brace clips.

Preserve the push through83.4s and brace from84.5s. This does not accept the
transition: collision, speed, and full-body review must be performed separately.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def main():
    from scipy.interpolate import CubicHermiteSpline

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--variant", choices=("thumb", "index-middle"), default="index-middle")
    p.add_argument("--version", default="v2")
    p.add_argument("--left-shoulder-clearance-rad", type=float, default=0.0)
    a = p.parse_args()
    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test")
    push = source / f"push-armchair-{a.variant}-kinematic-v1/motion.npz"
    brace = source / "brace-dining-table-kinematic-v1/motion.npz"
    order_path = brace.parent / "metadata/motion_order.json"
    order = json.loads(order_path.read_text())
    push_order = json.loads((push.parent / "metadata/motion_order.json").read_text())
    for key in ("robot_joint_names", "robot_body_names"):
        if order[key] != push_order[key]:
            raise ValueError(f"source {key} differ; do not compose by array index")
    out = a.run / "retarget/candidates" / f"07_push_brace_connection_{a.variant}_{a.version}"
    out.mkdir(parents=True, exist_ok=False)
    # Archive inputs contain image features; load only the six motion arrays.
    with np.load(push) as z:
        if float(z["fps"]) != 50:
            raise ValueError("push source must be 50 Hz")
        first = {key: z[key].copy() for key in FIELDS}
    with np.load(brace) as z:
        if float(z["fps"]) != 50:
            raise ValueError("brace source must be 50 Hz")
        second = {key: z[key].copy() for key in FIELDS}
    if len(first["joint_pos"]) != 475 or len(second["joint_pos"]) != 375:
        raise ValueError("source frame count differs from the audited source windows")
    data = {key: np.concatenate((first[key][:425], second[key]), axis=0) for key in FIELDS}
    start, stop = round((83.4 - 75) * 50), round((84.5 - 75) * 50)
    q = data["joint_pos"].astype(float)
    left_velocity = (first["joint_pos"][start + 1] - first["joint_pos"][start - 1]) * 25
    right_velocity = (second["joint_pos"][51] - second["joint_pos"][49]) * 25
    t = np.arange(start, stop + 1) / 50
    spline = CubicHermiteSpline([t[0], t[-1]], [q[start], q[stop]], [left_velocity, right_velocity])
    q[start : stop + 1] = spline(t)
    # Connecting upper-body actions must not replace the accepted gait with a spline.
    protected = [
        i
        for i, name in enumerate(order["robot_joint_names"])
        if any(part in name for part in ("hip_", "knee_", "ankle_")) or name in ("waist_yaw_joint", "waist_roll_joint")
    ]
    q[:, protected] = data["joint_pos"][:, protected]
    if a.left_shoulder_clearance_rad:
        times = 75 + np.arange(len(q)) / 50
        u = np.clip((times - 83.9) / 0.5, 0, 1)
        bump = 16 * u**2 * (1 - u) ** 2
        q[:, order["robot_joint_names"].index("left_shoulder_roll_joint")] += a.left_shoulder_clearance_rad * bump
    motion = MotionFields(
        tuple(order["robot_joint_names"]), tuple(order["robot_body_names"]), **{**data, "joint_pos": q}, fps=50
    )
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    motion = refresh_fk(motion, model)
    # Preserve all stored fields outside the changed connection window exactly.
    restored = {}
    for key in FIELDS:
        values = getattr(motion, key).copy()
        values[:start] = data[key][:start]
        values[stop + 1 :] = data[key][stop + 1 :]
        restored[key] = values
    motion = replace(motion, **restored)
    motion.save(out / "motion.npz")
    (out / "motion_order.json").write_text(json.dumps(order, indent=2) + "\n")
    report = {
        "status": "diagnostic_not_accepted",
        "source_window_s": [75, 91],
        "connection_window_s": [83.4, 84.5],
        "variant": a.variant,
        "method": "cubic Hermite joint connection with original endpoint velocities; root/body world frame preserved",
        "inputs_sha256": {str(path): sha256(path) for path in (push, brace, order_path)},
        "peak_connection_speed_rad_s": float(abs(np.gradient(q, 0.02, axis=0)[start : stop + 1]).max()),
        "left_shoulder_clearance_rad": a.left_shoulder_clearance_rad,
        "clearance_window_s": [83.9, 84.4],
        "protected_joint_names": [order["robot_joint_names"][i] for i in protected],
        "pending": [
            "joint limit and per-joint speed audit",
            "whole-body scene collision and support checks",
            "continuous interactive visual review",
            "unified object/human state integration",
        ],
    }
    (out / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out, report["peak_connection_speed_rad_s"])


if __name__ == "__main__":
    main()
