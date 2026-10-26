"""Construct the second brace entry from the chair interaction delta only."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk
from robosimstudio.humanoid_real2sim.track.launcher import DEFAULT_ROBOT


def main():
    root = Path("/home/galbot/workspace/ScaleTrack/data/1019test")
    out = Path("outputs/humanoid-real2sim/2026-10-25/table_hand_clearance/runs")
    order = root / "brace-dining-table-wipes-kinematic-v1/metadata/motion_order.json"
    base = MotionFields.load(out / "brace_v2/core.npz", order)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["thumb", "index-middle"], required=True)
    a = ap.parse_args()
    push = np.load(out / a.variant / "motion.npz")
    source_times = push["recording_time_s"]
    seam = int(np.argmin(abs(source_times - 83.5)))
    delta = push["joint_pos"][seam] - base.joint_pos[0]
    velocity_delta = push["joint_vel"][seam] - base.joint_vel[0]
    q = base.joint_pos.astype(float).copy()
    for k in range(50):
        u = k / 50
        h0 = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
        h1 = u - 6 * u**3 + 8 * u**4 - 3 * u**5
        q[k] += h0 * delta + h1 * velocity_delta
    result = refresh_fk(replace(base, joint_pos=q), DEFAULT_ROBOT)
    # Refresh only the altered arm/hand portion; suffix remains exactly the common motion.
    joints = [
        i
        for i, n in enumerate(base.joint_names)
        if n.startswith(("left_", "right_")) and any(s in n for s in ["shoulder_", "elbow_", "wrist_", "hand_"])
    ]
    bodies = [
        i for i, n in enumerate(base.body_names) if any(s in n for s in ["shoulder_", "elbow_", "wrist_", "hand_"])
    ]
    values = {}
    for key in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
        value = getattr(base, key).copy()
        cols = joints if key.startswith("joint_") else bodies
        value[np.ix_(np.arange(52), cols)] = getattr(result, key)[np.ix_(np.arange(52), cols)]
        values[key] = value
    replace(base, **values).save(out / f"brace_{a.variant}_input.npz")
    (out / "pair.json").write_text(
        json.dumps(
            {
                "overlap_source_time_s": 83.5,
                "push_source_frame": seam,
                "common_brace_from_frame": 52,
                "common_brace_from_time_s": 1.04,
                "scope": "same root/legs and common brace suffix; initial arm/hand delta follows chair contact variant",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
