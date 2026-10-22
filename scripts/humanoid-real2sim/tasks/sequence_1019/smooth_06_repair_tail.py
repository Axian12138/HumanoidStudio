"""Taper the terminal correction of a pre-contact leg repair with a quintic."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-s", type=float, default=61.2)
    parser.add_argument("--end-s", type=float, default=61.4)
    args = parser.parse_args()
    base = MotionFields.load(args.baseline / "motion.npz", args.baseline / "motion_order.json")
    motion = MotionFields.load(args.candidate / "motion.npz", args.candidate / "motion_order.json")
    if motion.joint_names != base.joint_names or motion.joint_pos.shape != base.joint_pos.shape:
        raise ValueError("Baseline and candidate mismatch")
    cols = [
        motion.joint_names.index("left_" + n + "_joint")
        for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    time = 27 + np.arange(len(motion.joint_pos)) / motion.fps
    rows = np.flatnonzero((time >= args.start_s - 1e-7) & (time <= args.end_s + 1e-7))
    u = np.clip((time[rows] - args.start_s) / (args.end_s - args.start_s), 0, 1)
    weight = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
    q = motion.joint_pos.astype(float).copy()
    original = base.joint_pos[np.ix_(rows, cols)].astype(float)
    q[np.ix_(rows, cols)] = original + weight[:, None] * (q[np.ix_(rows, cols)] - original)
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(motion, joint_pos=q), robot)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "solver_snapshot.py")
    for name in (
        "motion_order.json",
        "source_sync.npz",
        "human_motion_world.npz",
        "body_prior.npz",
        "black_ribbon_hat.npz",
        "floor_strawberry.npz",
        "round_laundry_basket.npz",
        "drawer_slide.npz",
        "tracks_manifest.json",
    ):
        shutil.copy2(args.candidate / name, args.output / name)
    result.save(args.output / "motion.npz")
    report = {
        "status": "pending_full_repair_revalidation",
        "source_candidate": str(args.baseline),
        "parent_repair": str(args.candidate),
        "tail_window_s": [args.start_s, args.end_s],
        "method": "Quintic fade of correction relative to baseline, zero first and second derivative at taper endpoints.",
        "input_sha256": {str(p): sha256(p) for p in (args.baseline / "motion.npz", args.candidate / "motion.npz")},
        "limitations": [
            "A smooth taper does not guarantee collision clearance; rerun full100Hz gates and source review.",
            "Inherited hip-pelvis collision proxy overlap and native-shell audit remain distinct.",
        ],
    }
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
