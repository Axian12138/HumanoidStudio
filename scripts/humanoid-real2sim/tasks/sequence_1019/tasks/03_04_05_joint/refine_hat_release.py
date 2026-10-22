"""Delay finger extension until the wrist withdraws from the released hat."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--input-version", default="03_04_05_joint_v14")
    p.add_argument("--output-version", default="03_04_05_joint_v15")
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.input_version
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(parents=True, exist_ok=True)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = m.joint_pos.copy()
    t = 27 + np.arange(len(q)) / 50
    rows = (t >= 47.8) & (t <= 49)
    u = np.clip((t - 48.25) / 0.5, 0, 1)
    u = u * u * (3 - 2 * u)
    for j, n in enumerate(m.joint_names):
        if n.startswith("right_hand"):
            angle = 0 if "thumb_0" in n else (-0.3 if "thumb" in n else 0.4)
            q[rows, j] = angle * (1 - u[rows])
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(m, joint_pos=q), model)
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "manifest.json",
            "review.json",
            "collision_diagnostic.json",
            "protection_check.json",
        ]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    review = json.loads((src / "review.json").read_text())
    review["status"] = "hat_release_timing_candidate_not_passed"
    review["hat_release_correction"] = {
        "object_release_s": 48,
        "finger_open_s": [48.25, 48.75],
        "reason": "Earlier extension crossed the hat brim; delayed extension tests withdrawal geometry only, rack support remains unverified.",
    }
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")


if __name__ == "__main__":
    main()
