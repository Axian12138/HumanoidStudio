"""Reject the penetrating v12 approach while retaining the v13 drawer correction."""

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
    a = p.parse_args()
    src = a.run / "retarget/candidates/03_04_05_joint_v13"
    prior = a.run / "retarget/candidates/03_04_05_joint_v11"
    out = a.run / "retarget/candidates/03_04_05_joint_v14"
    out.mkdir(parents=True, exist_ok=True)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    old = MotionFields.load(prior / "motion.npz", prior / "motion_order.json")
    q = m.joint_pos.copy()
    cols = [
        m.joint_names.index("right_" + n + "_joint")
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    rows = np.arange(475, 601)
    q[np.ix_(rows, cols)] = old.joint_pos[np.ix_(rows, cols)]
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
    review["status"] = "best_partial_candidate_not_passed"
    review["rejected_hat_approach"] = {
        "candidate": "v12/v13",
        "reason": "Full collision audit increased pre-grasp hat penetration from8.93mm to25.45mm and121 deep contact records vs45; restorev11 right arm36.5-39s.",
        "restored_source": str(prior),
    }
    review.pop("hat_approach_fit", None)
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")


if __name__ == "__main__":
    main()
