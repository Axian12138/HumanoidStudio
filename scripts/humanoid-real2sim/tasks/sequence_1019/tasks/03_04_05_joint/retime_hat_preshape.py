"""Prepare the fixed Dex3 pinch before its approach crosses the hat brim."""

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
    p.add_argument("--input-version", default="03_04_05_joint_v28")
    p.add_argument("--output-version", default="03_04_05_joint_v30")
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.input_version
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(exist_ok=True)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = m.joint_pos.copy()
    time = 27 + np.arange(len(q)) / 50
    rows = np.flatnonzero((time >= 37) & (time < 39))
    u = np.clip((time[rows] - 37) / 0.3, 0, 1)
    close = u * u * (3 - 2 * u)
    for j, n in enumerate(m.joint_names):
        if n.startswith("right_hand"):
            q[rows, j] = (0 if "thumb_0" in n else (-0.3 if "thumb" in n else 0.4)) * close
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "review.json",
            "manifest.json",
            "protection_check.json",
            "collision_diagnostic.json",
        ]:
            shutil.copy2(f, out / f.name)
    refresh_fk(
        replace(m, joint_pos=q),
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml",
    ).save(out / "motion.npz")
    review = json.loads((src / "review.json").read_text())
    review.update(
        status="earlier_dex3_preshape_candidate_not_passed",
        preshape={
            "close_s": [37, 37.3],
            "reason": "The straightened index finger crosses the brim before38s; keep the same final grasp and test pre-shaping before approach.",
        },
    )
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")


if __name__ == "__main__":
    main()
