"""Delay Dex3 extension until wrist withdrawal clears the hat brim."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--input-version", required=True)
    p.add_argument("--output-version", required=True)
    p.add_argument("--open-start", type=float, default=47.94)
    p.add_argument("--open-end", type=float, default=48.14)
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.input_version
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(exist_ok=True)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = m.joint_pos.copy()
    time = 27 + np.arange(len(q)) / 50
    rows = np.flatnonzero((time >= 47.3) & (time <= 49.5))
    u = np.clip((time[rows] - a.open_start) / (a.open_end - a.open_start), 0, 1)
    closed = 1 - u * u * (3 - 2 * u)
    for j, n in enumerate(m.joint_names):
        if n.startswith("right_hand"):
            q[rows, j] = (0 if "thumb_0" in n else (-0.3 if "thumb" in n else 0.4)) * closed
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "review.json",
            "manifest.json",
            "protection_check.json",
            "collision_diagnostic.json",
        ]:
            shutil.copy2(f, out / f.name)
    result = refresh_fk(
        replace(m, joint_pos=q),
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml",
    )
    result.save(out / "motion.npz")
    review = json.loads((src / "review.json").read_text())
    review.update(
        status="delayed_dex3_extension_candidate_not_passed",
        finger_release_adaptation={
            "open_s": [a.open_start, a.open_end],
            "object_detachment_s": 47.64,
            "reason": "Human hand extension timing intersects the hat with Dex3; test wrist withdrawal before finger extension. Actual clearance must be measured.",
        },
    )
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")


if __name__ == "__main__":
    main()
