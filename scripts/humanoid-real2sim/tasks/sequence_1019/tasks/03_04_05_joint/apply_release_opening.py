"""Install a separately searched opening branch, retaining both end poses."""

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
    p.add_argument("--opening", type=Path, required=True)
    p.add_argument("--output-version", required=True)
    p.add_argument("--base-version", default="03_04_05_joint_v70")
    a = p.parse_args()
    root = a.run / "retarget/candidates"
    src = root / a.base_version
    post = MotionFields.load(root / "03_04_05_joint_v68/motion.npz", src / "motion_order.json")
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    opening = json.loads(a.opening.read_text())
    cols = [motion.joint_names.index(n) for n in opening["joint_names"]]
    q = motion.joint_pos.astype(float).copy()
    start, key, end = 47.64, 47.76, 49.7
    k0, km, ke = [round((t - 27) * 50) for t in [start, key, end]]
    value = np.array(opening["joint_pos"] if "joint_pos" in opening else opening["records"][0]["joint_pos"])
    for k in range(k0, km + 1):
        alpha = (k - k0) / (km - k0)
        q[k, cols] = q[k0, cols] * (1 - alpha) + value * alpha
    delta = value - post.joint_pos[km, cols]
    for k in range(km + 1, ke + 1):
        u = (k - km) / (ke - km)
        fade = 1 - (10 * u**3 - 15 * u**4 + 6 * u**5)
        q[k, cols] = post.joint_pos[k, cols] + delta * fade
    result = refresh_fk(
        replace(motion, joint_pos=q),
        Path(
            "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
        ),
    )
    out = root / a.output_version
    out.mkdir(exist_ok=False)
    for f in src.iterdir():
        if (
            f.is_file()
            and f.name not in ["motion.npz", "review.json", "manifest.json"]
            and not f.name.startswith(("collision_", "grasp_contact"))
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    report = {
        "status": "opening_branch_and_unvalidated_return",
        "base": str(src),
        "return_base": "03_04_05_joint_v68",
        "opening": str(a.opening),
        "window_s": [start, end],
        "key_s": key,
        "limitations": [
            "47.76+ blended return requires collision and speed repair",
            "100Hz full room/self/props and source review pending",
        ],
    }
    (out / "review.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
