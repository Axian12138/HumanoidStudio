"""Combine reviewed disjoint hat windows while retaining the clean withdrawal base."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    c = a.run / "retarget/candidates"
    base = c / "03_04_05_joint_v39"
    early = c / "03_04_05_joint_v40"
    out = c / "03_04_05_joint_v41"
    out.mkdir(exist_ok=False)
    motion = MotionFields.load(base / "motion.npz", base / "motion_order.json")
    grasp = MotionFields.load(early / "motion.npz", early / "motion_order.json")
    q = motion.joint_pos.copy()
    first = round((36.3 - 27) * 50)
    last = round((41.5 - 27) * 50)
    q[first : last + 1] = grasp.joint_pos[first : last + 1]
    assert np.max(abs(q[last] - motion.joint_pos[last])) < 1e-6
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in base.iterdir():
        if (
            f.is_file()
            and f.name not in ["motion.npz", "review.json", "manifest.json", "black_ribbon_hat.npz"]
            and not f.name.startswith("collision_")
        ):
            shutil.copy2(f, out / f.name)
    h = dict(np.load(base / "black_ribbon_hat.npz"))
    e = np.load(early / "black_ribbon_hat.npz")
    for key in ["position_w", "quaternion_xyzw"]:
        assert np.max(abs(h[key][last] - e[key][last])) < 1e-6
        h[key][: last + 1] = e[key][: last + 1]
    np.savez_compressed(out / "black_ribbon_hat.npz", **h)
    result.save(out / "motion.npz")
    review = json.loads((base / "review.json").read_text())
    review["status"] = "combined_hat_windows_pending_full_review"
    review["combined_windows"] = {
        "base": "v39 clean withdrawal and later finger seam",
        "early": "v40 sofa initial pose and 36.3-41.5 grasp window",
        "hat_prefix_end_s": 41.5,
        "early_q_window_s": [36.3, 41.5],
        "source_geometry_unresolved": True,
    }
    review["early_window_review"] = json.loads((early / "review.json").read_text())
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(review["combined_windows"])


if __name__ == "__main__":
    main()
