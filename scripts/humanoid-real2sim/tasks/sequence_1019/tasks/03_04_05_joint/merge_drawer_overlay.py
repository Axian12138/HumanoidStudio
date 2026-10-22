"""Apply a baseline-checked drawer arm/track overlay to a new combined candidate."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from refine_drawer_event_timing import MODEL

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("overlay", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    motion = MotionFields.load(a.base / "motion.npz", a.base / "motion_order.json")
    patch = np.load(a.overlay / "overlay.npz")
    rows, cols = patch["rows"], patch["columns"]
    if not np.array_equal(motion.joint_pos[np.ix_(rows, cols)], patch["baseline_joint_pos"]):
        raise ValueError("New base right arm differs inside the drawer repair window")
    old_drawer = np.load(a.base / "drawer_slide.npz")
    if not np.array_equal(old_drawer["position"], patch["baseline_drawer_position"]):
        raise ValueError("New base drawer track differs")
    q = motion.joint_pos.copy()
    q[np.ix_(rows, cols)] = patch["joint_pos"]
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    shutil.copytree(a.base, a.output)
    result.save(a.output / "motion.npz")
    shutil.copy2(a.overlay / "drawer_slide.npz", a.output / "drawer_slide.npz")
    report = {
        "status": "composed_pending_combined_validation",
        "base": str(a.base),
        "overlay": str(a.overlay),
        "overlay_sha256": sha256(a.overlay / "overlay.npz"),
        "base_motion_sha256": sha256(a.base / "motion.npz"),
        "output_motion_sha256": sha256(a.output / "motion.npz"),
        "window_s": [27 + float(rows[0]) / 50, 27 + float(rows[-1]) / 50],
        "changed_joint_names": [motion.joint_names[c] for c in cols],
        "scene_requirement": "Drawer joint range must include the new opening extent; use the private drawer_extent_228mm_diagnostic variant.",
        "scope": "Arm cells and drawer track only. Body FK/derivatives refreshed; unrelated q and tracks preserved.",
    }
    (a.output / "drawer_composition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
