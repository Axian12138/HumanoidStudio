"""Open both index/middle fingers before the chair contact, then hook the backrest.

Only four finger joints per hand change. The base, arms, wrists, thumbs, chair
trajectory and all source alignment sidecars are copied from the reviewed
thumb-face candidate.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from robosimstudio.humanoid_real2sim.paths import run_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

from fit_hoi import MODEL, ORDER, smoothstep


HERE = Path(__file__).resolve().parents[6]
DEFAULT_SOURCE = (HERE / "outputs/humanoid-real2sim/2026-10-20/"
                  "sequence_1019__armchair_split_feedback/runs/19-12-20.289127+0800/"
                  "chair_thumb_backface_flip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    out = args.out or (run_dir("sequence_1019/armchair_finger_entry")
                       / "chair_thumb_backface_flip_open")
    out.mkdir(parents=True, exist_ok=False)
    motion = MotionFields.load(source / "motion.npz", ORDER)
    joints = motion.joint_pos.astype(np.float64).copy()
    fps = float(motion.fps)
    time = 75.0 + np.arange(len(joints)) / fps
    # Original finger pose before 77.45 s, fully open from 78.20 through
    # 79.04 s, and original contact curl again by 79.30 s. Zero is straight
    # for both sides in this Dex3 MJCF; opposite joint signs are preserved.
    scale = (1 - smoothstep((time - 77.45) / .75)
             + smoothstep((time - 79.04) / .26))
    names = [f"{side}_hand_{digit}_{link}_joint"
             for side in ("left", "right")
             for digit in ("index", "middle")
             for link in ("0", "1")]
    columns = [motion.joint_names.index(name) for name in names]
    joints[:, columns] *= scale[:, None]
    refined = refresh_fk(replace(motion, joint_pos=joints.astype(np.float32)), MODEL)
    refined.save(out / "motion.npz")
    for name in ("object_estimate.npz", "human_motion_world.npz", "human.npz",
                 "source_sync.npz"):
        shutil.copy2(source / name, out / name)
    report = {
        "source": str(source),
        "motion": str(out / "motion.npz"),
        "changed_joints": names,
        "opening_window_s": [77.45, 78.20],
        "fully_open_window_s": [78.20, 79.04],
        "contact_curl_window_s": [79.04, 79.30],
        "base_arm_wrist_thumb_and_chair_unchanged": True,
        "fps": fps,
        "frames": len(joints),
    }
    (out / "finger_entry.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
