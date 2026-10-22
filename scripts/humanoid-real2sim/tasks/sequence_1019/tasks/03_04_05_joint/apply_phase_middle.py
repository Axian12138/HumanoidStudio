"""Retain early middle support and retract it only after the pinch can support gravity."""

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
    p.add_argument("--base", default="03_04_05_joint_v58")
    p.add_argument("--output", default="03_04_05_joint_v62")
    a = p.parse_args()
    r = a.run
    c = r / "retarget/candidates" / a.base
    out = r / "retarget/candidates" / a.output
    out.mkdir(exist_ok=False)
    motion = MotionFields.load(c / "motion.npz", c / "motion_order.json")
    family = json.loads((r / "review/scene/03_04_05_middle_rack_family.json").read_text())
    target = np.array(next(x for x in family["records"] if x["seed"] == 11)["q"])
    cols = [motion.joint_names.index(n) for n in family["joints"]]
    q = motion.joint_pos.copy().astype(float)
    delta = target - q[1032, cols]
    for k in range(len(q)):
        t = 27 + k / 50
        if 46.2 < t < 47.2:
            x = t - 46.2
            w = x * x * (3 - 2 * x)
        elif 47.2 <= t <= 47.64:
            w = 1
        elif 47.64 < t < 48.3:
            x = (48.3 - t) / 0.66
            w = x * x * (3 - 2 * x)
        else:
            w = 0
        q[k, cols] += w * delta
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in c.iterdir():
        if (
            f.is_file()
            and f.suffix in [".npz", ".json", ".xml"]
            and f.name not in ["motion.npz", "review.json", "manifest.json"]
            and not f.name.startswith(("collision_", "grasp_"))
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    (out / "review.json").write_text(
        json.dumps(
            {
                "status": "phase_middle_rack_clearance_not_accepted",
                "base": str(c),
                "window_s": [46.2, 48.3],
                "full_retraction_s": [47.2, 47.64],
                "joints": family["joints"],
                "delta_q": delta.tolist(),
                "early_support_unchanged": True,
                "thumb_index_arm_and_object_tracks_unchanged": True,
                "all_geometry_and_speed_checks_pending": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
