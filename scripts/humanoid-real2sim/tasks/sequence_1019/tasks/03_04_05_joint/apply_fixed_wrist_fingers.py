"""Apply a constant finger grasp while preserving the feasible wrist/object transform."""

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
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--output-version", default="03_04_05_joint_v55")
    p.add_argument("--input-version", default="03_04_05_joint_v50")
    p.add_argument("--family-json", default="03_04_05_fixed_wrist_finger_grasp_family.json")
    a = p.parse_args()
    r = a.run
    src = r / "retarget/candidates" / a.input_version
    out = r / "retarget/candidates" / a.output_version
    out.mkdir(exist_ok=False)
    family = json.loads((r / "review/scene" / a.family_json).read_text())
    seed = next(x for x in family["records"] if x["seed"] == a.seed)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    cols = [motion.joint_names.index(n) for n in family["joints"]]
    assert len(cols) in (2, 7) and all(n.startswith("right_hand") for n in family["joints"])
    q = motion.joint_pos.astype(float).copy()
    times = 27 + np.arange(len(q)) / 50
    target = np.array(seed["q"])
    release = 1032
    for k, t in enumerate(times):
        if 36.3 < t < 38.3:
            x = (t - 36.3) / 2
            w = x * x * (3 - 2 * x)
            q[k, cols] += (target - motion.joint_pos[600, cols]) * w
        elif 38.3 <= t <= 47.64:
            q[k, cols] = target
        elif 47.64 < t < 48.3:
            x = (48.3 - t) / (0.66)
            w = x * x * (3 - 2 * x)
            q[k, cols] += (target - motion.joint_pos[release, cols]) * w
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in src.iterdir():
        if (
            f.is_file()
            and f.suffix in (".npz", ".json", ".xml")
            and f.name not in ("motion.npz", "review.json", "manifest.json")
            and not f.name.startswith("collision_")
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    review = {
        "status": "fixed_wrist_finger_candidate_not_accepted",
        "base": str(src),
        "seed": seed,
        "hold_window_s": [38.3, 47.64],
        "all_collision_speed_source_checks_pending": True,
        "non_finger_q_exact": bool(
            np.array_equal(
                q[:, [i for i in range(q.shape[1]) if i not in cols]],
                motion.joint_pos[:, [i for i in range(q.shape[1]) if i not in cols]],
            )
        ),
    }
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(review)


if __name__ == "__main__":
    main()
