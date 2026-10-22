"""Copy explicitly bounded left-arm patches and recompute complete motion fields."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    order = args.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    base = MotionFields.load(args.base, order)
    q = base.joint_pos.copy()
    left = [
        i
        for i, name in enumerate(base.joint_names)
        if name.startswith("left_") and any(x in name for x in ["shoulder_", "elbow_", "wrist_", "hand_"])
    ]
    used = np.zeros(len(q), dtype=bool)
    records = []
    for patch in json.loads(args.patches.read_text()):
        path = Path(patch["motion"])
        motion = MotionFields.load(path, order)
        start, end = patch["source_range_s"]
        first, last = [round((time - 108) * base.fps) for time in [start, end]]
        if first < 50 or last >= len(q) or last < first or used[first : last + 1].any():
            raise ValueError("patch crosses protected overlap, lies outside motion, or overlaps another patch")
        if motion.joint_names != base.joint_names or motion.joint_pos.shape != q.shape:
            raise ValueError("motion contracts differ")
        used[first : last + 1] = True
        indices = np.ix_(np.arange(first, last + 1), left)
        q[indices] = motion.joint_pos[indices]
        records.append(
            {
                "motion": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_range_s": [start, end],
                "max_joint_change_rad": float(np.max(abs(q[indices] - base.joint_pos[indices]))),
            }
        )
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(base, joint_pos=q), robot)
    result.save(args.out / "motion.npz")
    report = {
        "status": "composed candidate; full audit required",
        "base": str(args.base.resolve()),
        "base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "patches": records,
        "columns": [base.joint_names[i] for i in left],
        "protected_first50_exact": bool(np.array_equal(q[:50], base.joint_pos[:50])),
        "other_columns_exact": bool(
            np.array_equal(
                q[:, [i for i in range(q.shape[1]) if i not in left]],
                base.joint_pos[:, [i for i in range(q.shape[1]) if i not in left]],
            )
        ),
        "root_position_exact": bool(np.array_equal(result.body_pos_w[:, 0], base.body_pos_w[:, 0])),
    }
    (args.out / "composition.json").write_text(json.dumps(report, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
