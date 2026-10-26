"""Enforce the user's zero-axial-thumb contract and bound three wrist spline samples."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    import mujoco
    from fit import MODEL

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    src = r / "retarget/v7"
    out = r / "retarget/v8"
    shutil.copytree(src, out)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = m.joint_pos.astype(float).copy()
    j = m.joint_names.index("right_hand_thumb_0_joint")
    before = q[:, j].copy()
    q[:, j] = 0
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    wr = m.joint_names.index("right_wrist_roll_joint")
    lo, hi = model.joint("right_wrist_roll_joint").range
    original = q[:, wr].copy()
    q[:, wr] = np.clip(q[:, wr], lo + 1e-6, hi - 1e-6)
    refresh_fk(replace(m, joint_pos=q), MODEL).save(out / "motion.npz")
    report = json.loads((src / "review.json").read_text())
    report.update(
        status="candidate_thumb_axial_locked_pending_audits",
        thumb_root={
            "joint": "right_hand_thumb_0_joint",
            "fixed_angle_rad": 0,
            "max_abs_before_rad": float(abs(before).max()),
            "max_abs_after_rad": 0,
            "bending_only": ["right_hand_thumb_1_joint", "right_hand_thumb_2_joint"],
        },
        wrist_boundary_projection={
            "frames": np.flatnonzero(q[:, wr] != original).tolist(),
            "max_correction_rad": float(abs(q[:, wr] - original).max()),
        },
    )
    (out / "review.json").write_text(json.dumps(report, indent=2))
    print(out)


if __name__ == "__main__":
    main()
