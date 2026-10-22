"""Refresh FK and compare each protected joint and root, then hash candidate files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    import mujoco

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--candidate", required=True)
    p.add_argument("--allow-waist", action="store_true")
    a = p.parse_args()
    base = a.run / "retarget/candidates" / a.candidate
    source = a.run / "retarget/candidates/03_04_05_joint_v0"
    m = MotionFields.load(base / "motion.npz", base / "motion_order.json")
    prior = MotionFields.load(source / "motion.npz", source / "motion_order.json")
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    fk = refresh_fk(m, model)
    mj_model = mujoco.MjModel.from_xml_path(str(model))
    limits = np.array([mj_model.joint(n).range for n in m.joint_names])
    violation = np.maximum(limits[:, 0] - m.joint_pos, m.joint_pos - limits[:, 1])
    per = {n: float(abs(m.joint_pos[:, i] - prior.joint_pos[:, i]).max()) for i, n in enumerate(m.joint_names)}
    protected = [
        n
        for n in m.joint_names
        if not n.startswith(("left_hand", "right_hand"))
        and not (a.allow_waist and n.startswith("waist_"))
        and not any(n.startswith(s + "_" + part) for s in ("left", "right") for part in ("shoulder", "elbow", "wrist"))
    ]
    after59 = {
        n: float(abs(m.joint_pos[1600:, i] - prior.joint_pos[1600:, i]).max())
        for i, n in enumerate(m.joint_names)
        if not n.startswith("left_") or any(s in n for s in ("hip", "knee", "ankle"))
    }
    checks = {
        "candidate": a.candidate,
        "finite_joint_positions": bool(np.isfinite(m.joint_pos).all()),
        "joint_limit_violation_max_rad": float(max(0, violation.max())),
        "joint_limit_violation_samples_over_1e5": int((violation > 1e-5).sum()),
        "per_joint_max_speed_rad_s": {n: float(abs(m.joint_vel[:, i]).max()) for i, n in enumerate(m.joint_names)},
        "per_joint_max_difference_rad": per,
        "protected_joint_names": protected,
        "waist_exception_requested": a.allow_waist,
        "waist_max_change_rad": max(per[n] for n in m.joint_names if n.startswith("waist_")),
        "feet_world_position_difference_m": {
            n: float(abs(m.body_pos_w[:, m.body_names.index(n)] - prior.body_pos_w[:, prior.body_names.index(n)]).max())
            for n in ["left_ankle_roll_link", "right_ankle_roll_link"]
        },
        "protected_max_difference_rad": max(per[n] for n in protected),
        "after59_protected_per_joint_difference_rad": after59,
        "after59_protected_max_difference_rad": max(after59.values()),
        "root_position_max_difference_m": float(abs(m.body_pos_w[:, 0] - prior.body_pos_w[:, 0]).max()),
        "root_quaternion_numeric_max_difference": float(abs(m.body_quat_w[:, 0] - prior.body_quat_w[:, 0]).max()),
        "stored_vs_refreshed_fk_position_max_m": float(abs(m.body_pos_w - fk.body_pos_w).max()),
        "stored_vs_refreshed_fk_quaternion_numeric_max": float(abs(m.body_quat_w - fk.body_quat_w).max()),
        "interpretation": "Numerical protection only; left-held strawberry intentionally refined through06, dynamic feasibility not implied.",
    }
    (base / "protection_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in base.glob("*.npz")}
    (base / "manifest.json").write_text(
        json.dumps(
            {"status": "candidate_not_passed", "outputs": hashes, "protection_check": "protection_check.json"}, indent=2
        )
        + "\n"
    )
    print({k: v for k, v in checks.items() if not isinstance(v, (dict, list))})


if __name__ == "__main__":
    main()
