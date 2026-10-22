"""Audit hard limits and source-contact foot drift on the immutable clip prior."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    folder = a.run / "retarget/body_prior/10_flower_bouquet"
    m = np.load(folder / "body_prior.npz")
    h = np.load(folder / "human.npz")
    order = json.loads((folder / "motion_order.json").read_text())
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    indices = np.array([model.joint(n).id for n in order["robot_joint_names"]])
    low, high = model.jnt_range[indices].T
    q = m["joint_pos"]
    violation = np.maximum(low - q, q - high)
    velocity = m["joint_vel"]
    acceleration = np.gradient(velocity, 0.02, axis=0)
    report = {
        "status": "numeric_diagnostic_not_visual_approval",
        "hard_limit_violations": int((violation > 1e-5).sum()),
        "max_joint_velocity_rad_s": float(abs(velocity).max()),
        "max_joint_acceleration_rad_s2": float(abs(acceleration).max()),
        "hardware_velocity_limits": "not inferred from position ranges; per-joint limits pending contract lookup",
        "support_drift": {},
        "wrists": {},
    }
    for side in ["left", "right"]:
        ids = [i for i, n in enumerate(order["robot_joint_names"]) if n.startswith(side + "_wrist")]
        report["wrists"][side] = {
            order["robot_joint_names"][i]: {
                "min_rad": float(q[:, i].min()),
                "max_rad": float(q[:, i].max()),
                "max_velocity_rad_s": float(abs(velocity[:, i]).max()),
            }
            for i in ids
        }
        bi = order["robot_body_names"].index(side + "_ankle_roll_link")
        si = 0 if side == "left" else 1
        c = h["foot_contact"][:, si].astype(bool)
        edges = np.diff(np.r_[False, c, False].astype(int))
        intervals = []
        for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            pos = m["body_pos_w"][begin:end, bi]
            drift = np.linalg.norm(pos - pos[0], axis=1)
            worst = int(drift.argmax()) + int(begin)
            intervals.append(
                {
                    "start_s": 108 + begin * 0.02,
                    "end_s": 108 + (end - 1) * 0.02,
                    "max_origin_drift_m": float(drift.max()),
                    "worst_source_s": 108 + worst * 0.02,
                    "note": "ankle-origin drift; toe pivot can legitimately move ankle, not sole collision clearance",
                }
            )
        report["support_drift"][side] = intervals
    (a.run / "review/stage3/10_body_prior_numeric.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k not in ["support_drift", "wrists"]})


if __name__ == "__main__":
    main()
