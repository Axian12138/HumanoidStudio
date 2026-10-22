"""Broaden the collision-avoidance correction and smooth the local left-arm seam."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    from scipy.ndimage import gaussian_filter1d

    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("repaired", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=False)
    for p in a.repaired.iterdir():
        if p.is_file() and (
            p.suffix == ".npz"
            or p.name in ["motion_order.json", "tracks.json", "grasp_design_input.json", "geometry_config.json"]
        ):
            shutil.copy2(p, a.output / p.name)
    shutil.copy2(__file__, a.output / "smooth_connection_source.py")
    b = MotionFields.load(a.baseline / "motion.npz", a.baseline / "motion_order.json")
    r = MotionFields.load(a.repaired / "motion.npz", a.repaired / "motion_order.json")
    q = b.joint_pos.astype(float).copy()
    ids = [
        i
        for i, n in enumerate(b.joint_names)
        if n.startswith("left_") and any(x in n for x in ["shoulder", "elbow", "wrist"])
    ]
    correction = gaussian_filter1d(r.joint_pos[:, ids] - b.joint_pos[:, ids], 3, axis=0) * 2
    smoothed = gaussian_filter1d(b.joint_pos[:, ids], 1.5, axis=0) + correction
    for k in range(370, 411):
        u = min((k - 370) / 10, (411 - k) / 10, 1)
        w = u * u * (3 - 2 * u)
        q[k, ids] = q[k, ids] * (1 - w) + smoothed[k] * w
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(b, joint_pos=q), model)
    result.save(a.output / "motion.npz")
    (a.output / "connection_repair.json").write_text(
        json.dumps(
            {
                "baseline": str(a.baseline),
                "repaired": str(a.repaired),
                "gaussian_sigma_frames": {"baseline": 1.5, "correction": 3},
                "correction_gain": 2,
                "changed_window_s": [97.4, 98.22],
                "scope": "candidate left arm only; original accepted files untouched",
                "maximum_change_rad": float(abs(q - b.joint_pos).max()),
                "maximum_left_arm_acceleration_rad_s2": float(
                    abs(np.gradient(result.joint_vel[:, ids], 0.02, axis=0)).max()
                ),
                "accepted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
