"""Smooth a geometric entry solution while fixing both seam poses and velocities."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    import mujoco
    from scipy.optimize import lsq_linear

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="v3")
    p.add_argument("--weight", type=float, default=40)
    a = p.parse_args()
    r = a.run
    src = r / "retarget/v2"
    out = r / "retarget" / a.version
    shutil.copytree(src, out)
    b = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = b.joint_pos.astype(float).copy()
    cols = [
        i
        for i, n in enumerate(b.joint_names)
        if n.startswith("right_") and any(s in n for s in ["shoulder", "elbow", "wrist"])
    ]
    st = Path("/home/galbot/workspace/ScaleTrack")
    ch = np.load(st / "data/1019test/push-armchair-thumb-kinematic-v1/motion.npz")
    old = np.load(st / "data/1019test/brace-dining-table-kinematic-v1/motion.npz")
    n = 81
    d2 = np.diff(np.eye(n), n=2, axis=0)
    system = np.vstack([np.eye(n), np.sqrt(a.weight) * d2])
    fixed = np.array([0, 1, 79, 80])
    free = np.array([i for i in range(n) if i not in fixed])
    values = q[np.ix_(fixed, cols)].copy()
    values[1] = values[0] + (ch["joint_pos"][425, cols] - ch["joint_pos"][424, cols])
    values[2] = values[3] - (old["joint_pos"][81, cols] - old["joint_pos"][79, cols]) / 2
    x = q[:n, cols].copy()
    x[fixed] = values
    model = st / "source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    m = mujoco.MjModel.from_xml_path(str(model))
    for j, col in enumerate(cols):
        lo, hi = m.joint(b.joint_names[col]).range
        values[:, j] = np.clip(values[:, j], lo, hi)
        x[fixed, j] = values[:, j]
        rhs = np.r_[q[:n, col], np.zeros(n - 2)] - system[:, fixed] @ values[:, j]
        fit = lsq_linear(system[:, free], rhs, bounds=(lo + 1e-6, hi - 1e-6), tol=1e-10)
        assert fit.success
        x[free, j] = fit.x
    q[:n, cols] = x
    model = st / "source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    refresh_fk(replace(b, joint_pos=q), model).save(out / "motion.npz")
    report = {
        "source": "v2",
        "method": "quadratic second-difference penalty, first/last poses and adjacent velocities fixed",
        "weight": a.weight,
        "active_arm_peak_acceleration_before_rad_s2": float(
            abs(np.diff(b.joint_pos[:83, cols], n=2, axis=0) * 2500).max()
        ),
        "active_arm_peak_acceleration_after_rad_s2": float(abs(np.diff(q[:83, cols], n=2, axis=0) * 2500).max()),
        "max_joint_correction_rad": float(abs(q - b.joint_pos).max()),
    }
    (out / "smoothing.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
