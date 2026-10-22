"""Minimal local joint adjustment under an explicit adjacent-frame speed contract."""

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
    from scipy.optimize import Bounds, LinearConstraint, minimize

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--input-version", required=True)
    p.add_argument("--output-version", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--prefix", default="right_")
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.input_version
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(exist_ok=False)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = motion.joint_pos.astype(float).copy()
    rows = np.arange(round((a.start - 27) * 50), round((a.end - 27) * 50) + 1)
    D = np.diff(np.eye(len(rows)), axis=0)
    limit = 2.9999 / 50
    reports = []
    for j, name in enumerate(motion.joint_names):
        if not name.startswith(a.prefix) or "ankle" in name or "knee" in name or "hip" in name:
            continue
        y = q[rows, j].copy()
        if abs(np.diff(y)).max() <= limit:
            continue
        low = np.full(len(rows), -np.inf)
        high = np.full(len(rows), np.inf)
        low[[0, -1]] = high[[0, -1]] = y[[0, -1]]

        def objective(x, y=y):
            error = x - y
            smooth = D @ error
            return 0.5 * (error @ error + smooth @ smooth), error + D.T @ smooth

        fit = minimize(
            objective,
            y,
            method="SLSQP",
            jac=True,
            bounds=Bounds(low, high),
            constraints=[LinearConstraint(D, -limit, limit)],
            options={"ftol": 1e-12, "maxiter": 100},
        )
        if not fit.success:
            raise RuntimeError(f"{name}: {fit.message}")
        q[rows, j] = fit.x
        reports.append(
            {
                "joint": name,
                "max_delta_rad": float(abs(fit.x - y).max()),
                "before_speed_rad_s": float(abs(np.diff(y)).max() * 50),
                "after_speed_rad_s": float(abs(np.diff(fit.x)).max() * 50),
            }
        )
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in src.iterdir():
        if (
            f.is_file()
            and f.name not in ["motion.npz", "review.json", "manifest.json"]
            and not f.name.startswith("collision_")
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    review = json.loads((src / "review.json").read_text())
    review["speed_projection"] = {
        "window_s": [a.start, a.end],
        "contract": "retarget3 3rad/s fallback, not hardware specification",
        "joints": reports,
        "collision_reaudit_required": True,
    }
    review["status"] = "speed_projection_pending_geometry"
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(review["speed_projection"])


if __name__ == "__main__":
    main()
