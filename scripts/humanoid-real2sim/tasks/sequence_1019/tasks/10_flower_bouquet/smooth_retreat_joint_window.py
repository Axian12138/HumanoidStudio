"""Jointly smooth a local retreat with explicit velocity and acceleration bounds."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import Bounds, LinearConstraint, minimize

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion", "order", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--window", nargs=2, type=float, default=[120.4, 121.8])
    parser.add_argument("--acceleration", type=float, default=60.0)
    parser.add_argument("--curvature-weight", type=float, default=10.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    motion = MotionFields.load(args.motion, args.order)
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(robot))
    first, last = [round((t - 108) * motion.fps) for t in args.window]
    size = last - first - 1
    embed = np.zeros((size + 4, size))
    embed[2:-2] = np.eye(size)
    d1, d2 = np.diff(embed, axis=0), np.diff(embed, n=2, axis=0)
    q = motion.joint_pos.copy()
    rows = []
    for column, name in enumerate(motion.joint_names):
        if not name.startswith("left_") or not any(part in name for part in ("shoulder_", "elbow_", "wrist_", "hand_")):
            continue
        fixed = motion.joint_pos[first - 1 : last + 2, column].astype(float).copy()
        target = fixed[2:-2].copy()
        fixed[2:-2] = 0
        c1, c2 = np.diff(fixed), np.diff(fixed, n=2)
        speed = (2.95 if "hand_" in name else 6.0) / motion.fps
        accel = args.acceleration / motion.fps**2
        constraint = LinearConstraint(
            np.vstack([d1, d2]),
            np.r_[-speed - c1, -accel - c2],
            np.r_[speed - c1, accel - c2],
        )
        lower, upper = model.joint(name).range

        def fun(y, target=target, c2=c2):
            error = y - target
            curvature = d2 @ y + c2
            return 0.5 * error @ error + args.curvature_weight * curvature @ curvature

        def jac(y, target=target, c2=c2):
            return y - target + 2 * args.curvature_weight * d2.T @ (d2 @ y + c2)

        solution = minimize(
            fun,
            target,
            jac=jac,
            method="SLSQP",
            bounds=Bounds(np.full(size, lower), np.full(size, upper)),
            constraints=[constraint],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not solution.success:
            raise RuntimeError((name, solution.message))
        q[first + 1 : last, column] = solution.x
        rows.append(
            {"joint": name, "max_delta_rad": float(abs(solution.x - target).max()), "cost": float(solution.fun)}
        )
    refresh_fk(replace(motion, joint_pos=q), robot).save(args.out / "motion.npz")
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "status": "unreviewed smoothing candidate; geometry and source checks required",
                "seed_sha256": sha256(args.motion),
                "window_s": args.window,
                "acceleration_bound_rad_s2": args.acceleration,
                "curvature_weight": args.curvature_weight,
                "acceleration_scope": "Explicit smoothing planning bound, not an identified hardware rating.",
                "preserved": "Root, legs, waist, right arm and all poses outside window; first and last window joint positions.",
                "rows": rows,
            },
            indent=2,
        )
    )
    print(args.out)


if __name__ == "__main__":
    main()
