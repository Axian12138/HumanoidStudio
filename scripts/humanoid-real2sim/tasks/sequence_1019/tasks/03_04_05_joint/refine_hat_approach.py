"""Keep the established hat contact frame and repair its approach only."""

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
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    src = a.run / "retarget/candidates/03_04_05_joint_v11"
    out = a.run / "retarget/candidates/03_04_05_joint_v12"
    out.mkdir(parents=True, exist_ok=True)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    review = json.loads((src / "review.json").read_text())
    center = np.array(review["hat_grasp_fit"]["fixed_center_local_m"])
    floor = np.array(np.load(src / "black_ribbon_hat.npz")["position_w"][600])
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(model)
    cols = [
        motion.joint_names.index("right_" + n + "_joint")
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    wrist = model.body("right_wrist_yaw_link").id
    q = motion.joint_pos.astype(float).copy()
    time = 27 + np.arange(len(q)) / 50
    rows = np.flatnonzero((time >= 36.5) & (time <= 39.0))
    t = time[rows]
    prior = q[rows][:, cols].copy()
    knots = [36.5, 37.1, 37.8, 38.4, 39.0]

    def values(x):
        return prior + CubicSpline(knots, np.vstack([np.zeros(7), x.reshape(3, 7), np.zeros(7)]), bc_type="clamped")(t)

    def points(v):
        result = []
        for k, arm in zip(rows, v, strict=True):
            d.qpos[:3] = motion.body_pos_w[k, 0]
            d.qpos[3:7] = motion.body_quat_w[k, 0]
            d.qpos[addr] = q[k]
            d.qpos[addr[cols]] = arm
            mujoco.mj_forward(model, d)
            result.append(d.xpos[wrist] + d.xmat[wrist].reshape(3, 3) @ center)
        return np.array(result)

    before = points(prior)
    target = CubicSpline(
        [36.5, 37.9, 38.4, 39.0],
        np.array([before[0], floor + np.array([0, 0, 0.12]), floor + np.array([0, 0, 0.065]), floor]),
        bc_type="clamped",
    )(t)
    weight = np.minimum(np.clip((t - 36.5) / 0.5, 0, 1), np.clip((39.0 - t) / 0.25, 0, 1))
    lo = np.array([model.joint(motion.joint_names[c]).range[0] for c in cols])
    hi = np.array([model.joint(motion.joint_names[c]).range[1] for c in cols])

    def residual(x):
        v = values(x)
        return np.r_[
            ((points(v) - target) * weight[:, None] * 150).ravel(),
            ((v - prior) * 0.8).ravel(),
            (np.diff(v - prior, n=2, axis=0) * 100).ravel(),
            (np.maximum(lo - v, 0) * 200).ravel(),
            (np.maximum(v - hi, 0) * 200).ravel(),
        ]

    fit = least_squares(residual, np.zeros(21), max_nfev=50, ftol=1e-5)
    v = values(fit.x)
    q[np.ix_(rows, cols)] = v
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in src.iterdir():
        if f.is_file() and f.name not in ["motion.npz", "review.json", "manifest.json", "collision_diagnostic.json"]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    review["status"] = "hat_approach_refined_candidate_not_passed"
    review["hat_approach_fit"] = {
        "window_s": [36.5, 39.0],
        "contact_frame_preserved_max_joint_difference_rad": float(
            abs(result.joint_pos[600] - motion.joint_pos[600]).max()
        ),
        "before_max_path_error_m": float(np.linalg.norm(before - target, axis=1).max()),
        "after_max_path_error_m": float(np.linalg.norm(points(v) - target, axis=1).max()),
        "max_arm_change_rad": float(abs(v - prior).max()),
        "solver_success": bool(fit.success),
        "nfev": fit.nfev,
        "collision_audit_pending": True,
    }
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(review["hat_approach_fit"])


if __name__ == "__main__":
    main()
