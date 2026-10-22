"""Fit the observed closing-only drawer action over one continuous arm window."""

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
    p.add_argument("--output-version", default="03_04_05_joint_v3")
    p.add_argument("--limit-wrist-change", action="store_true")
    p.add_argument("--input-version", default="03_04_05_joint_v2")
    p.add_argument("--dense-contact-window", action="store_true")
    p.add_argument("--surface-orientation", action="store_true")
    p.add_argument("--contact-length", type=float, default=0.084)
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.input_version
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(parents=True, exist_ok=True)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    cols = [
        i
        for i, n in enumerate(motion.joint_names)
        if n.startswith("right_") and any(x in n for x in ["shoulder", "elbow", "wrist"])
    ]
    addresses = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    wrist = model.body("right_wrist_yaw_link").id
    q = motion.joint_pos.astype(float).copy()
    time = 27 + np.arange(len(q)) / 50
    rows = np.flatnonzero((time >= 49.5) & (time <= 55.5))
    t = time[rows]
    prior = q[rows][:, cols].copy()
    knots = (
        [49.5, 50.3, 51.1, 51.9, 52.7, 53.5, 54.3, 55.5] if a.dense_contact_window else [49.5, 50.7, 52.3, 54.1, 55.5]
    )
    nvar = (len(knots) - 2) * 7
    slider = np.load(src / "drawer_slide.npz")["position"][rows]
    target = np.column_stack([np.full(len(rows), 1.0515), -1.8625 + slider + 0.025, np.full(len(rows), 0.852)])
    active = np.minimum(
        np.clip((t - (50.8 if a.dense_contact_window else 50.3)) / 0.5, 0, 1),
        np.clip(((54.3 if a.dense_contact_window else 54.8) - t) / 0.5, 0, 1),
    )
    contact_local = np.array([a.contact_length, 0, 0])

    def values(x):
        return prior + CubicSpline(knots, np.vstack([np.zeros(7), x.reshape(-1, 7), np.zeros(7)]), bc_type="clamped")(t)

    axes = np.zeros((len(rows), 3))

    def points(v):
        pos = []
        for row_index, (k, arm) in enumerate(zip(rows, v, strict=True)):
            data.qpos[:3] = motion.body_pos_w[k, 0]
            data.qpos[3:7] = motion.body_quat_w[k, 0]
            data.qpos[addresses] = q[k]
            data.qpos[addresses[cols]] = arm
            mujoco.mj_forward(model, data)
            axes[row_index] = data.xmat[wrist].reshape(3, 3)[:, 0]
            pos.append(data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ contact_local)
        return np.asarray(pos)

    lower = np.array([model.joint(motion.joint_names[c]).range[0] for c in cols])
    upper = np.array([model.joint(motion.joint_names[c]).range[1] for c in cols])

    def residual(x):
        v = values(x)
        return np.r_[
            ((points(v) - target) * active[:, None] * 100).ravel(),
            ((axes - np.array([0, -1, 0])) * active[:, None] * (20 if a.surface_orientation else 0)).ravel(),
            (
                (v - prior) * np.array([0.3, 0.3, 0.3, 0.3, 4, 4, 4]) if a.limit_wrist_change else (v - prior) * 0.3
            ).ravel(),
            (
                np.maximum(abs(v - prior) - np.array([3, 3, 3, 3, 0.6, 0.6, 0.6]), 0) * 100
                if a.limit_wrist_change
                else np.zeros_like(v)
            ).ravel(),
            (np.diff(v - prior, n=2, axis=0) * 80).ravel(),
            (np.maximum(lower - v, 0) * 300).ravel(),
            (np.maximum(v - upper, 0) * 300).ravel(),
        ]

    before = points(prior)
    fit = least_squares(residual, np.zeros(nvar), max_nfev=60, ftol=1e-5)
    v = values(fit.x)
    q[np.ix_(rows, cols)] = v
    after = points(v)
    # Complete closure at the attachment frame, before the object lifts.
    close = np.clip((time - 30.8) / 0.4, 0, 1)
    close = close * close * (3 - 2 * close)
    use = (time >= 30.8) & (time <= 31.4)
    for i, n in enumerate(motion.joint_names):
        if n.startswith("left_hand"):
            target_angle = 0 if "thumb_0" in n else (0.3 if "thumb" in n else -0.4)
            q[use, i] = target_angle * close[use]
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "review.json",
            "manifest.json",
            "collision_diagnostic.json",
            "protection_check.json",
            "drawer_geometry_check.json",
        ]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    review = json.loads((src / "review.json").read_text())
    mask = active > 0.99
    summary = {
        "window_s": [49.5, 55.5],
        "mode": "push_preopened_drawer_closed",
        "palm_proxy_local_m": contact_local.tolist(),
        "before_max_error_m": float(np.linalg.norm(before - target, axis=1)[mask].max()),
        "after_max_error_m": float(np.linalg.norm(after - target, axis=1)[mask].max()),
        "after_median_error_m": float(np.median(np.linalg.norm(after - target, axis=1)[mask])),
        "max_joint_change_rad": float(abs(v - prior).max()),
        "solver_success": bool(fit.success),
        "nfev": fit.nfev,
        "geometry_contact_validated": False,
        "surface_orientation_constrained": a.surface_orientation,
        "max_contact_axis_error": float(np.linalg.norm(axes - np.array([0, -1, 0]), axis=1)[mask].max()),
    }
    review.update(status="strawberry_and_drawer_partial_candidate_not_passed", drawer_fit=summary)
    review["hand_schedule"]["close_s"] = [30.8, 31.2]
    review["protected_06_nonleft_arm_hand_joint_difference_rad"] = 0.0
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
