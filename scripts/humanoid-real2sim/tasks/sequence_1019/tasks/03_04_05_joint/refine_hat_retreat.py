"""Withdraw along the outward hat-brim normal before rejoining the source wrist path."""

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
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--input-version", default="03_04_05_joint_v14")
    p.add_argument("--output-version", default="03_04_05_joint_v16")
    p.add_argument("--hold-orientation", action="store_true")
    p.add_argument("--release-time", type=float, default=48.0)
    p.add_argument("--retime-fingers", action="store_true")
    p.add_argument("--world-stationary-start", action="store_true")
    p.add_argument("--retreat-direction", choices=["hat_normal", "palm_backward", "center_away"], default="hat_normal")
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.input_version
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(parents=True, exist_ok=True)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    review = json.loads((src / "review.json").read_text())
    center = np.array(review["hat_grasp_fit"]["fixed_center_local_m"])
    hat = np.load(src / "black_ribbon_hat.npz")
    release = a.release_time
    release_frame = round((release - 27) * 50)
    normal = Rotation.from_quat(hat["quaternion_xyzw"][release_frame]).apply([0, 0, 1])
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
    rows = np.flatnonzero((time >= release) & (time <= 49.5))
    t = time[rows]
    prior = q[rows][:, cols].copy()
    knots = (
        [release, release + 0.1, release + 0.3, release + 0.75, 49.5]
        if a.hold_orientation
        else [release, release + 0.2, release + 0.5, 49, 49.5]
    )
    original_rotation = Rotation.from_quat(
        motion.body_quat_w[release_frame, motion.body_names.index("right_wrist_yaw_link")][[1, 2, 3, 0]]
    )
    if a.retreat_direction == "palm_backward":
        normal = original_rotation.apply([-1, 0, 0])
    elif a.retreat_direction == "center_away":
        normal = (
            motion.body_pos_w[release_frame, motion.body_names.index("right_wrist_yaw_link")]
            - hat["position_w"][release_frame]
        )
        normal /= np.linalg.norm(normal)
    rotation_errors = np.zeros((len(rows), 3))
    boundaries = ((1, -(prior[1] - prior[0]) * 50), (1, np.zeros(7))) if a.hold_orientation else "clamped"

    def values(x):
        return prior + CubicSpline(knots, np.vstack([np.zeros(7), x.reshape(3, 7), np.zeros(7)]), bc_type=boundaries)(t)

    def points(v):
        result = []
        for row_index, (k, arm) in enumerate(zip(rows, v, strict=True)):
            d.qpos[:3] = motion.body_pos_w[k, 0]
            d.qpos[3:7] = motion.body_quat_w[k, 0]
            d.qpos[addr] = q[k]
            d.qpos[addr[cols]] = arm
            mujoco.mj_forward(model, d)
            rotation_errors[row_index] = (
                Rotation.from_matrix(d.xmat[wrist].reshape(3, 3)) * original_rotation.inv()
            ).as_rotvec()
            result.append(d.xpos[wrist] + d.xmat[wrist].reshape(3, 3) @ center)
        return np.array(result)

    before = points(prior)
    if a.world_stationary_start:
        k = release_frame
        d.qpos[:3] = motion.body_pos_w[k, 0]
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = q[k]
        mujoco.mj_forward(model, d)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, d, jacp, jacr, before[0], wrist)
        dofs = np.array([model.joint(motion.joint_names[c]).dofadr[0] for c in cols])
        jac = np.vstack([jacp[:, dofs], 0.2 * jacr[:, dofs]])
        prior_arm_velocity = (prior[1] - prior[0]) * 50
        full_velocity = np.r_[(before[1] - before[0]) * 50, 0.2 * rotation_errors[1] * 50]
        non_arm_velocity = full_velocity - jac @ prior_arm_velocity
        desired_arm_velocity = np.linalg.lstsq(jac, -non_arm_velocity, rcond=1e-4)[0]
        boundaries = ((1, desired_arm_velocity - prior_arm_velocity), (1, np.zeros(7)))
    target = CubicSpline(
        [release, release + 0.3, release + 0.75, 49.5],
        np.array([before[0], before[0] + 0.15 * normal, before[0] + 0.18 * normal, before[-1]]),
        bc_type="clamped",
    )(t)
    weight = np.minimum(np.clip((t - release) / 0.08, 0, 1), np.clip((49.5 - t) / 0.3, 0, 1))
    lo = np.array([model.joint(motion.joint_names[c]).range[0] for c in cols])
    hi = np.array([model.joint(motion.joint_names[c]).range[1] for c in cols])

    def residual(x):
        v = values(x)
        return np.r_[
            ((points(v) - target) * weight[:, None] * 200).ravel(),
            (
                rotation_errors * np.clip((release + 0.6 - t) / 0.2, 0, 1)[:, None] * (45 if a.hold_orientation else 0)
            ).ravel(),
            ((v - prior) * 0.8).ravel(),
            (np.diff(v - prior, n=2, axis=0) * 100).ravel(),
            (np.maximum(lo - v, 0) * 200).ravel(),
            (np.maximum(v - hi, 0) * 200).ravel(),
        ]

    fit = least_squares(residual, np.zeros(21), max_nfev=50, ftol=1e-5)
    v = np.clip(values(fit.x), lo, hi)
    q[np.ix_(rows, cols)] = v
    if a.retime_fingers:
        finger_rows = np.flatnonzero((time >= release - 0.2) & (time <= 49.5))
        u = np.clip((time[finger_rows] - (release - 0.2)) / 0.25, 0, 1)
        closed = 1 - u * u * (3 - 2 * u)
        for j, name in enumerate(motion.joint_names):
            if name.startswith("right_hand"):
                angle = 0 if "thumb_0" in name else (-0.3 if "thumb" in name else 0.4)
                q[finger_rows, j] = angle * closed
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in src.iterdir():
        if f.is_file() and f.name not in ["motion.npz", "review.json", "manifest.json", "collision_diagnostic.json"]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    review["status"] = "hat_retreat_candidate_not_passed"
    review["hat_retreat_fit"] = {
        "window_s": [release, 49.5],
        "release_arm_frame_preserved_max_joint_difference_rad": float(
            abs(result.joint_pos[release_frame, cols] - motion.joint_pos[release_frame, cols]).max()
        ),
        "fingers_retimed": a.retime_fingers,
        "world_stationary_start": a.world_stationary_start,
        "retreat_direction": a.retreat_direction,
        "retreat_unit_world": normal.tolist(),
        "before_max_path_error_m": float(np.linalg.norm(before - target, axis=1).max()),
        "after_max_path_error_m": float(np.linalg.norm(points(v) - target, axis=1).max()),
        "max_arm_change_rad": float(abs(v - prior).max()),
        "solver_success": bool(fit.success),
        "nfev": fit.nfev,
        "collision_audit_pending": True,
    }
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(review["hat_retreat_fit"])


if __name__ == "__main__":
    main()
