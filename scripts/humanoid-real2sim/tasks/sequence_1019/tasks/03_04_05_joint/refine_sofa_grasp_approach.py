"""Fit the same hat grasp transform to the registered sofa's settled initial pose."""

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
    p.add_argument("--input-version", required=True)
    p.add_argument("--output-version", required=True)
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.input_version
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(exist_ok=True)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    hat = np.load(src / "black_ribbon_hat.npz")
    q = motion.joint_pos.astype(float).copy()
    time = 27 + np.arange(len(q)) / 50
    wi = motion.body_names.index("right_wrist_yaw_link")
    wr = Rotation.from_quat(motion.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    g = 600
    center = wr[g].inv().apply(hat["position_w"][g] - motion.body_pos_w[g, wi])
    relative = wr[g].inv() * Rotation.from_quat(hat["quaternion_xyzw"][g])
    rest = np.array(
        json.loads((a.run / "review/scene/03_04_05_registered_sofa_support.json").read_text())["settled_hat_qpos"]
    )
    target = rest[:3]
    targetR = Rotation.from_quat(rest[3:7][[1, 2, 3, 0]])
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(model)
    names = [
        "right_" + s + "_joint"
        for s in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    cols = [motion.joint_names.index(n) for n in names]
    assert len(cols) == 7 and all("ankle" not in n for n in names)
    addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    w = model.body("right_wrist_yaw_link").id
    rows = np.flatnonzero((time >= 36.5) & (time <= 41.5))
    t = time[rows]
    prior = q[rows][:, cols].copy()
    knots = [36.5, 37.0, 37.4, 37.8, 38.3, 39, 40.2, 41.5]
    away = motion.body_pos_w[g, wi] - hat["position_w"][g]
    away /= np.linalg.norm(away)
    u = np.clip((38.3 - t) / 1.0, 0, 1)
    desired = target + (0.16 * u * u * (3 - 2 * u))[:, None] * away
    weight = np.clip((t - 37.0) / 0.3, 0, 1) * np.where(t <= 39, 1, np.exp(-0.5 * ((t - 39) / 0.12) ** 2))
    lo = np.array([model.joint(n).range[0] for n in names])
    hi = np.array([model.joint(n).range[1] for n in names])

    def values(x):
        return prior + CubicSpline(knots, np.vstack([np.zeros(7), x.reshape(-1, 7), np.zeros(7)]), bc_type="clamped")(t)

    def poses(v):
        ps = []
        rs = []
        for k, arm in zip(rows, v, strict=True):
            d.qpos[:3] = motion.body_pos_w[k, 0]
            d.qpos[3:7] = motion.body_quat_w[k, 0]
            d.qpos[addr] = q[k]
            d.qpos[addr[cols]] = arm
            mujoco.mj_forward(model, d)
            R = Rotation.from_matrix(d.xmat[w].reshape(3, 3))
            ps.append(d.xpos[w] + R.apply(center))
            rs.append(((R * relative) * targetR.inv()).as_rotvec())
        return np.array(ps), np.array(rs)

    def residual(x):
        v = values(x)
        pos, rot = poses(v)
        return np.r_[
            ((pos - desired) * weight[:, None] * 350).ravel(),
            (rot * weight[:, None] * 60).ravel(),
            (pos[np.argmin(abs(t - 39))] - target) * 2000,
            rot[np.argmin(abs(t - 39))] * 500,
            ((v - prior) * 0.3).ravel(),
            (np.diff(v - prior, n=2, axis=0) * 80).ravel(),
            (np.maximum(lo - v, 0) * 400).ravel(),
            (np.maximum(v - hi, 0) * 400).ravel(),
        ]

    fit = least_squares(residual, np.zeros((len(knots) - 2) * 7), max_nfev=80, ftol=1e-5)
    v = np.clip(values(fit.x), lo, hi)
    q[np.ix_(rows, cols)] = v
    finger_rows = np.flatnonzero((time >= 37) & (time < 39))
    u = np.clip((time[finger_rows] - 37) / 0.3, 0, 1)
    close = u * u * (3 - 2 * u)
    for j, name in enumerate(motion.joint_names):
        if name.startswith("right_hand"):
            q[finger_rows, j] = (0 if "thumb_0" in name else (-0.3 if "thumb" in name else 0.4)) * close
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    R = Rotation.from_quat(result.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    pos = hat["position_w"].copy()
    quat = hat["quaternion_xyzw"].copy()
    pos[:g] = target
    quat[:g] = targetR.as_quat()
    end = 725
    pos[g : end + 1] = result.body_pos_w[g : end + 1, wi] + R[g : end + 1].apply(center)
    quat[g : end + 1] = (R[g : end + 1] * relative).as_quat()
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "black_ribbon_hat.npz",
            "review.json",
            "manifest.json",
            "protection_check.json",
            "collision_diagnostic.json",
        ]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    np.savez_compressed(
        out / "black_ribbon_hat.npz", position_w=pos, quaternion_xyzw=quat, fps=np.array(50), source_time_s=time
    )
    review = json.loads((src / "review.json").read_text())
    summary = {
        "window_s": [36.5, 41.5],
        "initial_pose_source": "registered sofa surface gravity rest, rigid heightfield patch",
        "grasp_s": 39,
        "position_error_m": float(np.linalg.norm(pos[g] - target)),
        "orientation_error_rad": float((Rotation.from_quat(quat[g]) * targetR.inv()).magnitude()),
        "max_arm_change_rad": float(abs(v - prior).max()),
        "fixed_grasp_preserved": True,
        "solver_success": bool(fit.success),
        "nfev": fit.nfev,
        "source_event_limit": "Head/right views show established grasp by38s, lift around39s; precise first touch remains unresolved.",
    }
    review.update(status="registered_sofa_grasp_candidate_not_passed", registered_sofa_grasp=summary)
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
