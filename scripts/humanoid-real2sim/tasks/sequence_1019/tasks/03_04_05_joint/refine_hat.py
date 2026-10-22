"""Fit the sofa hat grasp with a fixed geometric hand design; rack remains pending."""

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
    a = p.parse_args()
    src = a.run / "retarget/candidates/03_04_05_joint_v7"
    out = a.run / "retarget/candidates/03_04_05_joint_v8"
    out.mkdir(parents=True, exist_ok=True)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = motion.joint_pos.astype(float).copy()
    time = 27 + np.arange(len(q)) / 50
    names = tuple(
        "right_" + n + "_joint"
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    )
    cols = [motion.joint_names.index(n) for n in names]
    assert len(cols) == 7
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    wrist = model.body("right_wrist_yaw_link").id
    geometry = json.loads((a.run / "review/stage3/03_04_05_joint_v0_black_ribbon_hat_grasp_geometry.json").read_text())[
        "records"
    ][0]
    center = np.array(geometry["center_local_m"])
    g = 600
    r = 1050
    wi = motion.body_names.index("right_wrist_yaw_link")
    original_wr = Rotation.from_quat(motion.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    relative = original_wr[g].inv()
    target = np.array([2.005000114, 0.819999993, 0.443100005])
    rows = np.flatnonzero((time >= 36.5) & (time <= 41.5))
    t = time[rows]
    prior = q[rows][:, cols].copy()
    knots = [36.5, 37.7, 39, 40.2, 41.5]
    weight = np.exp(-0.5 * ((t - 39) / 0.18) ** 2)

    def values(x):
        return prior + CubicSpline(knots, np.vstack([np.zeros(7), x.reshape(3, 7), np.zeros(7)]), bc_type="clamped")(t)

    def poses(v):
        ps = []
        rs = []
        for k, arm in zip(rows, v, strict=True):
            data.qpos[:3] = motion.body_pos_w[k, 0]
            data.qpos[3:7] = motion.body_quat_w[k, 0]
            data.qpos[addr] = q[k]
            data.qpos[addr[cols]] = arm
            mujoco.mj_forward(model, data)
            rot = Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))
            ps.append(data.xpos[wrist] + rot.apply(center))
            rs.append((rot * relative).as_rotvec())
        return np.array(ps), np.array(rs)

    lower = np.array([model.joint(n).range[0] for n in names])
    upper = np.array([model.joint(n).range[1] for n in names])

    def residual(x):
        v = values(x)
        pos, rot = poses(v)
        return np.r_[
            ((pos - target) * weight[:, None] * 250).ravel(),
            (rot * weight[:, None] * 15).ravel(),
            ((v - prior) * 0.5).ravel(),
            (np.diff(v - prior, n=2, axis=0) * 80).ravel(),
            (np.maximum(lower - v, 0) * 300).ravel(),
            (np.maximum(v - upper, 0) * 300).ravel(),
        ]

    fit = least_squares(residual, np.zeros(21), max_nfev=60, ftol=1e-5)
    v = values(fit.x)
    q[np.ix_(rows, cols)] = v

    def smooth(start, end):
        u = np.clip((time - start) / (end - start), 0, 1)
        return u * u * (3 - 2 * u)

    engage = smooth(38.5, 39) * (1 - smooth(47.8, 48.3))
    window = smooth(36.5, 37) * (1 - smooth(49, 49.5))
    for j, n in enumerate(motion.joint_names):
        if n.startswith("right_hand"):
            angle = 0 if "thumb_0" in n else (-0.3 if "thumb" in n else 0.4)
            q[:, j] = (1 - window) * q[:, j] + window * engage * angle
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    wp = result.body_pos_w[:, wi]
    wr = Rotation.from_quat(result.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    # Keep the source sofa root until grasp; ramp the submillimetre residual over
    # the approach. No claim that the later frozen root has physical rack support.
    pos = np.repeat(target[None], len(time), axis=0)
    quat = np.repeat([[0.0, 0.0, 0.0, 1.0]], len(time), axis=0)
    attached = wp[g : r + 1] + wr[g : r + 1].apply(center)
    delta = attached[0] - target
    pos[:g] += smooth(38.5, 39)[:g, None] * delta
    pos[g : r + 1] = attached
    quat[g : r + 1] = (wr[g : r + 1] * relative).as_quat()
    pos[r + 1 :] = pos[r]
    quat[r + 1 :] = quat[r]
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "black_ribbon_hat.npz",
            "review.json",
            "manifest.json",
            "collision_diagnostic.json",
        ]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    np.savez_compressed(
        out / "black_ribbon_hat.npz", position_w=pos, quaternion_xyzw=quat, fps=np.array(50), source_time_s=time
    )
    review = json.loads((src / "review.json").read_text())
    review.update(
        status="hat_grasp_partial_candidate_not_passed",
        hat_grasp_fit={
            "source_s": 39,
            "window_s": [36.5, 41.5],
            "max_arm_change_rad": float(abs(v - prior).max()),
            "grasp_root_change_from_sofa_m": delta.tolist(),
            "grasp_orientation_residual_rad": float(np.linalg.norm((wr[g] * relative).as_rotvec())),
            "fixed_center_local_m": center.tolist(),
            "hand_close_s": [38.5, 39],
            "hand_release_s": [47.8, 48.3],
            "solver_success": bool(fit.success),
            "nfev": fit.nfev,
            "rack_support_verified": False,
            "collision_audit_pending": True,
        },
    )
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(review["hat_grasp_fit"])


if __name__ == "__main__":
    main()
