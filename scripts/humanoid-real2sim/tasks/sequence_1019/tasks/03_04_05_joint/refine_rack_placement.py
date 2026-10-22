"""Fit the actual hat cavity onto the source rack and simulate its released support."""

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
    p.add_argument("--run", required=True, type=Path)
    a = p.parse_args()
    src = a.run / "retarget/candidates/03_04_05_joint_v17"
    out = a.run / "retarget/candidates/03_04_05_joint_v18"
    out.mkdir(parents=True, exist_ok=True)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    hat = np.load(src / "black_ribbon_hat.npz")
    q = m.joint_pos.copy().astype(float)
    time = 27 + np.arange(len(q)) / 50
    wi = m.body_names.index("right_wrist_yaw_link")
    wr = Rotation.from_quat(m.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    release = 1050
    center = wr[release].inv().apply(hat["position_w"][release] - m.body_pos_w[release, wi])
    relative = wr[release].inv() * Rotation.from_quat(hat["quaternion_xyzw"][release])
    target = np.array([2.150866718016237, -1.5736, 0.9271032539115094])
    rotation = Rotation.from_euler("y", -70, degrees=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in m.joint_names])
    cols = [
        m.joint_names.index("right_" + n + "_joint")
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    wrist = model.body("right_wrist_yaw_link").id
    rows = np.flatnonzero((time >= 44.5) & (time <= 49.5))
    t = time[rows]
    prior = q[rows][:, cols].copy()
    knots = [44.5, 45.5, 46.5, 47.3, 48, 48.7, 49.5]
    weight = np.exp(-0.5 * ((t - 48) / 0.12) ** 2)

    def values(x):
        return prior + CubicSpline(knots, np.vstack([np.zeros(7), x.reshape(-1, 7), np.zeros(7)]), bc_type="clamped")(t)

    def poses(v):
        ps = []
        rs = []
        for k, arm in zip(rows, v, strict=True):
            d.qpos[:3] = m.body_pos_w[k, 0]
            d.qpos[3:7] = m.body_quat_w[k, 0]
            d.qpos[addr] = q[k]
            d.qpos[addr[cols]] = arm
            mujoco.mj_forward(model, d)
            R = Rotation.from_matrix(d.xmat[wrist].reshape(3, 3))
            ps.append(d.xpos[wrist] + R.apply(center))
            rs.append(((R * relative) * rotation.inv()).as_rotvec())
        return np.array(ps), np.array(rs)

    lower = np.array([model.joint(m.joint_names[c]).range[0] for c in cols])
    upper = np.array([model.joint(m.joint_names[c]).range[1] for c in cols])

    def residual(x):
        v = values(x)
        pos, rot = poses(v)
        return np.r_[
            ((pos - target) * weight[:, None] * 350).ravel(),
            (rot * weight[:, None] * 80).ravel(),
            ((v - prior) * 0.3).ravel(),
            (np.diff(v - prior, n=2, axis=0) * 80).ravel(),
            (np.maximum(lower - v, 0) * 400).ravel(),
            (np.maximum(v - upper, 0) * 400).ravel(),
        ]

    fit = least_squares(residual, np.zeros(35), max_nfev=70, ftol=1e-5)
    v = values(fit.x)
    q[np.ix_(rows, cols)] = v
    result = refresh_fk(replace(m, joint_pos=q), MODEL)
    newR = Rotation.from_quat(result.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    pos = hat["position_w"].copy()
    quat = hat["quaternion_xyzw"].copy()
    start = 875
    pos[start : release + 1] = result.body_pos_w[start : release + 1, wi] + newR[start : release + 1].apply(center)
    quat[start : release + 1] = (newR[start : release + 1] * relative).as_quat()
    physics = mujoco.MjModel.from_xml_path(str(a.run / "scene/03_04_05/hat_rack_physics.xml"))
    pd = mujoco.MjData(physics)
    pd.qpos[:3] = pos[release]
    pd.qpos[3:7] = quat[release][[3, 0, 1, 2]]
    mujoco.mj_forward(physics, pd)
    initial_penetration = min([float(c.dist) for c in pd.contact] or [0])
    samples = []
    for frame in range(release + 1, len(q)):
        for _ in range(10):
            mujoco.mj_step(physics, pd)
        pos[frame] = pd.qpos[:3]
        quat[frame] = pd.qpos[3:7][[1, 2, 3, 0]]
        if frame % 50 == 0:
            samples.append(
                {
                    "source_s": 27 + frame / 50,
                    "position_m": pos[frame].tolist(),
                    "speed": float(np.linalg.norm(pd.qvel)),
                }
            )
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "black_ribbon_hat.npz",
            "manifest.json",
            "review.json",
            "collision_diagnostic.json",
            "protection_check.json",
        ]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    np.savez_compressed(
        out / "black_ribbon_hat.npz", position_w=pos, quaternion_xyzw=quat, fps=np.array(50), source_time_s=time
    )
    review = json.loads((src / "review.json").read_text())
    summary = {
        "window_s": [44.5, 49.5],
        "target_release_position": target.tolist(),
        "release_position_error_m": float(np.linalg.norm(pos[release] - target)),
        "release_orientation_error_rad": float((Rotation.from_quat(quat[release]) * rotation.inv()).magnitude()),
        "max_arm_change_rad": float(abs(v - prior).max()),
        "solver_success": bool(fit.success),
        "initial_physics_penetration_m": initial_penetration,
        "after_release_model": "actual hat and306 authored rack collision segments, MuJoCo gravity2ms dt, no robot force",
        "physics_samples": samples,
        "minimum_released_root_height_m": float(pos[release:, 2].min()),
        "last5s_drift_m": float(np.linalg.norm(pos[-1] - pos[-251])),
    }
    review.update(status="rack_support_candidate_not_passed", rack_placement=summary)
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print({k: v for k, v in summary.items() if k != "physics_samples"})


if __name__ == "__main__":
    main()
