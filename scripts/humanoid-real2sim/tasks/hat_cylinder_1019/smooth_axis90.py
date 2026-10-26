"""Smooth a multi-start candidate; rebuild FK and rigid held-object estimates, then audit separately."""


def main():
    import argparse
    import json
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import numpy as np
    from fit import ARM, MODEL, smooth
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--input-version", required=True)
    p.add_argument("--output-version", required=True)
    p.add_argument("--sigma", type=float, default=6)
    a = p.parse_args()
    r = a.run
    src = r / "retarget" / a.input_version
    out = r / "retarget" / a.output_version
    shutil.copytree(src, out)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = m.joint_pos.astype(float).copy()
    cols = [m.joint_names.index("right_" + x + "_joint") for x in ARM]
    q[:, cols] = gaussian_filter1d(q[:, cols], a.sigma, axis=0, mode="nearest")
    times = 35 + np.arange(len(q)) / 50
    for finger in ["middle", "index"]:
        col = m.joint_names.index("right_hand_" + finger + "_1_joint")
        for i, t in enumerate(times):
            opening = smooth(t, 35, 36.3) * (1 - smooth(t, 48.8, 50))
            closure = smooth(t, 38.3, 39.05) * (1 - smooth(t, 47.64, 48.39))
            q[i, col] += opening * (1 - closure) * 1.74
    res = refresh_fk(replace(m, joint_pos=q), MODEL)
    # Keep the actual release wrist fixed while fingers open, despite moving root.
    import mujoco
    from scipy.interpolate import BPoly
    from scipy.optimize import least_squares

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in m.joint_names])
    wi0 = m.body_names.index("right_wrist_yaw_link")
    bid = model.body("right_wrist_yaw_link").id
    targetp = res.body_pos_w[632, wi0]
    targetr = R.from_quat(res.body_quat_w[632, wi0, [1, 2, 3, 0]])
    lo = np.array([model.joint(m.joint_names[j]).range[0] for j in cols])
    hi = np.array([model.joint(m.joint_names[j]).range[1] for j in cols])
    for i in range(633, 671):
        d.qpos[:3] = m.body_pos_w[i, 0]
        d.qpos[3:7] = m.body_quat_w[i, 0]
        d.qpos[addr] = q[i]

        def residual(x, i=i):
            d.qpos[addr[cols]] = x
            mujoco.mj_forward(model, d)
            rot = R.from_matrix(d.xmat[bid].reshape(3, 3))
            return np.r_[
                (d.xpos[bid] - targetp) * 250, (rot * targetr.inv()).as_rotvec() * 25, (x - q[i - 1, cols]) * 0.1
            ]

        fit = least_squares(residual, np.clip(q[i - 1, cols], lo + 1e-7, hi - 1e-7), bounds=(lo, hi))
        q[i, cols] = fit.x
    base = MotionFields.load(r / "retarget/baseline/motion.npz", src / "motion_order.json")
    curve = BPoly.from_derivatives(
        [times[670], times[-1]],
        [
            [q[670, cols], (q[670, cols] - q[669, cols]) * 50, np.zeros(7)],
            [base.joint_pos[-1, cols], base.joint_vel[-1, cols], np.zeros(7)],
        ],
    )
    q[670:, cols] = curve(times[670:])
    q[:, cols] = np.clip(q[:, cols], lo, hi)
    res = refresh_fk(replace(m, joint_pos=q), MODEL)
    res.save(out / "motion.npz")
    design = json.load(open(r / "review" / ("grasp_design_" + a.input_version + ".json")))
    relative = R.from_quat(design["relative_hat_rotation_xyzw"])
    center = np.array(design["selected"]["center"])
    wi = m.body_names.index("right_wrist_yaw_link")
    wr = R.from_quat(res.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    nr = wr * relative
    np_ = res.body_pos_w[:, wi] + wr.apply(center) - nr.apply([0, 0, 0.11])
    hat = np.load(r / "retarget/baseline/scene_tracks/black_ribbon_hat_free.npz")
    data = {k: hat[k] for k in hat.files}
    hp = data["position_w"].copy()
    hr = R.from_quat(data["quaternion_xyzw"])
    pp = hp.copy()
    qq = hr.as_quat().copy()
    release = 632
    for i in range(len(q)):
        t = 35 + i / 50
        if 38.0 <= t <= 47.64:
            alpha = smooth(t, 38.0, 38.3)
            pp[i] = hp[i] + alpha * (np_[i] - hp[i])
            qq[i] = (R.from_rotvec((nr[i] * hr[i].inv()).as_rotvec() * alpha) * hr[i]).as_quat()
        elif t > 47.64:
            pp[i] = np_[release]
            qq[i] = nr[release].as_quat()
    data.update(position_w=pp, quaternion_xyzw=qq)
    np.savez_compressed(out / "scene_tracks/black_ribbon_hat_free.npz", **data)
    report = json.load(open(src / "review.json"))
    report.update(
        status="smoothed_candidate_pending_audits",
        smoothing_sigma_frames=a.sigma,
        parent=str(src),
        object_deviation_max_m=float(np.linalg.norm(pp - hp, axis=1).max()),
        release_deviation_m=float(np.linalg.norm(pp[release] - hp[release])),
    )
    (out / "review.json").write_text(json.dumps(report, indent=2))
    print(report["release_deviation_m"])


if __name__ == "__main__":
    main()
