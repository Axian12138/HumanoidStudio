"""Restore human-derived wrist posture during free carry, protecting acquisition and release."""


def main():
    import argparse
    import json
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import mujoco
    import numpy as np
    from fit import ARM, MODEL, smooth
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="natural_v4")
    a = p.parse_args()
    r = a.run
    src = r / "retarget/accepted_grasp"
    out = r / "retarget" / a.version
    shutil.copytree(src, out)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    prior = MotionFields.load(r / "retarget/human_body_prior/motion.npz", src / "motion_order.json")
    q = motion.joint_pos.astype(float).copy()
    cols = [motion.joint_names.index("right_" + x + "_joint") for x in ARM]
    bodycols = cols[:4]
    wristcols = cols[4:]
    tt = 35 + np.arange(len(q)) / 50
    blend = np.array([smooth(t, 39.3, 40.0) * (1 - smooth(t, 44.8, 45.8)) for t in tt])
    q[:, wristcols] += (prior.joint_pos[:, wristcols] - q[:, wristcols]) * blend[:, None]
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    w = model.body("right_wrist_yaw_link").id
    lo = np.array([model.joint(motion.joint_names[j]).range[0] for j in bodycols])
    hi = np.array([model.joint(motion.joint_names[j]).range[1] for j in bodycols])
    hat = np.load(src / "scene_tracks/black_ribbon_hat_free.npz")
    data = {k: hat[k].copy() for k in hat.files}
    hr = R.from_quat(data["quaternion_xyzw"])
    centers = data["position_w"] + hr.apply([0, 0, 0.11])
    design = json.load(open(r / "review/grasp_design_enclosing_v1.json"))
    center = np.array(design["selected"]["center"])
    relative = R.from_quat(design["relative_hat_rotation_xyzw"])
    records = []
    torso = [
        g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("torso_link").id and model.geom_contype[g]
    ]
    elbow = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == model.body("right_elbow_link").id and model.geom_contype[g]
    ]
    for i, t in enumerate(tt):
        if blend[i] == 0:
            continue
        d.qpos[:3] = motion.body_pos_w[i, 0]
        d.qpos[3:7] = motion.body_quat_w[i, 0]
        d.qpos[addr] = q[i]
        old = motion.joint_pos[i, bodycols]
        target = centers[i]

        def fun(x, old=old, target=target):
            d.qpos[addr[bodycols]] = x
            mujoco.mj_forward(model, d)
            wr = R.from_matrix(d.xmat[w].reshape(3, 3))
            actual = d.xpos[w] + wr.apply(center)
            return np.r_[
                (actual - target) * 500,
                (x - old) * 0.5,
                [
                    min(mujoco.mj_geomDistance(model, d, g, h, 0.1, None) - 0.003, 0) * 1000
                    for g in elbow
                    for h in torso
                ],
            ]

        step_lo = np.maximum(lo, q[i - 1, bodycols] - 0.16)
        step_hi = np.minimum(hi, q[i - 1, bodycols] + 0.16)
        fit = least_squares(
            fun, np.clip(q[i - 1, bodycols], step_lo + 1e-7, step_hi - 1e-7), bounds=(step_lo, step_hi), max_nfev=70
        )
        q[i, bodycols] = fit.x
        records.append([float(t), float(np.linalg.norm(fun(fit.x)[:3]) / 500)])
    res = refresh_fk(replace(motion, joint_pos=q), MODEL)
    res.save(out / "motion.npz")
    wi = motion.body_names.index("right_wrist_yaw_link")
    wr = R.from_quat(res.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    newr = wr * relative
    newp = res.body_pos_w[:, wi] + wr.apply(center) - newr.apply([0, 0, 0.11])
    mask = blend > 0
    data["position_w"][mask] = newp[mask]
    data["quaternion_xyzw"][mask] = newr.as_quat()[mask]
    np.savez_compressed(out / "scene_tracks/black_ribbon_hat_free.npz", **data)
    active = (tt >= 40) & (tt <= 44.8)
    report = {
        "status": "candidate_pending_checks",
        "human_reference": "unchanged human-derived body prior wrist joints; not a new raw human wrist estimate",
        "carry_blend_s": [39.3, 40, 44.8, 45.8],
        "source_grasp": str(src),
        "grasp_and_release_joint_positions_exact": bool(np.array_equal(q[~mask], motion.joint_pos[~mask])),
        "fingers_exact": bool(
            np.array_equal(
                q[:, [j for j, n in enumerate(motion.joint_names) if "_hand_" in n]],
                motion.joint_pos[:, [j for j, n in enumerate(motion.joint_names) if "_hand_" in n]],
            )
        ),
        "carry_wrist_prior_rmse_deg_before": float(
            np.rad2deg(
                np.sqrt(np.mean((motion.joint_pos[active][:, wristcols] - prior.joint_pos[active][:, wristcols]) ** 2))
            )
        ),
        "carry_wrist_prior_rmse_deg_after": float(
            np.rad2deg(np.sqrt(np.mean((q[active][:, wristcols] - prior.joint_pos[active][:, wristcols]) ** 2)))
        ),
        "max_cylinder_center_error_m": max(x[1] for x in records),
        "center_errors": records,
        "physics_verified": False,
    }
    (out / "natural_carry_review.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "center_errors"})


if __name__ == "__main__":
    main()
