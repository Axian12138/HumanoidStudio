"""Single-frame registered rack reachability probe."""


def main():
    import json
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    r = Path("outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800")
    b = r / "retarget/candidates/03_04_05_joint_v17"
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    z = np.load(b / "motion.npz")
    h = np.load(b / "black_ribbon_hat.npz")
    o = json.load(open(b / "motion_order.json"))
    cols = [
        i
        for i, n in enumerate(o["robot_joint_names"])
        if (n.startswith("right_") and any(k in n for k in ["shoulder", "elbow", "wrist"])) or n.startswith("waist_")
    ]
    addr = np.array([m.joint(n).qposadr[0] for n in o["robot_joint_names"]])
    w = m.body("right_wrist_yaw_link").id
    k = 1050
    wi = o["robot_body_names"].index("right_wrist_yaw_link")
    wr = Rotation.from_quat(z["body_quat_w"][k, wi][[1, 2, 3, 0]])
    relative = wr.inv() * Rotation.from_quat(h["quaternion_xyzw"][k])
    center = wr.inv().apply(h["position_w"][k] - z["body_pos_w"][k, wi])
    targetR = Rotation.from_euler("y", -70, degrees=True)
    target = np.array([2.150866718016237, -1.5736, 0.9271032539115094]) + np.array(
        json.loads((r / "scene/03_04_05/early_rack_registration.json").read_text())["translation_world_m"]
    )
    d.qpos[:3] = z["body_pos_w"][k, 0]
    d.qpos[3:7] = z["body_quat_w"][k, 0]
    d.qpos[addr] = z["joint_pos"][k]
    lower = np.array([m.joint(o["robot_joint_names"][c]).range[0] for c in cols])
    upper = np.array([m.joint(o["robot_joint_names"][c]).range[1] for c in cols])
    prior = z["joint_pos"][k, cols]
    waist = np.array([o["robot_joint_names"][c].startswith("waist_") for c in cols])
    lower[waist] = np.maximum(lower[waist], prior[waist] - 0.35)
    upper[waist] = np.minimum(upper[waist], prior[waist] + 0.35)

    def residual(x):
        d.qpos[addr[cols]] = x
        mujoco.mj_forward(m, d)
        R = Rotation.from_matrix(d.xmat[w].reshape(3, 3))
        return np.r_[(d.xpos[w] + R.apply(center) - target) * 3, ((R * relative) * targetR.inv()).as_rotvec()]

    results = []
    for yaw in [-180, -135, -90, 0]:
        targetR = Rotation.from_euler("y", -70, degrees=True) * Rotation.from_euler("z", yaw, degrees=True)
        fit = least_squares(
            residual, np.clip(prior, lower + 1e-6, upper - 1e-6), bounds=(lower, upper), max_nfev=250, diff_step=1e-4
        )
        err = residual(fit.x)
        results.append(
            {
                "yaw": yaw,
                "q": fit.x.tolist(),
                "pos_error": float(np.linalg.norm(err[:3]) / 3),
                "rot_error": float(np.linalg.norm(err[3:])),
                "nfev": fit.nfev,
                "success": bool(fit.success),
            }
        )
        print(results[-1], flush=True)
    (r / "review/stage3/03_04_05_registered_rack_pose_family_waist035.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
