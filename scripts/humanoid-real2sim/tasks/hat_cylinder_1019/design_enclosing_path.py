"""Task-local reproducible geometric grasp design; output remains a candidate."""


def main():
    import argparse
    import json
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    r = p.parse_args().run
    m = mujoco.MjModel.from_xml_path(str(r / "scene/robot_props_cylinder_hat.xml"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    names = json.load(open(r / "retarget/baseline/motion_order.json"))["robot_joint_names"]
    ad = np.array([m.joint(n).qposadr[0] for n in names])
    z = np.load(r / "retarget/seed/motion.npz")
    k = 250
    d.qpos[:3] = z["body_pos_w"][k, 0]
    d.qpos[3:7] = z["body_quat_w"][k, 0]
    d.qpos[ad] = z["joint_pos"][k]
    h = np.load(r / "retarget/seed/scene_tracks/black_ribbon_hat_free.npz")
    free = m.joint("black_ribbon_hat_free").qposadr[0]
    d.qpos[free : free + 3] = h["position_w"][k]
    d.qpos[free + 3 : free + 7] = h["quaternion_xyzw"][k, [3, 0, 1, 2]]
    mujoco.mj_forward(m, d)
    w = m.body("right_wrist_yaw_link").id
    wp = d.xpos[w].copy()
    wr = R.from_matrix(d.xmat[w].reshape(3, 3))
    ogs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("black_ribbon_hat").id and m.geom_contype[g]]
    for part in ["thumb_2", "middle_1", "index_1"]:
        gs = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == m.body("right_hand_" + part + "_link").id and m.geom_contype[g]
        ]
        records = []
        for g in gs:
            for o in ogs:
                ft = np.zeros(6)
                dist = mujoco.mj_geomDistance(m, d, g, o, 0.2, ft)
                records.append((dist, wr.inv().apply(ft[:3] - wp), wr.inv().apply(ft[3:] - wp), o))
        print(part, min(records, key=lambda x: x[0]))
    from scipy.optimize import least_squares

    hc = [
        names.index("right_hand_" + x + "_joint")
        for x in ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
    ]
    parts = ["right_wrist_yaw_link"] + [
        "right_hand_" + x + "_link" for x in ["thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
    ]
    groups = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(p).id and m.geom_contype[g]] for p in parts]
    design = json.load(open(r / "review/grasp_design_axis90_v9.json"))
    rel = R.from_quat(design["relative_hat_rotation_xyzw"])
    hr = wr * rel
    records = []
    best = json.load(open(r / "review/enclosing_grip_search.json"))[0]
    center = np.array(best["center"][:2])
    closed = np.array(best["hand"])
    records = []
    for alpha in np.linspace(0, 1, 41):
        d.qpos[ad[hc]] = closed * alpha

        def measure(c):
            hp = wp + wr.apply([*c, 0]) - hr.apply([0, 0, 0.11])
            d.qpos[free : free + 3] = hp
            d.qpos[free + 3 : free + 7] = hr.as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(m, d)
            return np.array(
                [min(mujoco.mj_geomDistance(m, d, g, o, 0.2, None) for g in gs for o in ogs) for gs in groups]
            )

        def fun(c):
            return np.r_[np.minimum(measure(c) - 0.00025, 0) * 1000, (c - center) * 10]

        fits = [
            least_squares(fun, center + offset, bounds=([0.08, -0.01], [0.19, 0.11]), max_nfev=70, diff_step=1e-4)
            for offset in [np.array([0, 0]), np.array([0, 0.02]), np.array([0.02, 0]), np.array([-0.02, 0])]
        ]
        fit = min(fits, key=lambda f: np.linalg.norm(f.fun))
        records.append({"alpha": float(alpha), "center": [*fit.x, 0], "min_gap": float(measure(fit.x).min())})
    print(json.dumps(records, indent=2))
    (r / "review/enclosing_center_path.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
