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
    cyl = m.geom("cylinder_hat_67").id
    records = []
    for f in [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]:
        for th in [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]:
            hand = np.array([0, -th, -th, f, f, f, f])
            d.qpos[ad[hc]] = hand

            def measure(c):
                hp = wp + wr.apply([*c, 0]) - hr.apply([0, 0, 0.11])
                d.qpos[free : free + 3] = hp
                d.qpos[free + 3 : free + 7] = hr.as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(m, d)
                ds = np.array(
                    [min(mujoco.mj_geomDistance(m, d, g, o, 0.2, None) for g in gs for o in ogs) for gs in groups]
                )
                norms = []
                cd = []
                for j in [2, 4, 6]:
                    ft = np.zeros(6)
                    gap = mujoco.mj_geomDistance(m, d, groups[j][0], cyl, 0.2, ft)
                    pt = wr.inv().apply(ft[3:] - wp)[:2] - c
                    norms.append(pt / max(np.linalg.norm(pt), 1e-7))
                    cd.append(gap)
                return ds, np.array(cd), np.array(norms)

            def fun(c):
                ds, cd, norm = measure(c)
                return np.r_[
                    (cd - 0.001) * 150, np.minimum(ds - 0.0003, 0) * 600, (norm[0] + (norm[1] + norm[2]) / 2) * 3
                ]

            for seed in [[0.12, 0.035], [0.09, 0.06], [0.14, 0.01]]:
                fit = least_squares(fun, seed, bounds=([0.04, -0.04], [0.19, 0.12]), max_nfev=50, diff_step=1e-4)
                ds, cd, norm = measure(fit.x)
                records.append(
                    {
                        "hand": hand.tolist(),
                        "center": [*fit.x, 0],
                        "score": float(np.linalg.norm(fun(fit.x))),
                        "min_gap": float(ds.min()),
                        "tip_gaps": cd.tolist(),
                        "opposition_dot": float(norm[0] @ norm[1]),
                    }
                )
    records.sort(key=lambda a: a["score"])
    print(json.dumps(records[:8], indent=2))
    (r / "review/enclosing_grip_search.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
