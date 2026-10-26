"""Audit open approach, actual cylinder-axis alignment and opposing contact regions."""


def main():
    import argparse
    import json
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="enclosing_v4")
    a = p.parse_args()
    r = a.run
    c = r / "retarget" / a.version
    z = np.load(c / "motion.npz")
    n = json.load(open(c / "motion_order.json"))["robot_joint_names"]
    model = mujoco.MjModel.from_xml_path(str(r / "scene/robot_props_cylinder_hat.xml"))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(model)
    addr = np.array([model.joint(name).qposadr[0] for name in n])
    w = model.body("right_wrist_yaw_link").id
    free = model.joint("black_ribbon_hat_free").qposadr[0]
    cyl = model.geom("cylinder_hat_67").id
    h = np.load(c / "scene_tracks/black_ribbon_hat_free.npz")
    design = json.load(open(r / "review/grasp_design_enclosing_v1.json"))
    center = np.array(design["selected"]["center"])
    records = []
    for k in range(215, 633, 5):
        d.qpos[:3] = z["body_pos_w"][k, 0]
        d.qpos[3:7] = z["body_quat_w"][k, 0]
        d.qpos[addr] = z["joint_pos"][k]
        hr = R.from_quat(h["quaternion_xyzw"][k])
        hp = h["position_w"][k]
        d.qpos[free : free + 3] = hp
        d.qpos[free + 3 : free + 7] = hr.as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(model, d)
        wp = d.xpos[w]
        wr = R.from_matrix(d.xmat[w].reshape(3, 3))
        actual = wr.inv().apply(hp + hr.apply([0, 0, 0.11]) - wp)
        axis = wr.inv().apply(hr.apply([0, 1, 0]))
        norms = []
        gaps = []
        for part in ["thumb_2", "middle_1", "index_1"]:
            gs = [
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == model.body("right_hand_" + part + "_link").id and model.geom_contype[g]
            ]
            candidates = []
            for g in gs:
                ft = np.zeros(6)
                gap = mujoco.mj_geomDistance(model, d, g, cyl, 0.2, ft)
                candidates.append((gap, ft.copy()))
            gap, ft = min(candidates, key=lambda x: x[0])
            radial = wr.inv().apply(ft[3:] - wp)[:2] - actual[:2]
            norms.append(radial / np.linalg.norm(radial))
            gaps.append(gap)
        records.append(
            {
                "source_s": 35 + k / 50,
                "center_error_m": float(np.linalg.norm(actual - center)),
                "axis_error_deg": float(np.rad2deg(np.arccos(np.clip(abs(axis[2]), 0, 1)))),
                "thumb_index_opposition_dot": float(norms[0] @ norms[2]),
                "tip_gaps_m": gaps,
            }
        )
    hc = [
        n.index("right_hand_" + part + "_joint")
        for part in ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
    ]
    report = {
        "candidate": str(c),
        "open_approach_interval_s": [36.3, 38.3],
        "max_abs_open_approach_finger_rad": float(abs(z["joint_pos"][65:166, hc]).max()),
        "thumb_root_max_abs_rad": float(abs(z["joint_pos"][:, hc[0]]).max()),
        "thumb_root_max_abs_speed_rad_s": float(abs(z["joint_vel"][:, hc[0]]).max()),
        "closure_command_s": [38.3, 39.05],
        "release_command_s": [47.64, 48.1],
        "hold_center_error_max_m": max(x["center_error_m"] for x in records),
        "hold_axis_error_max_deg": max(x["axis_error_deg"] for x in records),
        "opposition_dot_range": [
            min(x["thumb_index_opposition_dot"] for x in records),
            max(x["thumb_index_opposition_dot"] for x in records),
        ],
        "records": records,
        "physics_verified": False,
        "held_alignment_is_imposed_attachment": True,
    }
    (r / "review" / ("enclosure_" + a.version + ".json")).write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "records"})


if __name__ == "__main__":
    main()
