"""Compare explicit hand designs using source collision meshes at a contact frame.

Only the object's fixed wrist-relative translation is fitted; each hand design
is fixed before solving and recorded. This is a geometric probe, not force QA.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", default="03_04_05_joint_v1")
    p.add_argument("--object", choices=["floor_strawberry", "black_ribbon_hat"], default="floor_strawberry")
    a = p.parse_args()
    base = a.run / "retarget/candidates" / a.candidate
    model = mujoco.MjModel.from_xml_path(str(a.run / "scene/03_04_05/robot_props_drawer.xml"))
    data = mujoco.MjData(model)
    z = np.load(base / "motion.npz")
    order = json.loads((base / "motion_order.json").read_text())
    addr = np.array([model.joint(n).qposadr[0] for n in order["robot_joint_names"]])
    side = "left" if a.object == "floor_strawberry" else "right"
    sign = 1 if side == "left" else -1
    k = 210 if side == "left" else 600
    data.qpos[:3] = z["body_pos_w"][k, 0]
    data.qpos[3:7] = z["body_quat_w"][k, 0]
    data.qpos[addr] = z["joint_pos"][k]
    oid = model.body(a.object).id
    wrist = model.body(side + "_wrist_yaw_link").id
    obj_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == oid and model.geom_contype[g]]
    groups = {}
    for body_name in [
        "left_hand_thumb_2_link",
        "left_hand_index_1_link",
        "left_hand_middle_1_link",
        "left_wrist_yaw_link",
        "left_hand_thumb_1_link",
        "left_hand_index_0_link",
        "left_hand_middle_0_link",
    ]:
        name = body_name.replace("left_", side + "_")
        bid = model.body(name).id
        groups[name] = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_contype[g]]
    free = model.joint(a.object + "_free").qposadr[0]
    seed = np.array([0.15, -0.065, 0] if side == "left" else [0.1384, 0.0251, 0.1470])
    lo = np.array([0.11, -0.12, -0.045]) if side == "left" else seed - 0.12
    hi = np.array([0.22, -0.025, 0.045]) if side == "left" else seed + 0.12
    records = []
    # Fixed candidate hand vectors; angle families are declared, not free IK variables.
    for thumb, fingers in [(0.3, 0.4), (0.35, 0.45), (0.4, 0.5), (0.45, 0.65), (0.2, 0.3)]:
        for i, name in enumerate(order["robot_joint_names"]):
            if name.startswith(side + "_hand"):
                data.qpos[addr[i]] = 0 if "thumb_0" in name else (sign * thumb if "thumb" in name else -sign * fingers)
        mujoco.mj_forward(model, data)
        wp, rot = data.xpos[wrist].copy(), data.xmat[wrist].reshape(3, 3).copy()

        def distances(center, wp=wp, rot=rot):
            data.qpos[free : free + 3] = wp + rot @ center
            data.qpos[free + 3 : free + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            return np.array(
                [
                    min(mujoco.mj_geomDistance(model, data, x, y, 0.15, None) for x in gs for y in obj_geoms)
                    for gs in groups.values()
                ]
            )

        def residual(center):
            d = distances(center)
            return np.r_[d[:3] * 100, np.minimum(d[3:] - 0.001, 0) * 200, (center - seed) * 0.1]

        fit = least_squares(residual, seed, bounds=(lo, hi), max_nfev=60, diff_step=0.0001)
        ds = distances(fit.x)
        records.append(
            {
                "thumb_1_2_rad": sign * thumb,
                "index_middle_0_1_rad": -sign * fingers,
                "center_local_m": fit.x.tolist(),
                "distance_by_body_m": dict(zip(groups, ds.tolist(), strict=True)),
                "score": float(np.linalg.norm(residual(fit.x))),
                "target_contacts_within_3mm": bool(np.max(abs(ds[:3])) < 0.003),
                "noncontact_penetration_under_3mm": bool(np.min(ds[3:]) > -0.003),
            }
        )
    out = a.run / ("review/stage3/" + a.candidate + "_" + a.object + "_grasp_geometry.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source_s": 27 + k / 50,
                "object": a.object,
                "method": "actual MJCF signed geom distances at fixed hand designs",
                "records": records,
                "physics_verified": False,
                "sweep_verified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
