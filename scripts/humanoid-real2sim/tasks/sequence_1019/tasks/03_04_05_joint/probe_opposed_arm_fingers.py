"""Search local grasp postures against both the hat and measured sofa surface."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    c = r / "retarget/candidates/03_04_05_joint_v50"
    motion = np.load(c / "motion.npz")
    order = json.loads((c / "motion_order.json").read_text())
    names = order["robot_joint_names"]
    model = mujoco.MjModel.from_xml_path(str(r / "retarget/candidates/03_04_05_joint_v44/sofa_collision_model.xml"))
    d = mujoco.MjData(model)
    k = 600
    addr = np.array([model.joint(n).qposadr[0] for n in names])
    arm = [
        "right_" + n + "_joint"
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    cols = [names.index(n) for n in arm] + [j for j, n in enumerate(names) if n.startswith("right_hand")]
    lo = np.array([model.joint(names[j]).range[0] for j in cols])
    hi = np.array([model.joint(names[j]).range[1] for j in cols])
    prior = motion["joint_pos"][k, cols].astype(float)
    hat = np.load(c / "black_ribbon_hat.npz")
    ha = model.joint("black_ribbon_hat_free").qposadr[0]
    hb = model.body("black_ribbon_hat").id
    hand = [
        model.body(n).id for n in order["robot_body_names"] if n.startswith("right_hand") or n == "right_wrist_yaw_link"
    ]
    hm = {b: i for i, b in enumerate(hand)}
    hatgeoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == hb and model.geom_contype[i]]
    tips = [
        [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body(n).id and model.geom_contype[i]]
        for n in ["right_hand_thumb_2_link", "right_hand_index_1_link"]
    ]
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] not in {*hand, hb} and model.geom(gi).name != "registered_sofa_top":
            model.geom_contype[gi] = 0
            model.geom_conaffinity[gi] = 0
    from scipy.interpolate import RegularGridInterpolator

    sg = model.geom("registered_sofa_top").id
    hf = int(model.geom_dataid[sg])
    nr, nc = int(model.hfield_nrow[hf]), int(model.hfield_ncol[hf])
    hs = model.hfield_size[hf]
    start = int(model.hfield_adr[hf])
    height = model.hfield_data[start : start + nr * nc].reshape(nr, nc) * hs[2] + model.geom_pos[sg, 2]
    top = RegularGridInterpolator(
        (
            np.linspace(-hs[1], hs[1], nr) + model.geom_pos[sg, 1],
            np.linspace(-hs[0], hs[0], nc) + model.geom_pos[sg, 0],
        ),
        height,
        bounds_error=False,
        fill_value=-np.inf,
    )
    vertices = []
    for gi in range(model.ngeom):
        if (
            int(model.geom_bodyid[gi]) in hm
            and model.geom_contype[gi]
            and model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            mi = int(model.geom_dataid[gi])
            va = int(model.mesh_vertadr[mi])
            vn = int(model.mesh_vertnum[mi])
            vertices.append((gi, hm[int(model.geom_bodyid[gi])], model.mesh_vert[va : va + vn].copy()))
    wi = model.body("right_wrist_yaw_link").id
    wp = motion["body_pos_w"][k, order["robot_body_names"].index("right_wrist_yaw_link")]

    def measure(x):
        d.qpos[:3] = motion["body_pos_w"][k, 0]
        d.qpos[3:7] = motion["body_quat_w"][k, 0]
        d.qpos[addr] = motion["joint_pos"][k]
        d.qpos[addr[cols]] = x
        d.qpos[ha : ha + 3] = hat["position_w"][k]
        d.qpos[ha + 3 : ha + 7] = hat["quaternion_xyzw"][k][[3, 0, 1, 2]]
        mujoco.mj_forward(model, d)
        depth = np.zeros((2, len(hand)))
        for con in d.contact:
            b1, b2 = map(int, model.geom_bodyid[[con.geom1, con.geom2]])
            target = hb if hb in (b1, b2) else (0 if 0 in (b1, b2) else -1)
            other = b2 if b1 == target else b1
            if target >= 0 and other in hm:
                depth[int(target == 0), hm[other]] = max(depth[int(target == 0), hm[other]], -con.dist - 0.001)
        for gi, bi, verts in vertices:
            world = verts @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]
            depth[1, bi] = max(depth[1, bi], float(np.max(top(world[:, [1, 0]]) + 0.003 - world[:, 2])))
        distance = [min(mujoco.mj_geomDistance(model, d, t, g, 0.1, None) for t in ts for g in hatgeoms) for ts in tips]
        normals = []
        for ts in tips:
            best = (1.0, None)
            for tg in ts:
                for hg in hatgeoms:
                    segment = np.zeros(6)
                    signed = mujoco.mj_geomDistance(model, d, tg, hg, 0.1, segment)
                    if signed < best[0] and abs(signed) > 1e-10:
                        normal = (segment[3:] - segment[:3]) / signed
                        normal /= max(np.linalg.norm(normal), 1e-10)
                        best = (signed, normal)
            normals.append(best[1])
        dot = float(np.dot(*normals)) if all(n is not None for n in normals) else 1.0
        return depth, np.array(distance), d.xpos[wi].copy(), dot

    def residual(x):
        dep, dis, w, dot = measure(x)
        return np.r_[
            (dep * 2500).ravel(),
            np.maximum(dis - 0.001, 0) * 1500,
            (x - prior) * 0.2,
            (w - wp) * 100,
            max(dot + 0.3, 0) * 15,
        ]

    rng = np.random.default_rng(923)
    records = []
    for index in range(12):
        initial = prior.copy() if index == 0 else prior + rng.normal(0, 0.45, len(prior))
        initial = np.clip(initial, lo + 1e-6, hi - 1e-6)
        fit = least_squares(residual, initial, bounds=(lo, hi), max_nfev=120, ftol=1e-6)
        dep, dis, w, dot = measure(fit.x)
        record = {
            "seed": index,
            "q": fit.x.tolist(),
            "max_hat_penetration_m": float(dep[0].max() + 0.001),
            "max_sofa_margin_excess_plus_1mm_m": float(dep[1].max() + 0.001),
            "tip_distance_m": dis.tolist(),
            "closest_thumb_index_normal_dot": dot,
            "wrist_position_w": w.tolist(),
            "wrist_delta_m": float(np.linalg.norm(w - wp)),
            "cost": fit.cost,
            "solver_success": bool(fit.success),
        }
        records.append(record)
        print(record, flush=True)
    (r / "review/scene/03_04_05_opposed_arm_finger_family.json").write_text(
        json.dumps(
            {
                "source_s": 39,
                "joints": [names[j] for j in cols],
                "records": records,
                "status": "posture_candidates_not_continuous_grasp_acceptance",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
