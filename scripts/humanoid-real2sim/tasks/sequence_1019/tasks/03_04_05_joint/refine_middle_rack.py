"""Preserve thumb/index pinch and search a constant middle posture clear of sofa/rack."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import least_squares

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    c = r / "retarget/candidates/03_04_05_joint_v58"
    motion = np.load(c / "motion.npz")
    hat = np.load(c / "black_ribbon_hat.npz")
    order = json.loads((c / "motion_order.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(r / "retarget/candidates/03_04_05_joint_v56/sofa_collision_model.xml"))
    d = mujoco.MjData(m)
    names = ["right_hand_middle_0_joint", "right_hand_middle_1_joint"]
    cols = [order["robot_joint_names"].index(n) for n in names]
    addr = np.array([m.joint(n).qposadr[0] for n in order["robot_joint_names"]])
    ha = m.joint("black_ribbon_hat_free").qposadr[0]
    hb = m.body("black_ribbon_hat").id
    rb = m.body("coat_rack_support").id
    hand = [m.body(n).id for n in ["right_hand_middle_0_link", "right_hand_middle_1_link"]]
    hm = {b: i for i, b in enumerate(hand)}
    sg = m.geom("registered_sofa_top").id
    hf = int(m.geom_dataid[sg])
    nr, nc = int(m.hfield_nrow[hf]), int(m.hfield_ncol[hf])
    start = int(m.hfield_adr[hf])
    size = m.hfield_size[hf]
    height = m.hfield_data[start : start + nr * nc].reshape(nr, nc) * size[2] + m.geom_pos[sg, 2]
    top = RegularGridInterpolator(
        (
            np.linspace(-size[1], size[1], nr) + m.geom_pos[sg, 1],
            np.linspace(-size[0], size[0], nc) + m.geom_pos[sg, 0],
        ),
        height,
        bounds_error=False,
        fill_value=-np.inf,
    )
    verts = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) in hm and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = int(m.geom_dataid[g])
            v = int(m.mesh_vertadr[mi])
            n = int(m.mesh_vertnum[mi])
            verts.append((g, hm[int(m.geom_bodyid[g])], m.mesh_vert[v : v + n].copy()))
    times = [39, 47.5, 47.54, 47.58, 47.6, 47.62, 47.64]
    prior = motion["joint_pos"][600, cols].astype(float)
    lo = np.array([m.joint(n).range[0] for n in names])
    hi = np.array([m.joint(n).range[1] for n in names])

    def measure(q):
        depths = np.zeros((len(times), 3, 2))
        for ti, t in enumerate(times):
            k = round((t - 27) * 50)
            d.qpos[:3] = motion["body_pos_w"][k, 0]
            d.qpos[3:7] = motion["body_quat_w"][k, 0]
            d.qpos[addr] = motion["joint_pos"][k]
            d.qpos[addr[cols]] = q
            d.qpos[ha : ha + 3] = hat["position_w"][k]
            d.qpos[ha + 3 : ha + 7] = hat["quaternion_xyzw"][k][[3, 0, 1, 2]]
            mujoco.mj_forward(m, d)
            for con in d.contact:
                b1, b2 = map(int, m.geom_bodyid[[con.geom1, con.geom2]])
                other = b1 if b1 in hm else b2 if b2 in hm else -1
                if other < 0:
                    continue
                target = b2 if b1 == other else b1
                if target in (hb, rb):
                    depths[ti, int(target == rb), hm[other]] = max(
                        depths[ti, int(target == rb), hm[other]], -con.dist + (0.001 if target == rb else -0.001)
                    )
            if t == 39:
                for g, bi, v in verts:
                    world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                    depths[ti, 2, bi] = max(
                        depths[ti, 2, bi], float(np.max(top(world[:, [1, 0]]) + 0.002 - world[:, 2]))
                    )
        return depths

    def residual(q):
        return np.r_[measure(q).ravel() * 2500, (q - prior) * 0.2]

    records = []
    rng = np.random.default_rng(455)
    for seed in range(12):
        x = prior if seed == 0 else rng.uniform(lo, hi)
        fit = least_squares(residual, x, bounds=(lo, hi), max_nfev=100, diff_step=1e-4)
        depth = measure(fit.x)
        rec = {
            "seed": seed,
            "q": fit.x.tolist(),
            "max_excess_m": float(depth.max()),
            "depths": depth.tolist(),
            "cost": float(fit.cost),
            "success": bool(fit.success),
        }
        records.append(rec)
        print(rec, flush=True)
    (r / "review/scene/03_04_05_middle_rack_family.json").write_text(
        json.dumps(
            {"status": "diagnostic", "joints": names, "base": str(c), "times": times, "records": records}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
