"""Screen palm roll about the rotated crown axis using neutral wrists and authored clearances."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from fit import ARM, PARTS
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    m = mujoco.MjModel.from_xml_path(str(r / "scene/robot_props_cylinder_hat.xml"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    z = np.load(r / "retarget/baseline/motion.npz")
    names = json.load(open(r / "retarget/baseline/motion_order.json"))["robot_joint_names"]
    ad = np.array([m.joint(n).qposadr[0] for n in names])
    ac = [names.index("right_" + x + "_joint") for x in ARM]
    hc = [names.index("right_hand_" + x + "_joint") for x in PARTS]
    w = m.body("right_wrist_yaw_link").id
    o = m.body("black_ribbon_hat").id
    free = m.joint("black_ribbon_hat_free").qposadr[0]
    hat = np.load(r / "retarget/baseline/scene_tracks/black_ribbon_hat_free.npz")
    k = 165
    hr = R.from_quat(hat["quaternion_xyzw"][k])
    hp = hat["position_w"][k]
    d.qpos[:3] = z["body_pos_w"][k, 0]
    d.qpos[3:7] = z["body_quat_w"][k, 0]
    d.qpos[ad] = z["joint_pos"][k]
    d.qpos[ad[hc]] = [0, -0.7, -0.7, 1, 1, 1, 1]
    d.qpos[free : free + 3] = hp
    d.qpos[free + 3 : free + 7] = hr.as_quat()[[3, 0, 1, 2]]
    lo = np.array([m.joint(names[j]).range[0] for j in ac])
    hi = np.array([m.joint(names[j]).range[1] for j in ac])
    center = np.array([0.0898168, 0.098252, 0])
    rel0 = R.from_matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    hands = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("right_hand") and m.geom_contype[g]]
    ogs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == o and m.geom_contype[g]]
    records = []
    for sign in [90, -90]:
        for roll in range(-180, 181, 15):
            rel = R.from_euler("z", roll, degrees=True) * rel0 * R.from_euler("z", sign, degrees=True)
            tr = hr * rel.inv()
            tp = hp + hr.apply([0, 0, 0.11]) - tr.apply(center)

            def fun(x, tp=tp, tr=tr):
                d.qpos[ad[ac]] = x
                mujoco.mj_forward(m, d)
                wr = R.from_matrix(d.xmat[w].reshape(3, 3))
                return np.r_[
                    (d.xpos[w] - tp) * 250,
                    (wr * tr.inv()).as_rotvec() * 25,
                    x[4:] * 4,
                    (x - z["joint_pos"][k, ac]) * 0.1,
                ]

            seed = z["joint_pos"][k, ac].copy()
            seed[4:] = 0
            fit = least_squares(fun, np.clip(seed, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=100)
            res = fun(fit.x)
            gap = min(mujoco.mj_geomDistance(m, d, g, h, 0.1, None) for g in hands for h in ogs)
            records.append(
                {
                    "yaw": sign,
                    "roll": roll,
                    "q": fit.x.tolist(),
                    "wrist_norm": float(np.linalg.norm(fit.x[4:])),
                    "position_error_m": float(np.linalg.norm(res[:3]) / 250),
                    "orientation_error_rad": float(np.linalg.norm(res[3:6]) / 25),
                    "hand_hat_min_m": gap,
                }
            )
    records.sort(
        key=lambda x: x["wrist_norm"] + max(0.0001 - x["hand_hat_min_m"], 0) * 1000 + x["position_error_m"] * 100
    )
    (r / "review/axis90_grasp_roll_probe.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records[:5], indent=2))


if __name__ == "__main__":
    main()
