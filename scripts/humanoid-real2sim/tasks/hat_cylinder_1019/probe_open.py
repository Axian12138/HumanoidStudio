"""Screen thumb opening paths against the actual cylinder collision geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
from fit import PARTS


def main():
    import mujoco
    from scipy.spatial.transform import Rotation as R

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    r = p.parse_args().run
    m = mujoco.MjModel.from_xml_path(str(r / "scene/robot_props_cylinder_hat.xml"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    design = json.loads((r / "review/grasp_design_v6.json").read_text())
    closed = np.array(design["selected"]["hand"])
    center = np.array(design["selected"]["center"])
    rel = R.from_quat(design["relative_hat_rotation_xyzw"])
    ha = [m.joint("right_hand_" + n + "_joint").qposadr[0] for n in PARTS]
    w = m.body("right_wrist_yaw_link").id
    oid = m.body("black_ribbon_hat").id
    f = m.joint("black_ribbon_hat_free").qposadr[0]
    ogs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == oid and m.geom_contype[g]]
    hgs = [
        g
        for g in range(m.ngeom)
        if m.body(m.geom_bodyid[g]).name in ["right_hand_thumb_1_link", "right_hand_thumb_2_link"] and m.geom_contype[g]
    ]
    d.qpos[ha] = closed
    mujoco.mj_forward(m, d)
    wr = R.from_matrix(d.xmat[w].reshape(3, 3))
    hr = wr * rel
    d.qpos[f : f + 3] = d.xpos[w] + wr.apply(center) - hr.apply([0, 0, 0.11])
    d.qpos[f + 3 : f + 7] = hr.as_quat()[[3, 0, 1, 2]]
    rows = []
    for t0 in [0.0]:
        for t1 in [-1.04, -0.9, -0.7, -0.3, 0]:
            for t2 in [-1.04, -0.9, -0.7, -0.3, 0]:
                op = np.array([t0, t1, t2, 0, 0, 0, 0])
                ds = []
                for u in np.linspace(0, 1, 16):
                    d.qpos[ha] = op * (1 - u) + closed * u
                    mujoco.mj_forward(m, d)
                    ds.append(min(mujoco.mj_geomDistance(m, d, g, o, 0.2, None) for g in hgs for o in ogs))
                rows.append(
                    {"open": op.tolist(), "min": min(ds), "initial": ds[0], "score": min(ds) + min(ds[0], 0.03)}
                )
    rows.sort(key=lambda x: x["score"], reverse=True)
    print(rows[:8])
    (r / "review/thumb_open_probe.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
