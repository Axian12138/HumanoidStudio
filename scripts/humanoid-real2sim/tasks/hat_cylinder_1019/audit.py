"""100 Hz interpolated signed geometry audit for the isolated cylindrical-hat clip."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="v1")
    a = p.parse_args()
    r = a.run
    c = r / "retarget" / a.version
    m = mujoco.MjModel.from_xml_path(str(r / "scene/robot_props_cylinder_hat.xml"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    z = np.load(c / "motion.npz")
    names = json.loads((c / "motion_order.json").read_text())["robot_joint_names"]
    addr = [m.joint(n).qposadr[0] for n in names]
    h = np.load(c / "scene_tracks/black_ribbon_hat_free.npz")
    ts = 35 + np.arange(len(z["joint_pos"])) / 50
    tt = np.linspace(35, 50, 1501)
    roots = Slerp(ts, R.from_quat(z["body_quat_w"][:, 0][:, [1, 2, 3, 0]]))(tt).as_quat()[:, [3, 0, 1, 2]]
    hr = Slerp(ts, R.from_quat(h["quaternion_xyzw"]))(tt).as_quat()[:, [3, 0, 1, 2]]
    free = m.joint("black_ribbon_hat_free").qposadr[0]
    oid = m.body("black_ribbon_hat").id
    og = [g for g in range(m.ngeom) if m.geom_bodyid[g] == oid and m.geom_contype[g]]
    groups = {
        m.body(b).name: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]
        for b in range(m.nbody)
        if m.body(b).name.startswith("right_hand_")
        or m.body(b).name
        in ["right_wrist_yaw_link", "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link"]
    }
    groups = {k: v for k, v in groups.items() if v}
    rows = []
    for i, t in enumerate(tt):
        d.qpos[addr] = [np.interp(t, ts, z["joint_pos"][:, j]) for j in range(len(addr))]
        d.qpos[:3] = [np.interp(t, ts, z["body_pos_w"][:, 0, j]) for j in range(3)]
        d.qpos[3:7] = roots[i]
        d.qpos[free : free + 3] = [np.interp(t, ts, h["position_w"][:, j]) for j in range(3)]
        d.qpos[free + 3 : free + 7] = hr[i]
        mujoco.mj_forward(m, d)
        ds = [min(mujoco.mj_geomDistance(m, d, g, o, 0.2, None) for g in gs for o in og) for gs in groups.values()]
        rows.append(ds)
    arr = np.array(rows)
    bad = np.argwhere(arr < -0.001)
    report = {
        "method": "100Hz linear joints/root position and SLERP roots; signed source MJCF convex collision-piece distance, not triangle containment or physical contact",
        "times_s": tt.tolist(),
        "bodies": list(groups),
        "distances_m": arr.tolist(),
        "minimum_by_body_m": dict(zip(groups, arr.min(axis=0).tolist(), strict=True)),
        "penetration_over_1mm_samples": len(bad),
        "bad_time_range_s": [float(tt[bad[:, 0].min()]), float(tt[bad[:, 0].max()])] if len(bad) else None,
        "physics_verified": False,
    }
    (r / "review" / ("sweep_" + a.version + ".json")).write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k not in ["times_s", "bodies", "distances_m"]}, flush=True)


if __name__ == "__main__":
    main()
