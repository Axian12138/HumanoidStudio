"""100 Hz right upper-limb collision-proxy versus drawer check."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="v2")
    a = p.parse_args()
    c = a.run / "retarget" / a.version
    config = json.loads((a.run / "scene/config.json").read_text())
    m = mujoco.MjModel.from_xml_path(config["robot_and_early_props"])
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    z = np.load(c / "motion.npz")
    names = json.loads((c / "motion_order.json").read_text())["robot_joint_names"]
    addr = [m.joint(n).qposadr[0] for n in names]
    ts = 49.5 + np.arange(len(z["joint_pos"])) / 50
    tt = np.linspace(ts[0], ts[-1], 2 * len(ts) - 1)
    rot = Slerp(ts, Rotation.from_quat(z["body_quat_w"][:, 0, [1, 2, 3, 0]]))(tt).as_quat()
    qdrawer = np.load(c / "scene_tracks/drawer_slide.npz")["position"]
    drawer = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("drawer").id and m.geom_contype[g]]
    arm = [
        g
        for g in range(m.ngeom)
        if m.geom_contype[g]
        and m.body(m.geom_bodyid[g]).name.startswith(("right_hand", "right_wrist", "right_elbow", "right_shoulder"))
    ]
    records = []
    for i, t in enumerate(tt):
        d.qpos[:3] = [np.interp(t, ts, z["body_pos_w"][:, 0, j]) for j in range(3)]
        d.qpos[3:7] = rot[i, [3, 0, 1, 2]]
        d.qpos[addr] = [np.interp(t, ts, z["joint_pos"][:, j]) for j in range(len(addr))]
        d.qpos[m.joint("drawer_slide").qposadr[0]] = np.interp(t, ts, qdrawer)
        mujoco.mj_forward(m, d)
        for g in arm:
            distance = min(mujoco.mj_geomDistance(m, d, g, h, 0.2, None) for h in drawer)
            if distance < 0:
                records.append(
                    {"source_s": float(t), "body": m.body(m.geom_bodyid[g]).name, "distance_m": float(distance)}
                )
    report = {
        "samples": len(tt),
        "scope": "right shoulder through fingertips, collision proxies; independent actual hand surfaces checked separately",
        "minimum_distance_m": min([x["distance_m"] for x in records], default=0.0),
        "penetrations_over_1mm": sum(x["distance_m"] < -0.001 for x in records),
        "records": records,
    }
    (a.run / "review/drawer_proxy_sweep.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "records"})


if __name__ == "__main__":
    main()
