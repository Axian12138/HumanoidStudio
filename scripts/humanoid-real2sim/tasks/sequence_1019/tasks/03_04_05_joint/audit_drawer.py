"""Measure actual right-hand/forearm collision distances to the moving drawer."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    a = p.parse_args()
    base = a.run / "retarget/candidates" / a.candidate
    m = mujoco.MjModel.from_xml_path(str(a.run / "scene/03_04_05/robot_props_drawer.xml"))
    d = mujoco.MjData(m)
    motion = np.load(base / "motion.npz")
    order = json.loads((base / "motion_order.json").read_text())
    addr = np.array([m.joint(n).qposadr[0] for n in order["robot_joint_names"]])
    drawer = np.load(base / "drawer_slide.npz")["position"]
    da = m.joint("drawer_slide").qposadr[0]
    dg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("drawer").id and m.geom_contype[g]]
    names = [
        "right_wrist_yaw_link",
        "right_hand_thumb_2_link",
        "right_hand_index_1_link",
        "right_hand_middle_1_link",
        "right_wrist_pitch_link",
        "right_elbow_link",
    ]
    groups = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(n).id and m.geom_contype[g]] for n in names}
    records = []
    for k in range(1150, 1401, 2):
        d.qpos[:3] = motion["body_pos_w"][k, 0]
        d.qpos[3:7] = motion["body_quat_w"][k, 0]
        d.qpos[addr] = motion["joint_pos"][k]
        d.qpos[da] = drawer[k]
        mujoco.mj_forward(m, d)
        dist = {
            n: min(mujoco.mj_geomDistance(m, d, x, y, 0.3, None) for x in gs for y in dg) for n, gs in groups.items()
        }
        records.append({"source_s": 27 + k / 50, "distance_by_body_m": dist})
    active = [r for r in records if 51.3 <= r["source_s"] <= 53.8]
    summary = {
        n: {
            "min_m": min(r["distance_by_body_m"][n] for r in active),
            "max_m": max(r["distance_by_body_m"][n] for r in active),
        }
        for n in names
    }
    result = {
        "status": "diagnostic_not_passed",
        "sample_step_s": 0.04,
        "contact_window_s": [51.3, 53.8],
        "signed_distance_summary": summary,
        "records": records,
        "scope": "authored box drawer, actual robot hand collision meshes; no physics/contact forces or room collision",
    }
    (base / "drawer_geometry_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
