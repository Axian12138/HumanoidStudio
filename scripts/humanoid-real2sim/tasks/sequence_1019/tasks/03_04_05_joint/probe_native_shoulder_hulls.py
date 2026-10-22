"""Screen a conservative torso/upper-arm mesh hull pair without changing shared assets."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    import mujoco

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    tree = ET.parse(MODEL)
    tree.getroot().find("compiler").set("meshdir", str(MODEL.parent / "meshes"))
    for body in ["torso_link", "right_shoulder_yaw_link"]:
        ET.SubElement(
            tree.getroot().find('.//body[@name="' + body + '"]'),
            "geom",
            name="native_hull_" + body,
            type="mesh",
            mesh=body,
            contype="8",
            conaffinity="8",
            density="0",
            group="5",
        )
    path = r / "scene/03_04_05/robot_native_shoulder_hulls_diagnostic.xml"
    tree.write(path)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    c = r / "retarget/candidates/03_04_05_joint_v58"
    motion = np.load(c / "motion.npz")
    order = json.loads((c / "motion_order.json").read_text())
    addr = np.array([m.joint(n).qposadr[0] for n in order["robot_joint_names"]])
    g1 = m.geom("native_hull_torso_link").id
    g2 = m.geom("native_hull_right_shoulder_yaw_link").id
    rows = []
    for k in range(565, 1033):
        d.qpos[:3] = motion["body_pos_w"][k, 0]
        d.qpos[3:7] = motion["body_quat_w"][k, 0]
        d.qpos[addr] = motion["joint_pos"][k]
        mujoco.mj_forward(m, d)
        dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.1, None)
        rows.append({"source_s": 27 + k / 50, "signed_gap_m": float(dist)})
    out = {"status": "conservative_hull_pair_probe_not_native_mesh_acceptance", "model": str(path), "records": rows}
    (r / "review/scene/03_04_05_native_shoulder_hull_probe.json").write_text(json.dumps(out, indent=2) + "\n")
    hits = [x for x in rows if x["signed_gap_m"] < 0]
    print(
        {
            "n": len(hits),
            "range": [hits[0]["source_s"], hits[-1]["source_s"]] if hits else None,
            "min": min(x["signed_gap_m"] for x in rows),
        }
    )


if __name__ == "__main__":
    main()
