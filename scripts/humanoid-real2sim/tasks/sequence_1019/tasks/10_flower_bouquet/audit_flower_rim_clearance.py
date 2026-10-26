"""Measure held flower versus basket convex meshes through the rim crossing."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.run / "inputs/parent.json").read_text())
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    names = {}
    for name in ["flower_bouquet", "small_wicker_basket"]:
        body = ET.SubElement(world, "body", name=name)
        ET.SubElement(body, "freejoint", name=name + "_free")
        names[name] = []
        for j, f in enumerate(sorted(Path(info["parent"]["assets"]).glob(name + "_hull*.obj"))):
            n = name + str(j)
            names[name].append(n)
            ET.SubElement(asset, "mesh", name=n, file=str(f))
            ET.SubElement(body, "geom", name=n, type="mesh", mesh=n, mass=".01")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    result = {}
    for label, folder in [("parent", info["original_tracks"]), ("candidate", info["parent"]["tracks"])]:
        tracks = {n: np.load(Path(folder) / (n + ".npz")) for n in names}
        rows = []
        for t in np.arange(120.2, 122.651, 0.01):
            frame = (t - 108) * 50
            k = int(frame)
            u = frame - k
            for name, tr in tracks.items():
                addr = m.joint(name + "_free").qposadr[0]
                d.qpos[addr : addr + 3] = (1 - u) * tr["position_w"][k] + u * tr["position_w"][k + 1]
                rot = Slerp([0, 1], R.from_quat(tr["quaternion_xyzw"][k : k + 2]))([u]).as_quat()[0]
                d.qpos[addr + 3 : addr + 7] = rot[[3, 0, 1, 2]]
            mujoco.mj_forward(m, d)
            pairs = [
                (mujoco.mj_geomDistance(m, d, m.geom(f).id, m.geom(b).id, 1, None), f, b)
                for f in names["flower_bouquet"]
                for b in names["small_wicker_basket"]
            ]
            nearest = min(pairs)
            rows.append({"time_s": float(t), "gap_m": float(nearest[0]), "flower": nearest[1], "basket": nearest[2]})
        crossing = [x for x in rows if 120.9 - 1e-6 <= x["time_s"] <= 121.5 + 1e-6]
        result[label] = {
            "crossing_min_gap_m": min(x["gap_m"] for x in crossing),
            "all_min_gap_m": min(x["gap_m"] for x in rows),
            "negative_samples": sum(x["gap_m"] < 0 for x in rows),
            "closest": min(rows, key=lambda x: x["gap_m"]),
            "records": rows,
        }
    result["scope"] = (
        "100Hz convex collision meshes of both complete props; not dynamics or exact between-sample proof. Crossing window120.9-121.5s, full adjusted path120.2-122.65s."
    )
    (a.run / "review/flower_basket_clearance.json").write_text(json.dumps(result, indent=2))
    print({k: {n: v for n, v in row.items() if n != "records"} for k, row in result.items() if isinstance(row, dict)})


if __name__ == "__main__":
    main()
