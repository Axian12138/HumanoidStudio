"""Export each authored rack segment as a fixed collision mesh without hull bridging."""

import argparse
import importlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    import mujoco

    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget3/src")
    importlib.import_module("retarget3.runtime")
    from pxr import Gf, Usd, UsdGeom

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    out = a.run / "scene/03_04_05"
    dest = out / "rack_parts"
    dest.mkdir(exist_ok=True)
    source = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/scenes/tianze_sim_data/IKEA_LivingRoom_Mesh_1012/IKEA_LivingRoom_Mesh_1012_drawer_renderable_v4.usd"
    )
    stage = Usd.Stage.Open(str(source))
    cache = UsdGeom.XformCache()
    root = ET.parse(out / "robot_props_drawer.xml").getroot()
    asset = root.find("asset")
    body = ET.SubElement(root.find("worldbody"), "body", name="coat_rack_support")
    records = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith(
            "/World/MeasuredFurniture/CoatRack/Geometry/"
        ):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = cache.GetLocalToWorldTransform(prim)
        vertices = np.array([matrix.Transform(Gf.Vec3d(*map(float, v))) for v in mesh.GetPointsAttr().Get()])
        center = vertices.mean(axis=0)
        v = vertices - center
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        faces = []
        i = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            ids = indices[i : i + count]
            for j in range(1, count - 1):
                faces.append([ids[k] + 1 for k in (0, j, j + 1)])
            i += count
        name = "rack_" + str(len(records))
        file = dest / (name + ".obj")
        with file.open("w") as f:
            for row in v:
                f.write("v " + " ".join(map(str, row)) + "\n")
            for row in faces:
                f.write("f " + " ".join(map(str, row)) + "\n")
        ET.SubElement(asset, "mesh", name=name, file=str(file.resolve()))
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            pos=" ".join(map(str, center)),
            group="3",
            rgba="0 0 0 0",
            friction=".8 .005 .001",
        )
        records.append(
            {
                "geom": name,
                "source_prim": str(prim.GetPath()),
                "center_world_m": center.tolist(),
                "bounds_world_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            }
        )
    path = out / "robot_props_drawer_with_rack.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    (out / "rack_support.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "parts": records,
                "ngeom_compiled": model.ngeom,
                "visual": "existing static room retains visual; these are collision-only authored parts",
                "scope": "Each per-segment mesh convex hull; no cross-hook convexification.",
            },
            indent=2,
        )
        + "\n"
    )
    print(path, len(records), model.ngeom)


if __name__ == "__main__":
    main()
