"""Extract the authored IKEA 1012 coffee table into RSS visual/collision OBJs."""

from __future__ import annotations

import argparse
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


SOURCE = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/scenes/tianze_sim_data/IKEA_LivingRoom_Mesh_1012/IKEA_LivingRoom_Mesh_1012_drawer_renderable_v4.usd")
TABLE = "/World/MeasuredFurniture/CoffeeTable"


def triangles(mesh):
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    faces = []
    offset = 0
    for count in counts:
        polygon = indices[offset:offset + count]
        faces.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1))
        offset += count
    return np.asarray(faces, dtype=np.int32)


def write_obj(path, parts):
    with path.open("w") as stream:
        vertex_offset = 0
        for name, vertices, faces in parts:
            stream.write(f"o {name}\n")
            for x, y, z in vertices:
                stream.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
            for face in faces:
                stream.write("f " + " ".join(str(int(index) + vertex_offset + 1) for index in face) + "\n")
            vertex_offset += len(vertices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=example_dir(__file__) / "assets")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(SOURCE))
    if stage is None:
        raise RuntimeError(f"cannot open {SOURCE}")
    root = stage.GetPrimAtPath(TABLE)
    if not root.IsValid():
        raise RuntimeError(f"missing {TABLE}")
    cache = UsdGeom.XformCache()
    visual_parts = []
    collision_parts = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = cache.GetLocalToWorldTransform(prim)
        vertices = np.asarray([tuple(transform.Transform(point)) for point in mesh.GetPointsAttr().Get()], dtype=np.float64)
        faces = triangles(mesh)
        part = (prim.GetName(), vertices, faces)
        visual_parts.append(part)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_parts.append(part)
    if len(visual_parts) != 56 or len(collision_parts) != 8:
        raise ValueError(f"unexpected coffee table parts: visual={len(visual_parts)}, collision={len(collision_parts)}")
    write_obj(args.out / "coffee_table_visual.obj", visual_parts)
    for index, part in enumerate(collision_parts):
        write_obj(args.out / f"coffee_table_collision_{index:02d}.obj", [part])
    print(f"exported {len(visual_parts)} visual and {len(collision_parts)} collision parts to {args.out}")


if __name__ == "__main__":
    main()
