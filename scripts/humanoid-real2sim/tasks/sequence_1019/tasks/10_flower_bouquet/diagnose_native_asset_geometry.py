"""Read authored mesh components and preserve an independent collision inventory."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from pxr import Usd, UsdGeom

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    source = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/objects/1012/flower_bouquet_v2.usd"
    )
    scene = args.run / "scene/10_flower_bouquet"
    scale = np.asarray(json.loads((scene / "flower_registered_candidate.json").read_text())["native_scale"])
    stage = Usd.Stage.Open(str(source))
    cache = UsdGeom.XformCache()
    records = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        geom = UsdGeom.Mesh(prim)
        raw = np.asarray(geom.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim))
        vertices = (raw @ transform[:3, :3] + transform[3, :3]) * scale
        indices = np.asarray(geom.GetFaceVertexIndicesAttr().Get())
        faces, offset = [], 0
        for count in geom.GetFaceVertexCountsAttr().Get():
            face = indices[offset : offset + count]
            faces.extend([face[[0, i, i + 1]] for i in range(1, count - 1)])
            offset += count
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        collision = "/CollisionSet/" in str(prim.GetPath())
        name = ("collision_" if collision else "visual_") + prim.GetName()
        mesh.export(args.out / (name + ".obj"))
        records.append(
            {
                "prim": str(prim.GetPath()),
                "collision_set": collision,
                "visibility": str(geom.ComputeVisibility()),
                "mesh": name + ".obj",
                "vertices": len(vertices),
                "faces": len(faces),
                "bounds": mesh.bounds.tolist(),
                "area_m2": float(mesh.area),
                "signed_volume_m3": float(mesh.volume),
                "uniform_volume_com_m": mesh.center_mass.tolist(),
                "watertight": bool(mesh.is_watertight),
            }
        )
    summaries = {}
    for kind, wanted in [("authored_visual", False), ("authored_collision", True)]:
        selected = [x for x in records if x["collision_set"] == wanted]
        volume = np.array([x["signed_volume_m3"] for x in selected])
        centers = np.array([x["uniform_volume_com_m"] for x in selected])
        summaries[kind] = {
            "components": len(selected),
            "vertices": sum(x["vertices"] for x in selected),
            "summed_signed_volume_m3": float(volume.sum()),
            "volume_weighted_center_m": (volume @ centers / volume.sum()).tolist(),
            "summed_area_m2": sum(x["area_m2"] for x in selected),
        }
    registered = trimesh.load(scene / "registered_assets/flower_bouquet.obj", force="mesh", process=False)
    summaries["registered_original_visual"] = {
        "vertices": len(registered.vertices),
        "volume_m3": float(registered.volume),
        "uniform_com_m": registered.center_mass.tolist(),
        "area_m2": float(registered.area),
    }
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "status": "diagnostic inventory, no physical parameter or asset replacement",
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "native_scale": scale.tolist(),
                "summaries": summaries,
                "meshes": records,
                "mass_limit": "Uniform solid component volumes and areas do not measure real paper, wire, or artificial flower mass.",
            },
            indent=2,
        )
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
