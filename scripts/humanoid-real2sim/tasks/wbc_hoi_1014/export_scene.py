"""Export source USD room geometry and recorded fruit meshes for review."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation


def transform(pose):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    t[:3, 3] = pose[:3]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    for idx, (label, sid) in enumerate((("01_1014", 25), ("02_1014", 48))):
        folder = run / label
        source = run / "IKEA_0907_100Scenes_IsaacSim/scenes" / f"scene_{sid}.usda"
        stage = Usd.Stage.Open(str(source))
        cache = UsdGeom.XformCache()
        meshes = []
        rows = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if (
                not prim.IsA(UsdGeom.Mesh)
                or "/Tasks/" in path
                or "/Task/" in path
                or "OmniverseKitViewportCameraMesh" in path
            ):
                continue
            img = UsdGeom.Imageable(prim)
            if img.ComputeVisibility() == "invisible" or img.ComputePurpose() in (
                "guide",
                "proxy",
            ):
                continue
            geo = UsdGeom.Mesh(prim)
            points = np.asarray(geo.GetPointsAttr().Get(), float)
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            points = points @ matrix[:3, :3] + matrix[3, :3] + [12 * idx, 0, 0]
            counts = geo.GetFaceVertexCountsAttr().Get()
            inds = geo.GetFaceVertexIndicesAttr().Get()
            faces = []
            cursor = 0
            for n in counts:
                ids = inds[cursor : cursor + n]
                faces.extend((ids[0], ids[j], ids[j + 1]) for j in range(1, n - 1))
                cursor += n
            if not faces:
                continue
            color = np.array([0.65, 0.65, 0.62])
            dc = geo.GetDisplayColorAttr().Get()
            if dc:
                color = np.array(dc[0])
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if mat:
                for child in mat.GetPrim().GetChildren():
                    value = child.GetAttribute("inputs:diffuseColor").Get()
                    if value is not None:
                        color = np.array(value)
                        break
            mesh = trimesh.Trimesh(points, faces, process=False)
            mesh.visual.face_colors = np.r_[np.clip(color * 255, 0, 255), 255].astype(
                np.uint8
            )
            meshes.append(mesh)
            rows.append(
                {"prim": path, "vertices": len(points), "bounds": mesh.bounds.tolist()}
            )
        if not meshes:
            raise ValueError("No room meshes")
        trimesh.util.concatenate(meshes).export(folder / "room.glb")
        cfg = json.loads((folder / "object_config.json").read_text())
        json.loads((folder / "meta.json").read_text())
        obj = next(o for o in cfg["objects"] if o.get("trajectory_index") == 0)
        pieces = []
        for g in obj["geoms"]:
            if "mesh_asset" not in g or g.get("group") == 3:
                continue
            asset = cfg["assets"][g["mesh_asset"]]
            path = run / asset["path"]
            if not path.exists():
                # Explicit diagnostic marker; never presented as the missing fruit mesh.
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.045)
                mesh.visual.face_colors = [255, 185, 40, 180]
                pieces.append(mesh)
                continue
            mesh = trimesh.load(path, force="mesh", process=False)
            mesh.vertices *= np.asarray(g.get("mesh_scale", [1, 1, 1]))
            # MuJoCo asset recentering is already encoded in geom local_pose.
            mesh.apply_transform(
                transform(g["local_pose"])
                @ np.linalg.inv(transform(g["mesh_pos"] + g["mesh_quat"]))
            )
            mesh.visual.face_colors = (
                [235, 195, 45, 255] if idx == 0 else [105, 180, 50, 255]
            )
            pieces.append(mesh)
        if not pieces:
            raise ValueError("No fruit mesh")
        trimesh.util.concatenate(pieces).export(folder / "object.glb")
        (folder / "scene_audit.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "mesh_count": len(rows),
                    "meshes": rows,
                    "notes": "Original Room_assets USD layout; flat source diffuse colors; texture maps omitted. Asset package cannot certify capture-byte identity.",
                },
                indent=2,
            )
        )
        print(label, len(rows), obj["id"])


if __name__ == "__main__":
    main()
