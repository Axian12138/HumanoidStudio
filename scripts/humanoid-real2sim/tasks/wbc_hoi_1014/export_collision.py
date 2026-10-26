"""Export original visible room triangles without display offsets for audit."""

import argparse
from pathlib import Path

import numpy as np


def main():
    from pxr import Usd, UsdGeom

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("usd", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    stage = Usd.Stage.Open(str(args.usd.resolve()))
    cache = UsdGeom.XformCache()
    vertices = []
    faces = []
    owners = []
    paths = []
    offset = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (
            not prim.IsA(UsdGeom.Mesh)
            or "/Tasks/" in path
            or "/Task/" in path
            or "OmniverseKitViewportCameraMesh" in path
        ):
            continue
        im = UsdGeom.Imageable(prim)
        if im.ComputeVisibility() == "invisible" or im.ComputePurpose() in ("guide", "proxy"):
            continue
        geo = UsdGeom.Mesh(prim)
        v = np.asarray(geo.GetPointsAttr().Get(), float)
        mat = np.asarray(cache.GetLocalToWorldTransform(prim))
        v = v @ mat[:3, :3] + mat[3, :3]
        indices = geo.GetFaceVertexIndicesAttr().Get()
        tri = []
        cursor = 0
        for count in geo.GetFaceVertexCountsAttr().Get():
            ids = indices[cursor : cursor + count]
            tri.extend((ids[0], ids[j], ids[j + 1]) for j in range(1, count - 1))
            cursor += count
        if not tri:
            continue
        vertices.append(v)
        faces.append(np.asarray(tri) + offset)
        owners.extend([len(paths)] * len(tri))
        paths.append(path)
        offset += len(v)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vertices=np.concatenate(vertices),
        faces=np.concatenate(faces),
        mesh_indices=np.array(owners),
        mesh_paths=np.array(paths),
    )
    print("Exported", len(paths), "meshes;", sum(map(len, faces)), "triangles")


if __name__ == "__main__":
    main()
