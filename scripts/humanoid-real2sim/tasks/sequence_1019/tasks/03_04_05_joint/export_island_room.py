"""Export a room once, with explicit per-furniture world transforms and exclusions."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget3/src")
    import retarget3.runtime  # noqa: F401
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(spec["source_usd"])
    cache = UsdGeom.XformCache()
    rows, vertices, faces, counts = [], [], [], {}
    for prim in stage.Traverse():
        name = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or any(name.startswith(p) for p in spec["excluded_prefixes"]):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() in ("guide", "proxy"):
            continue
        rules = [rule for rule in spec["rules"] if name.startswith(rule["prefix"])]
        if len(rules) > 1:
            raise ValueError(f"overlapping transform rules: {name}")
        rule = rules[0] if rules else {"id": "unchanged_raw_usd", "transform": np.eye(4).tolist()}
        transform = np.asarray(rule["transform"])
        if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError(f"invalid world transform: {rule['id']}")
        mesh = UsdGeom.Mesh(prim)
        matrix = cache.GetLocalToWorldTransform(prim)
        points = np.asarray([matrix.Transform(Gf.Vec3d(*map(float, v))) for v in mesh.GetPointsAttr().Get()])
        raw_points_sha256 = hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()
        points = points @ transform[:3, :3].T + transform[:3, 3]
        if not np.isfinite(points).all():
            raise ValueError(f"nonfinite world points: {name}")
        offset = len(vertices)
        vertices.extend(points.tolist())
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        cursor = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            polygon = indices[cursor : cursor + count]
            for j in range(1, count - 1):
                faces.append([offset + polygon[k] + 1 for k in (0, j, j + 1)])
            cursor += count
        counts[rule["id"]] = counts.get(rule["id"], 0) + 1
        rows.append(
            {
                "prim": name,
                "rule": rule["id"],
                "transform": transform.tolist(),
                "vertex_offset": offset,
                "vertex_count": len(points),
                "raw_world_vertices_sha256": raw_points_sha256,
                "transformed_world_vertices_sha256": hashlib.sha256(
                    np.asarray(points, dtype="<f8").tobytes()
                ).hexdigest(),
                "face_indices_sha256": hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
                "bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
            }
        )
    if any(rule["id"] not in counts for rule in spec["rules"]):
        raise ValueError("a furniture transform rule matched no visible mesh")
    if not vertices or len({row["prim"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicated static geometry")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for point in vertices:
            stream.write("v " + " ".join(map(str, point)) + "\n")
        for face in faces:
            stream.write("f " + " ".join(map(str, face)) + "\n")
    report = {
        "status": "per_furniture_candidate_not_accepted",
        "source_usd_sha256": sha256(spec["source_usd"]),
        "spec_sha256": sha256(args.spec),
        "output_sha256": sha256(args.output),
        "rule_mesh_counts": counts,
        "vertices": len(vertices),
        "triangles": len(faces),
        "meshes": rows,
        "limitations": spec["limitations"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "meshes"}, indent=2))


if __name__ == "__main__":
    main()
