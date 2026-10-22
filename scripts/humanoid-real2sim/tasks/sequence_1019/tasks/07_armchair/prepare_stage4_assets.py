"""Decompose the chair and dining table into physical convex collision pieces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import vhacdx

from robosimstudio.humanoid_real2sim.paths import example_dir


TASK = example_dir(__file__)
SNAPSHOT = TASK.parents[1]
SOURCES = {
    "chair": TASK / "grey_armchair_visual.obj",
    "table": SNAPSHOT / "assets/morbylanga_table_1012.obj",
}
HULL_LIMITS = {"chair": 32, "table": 16}


def chair_semantic_hulls(mesh: trimesh.Trimesh) -> dict[str, trimesh.Trimesh]:
    """Cover the chair parts V-HACD drops from its open visual surface.

    The exported mesh has separate open seat, leg and backrest surfaces. V-HACD
    processes the seat but ignores the narrow legs and tall backrest. These
    spatially selected convex pieces retain their local OBJ coordinates.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    hulls = {}
    for x_side in (-1, 1):
        for y_side in (-1, 1):
            selected = vertices[
                (vertices[:, 0] * x_side > 0.19)
                & (vertices[:, 1] * y_side > 0.08)
                & (vertices[:, 2] < 0.445)
            ]
            hulls[f"leg_{'left' if x_side < 0 else 'right'}_"
                  f"{'front' if y_side < 0 else 'rear'}"] = trimesh.convex.convex_hull(selected)
        selected = vertices[
            (vertices[:, 0] * x_side >= 0)
            & (vertices[:, 1] > 0.14)
            & (vertices[:, 2] >= 0.42)
        ]
        hulls[f"back_{'left' if x_side < 0 else 'right'}"] = trimesh.convex.convex_hull(selected)
    return hulls


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"method": "V-HACD convex decomposition", "objects": {}}
    for name, source in SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        mesh = trimesh.load_mesh(source, force="mesh")
        pieces = vhacdx.compute_vhacd(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.uint32),
            maxConvexHulls=HULL_LIMITS[name],
            resolution=150000,
            minimumVolumePercentErrorAllowed=1.0,
            maxRecursionDepth=10,
            shrinkWrap=True,
        )
        paths = []
        skipped = []
        for i, (vertices, faces) in enumerate(pieces):
            hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            if len(hull.vertices) < 4 or not hull.is_watertight or hull.volume < 1e-7:
                # VHACD occasionally emits a zero-volume submillimeter
                # sliver around a table leg. It cannot be a MuJoCo collider.
                skipped.append({"index": i, "extents_m": hull.extents.tolist(),
                                "volume_m3": float(hull.volume)})
                continue
            path = out / f"{name}_collision_{i:02d}.obj"
            hull.export(path)
            paths.append(path.name)
        semantic = {}
        if name == "chair":
            for part, hull in chair_semantic_hulls(mesh).items():
                if not hull.is_watertight or hull.volume < 1e-6:
                    raise ValueError(f"chair {part} collider is degenerate")
                path = out / f"chair_collision_{part}.obj"
                hull.export(path)
                paths.append(path.name)
                semantic[part] = path.name
            all_bounds = np.asarray([trimesh.load_mesh(out / path).bounds for path in paths])
            low = np.min(all_bounds[:, 0], axis=0)
            high = np.max(all_bounds[:, 1], axis=0)
            if low[2] > 0.01 or high[2] < 0.80:
                raise ValueError(f"chair colliders omit feet or backrest: {low}, {high}")
        manifest["objects"][name] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "source_bounds_m": mesh.bounds.tolist(),
            "source_watertight": bool(mesh.is_watertight),
            "hull_limit": HULL_LIMITS[name],
            "hull_count": len(paths),
            "hulls": paths,
            "semantic_hulls": semantic,
            "skipped_degenerate_hulls": skipped,
        }
    (out / "collision_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
