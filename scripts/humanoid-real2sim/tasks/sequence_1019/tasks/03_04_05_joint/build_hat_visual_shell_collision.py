"""Isolated, cavity-preserving collision hypothesis from the authored visual shell."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate


def main():
    import mujoco
    import trimesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    scene = args.run.resolve() / "scene/03_04_05"
    out = scene / "hat_visual_shell_collision_v1"
    out.mkdir(exist_ok=False)
    source = scene / "hat_rack_physics_early_registered.xml"
    original = mujoco.MjModel.from_xml_path(str(source))
    bid = original.body("black_ribbon_hat").id
    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    body = root.find(".//body[@name='black_ribbon_hat']")
    removed = []
    for geom in list(body.findall("geom")):
        if geom.get("group") == "3":
            removed.append(geom.attrib.copy())
            body.remove(geom)
    # Preserve compiled old inertia for the first representation-only experiment.
    for inertial in body.findall("inertial"):
        body.remove(inertial)
    ET.SubElement(
        body,
        "inertial",
        mass=str(original.body_mass[bid]),
        pos=" ".join(map(str, original.body_ipos[bid])),
        quat=" ".join(map(str, original.body_iquat[bid])),
        diaginertia=" ".join(map(str, original.body_inertia[bid])),
    )
    parts = []
    cell_index = 0
    # Original meshes use 96 equal angular sectors and an elliptical y/x ratio.
    # At theta=0 the authored crown profile closes along the rotation axis.
    profiles = {3: [0, 1, 2, 3, 4, 5, 991, 282, 6, 7, 8, 9, 10, 11], 4: [0, 1, 2, 3], 5: [0, 1, 2, 3]}
    for part in range(3, 8):
        src = scene / f"black_ribbon_hat_visual_part_{part}.obj"
        visual = trimesh.load(src, process=False)
        cells = []
        if part in profiles:
            profile = visual.vertices[profiles[part]][:, [0, 2]]
            polygon = Polygon(profile)
            assert polygon.is_valid
            triangles = [t for t in triangulate(polygon) if polygon.covers(t)]
            assert abs(sum(t.area for t in triangles) - polygon.area) < 1e-12
            # Each triangle is swept through just one original angular interval.
            # Its convex cell cannot bridge opposite sides of the hat cavity.
            for k in range(96):
                angles = np.array([k, k + 1]) * 2 * np.pi / 96
                for triangle in triangles:
                    rz = np.array(triangle.exterior.coords)[:3]
                    vertices = np.concatenate(
                        [
                            np.column_stack([rz[:, 0] * np.cos(t), rz[:, 0] * np.sin(t) * (0.235 / 0.28), rz[:, 1]])
                            for t in angles
                        ]
                    )
                    cells.append(trimesh.convex.convex_hull(vertices))
        else:
            # Bow lobes are already convex to source float precision.
            cells = [visual.convex_hull]
        volume = sum(c.volume for c in cells)
        parts.append(
            {
                "part": part,
                "source": str(src),
                "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "source_watertight": bool(visual.is_watertight),
                "cells": len(cells),
                "all_cells_watertight": all(c.is_watertight for c in cells),
                "source_volume_m3": float(visual.volume),
                "sum_cell_volume_m3": float(volume),
                "sum_cell_volume_ratio": float(volume / visual.volume),
            }
        )
        for cell in cells:
            name = f"hat_shell_part{part}_cell{cell_index}"
            path = out / f"{name}.obj"
            cell.export(path)
            ET.SubElement(asset, "mesh", name=name, file=str(path))
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, group="3", rgba="0 0 0 0", mass="0")
            cell_index += 1
    tree.write(out / "hat_rack_physics.xml", encoding="unicode")
    check = mujoco.MjModel.from_xml_path(str(out / "hat_rack_physics.xml"))
    new_bid = check.body("black_ribbon_hat").id
    inertia_deltas = {
        key: float(np.max(np.abs(getattr(check, key)[new_bid] - getattr(original, key)[bid])))
        for key in ["body_mass", "body_ipos", "body_iquat", "body_inertia"]
    }
    assert max(inertia_deltas.values()) < 1e-12
    report = {
        "status": "isolated_collision_representation_candidate_not_accepted",
        "original_model": str(source),
        "original_model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "visual_changed": False,
        "original_collision_count": len(removed),
        "new_collision_count": cell_index,
        "inertia_policy": "Explicit original compiled inertia retained, to isolate collision representation; not measured physical mass distribution.",
        "inertia_delta": inertia_deltas,
        "parts": parts,
        "limitations": [
            "Cells use convex hulls within a single 3.75-degree source sector; surface and cavity error audit is required before rest conclusions.",
            "Volume sum alone is not a union/Hausdorff proof and does not rule out cell overlaps.",
            "All motion/grasp/release certificates remain bound to the old geometry and do not transfer.",
            "No source visual, rack geometry, hidden support or motion has been modified.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(out), "parts": parts, "inertia_delta": inertia_deltas}))


if __name__ == "__main__":
    main()
