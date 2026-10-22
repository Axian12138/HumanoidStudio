"""Build explicit source-motivated crown crease hypotheses without selecting one."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def deform(vertices, depth, axis):
    """Injective vertical deformation, fixed below band and fixed brim/band."""
    result = vertices.copy()
    cross = -np.sin(axis) * vertices[:, 0] + np.cos(axis) * vertices[:, 1]
    t = np.clip((vertices[:, 2] - 0.055) / 0.07, 0, 1)
    ramp = t * t * (3 - 2 * t)
    result[:, 2] -= depth * np.exp(-((cross / 0.026) ** 2)) * ramp
    return result


def main():
    import trimesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    scene = args.run / "scene/03_04_05"
    output = scene / "hat_crown_crease_family_v1"
    output.mkdir(exist_ok=False)
    original = trimesh.load(scene / "black_ribbon_hat_visual_part_3.obj", process=False)
    rows = []
    for depth in (0.015, 0.025, 0.035):
        for axis in (0.0, np.pi / 2):
            mesh = original.copy()
            mesh.vertices = deform(original.vertices, depth, axis)
            name = f"crease_{round(depth * 1000):02d}mm_axis{round(np.rad2deg(axis)):02d}"
            path = output / f"{name}.obj"
            mesh.export(path)
            rows.append(
                {
                    "name": name,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "maximum_crease_depth_parameter_m": depth,
                    "axis_rad": axis,
                    "maximum_actual_vertex_displacement_m": float(
                        np.linalg.norm(mesh.vertices - original.vertices, axis=1).max()
                    ),
                    "watertight": bool(mesh.is_watertight),
                    "minimum_analytic_dz_dz_bound": 1 - depth * 1.5 / 0.07,
                    "bounds": mesh.bounds.tolist(),
                    "volume_ratio": float(mesh.volume / original.volume),
                }
            )
    report = {
        "status": "unselected_crown_shape_hypotheses_not_motion_or_support_acceptance",
        "source_geometry": str(scene / "black_ribbon_hat_visual_part_3.obj"),
        "source_evidence": str(
            args.run / "inputs/03_04_05/hat_static_shape_camera/static_to_grasp_source_shape_evidence.json"
        ),
        "definition": "Vertical Gaussian trough across one crown axis, width26mm, ramp0 below55mm to1 above125mm. Brim, band and bows are not altered by this family.",
        "reason": "Source39.2-40.0 visibly shows a longitudinal ridge/concavity, with near side hand occlusion. A preformed crease remains possible; dynamics are not inferred.",
        "parameter_status": "15/25/35mm and two axes are diagnostic sensitivity samples, NOT measured depth estimates or uncertainty bounds.",
        "injectivity": "x/y unchanged; dz_new/dz positive for all variants by analytic bound. This preserves shell ordering, but is not a collision-decomposition certificate.",
        "variants": rows,
        "limitations": [
            "No candidate selected; fixed-camera internal feature comparison is still required.",
            "No collision proxy exported: old collision and interaction certificates must not be reused for these meshes.",
            "No global crown shrink, hand-fitted recess, hidden support, or dynamics tuning.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "variants": len(rows)}))


if __name__ == "__main__":
    main()
