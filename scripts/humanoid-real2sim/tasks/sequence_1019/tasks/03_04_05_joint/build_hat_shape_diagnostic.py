"""Create a visible/collision-consistent bounded shape hypothesis, retaining originals."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    source = r / "review/scene/03_04_05_hat_multiview_axis_shape_diagnostic.json"
    fit = json.loads(source.read_text())["variants"]["axis_scale"]
    scene = r / "scene/03_04_05"
    out = scene / "hat_axis_shape_diagnostic"
    out.mkdir(exist_ok=False)
    tree = ET.parse(scene / "hat_rack_physics_early_registered.xml")
    scale = fit["scale_xyz"]
    modified = []
    for mesh in tree.getroot().findall(".//asset/mesh"):
        if mesh.get("name", "").startswith("black_ribbon_hat"):
            mesh.set("scale", " ".join(map(str, scale)))
            modified.append(mesh.get("name"))
    tree.write(out / "hat_rack_physics.xml")
    report = {
        "status": "source_constrained_shape_hypothesis_not_accepted",
        "source_fit": str(source),
        "scale_xyz": scale,
        "pose_xyzw": fit["pose_xyzw"],
        "dimensions_m": [0.28 * scale[0], 0.235 * scale[1], 0.145 * scale[2]],
        "changed_meshes": modified,
        "mass_policy": "Original per-part masses unchanged; inertias recomputed from consistently scaled collision meshes.",
        "visual_collision_consistency": "Same affine transform applied to every visible and collision hat mesh.",
        "limitations": [
            "Shape and camera depth are correlated; these are estimated hypothesis dimensions, not physical measurements.",
            "47.7/48 source contours and bow trained; 48.3 withheld from shape fit.",
            "No hidden pocket, latch, COM relocation, or collision-only shrink added.",
            "Contact, native/RSS rest, robot grasp, and independent side-view review remain required.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
