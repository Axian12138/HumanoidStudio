"""Create explicit candidate collision/visual variants without editing source assets."""

import argparse
import json
from pathlib import Path


def main():
    import trimesh

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    assets = a.run / "scene/10_flower_bouquet/assets"
    candidate = json.loads((assets.parent / "basket_multiframe_candidate.json").read_text())
    dx, dy, dz = candidate["dimensions_world_aligned_m"]
    sx, sy = dy / 0.25, dx / 0.20
    mesh = trimesh.load(assets / "small_wicker_basket.obj", force="mesh", process=False)
    mesh.apply_scale((sx, sy, dz / 0.16))
    mesh.export(assets / "small_wicker_basket_estimate.obj")
    # 12 mm distinct walls; never merge the opening into a single convex hull.
    w = 0.012
    boxes = [
        ((0, 0, w / 2), (dy, dx, w)),
        ((dy / 2 - w / 2, 0, dz / 2), (w, dx, dz)),
        ((-dy / 2 + w / 2, 0, dz / 2), (w, dx, dz)),
        ((0, dx / 2 - w / 2, dz / 2), (dy - 2 * w, w, dz)),
        ((0, -dx / 2 + w / 2, dz / 2), (dy - 2 * w, w, dz)),
    ]
    for i, (center, size) in enumerate(boxes):
        m = trimesh.creation.box(size)
        m.apply_translation(center)
        m.export(assets / f"small_wicker_basket_estimate_hull{i}.obj")
    # Cabinet dimensions from authored room mesh, not robot reach.
    m = trimesh.creation.box((3.036 - 2.577, 1.140 + 0.624, 0.06))
    m.apply_translation(((-3.036 - 2.577) / 2, (1.140 - 0.624) / 2, 0.631))
    m.export(assets / "cabinet_support_v2.obj")
    m = trimesh.creation.box((0.459, 0.018, 1.0))
    m.apply_translation((-2.8065, -0.633, 1.161))
    m.export(assets / "cabinet_side_v2.obj")
    (assets / "geometry_v2.json").write_text(
        json.dumps(
            {
                "basket_scale_native": [sx, sy, dz / 0.16],
                "basket_dimensions_native_m": [dy, dx, dz],
                "source_candidate": str(assets.parent / "basket_multiframe_candidate.json"),
                "support_top_z_m": 0.661,
                "side_panel_y_m": -0.624,
                "representation": "rigid approximation, external appearance may differ; cavity maintained by separate walls",
                "position_uncertainty_m": 0.05,
                "dimension_uncertainty_m": 0.04,
                "mass_and_friction": "provisional; no interaction physics conclusion",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
