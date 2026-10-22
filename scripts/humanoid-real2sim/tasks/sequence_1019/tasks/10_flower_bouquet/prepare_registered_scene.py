"""Build a private, uncertainty-bounded scene candidate without changing old assets."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    folder = r / "scene/10_flower_bouquet"
    assets = folder / "assets"
    dest = folder / "registered_assets"
    dest.mkdir(exist_ok=True)
    room = np.array(json.loads((folder / "shared_level_room_floor_v3.json").read_text())["T_room_correction"])
    room_path = Path("outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/assets/ikea_1012_room.obj")
    for name, path in [
        ("room", room_path),
        ("cabinet_support", assets / "cabinet_support_v2.obj"),
        ("cabinet_side", assets / "cabinet_side_v2.obj"),
    ]:
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh.apply_transform(room)
        mesh.export(dest / (name + ".obj"))
    basket = json.loads((folder / "basket_registered_candidate.json").read_text())
    dx, dy, dz = basket["dimensions_world_aligned_m"]
    mesh = trimesh.load(assets / "small_wicker_basket.obj", force="mesh", process=False)
    mesh.apply_scale([dy / 0.25, dx / 0.20, dz / 0.16])
    mesh.export(dest / "small_wicker_basket.obj")
    w = 0.012
    for i, (center, size) in enumerate(
        [
            ((0, 0, w / 2), (dy, dx, w)),
            ((dy / 2 - w / 2, 0, dz / 2), (w, dx, dz)),
            ((-dy / 2 + w / 2, 0, dz / 2), (w, dx, dz)),
            ((0, dx / 2 - w / 2, dz / 2), (dy - 2 * w, w, dz)),
            ((0, -dx / 2 + w / 2, dz / 2), (dy - 2 * w, w, dz)),
        ]
    ):
        mesh = trimesh.creation.box(size)
        mesh.apply_translation(center)
        mesh.export(dest / f"small_wicker_basket_hull{i}.obj")
    point = json.loads((folder / "flower_point_triangulation.json").read_text())["stem_tip"]["position"]
    flower = {
        "position_m": point,
        "quaternion_xyzw": [0.6412218182, -0.0448385959, -0.0534365622, 0.7641783915],
        "status": "estimated initial support pose; stem tip multi-view fit and native asset, independent rest pending",
        "camera_lateral_uncertainty_m": 0.02,
        "geometry_uncertainty_m": 0.055,
    }
    (folder / "flower_registered_candidate.json").write_text(json.dumps(flower, indent=2))
    (dest / "provenance.json").write_text(
        json.dumps(
            {
                "room_source": str(room_path),
                "room_transform": room.tolist(),
                "basket_source": "basket_registered_candidate.json",
                "flower_source": "flower_registered_candidate.json",
                "status": "private candidate not accepted global registration",
            },
            indent=2,
        )
    )
    print(dest)


if __name__ == "__main__":
    main()
