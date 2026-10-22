"""Add the existing curated room's measured wall as a separate collision solid."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--authored-masses", action="store_true")
    args = parser.parse_args()
    folder = args.run / "scene/10_flower_bouquet"
    source = folder / "registered_assets_measured_support_v1"
    dest = folder / (
        "registered_assets_measured_support_v3" if args.authored_masses else "registered_assets_measured_support_v2"
    )
    dest.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.name != "provenance.json":
            (dest / path.name).symlink_to(path.resolve())
    transform = np.array(json.loads((folder / "shared_level_room_floor_v3.json").read_text())["T_room_correction"])
    low = np.array([-3.145, -3.89, 0.0])
    high = np.array([-3.04, 1.71, 2.35])
    wall = trimesh.creation.box(high - low)
    wall.apply_translation((low + high) / 2)
    wall.apply_transform(transform)
    wall.export(dest / "room_back_wall.obj")
    if args.authored_masses:
        (dest / "physical_properties.json").write_text(
            json.dumps(
                {
                    "mass_kg": {"flower_bouquet": 0.20, "small_wicker_basket": 0.25},
                    "source": "original USD /Asset physics:mass; independently inspected",
                    "evidence": "review/scene/10_independent_mass_properties.json",
                    "com_inertia": "not authored in USD; RSS uniform closed visual volume assumption remains unverified for real thin leaves/paper/wire",
                },
                indent=2,
            )
        )
    (dest / "provenance.json").write_text(
        json.dumps(
            {
                "status": "existing source-room wall collision added; no pose fit",
                "source": str(source),
                "measured_curated_bounds": [low.tolist(), high.tolist()],
                "evidence": "registered room.obj faces160813-160816, transformed back into curated coordinates",
                "room_correction": transform.tolist(),
                "limitation": "source-video wall registration and final rest require independent review",
            },
            indent=2,
        )
    )
    print(dest)


if __name__ == "__main__":
    main()
