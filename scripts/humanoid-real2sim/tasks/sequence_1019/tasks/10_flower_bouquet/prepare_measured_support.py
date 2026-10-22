"""Version the side panel proxy using the independently measured cabinet height."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run / "scene/10_flower_bouquet"
    source = folder / "registered_assets"
    dest = folder / "registered_assets_measured_support_v1"
    dest.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.name not in {"cabinet_side.obj", "provenance.json"}:
            (dest / path.name).symlink_to(path.resolve())
    transform = np.array(json.loads((folder / "shared_level_room_floor_v3.json").read_text())["T_room_correction"])
    side = trimesh.load(source / "cabinet_side.obj", force="mesh", process=False)
    side.apply_transform(np.linalg.inv(transform))
    old_bounds = side.bounds.copy()
    side.vertices[:, 2] = np.minimum(side.vertices[:, 2], 1.321688)
    measured_bounds = side.bounds.copy()
    side.apply_transform(transform)
    side.export(dest / "cabinet_side.obj")
    (dest / "provenance.json").write_text(
        json.dumps(
            {
                "status": "measured cabinet height correction; independent rest still required",
                "source": str(source),
                "source_side_sha256": hashlib.sha256((source / "cabinet_side.obj").read_bytes()).hexdigest(),
                "old_curated_bounds": old_bounds.tolist(),
                "measured_curated_bounds": measured_bounds.tolist(),
                "evidence": "review/scene/room_export_basis_comparison.json; tall cabinet height 1.321688 m",
                "room_correction": transform.tolist(),
            },
            indent=2,
        )
    )
    print(dest)


if __name__ == "__main__":
    main()
