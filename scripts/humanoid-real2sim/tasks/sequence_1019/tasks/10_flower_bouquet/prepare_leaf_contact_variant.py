"""Version a bounded planar leaf-contact proxy; never overwrite an accepted asset."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assets = args.run / "scene/10_flower_bouquet/registered_assets_paper_wrapper_v1"
    tracks = args.run / "retarget/hoi/10_flower_bouquet/tracks_v8"
    dest = args.out / "assets"
    dest.mkdir()
    track_dest = args.out / "tracks"
    track_dest.mkdir()
    flower = dict(np.load(tracks / "flower_bouquet.npz"))
    room = np.array(
        json.loads((args.run / "scene/10_flower_bouquet/shared_level_room_floor_v3.json").read_text())[
            "T_room_correction"
        ]
    )
    gap = json.loads((args.run / "review/scene/10_native31_inventory/visual_side_gap.json").read_text())
    depth = float(gap["nearest_visual_vertex_gap_m"])
    normal_world = room[:3, 1]
    rotation = Rotation.from_quat(flower["quaternion_xyzw"][850])
    normal_local = rotation.inv().apply(normal_world)
    mesh = trimesh.load(assets / "flower_bouquet_hull1.obj", force="mesh", process=False)
    values = mesh.vertices @ normal_local
    plane = values.min() + depth
    vertices = list(mesh.vertices[values >= plane])
    for i, j in mesh.edges_unique:
        if (values[i] - plane) * (values[j] - plane) < 0:
            weight = (plane - values[i]) / (values[j] - values[i])
            vertices.append(mesh.vertices[i] + weight * (mesh.vertices[j] - mesh.vertices[i]))
    variant = trimesh.convex.convex_hull(np.asarray(vertices))
    variant.export(dest / "flower_bouquet_hull1.obj")
    for path in assets.iterdir():
        if path.name != "flower_bouquet_hull1.obj":
            (dest / path.name).symlink_to(path.resolve())
    time = flower["source_time_s"]
    weight = np.clip((time - 123.2) / 0.2, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    flower["position_w"] = flower["position_w"].astype(float) - weight[:, None] * depth * normal_world
    np.savez_compressed(track_dest / "flower_bouquet.npz", **flower)
    (track_dest / "small_wicker_basket.npz").symlink_to((tracks / "small_wicker_basket.npz").resolve())
    (track_dest / "tracks.json").write_text(
        json.dumps(
            {"objects": [{"name": name, "path": name + ".npz"} for name in ["flower_bouquet", "small_wicker_basket"]]},
            indent=2,
        )
    )
    on_cap = np.abs(variant.vertices @ normal_local - plane) < 1e-7
    cap_vertices = variant.vertices[on_cap]
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "status": "unreviewed representation hypothesis, not measured leaf deformation",
                "source_assets": str(assets.resolve()),
                "source_tracks": str(tracks.resolve()),
                "depth_basis": "Precomputed native visual-to-cabinet-side nearest vertex gap at source125, not chosen from rest success",
                "cut_depth_m": depth,
                "normal_world": normal_world.tolist(),
                "normal_local": normal_local.tolist(),
                "cap_vertices_local": cap_vertices.tolist(),
                "cap_span_local_xyz_m": np.ptp(cap_vertices, axis=0).tolist(),
                "geometry_change": "Remove existing ellipsoid side cap and close with planar face; no new outward envelope or hidden support.",
                "pose_change": "Translate flower toward cabinet by measured visual gap after right release123.2-123.4 using smoothstep; preserve all pre123.2 poses, basket, and robot.",
                "mass_friction": "Unchanged authored masses, uniform visual inertia approximation, friction0.6.",
                "limitations": "Finite flattened contact represents compliant leaves using a rigid aggregate. Must independently check source projections, all contacts, and free rest; source leaf stiffness/contact footprint is unmeasured.",
            },
            indent=2,
        )
    )
    print(args.out)


if __name__ == "__main__":
    main()
