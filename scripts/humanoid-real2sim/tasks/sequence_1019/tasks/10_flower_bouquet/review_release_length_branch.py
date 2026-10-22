"""Project a source-estimated length/lean branch before changing any task assets."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run = args.run
    folder = run / "scene/10_flower_bouquet"
    assets = folder / "registered_assets_paper_wrapper_v1"
    report = json.loads((run / "review/scene/10_relative_flower_features_v1/report.json").read_text())
    branch = next(x for x in report["rows"] if x["fit_relative_marker_vectors"] and x["length_prior_sigma_m"] == 0.055)
    track = np.load(run / "retarget/hoi/10_flower_bouquet/tracks_v8/flower_bouquet.npz")
    old_r = Rotation.from_quat(track["quaternion_xyzw"][850])
    axis = np.array(branch["axis"])
    new_r = Rotation.align_vectors(axis[None], old_r.as_matrix()[:, 2][None])[0] * old_r
    new_p = np.array(branch["points_world"][0])
    scale = branch["length_m"] / 0.360078
    mesh = trimesh.load(assets / "flower_bouquet.obj", force="mesh", process=False)
    leaf = mesh.vertices[mesh.vertices[:, 2] > 0.36]
    paper = trimesh.load(assets / "flower_bouquet_hull0.obj", force="mesh", process=False).vertices
    new_paper = paper * [1, 1, scale]
    floor_corrected = new_p.copy()
    floor_corrected[2] = 0.6735 - new_r.apply(new_paper)[:, 2].min()
    cal = json.loads((folder / "head_marker_diagnostic_v4_header.json").read_text())
    heads = json.loads((run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for time in [123.4, 124.0, 124.6, 125.0]:
        h = heads[str(time)]
        H = np.eye(4)
        H[:3, :3] = C @ Rotation.from_quat(h["quaternion_xyzw_raw"]).as_matrix()
        H[:3, 3] = C @ h["position_raw"]
        H[2, 3] += 0.08245383203029633
        camera = H @ cal["T_head_rigid_camera_optical"]
        im = Image.open(run / f"inputs/10_flower_bouquet/header_views/{time}.jpg")
        draw = ImageDraw.Draw(im)
        for label, rotation, position, scaling, color in [
            ("old", old_r, track["position_w"][850], 1.0, "magenta"),
            ("new_length", new_r, floor_corrected, scale, "cyan"),
        ]:
            for vertices, width in [(leaf, 3), (paper, 2)]:
                world = rotation.apply(vertices * [1, 1, scaling]) + position
                local = (world - camera[:3, 3]) @ camera[:3, :3]
                uvw = local @ np.array(cal["K"]).T
                uv = uvw[:, :2] / uvw[:, 2:]
                hull = uv[ConvexHull(uv).vertices]
                draw.line([tuple(x) for x in np.vstack([hull, hull[0]])], fill=color, width=width)
            draw.text((15, 15 if label == "old" else 35), label, fill=color)
        im.save(args.out / f"{time}_envelopes.png")
    flower_points = []
    for file in assets.glob("flower_bouquet_hull*.obj"):
        m = trimesh.load(file, force="mesh", process=False)
        bary = np.array([(i / 6, j / 6, 1 - (i + j) / 6) for i in range(7) for j in range(7 - i)])
        points = np.einsum("ki,tij->tkj", bary, m.triangles).reshape(-1, 3) * [1, 1, scale]
        flower_points.append(new_r.apply(points) + floor_corrected)
    points = np.vstack(flower_points)
    gaps = {}
    for name in ["cabinet_side.obj", "room_back_wall.obj"]:
        m = trimesh.load(assets / name, force="mesh", process=False)
        gaps[name] = float(trimesh.proximity.closest_point(m, points)[1].min())
    output = {
        "status": "source-feature branch, not adopted and no rest test",
        "scale_relative_to_paper_v1_z": scale,
        "position_before_floor_constraint": new_p.tolist(),
        "position_after_floor_constraint": floor_corrected.tolist(),
        "quaternion_xyzw": new_r.as_quat().tolist(),
        "sampled_support_gap_m": gaps,
        "scope": "Cyan scaled native visual convex projected envelope versus magenta old. Full foliage appearance not fitted; images require independent review. Not exact occlusion-aware silhouette.",
    }
    (args.out / "report.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
