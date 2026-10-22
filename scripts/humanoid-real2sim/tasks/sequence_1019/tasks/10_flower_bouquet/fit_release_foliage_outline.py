"""Fit visible foliage envelopes, independently of support forces or rest success."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from PIL import Image, ImageDraw
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run = args.run
    folder = run / "scene/10_flower_bouquet"
    observations = {
        123.4: [
            [701, 0],
            [685, 120],
            [692, 168],
            [718, 196],
            [760, 217],
            [820, 197],
            [891, 153],
            [1000, 34],
            [995, 20],
            [914, 54],
            [873, 0],
        ],
        124.0: [[592, 0], [599, 94], [612, 149], [658, 180], [701, 188], [748, 166], [801, 121], [848, 65], [911, 0]],
        124.6: [
            [551, 0],
            [552, 35],
            [560, 80],
            [578, 108],
            [597, 139],
            [624, 151],
            [646, 156],
            [684, 150],
            [717, 125],
            [747, 85],
            [776, 50],
            [800, 0],
        ],
        125.0: [
            [470, 0],
            [478, 40],
            [485, 80],
            [510, 104],
            [520, 136],
            [565, 155],
            [600, 164],
            [637, 141],
            [670, 119],
            [696, 80],
            [720, 30],
            [721, 0],
        ],
    }
    # Convex outer envelopes, not pixel masks or individual branch correspondences.
    observed = {t: np.array(p)[ConvexHull(p).vertices] for t, p in observations.items()}
    cal = json.loads((folder / "head_marker_diagnostic_v4_header.json").read_text())
    heads = json.loads((run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    cameras = {}
    for time in observations:
        h = heads[str(time)]
        H = np.eye(4)
        H[:3, :3] = C @ Rotation.from_quat(h["quaternion_xyzw_raw"]).as_matrix()
        H[:3, 3] = C @ h["position_raw"]
        H[2, 3] += 0.08245383203029633
        cameras[time] = H @ cal["T_head_rigid_camera_optical"]
    branch = json.loads((run / "review/scene/10_release_length_branch_v1/report.json").read_text())
    rotation = Rotation.from_quat(branch["quaternion_xyzw"])
    position = np.array(branch["position_after_floor_constraint"])
    mesh = trimesh.load(folder / "registered_assets_paper_wrapper_v1/flower_bouquet.obj", force="mesh", process=False)
    vertices = mesh.vertices[mesh.vertices[:, 2] > 0.32]
    _, ids = np.unique(np.round(vertices / 0.004), axis=0, return_index=True)
    vertices = vertices[ids]

    def clip_top(poly):
        result = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
            if a[1] >= 0:
                result.append(a)
            if (a[1] >= 0) != (b[1] >= 0):
                result.append(a + (b - a) * (-a[1]) / (b[1] - a[1]))
        return np.array(result)

    def polygon(x, time):
        v = vertices.copy()
        v[:, :2] *= x[:2]
        blend = np.clip((v[:, 2] - 0.30) / 0.20, 0, 1)
        v[:, :2] += blend[:, None] * x[2:4]
        v = Rotation.from_euler("z", x[4]).apply(v)
        world = rotation.apply(v) + position
        cam = cameras[time]
        uvw = ((world - cam[:3, 3]) @ cam[:3, :3]) @ np.array(cal["K"]).T
        uv = uvw[:, :2] / uvw[:, 2:]
        return clip_top(uv[ConvexHull(uv).vertices])

    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    directions = np.array([np.cos(angles), np.sin(angles)]).T

    def signatures(poly):
        return np.max(poly @ directions.T, axis=0)

    def residual(x):
        return np.r_[
            np.concatenate([(signatures(polygon(x, t)) - signatures(observed[t])) / 12 for t in [123.4, 124.0]]),
            (x[:2] - 1) / 0.5,
            x[2:4] / 0.08,
            x[4] / 1.0,
        ]

    fits = []
    for roll in [0, np.pi / 2, -np.pi / 2]:
        fit = least_squares(
            residual,
            [1, 1, 0, 0, roll],
            bounds=([0.5, 0.5, -0.25, -0.25, -np.pi], [2.5, 2.5, 0.25, 0.25, np.pi]),
            max_nfev=300,
            diff_step=1e-4,
        )
        fits.append(fit)
    fit = min(fits, key=lambda f: np.linalg.norm(f.fun))
    rows = []
    for time in observations:
        pred = polygon(fit.x, time)
        errors = signatures(pred) - signatures(observed[time])
        rows.append(
            {
                "source_s": time,
                "partition": "fit" if time in [123.4, 124.0] else "holdout",
                "support_function_rms_px": float(np.sqrt(np.mean(errors**2))),
                "support_function_max_px": float(np.max(abs(errors))),
                "predicted_clipped_polygon": pred.tolist(),
            }
        )
        im = Image.open(run / f"inputs/10_flower_bouquet/header_views/{time}.jpg")
        draw = ImageDraw.Draw(im)
        for poly, color in [(observed[time], "yellow"), (pred, "cyan")]:
            draw.line([tuple(v) for v in np.vstack([poly, poly[0]])], fill=color, width=3)
        im.save(args.out / f"{time}_outline.png")
    report = {
        "status": "diagnostic_no_assets_changed",
        "parameters_sx_sy_offsetx_offsety_roll": fit.x.tolist(),
        "observations": observations,
        "rows": rows,
        "geometry_basis": "Paper shortened branch pose; native foliage original z extent retained, transverse scale/offset/roll fitted to visual envelope only.",
        "limitations": "Manual outer envelope includes convex empty space and partial upper clipping. Not evidence of solid leaf contact, no stability objective or dynamics test.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
