"""Compare source landmark residuals across a support-constrained lean family."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    folder = args.run / "scene/10_flower_bouquet"
    assets = folder / "registered_assets_paper_wrapper_v1"
    cal = json.loads((folder / "head_marker_diagnostic_v4_header.json").read_text())
    raw = json.loads((args.run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    K = np.array(cal["K"])
    cameras = []
    times = [123.4, 124.0, 124.6, 125.0]
    for time in times:
        row = raw[str(time)]
        head = np.eye(4)
        head[:3, :3] = C @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
        head[:3, 3] = C @ row["position_raw"]
        head[2, 3] += 0.08245383203029633
        cameras.append(head @ cal["T_head_rigid_camera_optical"])
    observed = np.array(
        [
            [[749, 435], [749, 220]],
            [[710, 408], [688, 190]],
            [[707, 390], [667, 169]],
            [[666, 387], [607, 168]],
        ]
    )
    room = np.array(json.loads((folder / "shared_level_room_floor_v3.json").read_text())["T_room_correction"])
    side = trimesh.load(assets / "cabinet_side.obj", force="mesh", process=False).vertices
    side_y = ((side - room[:3, 3]) @ room[:3, :3])[:, 1].max()
    leaf = trimesh.load(assets / "flower_bouquet_hull1.obj", force="mesh", process=False).vertices
    paper = trimesh.load(assets / "flower_bouquet_hull0.obj", force="mesh", process=False).vertices
    old = json.loads((folder / "placed_stem_supported_multiview.json").read_text())
    old_rotation = Rotation.from_quat(old["quaternion_xyzw"])

    def unpack(x, axis_x):
        axis = np.array([axis_x, x[1], np.sqrt(1 - axis_x**2 - x[1] ** 2)])
        rotation = Rotation.align_vectors(axis[None], old_rotation.as_matrix()[:, 2][None])[0] * old_rotation
        base = np.array([x[0], 0.0, 0.6735 - rotation.apply(paper)[:, 2].min()])
        leaf_y = ((rotation.apply(leaf) + base - room[:3, 3]) @ room[:3, :3])[:, 1].min()
        base[1] = (side_y + 0.0005 - leaf_y) / room[1, 1]
        return base, rotation, axis

    def project(base, axis):
        points = np.array([base, base + 0.360078 * axis])
        projected = []
        for camera in cameras:
            local = (points - camera[:3, 3]) @ camera[:3, :3]
            uvw = local @ K.T
            projected.append(uvw[:, :2] / uvw[:, 2:])
        return np.array(projected)

    rows = []
    for axis_x in [-0.35, -0.30, -0.27, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05]:

        def residual(x, axis_x=axis_x):
            base, _, axis = unpack(x, axis_x)
            return ((project(base, axis)[:3] - observed[:3]) / 20).ravel()

        fit = least_squares(residual, [-2.61, -0.145], bounds=([-2.8, -0.5], [-2.45, 0.05]))
        base, rotation, axis = unpack(fit.x, axis_x)
        uv = project(base, axis)
        rows.append(
            {
                "axis_x_fixed": axis_x,
                "base": base.tolist(),
                "axis": axis.tolist(),
                "quaternion_xyzw": rotation.as_quat().tolist(),
                "train_errors_px": np.linalg.norm(uv[:3] - observed[:3], axis=2).tolist(),
                "holdout125_errors_px": np.linalg.norm(uv[3] - observed[3], axis=1).tolist(),
                "projected_pixels": uv.tolist(),
            }
        )
    report = {
        "status": "diagnostic family, no tracks or assets changed",
        "source_times": times,
        "observed_pixels_tip_top": observed.tolist(),
        "annotation_uncertainty": "20px train from original fit; 125 top lip partly occluded and tip finite radius, approximate12px holdout annotation",
        "constraints": "Current exact paper and foliage proxies; 0.5mm initial floor and cabinet-side gap. X-axis lean varied independently, baseX and axisY fit only123.4/124/124.6.",
        "rows": rows,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            [
                {
                    "axis_x": x["axis_x_fixed"],
                    "holdout": x["holdout125_errors_px"],
                    "train_max": float(np.max(x["train_errors_px"])),
                }
                for x in rows
            ]
        )
    )


if __name__ == "__main__":
    main()
