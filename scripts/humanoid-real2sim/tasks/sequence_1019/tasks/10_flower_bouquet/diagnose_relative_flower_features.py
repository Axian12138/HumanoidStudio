"""Diagnose release geometry with observed relative marker-to-flower image vectors."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    scene = args.run / "scene/10_flower_bouquet"
    cal = json.loads((scene / "head_marker_diagnostic_v4_header.json").read_text())
    head = json.loads((args.run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    marker_projection = json.loads((args.run / "review/scene/10_release_marker_holdout_v1/projection.json").read_text())
    marker_observed = json.loads((args.run / "review/scene/10_release_image_delay_v2/report.json").read_text())["rows"]
    family = json.loads((args.run / "review/scene/10_release_pose_family_v1/report.json").read_text())
    times = family["source_times"]
    observed = np.array(family["observed_pixels_tip_top"])
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    K = np.array(cal["K"])
    cameras, common_residual = [], []
    for time in times:
        h = head[str(time)]
        H = np.eye(4)
        H[:3, :3] = C @ Rotation.from_quat(h["quaternion_xyzw_raw"]).as_matrix()
        H[:3, 3] = C @ h["position_raw"]
        H[2, 3] += 0.08245383203029633
        cameras.append(H @ cal["T_head_rigid_camera_optical"])
        measured = next(row for row in marker_observed if row["source_s"] == time)
        predicted = next(row for row in marker_projection if row["source_s"] == time)
        common_residual.append(np.mean(np.array(predicted["uv"])[:3] - np.array(measured["observed_uv"]), axis=0))
    common_residual = np.array(common_residual)

    def project(points):
        rows = []
        for camera in cameras:
            local = (points - camera[:3, 3]) @ camera[:3, :3]
            uvw = local @ K.T
            rows.append(uvw[:, :2] / uvw[:, 2:])
        return np.array(rows)

    def unpack(x):
        return np.array([[x[0], x[1], 0.68], [x[2], x[3], x[4]]])

    rows = []
    for relative in [False, True]:
        for length_sigma in [0.02, 0.055, 0.10]:

            def residual(x, relative=relative, length_sigma=length_sigma):
                points = unpack(x)
                errors = project(points) - observed
                if relative:
                    errors -= common_residual[:, None, :]
                return np.r_[
                    errors[:3].ravel() / 12,
                    (np.linalg.norm(points[1] - points[0]) - 0.360078) / length_sigma,
                ]

            fit = least_squares(
                residual,
                [-2.612, -0.291, -2.71, -0.343, 1.022],
                bounds=([-2.9, -0.5, -3.1, -0.7, 0.8], [-2.4, -0.1, -2.4, 0.0, 1.2]),
            )
            points = unpack(fit.x)
            uv = project(points)
            delta = points[1] - points[0]
            relative_error = uv - observed - common_residual[:, None, :]
            rows.append(
                {
                    "fit_relative_marker_vectors": relative,
                    "length_prior_sigma_m": length_sigma,
                    "points_world": points.tolist(),
                    "length_m": float(np.linalg.norm(delta)),
                    "axis": (delta / np.linalg.norm(delta)).tolist(),
                    "absolute_errors_px": np.linalg.norm(uv - observed, axis=2).tolist(),
                    "relative_marker_errors_px": np.linalg.norm(relative_error, axis=2).tolist(),
                    "projected_pixels": uv.tolist(),
                }
            )
    report = {
        "scope": "Diagnostic only. Fixed mount/K/head poses. Relative flower-to-marker image vectors cancel measured common image residual approximately, not a new per-frame camera.",
        "train_times": times[:3],
        "holdout_time": times[3],
        "floor_tip_z_prior_m": 0.68,
        "marker_prediction_minus_observation_px": common_residual.tolist(),
        "rows": rows,
        "limitations": "Tip floor height and native paper length remain explicit priors. Relative residual cancellation assumes locally common image displacement; different depth and rotation can violate it. No contact/physics pass.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
