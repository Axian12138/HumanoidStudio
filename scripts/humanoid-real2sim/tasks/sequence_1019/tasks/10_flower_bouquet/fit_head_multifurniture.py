"""Fit head optical extrinsics from explicitly labeled, fixed world landmarks.

This produces diagnostic candidates only. Held-out observations are never fitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def pose(x):
    from scipy.spatial.transform import Rotation

    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    result[:3, 3] = x[3:]
    return result


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-transform", type=Path)
    args = parser.parse_args()
    calibration = json.loads((args.run / "inputs/10_flower_bouquet/recording_calibration.json").read_text())
    k = np.array(calibration["records"]["/camera/head/color/camera_info"]["K"]).reshape(3, 3)
    observations = json.loads(args.observations.read_text())["observations"]
    initial = np.array(
        json.loads((args.run / "scene/10_flower_bouquet/head_multiframe_candidates.json").read_text())[0][
            "T_head_rigid_camera_optical"
        ]
    )
    if args.initial_transform:
        initial = np.array(json.loads(args.initial_transform.read_text()))
    x0 = np.r_[Rotation.from_matrix(initial[:3, :3]).as_rotvec(), initial[:3, 3]]

    def projection(x, observation):
        world_camera = np.array(observation["T_world_head_rigid"]) @ pose(x)
        local = (np.array(observation["world_points_m"]) - world_camera[:3, 3]) @ world_camera[:3, :3]
        pixels = local @ k.T
        return pixels[:, :2] / pixels[:, 2:], local[:, 2]

    def residual(x):
        values = []
        for obs in observations:
            if obs["split"] != "fit":
                continue
            pixels, depth = projection(x, obs)
            values.extend(((pixels - obs["image_points_px"]) / obs.get("annotation_sigma_px", 5)).ravel())
            values.extend(np.minimum(depth, 0) * 100)
        return values

    fit = least_squares(residual, x0, bounds=([-np.inf] * 3 + [-0.3] * 3, [np.inf] * 3 + [0.3] * 3), max_nfev=1000)
    rows = []
    for obs in observations:
        pixels, depth = projection(fit.x, obs)
        errors = np.linalg.norm(pixels - obs["image_points_px"], axis=1)
        rows.append(
            {
                "name": obs["name"],
                "split": obs["split"],
                "projected_pixels": pixels.tolist(),
                "errors_px": errors.tolist(),
                "depths_m": depth.tolist(),
                "max_error_px": float(errors.max()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "diagnostic_candidate_pending_independent_review",
                "observations": str(args.observations),
                "K": k.tolist(),
                "D_assumed": [0] * 14,
                "T_head_rigid_camera_optical": pose(fit.x).tolist(),
                "translation_norm_m": float(np.linalg.norm(fit.x[3:])),
                "success": bool(fit.success),
                "fit_message": fit.message,
                "observations_report": rows,
            },
            indent=2,
        )
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
