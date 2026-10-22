"""Check the placed basket pose against visible rim corners with a fixed camera."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run / "scene/10_flower_bouquet"
    cal = json.loads((folder / "head_marker_diagnostic_v3_header.json").read_text())
    raw = json.loads((args.run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    geometry = json.loads((folder / "basket_registered_candidate.json").read_text())
    dx, dy, dz = geometry["dimensions_world_aligned_m"]
    corners = np.array([[dx / 2, -dy / 2, dz], [dx / 2, dy / 2, dz], [-dx / 2, dy / 2, dz], [-dx / 2, -dy / 2, dz]])
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    observations = [
        {
            "time": 124.6,
            "pixels": [[571, 415], [766, 442], [805, 179], [626, 157]],
            "indices": [0, 1, 2, 3],
            "role": "fit",
        },
        {
            "time": 125.0,
            "pixels": [[525, 405], [724, 429], [745, 165], [574, 173]],
            "indices": [0, 1, 2, 3],
            "role": "fit",
        },
        {"time": 123.4, "pixels": [[631, 429], [814, 471]], "indices": [0, 1], "role": "holdout"},
    ]
    cameras = {}
    for obs in observations:
        row = raw[str(obs["time"])]
        H = np.eye(4)
        H[:3, :3] = C @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
        H[:3, 3] = C @ row["position_raw"]
        H[2, 3] += 0.08245383203029633
        cameras[obs["time"]] = H @ cal["T_head_rigid_camera_optical"]

    def project(x, obs):
        T = cameras[obs["time"]]
        world = Rotation.from_euler("z", x[2]).apply(corners[obs["indices"]]) + np.r_[x[:2], 0.661]
        camera = (world - T[:3, 3]) @ T[:3, :3]
        uvw = camera @ np.array(cal["K"]).T
        return uvw[:, :2] / uvw[:, 2:]

    def residual(x):
        return np.concatenate([(project(x, o) - o["pixels"]).ravel() / 12 for o in observations if o["role"] == "fit"])

    fit = least_squares(residual, [-2.62, -0.336, 0], bounds=([-2.85, -0.5, -0.4], [-2.4, -0.1, 0.4]))
    for obs in observations:
        obs["projected"] = project(fit.x, obs).tolist()
        obs["errors_px"] = np.linalg.norm(project(fit.x, obs) - obs["pixels"], axis=1).tolist()
    result = {
        "status": "manual rim-corner diagnostic; rounded corners and partial occlusion limit precision",
        "position": np.r_[fit.x[:2], 0.661].tolist(),
        "world_rectangle_yaw": float(fit.x[2]),
        "fixed_dimensions": [dx, dy, dz],
        "sigma_annotation_px": 12,
        "observations": observations,
    }
    (folder / "placed_basket_multiview.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
