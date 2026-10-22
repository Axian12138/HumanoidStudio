"""Fit basket rim in the estimated level scene; retain independent 112s holdout."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    room = np.array(
        json.loads((r / "scene/10_flower_bouquet/shared_level_room_floor_v3.json").read_text())["T_room_correction"]
    )
    mount = json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v3_header.json").read_text())
    E = np.array(mount["T_head_rigid_camera_optical"])
    K = np.array(mount["K"])
    raw = json.loads((r / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    cameras = []
    pixels = np.array(
        [
            [[608, 40], [676, 26], [688, 62], [610, 75]],
            [[398, 140], [489, 118], [511, 204], [393, 219]],
            [[126, 204], [276, 184], [272, 370], [72, 403]],
        ],
        float,
    )
    for time in [110.0, 111.0, 112.0]:
        h = raw[str(time)]
        T = np.eye(4)
        T[:3, :3] = C @ Rotation.from_quat(h["quaternion_xyzw_raw"]).as_matrix()
        T[:3, 3] = C @ h["position_raw"]
        T[2, 3] += 0.08245383203029633
        cameras.append(T @ E)
    base_z = 0.662 + room[2, 3]

    def corners(x):
        cx, cy, yaw, dx, dy, dz = x
        v = np.array([[-dx / 2, -dy / 2, dz], [-dx / 2, dy / 2, dz], [dx / 2, dy / 2, dz], [dx / 2, -dy / 2, dz]])
        return v @ Rotation.from_euler("z", yaw).as_matrix().T + [cx, cy, base_z]

    def project(x):
        points = corners(x)
        out = []
        for T in cameras:
            v = (points - T[:3, 3]) @ T[:3, :3]
            uv = v @ K.T
            out.append(uv[:, :2] / uv[:, 2:])
        return np.array(out)

    old = json.loads((r / "scene/10_flower_bouquet/basket_multiframe_candidate.json").read_text())
    center = room[:3, :3] @ old["position_m"] + room[:3, 3]
    x0 = np.r_[center[:2], 0.04, 0.36, 0.23, 0.245]
    fit = least_squares(
        lambda x: (project(x)[:2] - pixels[:2]).ravel(),
        x0,
        bounds=([-2.95, -0.65, -0.25, 0.25, 0.16, 0.16], [-2.4, -0.05, 0.25, 0.43, 0.32, 0.32]),
        max_nfev=1000,
    )
    cx, cy, yaw, dx, dy, dz = fit.x
    uv = project(fit.x)
    errors = np.linalg.norm(uv - pixels, axis=2)
    result = {
        "status": "estimated_geometry_candidate_with_camera_uncertainty",
        "position_m": [cx, cy, base_z],
        "quaternion_xyzw": Rotation.from_euler("z", yaw + np.pi / 2).as_quat().tolist(),
        "dimensions_world_aligned_m": [dx, dy, dz],
        "source_time_s": [110.0, 111.0, 112.0],
        "source_corner_pixels": pixels.tolist(),
        "projected_corners_px": uv.tolist(),
        "errors_px": errors.tolist(),
        "training_times": [110, 111],
        "holdout_times": [112],
        "camera_source": "head_marker_diagnostic_v3_header.json",
        "room_source": "shared_level_room_floor_v3.json",
        "geometry_uncertainty_m": 0.055,
        "camera_lateral_uncertainty_m": 0.02,
    }
    out = r / "scene/10_flower_bouquet/basket_registered_candidate.json"
    out.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
