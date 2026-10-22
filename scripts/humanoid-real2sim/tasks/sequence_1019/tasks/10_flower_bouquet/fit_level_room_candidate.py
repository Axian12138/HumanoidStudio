"""Diagnose shared room yaw/translation while preserving level support planes."""

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
    rows = json.loads((r / "inputs/10_flower_bouquet/multifurniture_observations_v6.json").read_text())["observations"]
    raw = json.loads((r / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    mount = json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v3_header.json").read_text())
    E = np.array(mount["T_head_rigid_camera_optical"])
    K = np.array(mount["K"])
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for o in rows:
        h = raw[str(float(o["source_time_s"]))]
        T = np.eye(4)
        T[:3, :3] = C @ Rotation.from_quat(h["quaternion_xyzw_raw"]).as_matrix()
        T[:3, 3] = C @ h["position_raw"]
        T[2, 3] += 0.08245383203029633
        o["camera"] = T @ E

    def pose(x):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", x[0]).as_matrix()
        T[:3, 3] = x[1:]
        return T

    def project(x, o):
        T = pose(x)
        p = np.array(o["world_points_m"]) @ T[:3, :3].T + T[:3, 3]
        c = o["camera"]
        v = (p - c[:3, 3]) @ c[:3, :3]
        q = v @ K.T
        return q[:, :2] / q[:, 2:], v[:, 2]

    def training(o):
        return o["split"] == "fit" or o["name"] in ["dining_table_near_corner_90.0", "dining_table_near_corner_92.0"]

    def residual(x):
        return np.concatenate(
            [
                ((project(x, o)[0] - o["image_points_px"]) / o["annotation_sigma_px"]).ravel()
                for o in rows
                if training(o)
            ]
        )

    fit = least_squares(residual, np.zeros(4), bounds=([-0.1, -0.3, -0.3, -0.15], [0.1, 0.3, 0.3, 0.15]))
    report = []
    for o in rows:
        uv, z = project(fit.x, o)
        error = np.linalg.norm(uv - o["image_points_px"], axis=1)
        report.append(
            {
                "name": o["name"],
                "fit": training(o),
                "errors_px": error.tolist(),
                "pixel_plane_error_m": (error * z / K[0, 0]).tolist(),
                "projected_pixels": uv.tolist(),
            }
        )
    out = r / "scene/10_flower_bouquet/level_room_candidate.json"
    out.write_text(
        json.dumps(
            {
                "status": "estimated_private_scene_candidate_not_global_registration_acceptance",
                "T_room_correction": pose(fit.x).tolist(),
                "head_mount_source": "head_marker_diagnostic_v3_header.json",
                "parameters": fit.x.tolist(),
                "observations": report,
            },
            indent=2,
        )
    )
    print([(o["name"], max(o["errors_px"])) for o in report])


if __name__ == "__main__":
    main()
