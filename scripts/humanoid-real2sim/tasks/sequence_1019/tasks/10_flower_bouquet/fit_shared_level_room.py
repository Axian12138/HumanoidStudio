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
    p.add_argument("--fixed-floor", action="store_true")
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
    rows = [o for o in rows if not o["name"].startswith("dining_table")]
    table = json.loads((r / "inputs/08_table_strawberry/table_marker_camera_observations.json").read_text())[
        "observations"
    ]
    for o in table:
        o["camera"] = np.array(o["T_world_head_rigid"]) @ E
        if o.get("type") == "line":
            o["world_points_m"] = o["world_endpoints_m"]
    rows.extend(table)

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
        return o["split"] == "fit"

    def error_components(x, o):
        uv, _ = project(x, o)
        if o.get("type") == "line":
            line = np.cross(np.r_[uv[0], 1], np.r_[uv[1], 1])
            line /= np.linalg.norm(line[:2])
            points = np.c_[o["image_points_px"], np.ones(len(o["image_points_px"]))]
            return points @ line
        return (uv - o["image_points_px"]).ravel()

    def residual(x):
        return np.concatenate([error_components(x, o) / o["annotation_sigma_px"] for o in rows if training(o)])

    fit = least_squares(
        residual,
        np.zeros(4),
        bounds=(
            [-0.1, -0.3, -0.3, -1e-12 if a.fixed_floor else -0.15],
            [0.1, 0.3, 0.3, 1e-12 if a.fixed_floor else 0.15],
        ),
    )
    report = []
    for o in rows:
        uv, z = project(fit.x, o)
        error = (
            np.abs(error_components(fit.x, o))
            if o.get("type") == "line"
            else np.linalg.norm(uv - o["image_points_px"], axis=1)
        )
        report.append(
            {
                "name": o["name"],
                "fit": training(o),
                "errors_px": error.tolist(),
                "pixel_plane_error_m": (error * np.mean(z) / K[0, 0]).tolist(),
                "projected_pixels": uv.tolist(),
            }
        )
    out = (
        r
        / "scene/10_flower_bouquet"
        / ("shared_level_room_floor_v3.json" if a.fixed_floor else "shared_level_room_v2.json")
    )
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
