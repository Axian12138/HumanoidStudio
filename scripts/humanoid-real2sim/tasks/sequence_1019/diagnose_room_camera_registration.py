"""Diagnose one shared room SE3 and head mount without changing scene assets."""

import argparse
import json
from pathlib import Path

import numpy as np


def transform(x):
    from scipy.spatial.transform import Rotation

    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    t[:3, 3] = x[3:]
    return t


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fixed-marker-mount", action="store_true")
    a = p.parse_args()
    folder = a.run / "inputs/10_flower_bouquet"
    observations = json.loads((folder / "multifurniture_observations_v6.json").read_text())["observations"]
    initial = json.loads((a.run / "scene/10_flower_bouquet/head_multifurniture_v6.json").read_text())
    if a.fixed_marker_mount:
        initial = json.loads((a.run / "scene/10_flower_bouquet/head_marker_diagnostic_v3_header.json").read_text())
        raw = json.loads((folder / "head_raw_pose_audit_header.json").read_text())["rows"]
        basis = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        for obs in observations:
            time_key = str(float(obs["name"].split("_")[-1]))
            row = raw[time_key]
            head = np.eye(4)
            head[:3, :3] = basis @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
            head[:3, 3] = basis @ row["position_raw"] + np.array([0, 0, 0.08245383203029633])
            obs["T_world_head_rigid"] = head.tolist()
    mount = np.array(initial["T_head_rigid_camera_optical"])
    k = np.array(initial["K"])
    x0 = np.r_[Rotation.from_matrix(mount[:3, :3]).as_rotvec(), mount[:3, 3], np.zeros(6)]
    fixed_parameters = x0[:6].copy()
    if a.fixed_marker_mount:
        x0 = np.zeros(6)

    def expand(x):
        return np.r_[fixed_parameters, x] if a.fixed_marker_mount else x

    def project(x, obs):
        x = expand(x)
        world_camera = np.array(obs["T_world_head_rigid"]) @ transform(x[:6])
        room = transform(x[6:])
        points = np.array(obs["world_points_m"]) @ room[:3, :3].T + room[:3, 3]
        local = (points - world_camera[:3, 3]) @ world_camera[:3, :3]
        pixels = local @ k.T
        return pixels[:, :2] / pixels[:, 2:], local[:, 2]

    results = []
    # Bounds are exploratory hypotheses, not acceptance tolerances.
    lower = np.r_[[-np.inf] * 3, [-0.3] * 3, [-0.35] * 3, [-0.5] * 3]
    if a.fixed_marker_mount:
        lower = lower[6:]
    upper = -lower
    for mode in ("original_training_only", "include_table_90_92_keep_96_holdout"):

        def training(obs, mode=mode):
            return obs["split"] == "fit" or (
                mode != "original_training_only"
                and obs["name"] in ("dining_table_near_corner_90.0", "dining_table_near_corner_92.0")
            )

        def residual(x):
            pieces = []
            for obs in observations:
                if training(obs):
                    pixels, depth = project(x, obs)
                    pieces.extend(((pixels - obs["image_points_px"]) / obs.get("annotation_sigma_px", 5)).ravel())
                    pieces.extend(np.minimum(depth, 0) * 100)
            return pieces

        fit = least_squares(residual, x0, bounds=(lower, upper), max_nfev=2500)
        rows = []
        for obs in observations:
            pixels, depth = project(fit.x, obs)
            error = np.linalg.norm(pixels - obs["image_points_px"], axis=1)
            rows.append(
                {
                    "name": obs["name"],
                    "used_for_fit": training(obs),
                    "errors_px": error.tolist(),
                    "max_error_px": float(error.max()),
                    "depths_m": depth.tolist(),
                }
            )
        results.append(
            {
                "mode": mode,
                "optimizer_success": bool(fit.success),
                "jacobian_singular_values": np.linalg.svd(fit.jac, compute_uv=False).tolist(),
                "T_head_camera": transform(expand(fit.x)[:6]).tolist(),
                "T_room_correction": transform(expand(fit.x)[6:]).tolist(),
                "parameters": fit.x.tolist(),
                "active_bounds": fit.active_mask.tolist(),
                "observations": rows,
            }
        )
    out = a.run / "review/scene/parent_room_camera_registration_diagnostic.json"
    if a.fixed_marker_mount:
        out = out.with_name("parent_room_fixed_marker_mount_diagnostic.json")
    out.write_text(
        json.dumps(
            {
                "status": "diagnostic_only_not_accepted",
                "scope": "one shared rigid room correction; no per-object movement or asset mutation",
                "exploratory_room_bounds": "rotation-vector components ±0.35 rad, translation ±0.5 m",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    for result in results:
        print(result["mode"], [(o["name"], round(o["max_error_px"], 2)) for o in result["observations"]])


if __name__ == "__main__":
    main()
