"""Fit drawer extension to the front edge; hold out two earlier source frames."""

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
    folder = r / "inputs/03_04_05/drawer_extent_holdout"
    cal = json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())
    raw = json.loads((folder / "head_raw_pose_audit_header.json").read_text())["rows"]
    K, E = np.array(cal["K"]), np.array(cal["T_head_rigid_camera_optical"])
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    # Hand labels on the narrow white front top edge, approximately its center.
    observed = {
        50.0: [[450, 445], [620, 412], [790, 379], [960, 346], [1120, 315]],
        50.5: [[434, 442], [610, 414], [790, 385], [970, 356], [1140, 328]],
        51.0: [[466, 431], [660, 412], [850, 391], [1040, 369], [1240, 350]],
    }
    cameras = {}
    for t in observed:
        row = raw[str(t)]
        T = np.eye(4)
        T[:3, :3] = C @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
        T[:3, 3] = C @ row["position_raw"]
        T[2, 3] += 0.08245383203029633
        cameras[t] = T @ E

    def errors(q, t, z=0.882):
        camera = cameras[t]
        world = np.array([[1.45, -1.88 + float(q), z], [0.653, -1.88 + float(q), z]])
        pc = (world - camera[:3, 3]) @ camera[:3, :3]
        uv = pc @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        n = np.array([-(uv[1, 1] - uv[0, 1]), uv[1, 0] - uv[0, 0]])
        n /= np.linalg.norm(n)
        return (np.array(observed[t]) - uv[0]) @ n

    fit = least_squares(lambda x: errors(x[0], 51.0), [0.19], bounds=([0], [0.35]))
    q = float(fit.x[0])
    rows = {
        str(t): {
            "observed_px": observed[t],
            "prior_signed_errors_px": errors(0.19, t).tolist(),
            "fit_signed_errors_px": errors(q, t).tolist(),
        }
        for t in observed
    }
    sensitivity = []
    for height in [0.872, 0.882, 0.892]:
        f = least_squares(lambda x, height=height: errors(x[0], 51.0, height), [q], bounds=([0], [0.35]))
        sensitivity.append({"top_z_m": height, "q_m": float(f.x[0])})
    report = {
        "status": "multi_frame_edge_diagnostic_not_full_scene_acceptance",
        "fit_source_s": 51.0,
        "holdout_source_s": [50.0, 50.5],
        "prior_extent_m": 0.19,
        "fit_extent_m": q,
        "fixed_top_z_m": 0.882,
        "annotation_uncertainty_px": 5,
        "fixed_camera": "original independently marker-fitted mount with Header-synchronized raw head pose",
        "rows": rows,
        "height_sensitivity": sensitivity,
        "limitations": [
            "Front edge alone cannot separate camera/furniture depth and height errors.",
            "50 and50.5 are assumed pre-push constant extension from observed hand approach.",
            "No drawer asset or motion has been changed by this fit.",
        ],
    }
    (folder / "extent_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
