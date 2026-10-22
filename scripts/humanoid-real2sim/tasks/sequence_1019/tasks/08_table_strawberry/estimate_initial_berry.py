"""Triangulate a pre-contact berry center estimate with held-out image checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    r = args.run
    views = json.loads((r / "inputs/08_table_strawberry/header_views/metadata.json").read_text())
    calibration = json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())
    K = np.array(calibration["K"])
    mount = np.array(calibration["T_head_rigid_camera_optical"])
    basis = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    cameras = {}
    for t, raw in views["poses"].items():
        T = np.eye(4)
        T[:3, :3] = basis @ Rotation.from_quat(raw["quaternion_xyzw_raw"]).as_matrix()
        T[:3, 3] = basis @ raw["position_raw"] + [0, 0, 0.082453832]
        cameras[t] = T @ mount
    observations = {"90.0": [784, 105], "91.0": [578, 136], "91.5": [662, 246], "92.0": [790, 315]}
    training = ["90.0", "91.0"]

    def project(point, time):
        T = cameras[time]
        local = (point - T[:3, 3]) @ T[:3, :3]
        pixel = K @ local
        return pixel[:2] / pixel[2]

    fit = least_squares(
        lambda point: np.concatenate([project(point, t) - observations[t] for t in training]), [-0.88, -2.18, 0.815]
    )
    rows = {
        t: {
            "observed": uv,
            "projected": project(fit.x, t).tolist(),
            "error_px": float(np.linalg.norm(project(fit.x, t) - uv)),
            "split": "fit" if t in training else "holdout",
        }
        for t, uv in observations.items()
    }
    random = np.random.default_rng(1019)
    samples = []
    for _ in range(500):
        jitter = {t: np.array(observations[t]) + random.normal(0, 6, 2) for t in training}
        result = least_squares(
            lambda point, jitter=jitter: np.concatenate([project(point, t) - jitter[t] for t in jitter]), fit.x
        )
        samples.append(result.x)
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    report = {
        "status": "estimated_camera_bound_initial_position_not_scene_acceptance",
        "center_world_m": fit.x.tolist(),
        "annotation_only_95pct_range_m": intervals.tolist(),
        "annotation_sigma_px": 6,
        "uncertainty_excludes": "mount estimation error, marker systematic error, visible silhouette versus 3D mesh-center bias",
        "observations": rows,
        "T_world_camera": {t: T.tolist() for t, T in cameras.items()},
        "excluded": "90.5 berry clipped at image top;92 already touching and used only holdout",
        "conservative_world_range_m": (intervals + np.array([[-0.02, -0.02, -0.02], [0.02, 0.02, 0.02]])).tolist(),
        "support_constraint": "not applied before shared gravity-preserving room registration is validated",
    }
    (r / "review/scene/08_marker_camera/berry_multiview.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k in ["center_world_m", "annotation_only_95pct_range_m", "conservative_world_range_m"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
