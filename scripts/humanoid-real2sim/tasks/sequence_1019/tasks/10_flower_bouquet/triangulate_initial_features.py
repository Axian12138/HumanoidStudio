"""Independent static flower landmarks; annotation-only uncertainty, no support fit."""

import argparse
import hashlib
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
    paths = [
        args.run / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json",
        args.run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json",
        args.run / "review/scene/10_initial_feature_triangulation_v1/report.json",
    ]
    cal, heads, old = [json.loads(path.read_text()) for path in paths]
    times = [111.0, 112.0, 113.0]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    cameras = []
    for time in times:
        head = heads["rows"][str(time)]
        H = np.eye(4)
        H[:3, :3] = C @ Rotation.from_quat(head["quaternion_xyzw_raw"]).as_matrix()
        H[:3, 3] = C @ head["position_raw"]
        H[2, 3] += 0.08245383203029633
        cameras.append(H @ cal["T_head_rigid_camera_optical"])
    K = np.array(cal["K"])

    def project(point):
        output = []
        for camera in cameras:
            local = (point - camera[:3, 3]) @ camera[:3, :3]
            uvw = K @ local
            output.append(uvw[:2] / uvw[2])
        return np.array(output)

    rng = np.random.default_rng(1019112)
    rows, samples = [], {}
    for old_row in old["rows"]:
        name = old_row["name"]
        observed = old_row["observed_pixels"]
        if name == "paper_lip":
            observed = [[798, 158], [752, 293], None]
        obs = np.array(observed[:2], dtype=float)
        fit = least_squares(lambda point, obs=obs: (project(point)[:2] - obs).ravel(), old_row["world_m"])
        uv = project(fit.x)
        # Independent image labeling noise only; fixed camera uncertainty is excluded.
        draws = []
        for _ in range(500):
            noisy = obs + rng.normal(0, 8.0, obs.shape)
            draw = least_squares(lambda point, noisy=noisy: (project(point)[:2] - noisy).ravel(), fit.x)
            draws.append(draw.x)
        samples[name] = np.array(draws)
        rows.append(
            {
                "name": name,
                "observed_pixels": observed,
                "world_m": fit.x.tolist(),
                "projected_pixels": uv.tolist(),
                "fit_errors_px": np.linalg.norm(uv[:2] - obs, axis=1).tolist(),
                "holdout113_error_px": None if observed[2] is None else float(np.linalg.norm(uv[2] - observed[2])),
                "annotation_only_world_95pct_m": np.percentile(draws, [2.5, 97.5], axis=0).tolist(),
                "use": "rejected correspondence" if name == "low_green_tip" else "diagnostic landmark",
            }
        )
    points = {row["name"]: np.array(row["world_m"]) for row in rows}
    lengths = {}
    for name in ["paper_lip", "left_white", "upper_white"]:
        draws = np.linalg.norm(samples[name] - samples["tip"], axis=1)
        lengths[name] = {
            "tip_distance_m": float(np.linalg.norm(points[name] - points["tip"])),
            "annotation_only_95pct_m": np.percentile(draws, [2.5, 97.5]).tolist(),
        }
    report = {
        "scope": "111/112 static triangulation, fixed Header camera mount; no floor, shape, support, length or dynamics constraints.",
        "correction": "Paper lip center re-read on 3x crops, replacing lower visible rim points. Partial rim occlusion prevents exact circle-center identification; 113 lip excluded.",
        "uncertainty": "500 seeded draws with independent 8px Gaussian annotation noise. Does not include camera calibration, common timing bias or feature-identity uncertainty; not a total confidence interval.",
        "holdout": "113 after first touch is only a consistency diagnostic, not guaranteed static ground truth.",
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "rows": rows,
        "lengths": lengths,
        "status": "diagnostic only; no asset or trajectory changed",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(lengths, indent=2))


if __name__ == "__main__":
    main()
