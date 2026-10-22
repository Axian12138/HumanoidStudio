"""Compare unchanged original and gravity-settled poses in independent marker cameras."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    review = r / "review/scene"
    old = json.loads((review / "03_04_05_hat_source_pose_comparison.json").read_text())
    hold = json.loads((review / "03_04_05_hat_pose_holdout/report.json").read_text())["records"]
    observations = {48: old["source_observations_px"], **{x["requested_s"]: x["observations_px"] for x in hold}}
    bow = {47.7: [812, 407], 48: [775, 390], 48.3: [705, 346]}
    vertices = np.array(
        [
            list(map(float, line.split()[1:]))
            for line in (r / "scene/03_04_05/black_ribbon_hat.obj").read_text().splitlines()
            if line.startswith("v ")
        ]
    )
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    poses = {key: np.array(rec["pose_xyzw"]) for key, rec in old["variants"].items() if not key.startswith("rss")}
    for name in [
        "03_04_05_hat_multiview_landmarks_fit_local_marker",
        "03_04_05_hat_original_hook_local_marker_contact_fit",
    ]:
        trace = np.load(review / (name + "_native_gravity.npz"))
        q = trace["qpos_wxyz"][-1]
        poses[name + "_settled"] = q[[0, 1, 2, 4, 5, 6, 3]]
    result = {"status": "source_projection_diagnostic_only", "camera_fit_used_hat_or_rack": False, "records": {}}
    for name, pose in poses.items():
        records = {}
        world = Rotation.from_quat(pose[3:]).apply(vertices) + pose[:3]
        bw = Rotation.from_quat(pose[3:]).apply([0, -0.077, 0.040]) + pose[:3]
        for t in [47.7, 48, 48.3]:
            camera = np.array(
                json.loads((review / f"parent_head{t:g}_marker_camera_pose_diagnostic.json").read_text())[
                    "fitted_T_world_camera"
                ]
            )
            c = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = c @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            poly = uv[ConvexHull(uv).vertices]
            obs = np.array(observations[t])
            delta = np.roll(poly, -1, axis=0) - poly
            u = np.clip(((obs[:, None] - poly) * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
            err = np.linalg.norm(obs[:, None] - (poly[None] + u[:, :, None] * delta), axis=2).min(axis=1)
            bc = (bw - camera[:3, 3]) @ camera[:3, :3]
            buv = K @ bc
            records[str(t)] = {
                "median_px": float(np.median(err)),
                "max_px": float(err.max()),
                "bow_error_px": float(np.linalg.norm(buv[:2] / buv[2] - bow[t])),
            }
        result["records"][name] = {"pose_xyzw": pose.tolist(), "projections": records}
    (review / "03_04_05_hat_settled_local_cameras.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
