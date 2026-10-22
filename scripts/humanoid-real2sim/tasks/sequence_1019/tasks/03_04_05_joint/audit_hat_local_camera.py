"""Reproject unchanged hat hypotheses using an independently fitted marker camera."""

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
    camera = np.array(
        json.loads((review / "parent_head48_marker_camera_pose_diagnostic.json").read_text())["fitted_T_world_camera"]
    )
    old = json.loads((review / "03_04_05_hat_source_pose_comparison.json").read_text())
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    v = np.array(
        [
            list(map(float, l.split()[1:]))
            for l in (r / "scene/03_04_05/black_ribbon_hat.obj").read_text().splitlines()
            if l.startswith("v ")
        ]
    )
    obs = np.array(old["source_observations_px"])
    variants = {name: {"pose_xyzw": rec["pose_xyzw"], "scale_xyz": [1, 1, 1]} for name, rec in old["variants"].items()}
    variants["shallow_cavity"] = json.loads((review / "03_04_05_hat_shallow_cavity_contact_fit.json").read_text())[
        "variants"
    ]["rigid"]
    result = {
        "status": "unchanged_hat_pose_projection_diagnostic_not_acceptance",
        "camera_source": "parent_head48_marker_camera_pose_diagnostic.json",
        "camera_fit_used_hat_or_rack": False,
        "source_observations_px": obs.tolist(),
        "records": {},
    }
    for name, rec in variants.items():
        pose = np.array(rec["pose_xyzw"])
        scale = np.array(rec.get("scale_xyz", [1, 1, 1]))
        world = Rotation.from_quat(pose[3:]).apply(v * scale) + pose[:3]
        c = (world - camera[:3, 3]) @ camera[:3, :3]
        uv = c @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        poly = uv[ConvexHull(uv).vertices]
        delta = np.roll(poly, -1, axis=0) - poly
        u = np.clip(((obs[:, None] - poly) * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
        err = np.linalg.norm(obs[:, None] - (poly[None] + u[:, :, None] * delta), axis=2).min(axis=1)
        result["records"][name] = {
            "median_px": float(np.median(err)),
            "max_px": float(err.max()),
            "errors_px": err.tolist(),
            "projected_polygon_px": poly.tolist(),
            "unchanged_pose_xyzw": pose.tolist(),
            "scale_xyz": scale.tolist(),
        }
    path = review / "03_04_05_hat_local_camera_reprojection.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print({k: {s: v[s] for s in ["median_px", "max_px"]} for k, v in result["records"].items()})


if __name__ == "__main__":
    main()
