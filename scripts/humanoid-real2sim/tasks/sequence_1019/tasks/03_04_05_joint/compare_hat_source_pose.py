"""Compare hat silhouettes with a fixed independently estimated source camera."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    folder = r / "review/scene/03_04_05_shared_room_validation"
    T = np.array(
        next(x for x in json.loads((folder / "projections.json").read_text())["records"] if x["source_s"] == 48)[
            "T_world_camera"
        ]
    )
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    v = np.array(
        [
            list(map(float, l.split()[1:]))
            for l in (r / "scene/03_04_05/black_ribbon_hat.obj").read_text().splitlines()
            if l.startswith("v ")
        ]
    )
    obs = np.array(
        [
            [824, 174],
            [859, 184],
            [881, 211],
            [893, 250],
            [910, 281],
            [879, 385],
            [832, 400],
            [782, 398],
            [737, 380],
            [709, 348],
            [698, 311],
            [702, 274],
            [718, 235],
            [748, 201],
            [788, 180],
        ]
    )
    timing = json.loads((r / "inputs/03_04_05/image_header_timestamp_audit.json").read_text())
    t = (timing["samples"]["48.0"]["header_time_ns"] - timing["origin_log_time_ns"]) / 1e9
    mj = np.load(r / "retarget/candidates/03_04_05_joint_v26/black_ribbon_hat.npz")
    rss = np.load(r / "review/scene/03_04_05_rss_hat_v26_fresh_30s.npz")
    ki = round((t - 27) * 50)
    kr = round((t - 47.64) * 50)
    variants = {
        "release_initial": np.r_[mj["position_w"][1032], mj["quaternion_xyzw"][1032]],
        "mujoco_at_image_header": np.r_[mj["position_w"][ki], mj["quaternion_xyzw"][ki]],
        "rss_at_image_header_old_reset": np.r_[rss["position_w"][kr], rss["quaternion_xyzw"][kr]],
        "rss_after30s_old_reset": np.r_[rss["position_w"][-1], rss["quaternion_xyzw"][-1]],
    }
    report = {
        "status": "source_silhouette_diagnostic_fixed_camera",
        "source_header_s": t,
        "source_observations_px": obs.tolist(),
        "annotation_uncertainty_px": 5,
        "camera_boundary": "early independent marker max21.32px; source silhouette alone is not a6D pose ground truth",
        "variants": {},
    }
    for name, pose in variants.items():
        w = Rotation.from_quat(pose[3:]).apply(v) + pose[:3]
        c = (w - T[:3, 3]) @ T[:3, :3]
        uv = c @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        poly = uv[ConvexHull(uv).vertices]
        delta = np.roll(poly, -1, axis=0) - poly
        u = np.clip(((obs[:, None] - poly) * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
        error = np.linalg.norm(obs[:, None] - (poly[None] + u[:, :, None] * delta), axis=2).min(axis=1)
        report["variants"][name] = {
            "pose_xyzw": pose.tolist(),
            "median_error_px": float(np.median(error)),
            "max_error_px": float(error.max()),
            "errors_px": error.tolist(),
            "projected_bbox": [uv.min(axis=0).tolist(), uv.max(axis=0).tolist()],
        }
        im = Image.open(r / "inputs/03_04_05/head_48_header.jpg")
        d = ImageDraw.Draw(im)
        d.line([tuple(x) for x in np.vstack([poly, poly[0]])], fill="red", width=2)
        for x, y in obs:
            d.ellipse((x - 3, y - 3, x + 3, y + 3), outline="lime", width=2)
        im.save(folder / f"head_48_hat_{name}.png")
    (r / "review/scene/03_04_05_hat_source_pose_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
