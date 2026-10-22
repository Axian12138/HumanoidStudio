"""Fit visible rack geometry only, with independently established fixed cameras."""

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
    review = r / "review/scene"
    source = json.loads((review / "03_04_05_rack_support_hypotheses/report.json").read_text())["records"]
    rec45 = next(x for x in source if x["source_s"] == 45 and x["variant"] == "early_registered")
    cams = {45: np.array(rec45["T_world_camera"])}
    for t in [48, 48.3]:
        cams[t] = np.array(
            json.loads((review / f"parent_head{t:g}_marker_camera_pose_diagnostic.json").read_text())[
                "fitted_T_world_camera"
            ]
        )
    points = {p["geom"]: np.array(p["world_m"]) for p in rec45["points"]}
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    axis = np.array([2.409627927, -1.469792849, 0.0])
    observations = {
        45: {"rack_30": [358, 210], "rack_55": [553, 178], "rack_80": [465, 94]},
        48: {"rack_30": [636, 183], "rack_80": [819, 106]},
        48.3: {"rack_30": [550, 170], "rack_80": [718, 84]},
    }
    base_obs = {
        x["source_s"]: np.array(x["observed_boundary_pixels"])
        for x in json.loads(
            (review / "03_04_05_rack_support_hypotheses/parent_base_45_48_consistency.json").read_text()
        )["frames"]
    }
    q = r / "inputs/03_04_05/hat_carry_side_source_headers"
    annotations = json.loads((q / "parent_rack_annotations.json").read_text())
    seed = json.loads((q / "parent_seed.json").read_text())
    mount = np.array(json.loads((q / "parent_bilateral_marker_fit.json").read_text())["T_head_camera_candidate"])
    for row in annotations["rows"]:
        t = row["encoded_pts_s"]
        cams[t] = np.array(seed["rows"][str(t)]["T_world_head"]) @ mount
        observations[t] = {
            "rack_30": row["ball_centers"]["left_visible_hook"],
            "rack_80": row["ball_centers"]["upper_right_visible_hook"],
        }
        base_obs[t] = np.array(row["visible_base_outer_arc_px"])
    train = [45.7, 46.0]
    theta = np.linspace(0, 2 * np.pi, 361)
    ring = np.column_stack([0.17 * np.cos(theta), 0.17 * np.sin(theta), np.full(len(theta), 0.026)])

    def project(world, t):
        cam = (world - cams[t][:3, 3]) @ cams[t][:3, :3]
        uv = cam @ K.T
        return uv[..., :2] / uv[..., 2:]

    def hook(x, t, n):
        point = Rotation.from_euler("z", x[2]).apply((points[n] - axis) * [x[3], x[3], 1]) + axis + [x[0], x[1], x[4]]
        return project(point, t)

    def base_errors(x, t):
        uv = project(ring + axis + [x[0], x[1], 0], t)
        return np.linalg.norm(base_obs[t][:, None] - uv[None], axis=2).min(axis=1)

    def residual(x):
        return np.r_[
            *[((hook(x, t, n) - uv) / 10) for t in train for n, uv in observations[t].items()],
            *[base_errors(x, t) / 10 for t in train],
            x[:2] / 0.05,
            x[2] / 0.6,
            (x[3] - 1) / 0.2,
            x[4] / 0.05,
        ]

    fit = least_squares(
        residual, [0, 0, 0, 1, 0], bounds=([-0.08, -0.08, -0.8, 0.7, -0.08], [0.08, 0.08, 0.8, 1.3, 0.08])
    )
    x = fit.x
    from PIL import Image, ImageDraw

    output_dir = review / "03_04_05_rack_carry_camera"
    output_dir.mkdir(exist_ok=True)
    for row in annotations["rows"]:
        t = row["encoded_pts_s"]
        im = Image.open(row["image"]).convert("RGB")
        draw = ImageDraw.Draw(im)
        uv = project(ring + axis + [x[0], x[1], 0], t)
        draw.line([tuple(v) for v in uv], fill="cyan", width=2)
        for point in base_obs[t]:
            u, v = point
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline="yellow", width=2)
        for name, observation in observations[t].items():
            u, v = hook(x, t, name)
            draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline="cyan", width=2)
            u, v = observation
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline="yellow", width=2)
        draw.text((15, 15), "cyan=prediction; yellow=source annotation; fixed marker camera", fill="white")
        im.save(output_dir / f"frame_{row['frame_index']}.png")
    report = {
        "status": "rack_source_only_joint_geometry_diagnostic_not_adopted",
        "trained_encoded_pts_s": train,
        "holdout_encoded_pts_s": [45.833333333333336],
        "external_source_s_checks": [45, 48, 48.3],
        "hat_used_in_fit": False,
        "annotations": str(q / "parent_rack_annotations.json"),
        "camera": str(q / "parent_bilateral_marker_fit.json"),
        "third_hidden_hook_measured": False,
        "camera_fit_changed": False,
        "xy_delta_m": x[:2].tolist(),
        "fit": {"yaw_rad": float(x[2]), "radial_scale": float(x[3]), "height_offset_m": float(x[4])},
        "records": {},
        "limitations": [
            "Camera annotation sensitivity is substantial at rack depth.",
            "48.3 tip observations manually approximate to about3px.",
            "Base height and radius remain assumed from authored geometry; no millimetric ground truth claim.",
        ],
    }
    for t, obs in observations.items():
        report["records"][str(t)] = {
            "tips": {
                n: {
                    "predicted_px": hook(x, t, n).tolist(),
                    "observed_px": uv,
                    "error_px": float(np.linalg.norm(hook(x, t, n) - uv)),
                }
                for n, uv in obs.items()
            }
        }
        if t in base_obs:
            report["records"][str(t)]["base_mean_px"] = float(base_errors(x, t).mean())
            report["records"][str(t)]["base_max_px"] = float(base_errors(x, t).max())
    rng = np.random.default_rng(4105)
    original_obs = {t: {n: np.array(uv) for n, uv in observations[t].items()} for t in train}
    original_base = {t: base_obs[t].copy() for t in train}
    fits = []
    for _ in range(32):
        for t in train:
            observations[t] = {n: uv + rng.uniform(-3, 3, 2) for n, uv in original_obs[t].items()}
            base_obs[t] = original_base[t] + rng.uniform(-4, 4, original_base[t].shape)
        perturbed = least_squares(residual, x, bounds=([-0.08, -0.08, -0.8, 0.7, -0.08], [0.08, 0.08, 0.8, 1.3, 0.08]))
        fits.append(perturbed.x.tolist())
    report["annotation_sensitivity"] = {
        "scope": "32 deterministic independently perturbed training annotations; fixed camera; not statistical CI and excludes camera uncertainty",
        "parameter_order": ["dx", "dy", "yaw", "radial", "height"],
        "min": np.min(fits, axis=0).tolist(),
        "max": np.max(fits, axis=0).tolist(),
        "samples": fits,
    }
    (review / "03_04_05_rack_carry_camera_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(json.dumps(report["records"], indent=2))


if __name__ == "__main__":
    main()
