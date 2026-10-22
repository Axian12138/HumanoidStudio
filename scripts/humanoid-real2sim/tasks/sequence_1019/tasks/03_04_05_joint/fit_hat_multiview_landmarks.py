"""Compare fixed shape and bounded scale against source contours with fixed cameras."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--local-marker-cameras", action="store_true")
    p.add_argument("--output-name")
    a = p.parse_args()
    r = a.run
    hold = json.loads((r / "review/scene/03_04_05_hat_pose_holdout/report.json").read_text())["records"]
    old = json.loads((r / "review/scene/03_04_05_hat_source_pose_comparison.json").read_text())
    fit = json.loads((r / "review/scene/03_04_05_hat_silhouette_fit.json").read_text())
    base = np.array(fit["pose_xyzw"])
    camera48 = np.array(
        next(
            x
            for x in json.loads((r / "review/scene/03_04_05_shared_room_validation/projections.json").read_text())[
                "records"
            ]
            if x["source_s"] == 48
        )["T_world_camera"]
    )
    records = [{"requested_s": 48, "T_world_camera": camera48, "observations_px": old["source_observations_px"]}, *hold]
    if a.local_marker_cameras:
        records = [rec for rec in records if rec["requested_s"] in (47.7, 48, 48.3)]
        for rec in records:
            stamp = f"{rec['requested_s']:g}"
            camera_path = r / f"review/scene/parent_head{stamp}_marker_camera_pose_diagnostic.json"
            rec["T_world_camera"] = json.loads(camera_path.read_text())["fitted_T_world_camera"]
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    v = np.array(
        [
            list(map(float, l.split()[1:]))
            for l in (r / "scene/03_04_05/black_ribbon_hat.obj").read_text().splitlines()
            if l.startswith("v ")
        ]
    )
    result = {
        "status": "multiview_bow_and_shape_diagnostic_not_asset_change",
        "fit_times_s": [47.7, 48.0],
        "held_out_s": 48.3,
        "fixed_marker_cameras": True,
        "variants": {},
        "bow_observations_px": {"47.7": [812, 407], "48": [775, 390], "48.3": [705, 346]},
        "bow_annotation_uncertainty_px": 8,
        "local_marker_cameras": a.local_marker_cameras,
        "camera_uncertainty": "Independent marker fit and held-out markers; annotation sensitivity is not a statistical confidence interval.",
    }
    for variant in ["rigid", "uniform", "crown_depth", "axis_scale"]:

        def shape(x, variant=variant):
            scale = np.ones(3)
            if variant == "uniform":
                scale[:] = x[6]
            if variant == "crown_depth":
                scale[2] = x[6]
            if variant == "axis_scale":
                scale[:] = x[6:9]
            return scale

        def errors(x, rec):
            camera = np.array(rec["T_world_camera"])
            obs = np.array(rec["observations_px"])
            world = (Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(base[3:])).apply(v * shape(x)) + base[:3] + x[:3]
            c = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = c @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            poly = uv[ConvexHull(uv).vertices]
            delta = np.roll(poly, -1, axis=0) - poly
            u = np.clip(((obs[:, None] - poly) * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
            return np.linalg.norm(obs[:, None] - (poly[None] + u[:, :, None] * delta), axis=2).min(axis=1)

        def residual(x, variant=variant):
            return np.r_[
                *[errors(x, rec) / 5 for rec in records if rec["requested_s"] != 48.3],
                *[bow_error(x, rec) / 8 for rec in records if rec["requested_s"] != 48.3],
                x[:3] / 0.1,
                x[3:6] / 0.7,
                ([] if variant == "rigid" else (x[6:] - 1) / 0.15),
            ]

        def bow_error(x, rec):
            camera = np.array(rec["T_world_camera"])
            local = np.array([0, -0.077, 0.040]) * shape(x)
            world = (Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(base[3:])).apply(local) + base[:3] + x[:3]
            pc = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = K @ pc
            return uv[:2] / uv[2] - np.array(result["bow_observations_px"][str(rec["requested_s"])])

        count = 0 if variant == "rigid" else (3 if variant == "axis_scale" else 1)
        x0 = np.r_[np.zeros(6), np.ones(count)]
        lo = np.r_[[-0.1] * 3, [-0.7] * 3, [0.7] * count]
        hi = np.r_[[0.1] * 3, [0.7] * 3, [1.3] * count]
        best = None
        for angle in [0, -0.15, 0.15]:
            initial = x0.copy()
            initial[4] = angle
            f = least_squares(residual, initial, bounds=(lo, hi), max_nfev=100)
            if best is None or f.cost < best.cost:
                best = f
        pose = np.r_[
            base[:3] + best.x[:3], (Rotation.from_rotvec(best.x[3:6]) * Rotation.from_quat(base[3:])).as_quat()
        ]
        result["variants"][variant] = {
            "pose_xyzw": pose.tolist(),
            "scale_xyz": shape(best.x).tolist(),
            "cost": best.cost,
            "solver_success": bool(best.success),
            "errors": {
                str(rec["requested_s"]): {
                    "median_px": float(np.median(errors(best.x, rec))),
                    "max_px": float(errors(best.x, rec).max()),
                    "bow_error_px": float(np.linalg.norm(bow_error(best.x, rec))),
                }
                for rec in records
            },
        }
    suffix = "_local_marker" if a.local_marker_cameras else ""
    (r / "review/scene" / (a.output_name or f"03_04_05_hat_multiview_landmarks_fit{suffix}.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
