"""Fit visible ribbon midline and silhouette with independent fixed cameras."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
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
    band_report = json.loads(
        (r / "scene/03_04_05/hat_visual_shell_collision_v1/source_band_observations.json").read_text()
    )
    band_observations = band_report["observations_px"]
    theta = np.linspace(0, 2 * np.pi, 721)[:-1]
    band_ring = np.column_stack(
        [
            0.095 * np.cos(theta),
            0.095 * (0.235 / 0.28) * np.sin(theta),
            np.full(len(theta), (0.031071428 + 0.047642857) / 2),
        ]
    )
    result = {
        "status": "crown_roof_height_family_fixed_band_brim_bow_not_adopted",
        "fit_times_s": [47.7, 48.0],
        "held_out_s": 48.3,
        "fixed_marker_cameras": True,
        "band_observations": band_report,
        "band_model": "Authored part5 outer radius linear midheight ring; nearest projected ring centerline, visibility must be checked independently",
        "variants": {},
        "bow_observations_px": {"47.7": [812, 407], "48": [775, 390], "48.3": [705, 346]},
        "bow_annotation_uncertainty_px": 8,
        "local_marker_cameras": a.local_marker_cameras,
        "camera_uncertainty": "Independent marker fit and held-out markers; annotation sensitivity is not a statistical confidence interval.",
    }
    original_parts = {
        i: trimesh.load(r / f"scene/03_04_05/black_ribbon_hat_visual_part_{i}.obj", process=False).vertices
        for i in range(3, 8)
    }
    phase = -1.4839047379963846
    bow_A = np.diag([1, 0.235 / 0.28, 1]) @ Rotation.from_euler("z", phase).as_matrix() @ np.diag([1, 0.28 / 0.235, 1])
    result["limitations"] = [
        "Crown-only roof height family; brim/band fixed. Convex silhouette cannot identify a concave crease.",
        "No collision geometry or dynamics changed; no candidate adopted.",
    ]
    for roof_height in [0.075, 0.09, 0.105, 0.12, 0.145]:
        variant = "rigid"
        points = []
        for part, original_vertices in original_parts.items():
            vv = original_vertices.copy()
            if part == 3:
                above = vv[:, 2] > 0.047642857
                vv[above, 2] = 0.047642857 + (vv[above, 2] - 0.047642857) * (
                    (roof_height - 0.047642857) / (0.145 - 0.047642857)
                )
            if part >= 6:
                vv = vv @ bow_A.T
            points.append(vv)
        v = np.vstack(points)

        def shape(x, variant=variant):
            scale = np.ones(3)
            if variant == "uniform":
                scale[:] = x[6]
            if variant == "crown_depth":
                scale[2] = x[6]
            if variant == "axis_scale":
                scale[:] = x[6:9]
            return scale

        def errors(x, rec, v=v):
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

        def band_error(x, rec):
            camera = np.array(rec["T_world_camera"])
            world = (
                (Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(base[3:])).apply(band_ring * shape(x))
                + base[:3]
                + x[:3]
            )
            pc = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = pc @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            points = np.array(band_observations[f"{rec['requested_s']:g}"])
            return np.linalg.norm(points[:, None] - uv[None], axis=2).min(axis=1)

        def residual(x, variant=variant):
            return np.r_[
                *[errors(x, rec) / 5 for rec in records if rec["requested_s"] != 48.3],
                *[bow_error(x, rec) / 8 for rec in records if rec["requested_s"] != 48.3],
                *[band_error(x, rec) / 3 for rec in records if rec["requested_s"] != 48.3],
                x[:3] / 0.1,
                x[3:6] / 0.7,
                ([] if variant == "rigid" else (x[6:] / np.pi if variant == "bow_phase" else (x[6:] - 1) / 0.15)),
            ]

        def bow_error(x, rec, variant=variant):
            camera = np.array(rec["T_world_camera"])
            local = bow_A @ np.array([0, -0.077, 0.040]) * shape(x)
            if variant == "bow_phase":
                local[1] /= 0.235 / 0.28
                local = Rotation.from_euler("z", x[6]).apply(local)
                local[1] *= 0.235 / 0.28
            world = (Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(base[3:])).apply(local) + base[:3] + x[:3]
            pc = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = K @ pc
            return uv[:2] / uv[2] - np.array(result["bow_observations_px"][str(rec["requested_s"])])

        count = 0 if variant == "rigid" else (3 if variant == "axis_scale" else 1)
        x0 = np.r_[np.zeros(6), np.ones(count)]
        lo = np.r_[[-0.1] * 3, [-0.7] * 3, [0.7] * count]
        hi = np.r_[[0.1] * 3, [0.7] * 3, [1.3] * count]
        if variant == "bow_phase":
            x0[6], lo[6], hi[6] = 0, -np.pi, np.pi
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
        result["variants"][f"roof_{roof_height:.3f}"] = {
            "roof_height_m": roof_height,
            "pose_xyzw": pose.tolist(),
            "scale_xyz": shape(best.x).tolist(),
            "bow_phase_rad": phase,
            "bow_phase_scope": "Landmark-only diagnostic; no asset mutation. Actual bow mesh must be transformed and rechecked before any use.",
            "cost": best.cost,
            "solver_success": bool(best.success),
            "errors": {
                str(rec["requested_s"]): {
                    "band_median_px": float(np.median(band_error(best.x, rec))),
                    "band_max_px": float(band_error(best.x, rec).max()),
                    "median_px": float(np.median(errors(best.x, rec))),
                    "max_px": float(errors(best.x, rec).max()),
                    "bow_error_px": float(np.linalg.norm(bow_error(best.x, rec))),
                }
                for rec in records
            },
        }
    suffix = "_local_marker" if a.local_marker_cameras else ""
    (r / "review/scene" / (a.output_name or f"03_04_05_hat_band_landmarks_fit{suffix}.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print({k: v["errors"] for k, v in result["variants"].items()})


if __name__ == "__main__":
    main()
