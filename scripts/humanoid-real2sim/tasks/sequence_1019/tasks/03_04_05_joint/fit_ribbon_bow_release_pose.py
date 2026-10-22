"""Fit visible ribbon midline and silhouette with independent fixed cameras."""

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
    base = np.array(
        json.loads((r / "review/scene/03_04_05_hat_band_landmarks_elliptic_bow_phase.json").read_text())["variants"][
            "bow_phase"
        ]["pose_xyzw"]
    )
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
            for l in (r / "scene/03_04_05/hat_source_ribbon_bow_v2/visual_union.obj").read_text().splitlines()
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
            0.095 * 1.1086393424796992 * np.cos(theta),
            0.095 * 1.1086393424796992 * (0.235 / 0.28) * np.sin(theta),
            np.full(len(theta), (0.031071428 + 0.047642857) / 2),
        ]
    )
    result = {
        "status": "ribbon_bow_pose_diagnostic_no_contact_constraint",
        "fit_times_s": [47.7, 48.0],
        "held_out_s": 48.3,
        "fixed_marker_cameras": True,
        "band_observations": band_report,
        "band_model": "Authored part5 outer radius linear midheight ring; nearest projected ring centerline, visibility must be checked independently",
        "variants": {},
        "bow_observations_px": {
            "48": {
                "band_entry": [753, 375],
                "loop_outer_left": [751, 378],
                "loop_lower": [755, 383],
                "loop_return": [771, 389],
                "junction": [779, 394],
                "lower_root": [785, 400],
                "lower_mid": [796, 412],
                "lower_tip": [810, 419],
                "right_mid": [798, 396],
            }
        },
        "bow_annotation_uncertainty_px": 4,
        "bow_residual_scale_px": 8,
        "bow_metric_scope": "48 bow_error_px is18-scalar residual norm, not maximum node error; absent other-time feature errors are null.",
        "local_marker_cameras": a.local_marker_cameras,
        "camera_uncertainty": "Independent marker fit and held-out markers; annotation sensitivity is not a statistical confidence interval.",
    }
    import mujoco
    import trimesh

    model = mujoco.MjModel.from_xml_path(str(r / "scene/03_04_05/hat_source_ribbon_bow_v2/model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hook = model.geom("rack_55").id
    center = data.geom_xpos[hook].copy()
    meshid = model.geom_dataid[hook]
    start, count = model.mesh_vertadr[meshid], model.mesh_vertnum[meshid]
    radius = float(np.linalg.norm(model.mesh_vert[start : start + count], axis=1).max())
    actual = trimesh.load(r / "scene/03_04_05/hat_source_ribbon_bow_v2/visual_union.obj", process=False)
    prox = trimesh.proximity.ProximityQuery(actual)

    def contact(x):
        rot = Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(base[3:])
        local = rot.inv().apply(center - base[:3] - x[:3])
        return float(prox.on_surface(local[None])[1][0]) - radius

    result["contact_scope"] = "Contact is measured after fitting, not imposed; source pose hypothesis only."
    result["hook_center_world"] = center.tolist()
    result["hook_radius_m"] = radius
    for variant in ["rigid"]:

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
                [],
                x[:3] / 0.1,
                x[3:6] / 0.7,
                ([] if variant == "rigid" else (x[6:] / np.pi if variant == "bow_phase" else (x[6:] - 1) / 0.15)),
            ]

        def bow_error(x, rec, variant=variant):
            if rec["requested_s"] != 48:
                return np.empty(0)
            report = json.loads((r / "scene/03_04_05/hat_source_ribbon_bow_v2/review.json").read_text())
            observations = {
                "band_entry": [753, 375],
                "loop_outer_left": [751, 378],
                "loop_lower": [755, 383],
                "loop_return": [771, 389],
                "junction": [779, 394],
                "lower_root": [785, 400],
                "lower_mid": [796, 412],
                "lower_tip": [810, 419],
                "right_mid": [798, 396],
            }
            local = np.array([report["local_nodes"][k] for k in observations])
            camera = np.array(rec["T_world_camera"])
            world = (Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(base[3:])).apply(local) + base[:3] + x[:3]
            pc = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = pc @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            return (uv - np.array(list(observations.values()))).ravel()

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
        result["variants"][variant] = {
            "pose_xyzw": pose.tolist(),
            "scale_xyz": shape(best.x).tolist(),
            "bow_phase_rad": None,
            "bow_phase_scope": "New15-cell ribbon geometry inhat_source_ribbon_bow_v2; no ellipsoid-center fit. Crown/brim/band fixed.",
            "cost": best.cost,
            "hook_surface_gap_m": contact(best.x),
            "solver_success": bool(best.success),
            "errors": {
                str(rec["requested_s"]): {
                    "band_median_px": float(np.median(band_error(best.x, rec))),
                    "band_max_px": float(band_error(best.x, rec).max()),
                    "median_px": float(np.median(errors(best.x, rec))),
                    "max_px": float(errors(best.x, rec).max()),
                    "bow_error_px": float(np.linalg.norm(bow_error(best.x, rec))) if rec["requested_s"] == 48 else None,
                }
                for rec in records
            },
        }
    suffix = "_local_marker" if a.local_marker_cameras else ""
    (r / "review/scene" / (a.output_name or f"03_04_05_hat_band_landmarks_fit{suffix}.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
