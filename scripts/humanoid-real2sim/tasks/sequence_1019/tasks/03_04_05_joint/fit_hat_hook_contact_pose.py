"""Compare fixed shape and bounded scale against source contours with fixed cameras."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output-name", default="03_04_05_hat_hook_contact_fit.json")
    p.add_argument("--shape-scale-z", type=float, default=1)
    p.add_argument("--shape-scale-xy", type=float, default=1)
    p.add_argument("--inside-cavity", action="store_true")
    p.add_argument("--wide-search", action="store_true")
    p.add_argument("--local-marker-cameras", action="store_true")
    p.add_argument("--model", type=Path)
    p.add_argument("--hook", default="rack_55")
    p.add_argument(
        "--gravity-align", action="store_true", help="Diagnostic single-hook COM alignment, not a full support solver"
    )
    a = p.parse_args()
    r = a.run
    hold = json.loads((r / "review/scene/03_04_05_hat_pose_holdout/report.json").read_text())["records"]
    old = json.loads((r / "review/scene/03_04_05_hat_source_pose_comparison.json").read_text())
    suffix = "_local_marker" if a.local_marker_cameras else ""
    fit = json.loads((r / f"review/scene/03_04_05_hat_multiview_landmarks_fit{suffix}.json").read_text())["variants"][
        "rigid" if a.shape_scale_z == 1 else "crown_depth"
    ]
    base = np.array(fit["pose_xyzw"])
    model_path = a.model or r / "scene/03_04_05/hat_rack_physics_hook_registered.xml"
    if a.shape_scale_z != 1 or a.shape_scale_xy != 1:
        import xml.etree.ElementTree as ET

        tree = ET.parse(model_path)
        for mesh in tree.getroot().findall(".//asset/mesh"):
            if mesh.get("name", "").startswith("black_ribbon_hat"):
                mesh.set("scale", f"{a.shape_scale_xy} {a.shape_scale_xy} {a.shape_scale_z}")
        model_path = r / "scene/03_04_05" / f"hat_xy_{a.shape_scale_xy:g}_depth_{a.shape_scale_z:g}_hook_registered.xml"
        tree.write(model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    hb = model.body("black_ribbon_hat").id
    hat_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == hb and model.geom_contype[i]]
    hook_geom = model.geom(a.hook).id
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
        "status": "source_and_hook_contact_pose_pending_dynamic_and_robot_checks",
        "fit_times_s": [47.7, 48.0],
        "held_out_s": 48.3,
        "fixed_marker_cameras": True,
        "variants": {},
        "bow_observations_px": {"47.7": [812, 407], "48": [775, 390], "48.3": [705, 346]},
        "bow_annotation_uncertainty_px": 8,
        "shape_scale_z": a.shape_scale_z,
        "shape_scale_xy": a.shape_scale_xy,
        "inside_cavity_constraint": a.inside_cavity,
        "single_hook_gravity_alignment": a.gravity_align,
        "model": str(model_path),
        "hook": a.hook,
        "local_marker_cameras": a.local_marker_cameras,
    }
    for variant in ["rigid"]:

        def shape(x, variant=variant):
            scale = np.array([a.shape_scale_xy, a.shape_scale_xy, 1.0])
            scale[2] = a.shape_scale_z
            if variant == "uniform":
                scale[:] = x[6]
            if variant == "crown_depth":
                scale[2] = x[6]
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
                contact_error(x) * 3000,
                x[:3] / 0.1,
                x[3:6] / 0.7,
                ([] if variant == "rigid" else [(x[6] - 1) / 0.15]),
            ]

        def contact_error(x):
            rotation = Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(base[3:])
            data.qpos[:3] = base[:3] + x[:3]
            data.qpos[3:7] = rotation.as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(model, data)
            distance = min(mujoco.mj_geomDistance(model, data, g, hook_geom, 0.1, None) for g in hat_geoms)
            penetration = max([max(0, -c.dist - 0.001) for c in data.contact] or [0])
            terms = [max(0, distance - 0.001), max(0, -distance - 0.001), penetration]
            if a.inside_cavity:
                local = rotation.inv().apply(data.geom_xpos[hook_geom] - data.qpos[:3])
                radial = np.linalg.norm(local[:2] / (np.array([0.084, 0.067]) * a.shape_scale_xy))
                terms.extend(
                    [max(0, radial - 1) * 0.05, max(0, 0.015 - local[2]), max(0, local[2] - 0.13 * a.shape_scale_z)]
                )
            if a.gravity_align:
                delta = data.xipos[hb] - data.geom_xpos[hook_geom]
                terms.extend([*delta[:2], max(0, delta[2] + 0.01)])
            return np.array(terms)

        def bow_error(x, rec):
            camera = np.array(rec["T_world_camera"])
            local = np.array([0, -0.077, 0.040]) * shape(x)
            world = (Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(base[3:])).apply(local) + base[:3] + x[:3]
            pc = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = K @ pc
            return uv[:2] / uv[2] - np.array(result["bow_observations_px"][str(rec["requested_s"])])

        x0 = np.r_[np.zeros(6), ([] if variant == "rigid" else [1])]
        lo = np.r_[
            [-0.2 if a.wide_search else -0.1] * 3,
            [-2.5 if a.wide_search else -0.7] * 3,
            ([] if variant == "rigid" else [0.7]),
        ]
        hi = np.r_[
            [0.2 if a.wide_search else 0.1] * 3,
            [2.5 if a.wide_search else 0.7] * 3,
            ([] if variant == "rigid" else [1.3]),
        ]
        best = None
        for angle in [0, -0.6, 0.6, -1.2, 1.2, -2.2, 2.2] if a.wide_search else [0, -0.6, -0.3, 0.3, 0.6]:
            initial = x0.copy()
            initial[3:6] = Rotation.from_quat(base[3:]).apply([0, 0, angle])
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
            "contact_error_m": contact_error(best.x).tolist(),
            "errors": {
                str(rec["requested_s"]): {
                    "median_px": float(np.median(errors(best.x, rec))),
                    "max_px": float(errors(best.x, rec).max()),
                    "bow_error_px": float(np.linalg.norm(bow_error(best.x, rec))),
                }
                for rec in records
            },
        }
    (r / "review/scene" / a.output_name).write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
