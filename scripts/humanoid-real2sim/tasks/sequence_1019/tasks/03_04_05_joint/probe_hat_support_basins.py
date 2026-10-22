"""Explore actual multi-point rack support; preserve original geometry and mass."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--mirror-settled", action="store_true")
    p.add_argument("--align-bow-settled", action="store_true")
    p.add_argument("--axis-shape", action="store_true")
    p.add_argument("--model", type=Path)
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--output", type=Path)
    p.add_argument("--centers-report", type=Path)
    p.add_argument("--pose-fit", type=Path)
    p.add_argument("--pose-variant", default="rigid")
    p.add_argument("--visual-obj", type=Path)
    a = p.parse_args()
    if a.output and a.output.exists():
        raise FileExistsError(a.output)
    r = a.run
    review = r / "review/scene"
    old = json.loads((review / "03_04_05_hat_source_pose_comparison.json").read_text())
    fit = json.loads((review / "03_04_05_hat_multiview_landmarks_fit_local_marker.json").read_text())["variants"][
        "rigid"
    ]
    if a.axis_shape:
        fit = json.loads((review / "03_04_05_hat_multiview_axis_shape_diagnostic.json").read_text())["variants"][
            "axis_scale"
        ]
    if a.pose_fit:
        fit = json.loads(a.pose_fit.read_text())["variants"][a.pose_variant]
    bow_local = np.array([0, -0.077, 0.04])
    if fit.get("bow_phase_rad", 0):
        bow_local[1] /= 0.235 / 0.28
        bow_local = Rotation.from_euler("z", fit["bow_phase_rad"]).apply(bow_local)
        bow_local[1] *= 0.235 / 0.28
    if "bow_local_m" in fit:
        bow_local = np.asarray(fit["bow_local_m"], dtype=float)
    scale = np.array(fit.get("scale_xyz", [1, 1, 1])) if a.axis_shape else np.ones(3)
    centers = [np.array(fit["pose_xyzw"]), np.array(old["variants"]["release_initial"]["pose_xyzw"])]
    if a.mirror_settled or a.align_bow_settled:
        centers = [
            np.array(x["final_pose_xyzw"])
            for x in json.loads((a.centers_report or review / "03_04_05_hat_support_basin_family.json").read_text())[
                "records"
            ]
            if x["rack_contacts"]
        ]
        if a.mirror_settled:
            for pose in centers:
                pose[3:] = (Rotation.from_quat(pose[3:]) * Rotation.from_euler("z", np.pi)).as_quat()
    model_path = r / (
        "scene/03_04_05/hat_axis_shape_diagnostic/hat_rack_physics.xml"
        if a.axis_shape
        else "scene/03_04_05/hat_rack_physics_early_registered.xml"
    )
    if a.model:
        model_path = a.model
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    ha = m.joint("black_ribbon_hat_free").qposadr[0]
    hb, rb = m.body("black_ribbon_hat").id, m.body("coat_rack_support").id
    v = np.array(
        [
            list(map(float, line.split()[1:]))
            for line in (a.visual_obj or r / "scene/03_04_05/black_ribbon_hat.obj").read_text().splitlines()
            if line.startswith("v ")
        ]
    )
    v *= scale
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    hold = json.loads((review / "03_04_05_hat_pose_holdout/report.json").read_text())["records"]
    obs = {48: old["source_observations_px"], **{x["requested_s"]: x["observations_px"] for x in hold}}
    bows = {47.7: [812, 407], 48: [775, 390], 48.3: [705, 346]}
    cams = {
        t: np.array(
            json.loads((review / f"parent_head{t:g}_marker_camera_pose_diagnostic.json").read_text())[
                "fitted_T_world_camera"
            ]
        )
        for t in bows
    }
    rng = np.random.default_rng(1019)
    records = []
    for seed in range(len(centers) if a.mirror_settled or a.align_bow_settled else a.seeds):
        initial = centers[seed % len(centers)].copy()
        if a.align_bow_settled:
            angles = np.linspace(-np.pi, np.pi, 361)
            rotations = Rotation.from_quat(initial[3:]) * Rotation.from_euler("z", angles[:, None])
            bw = rotations.apply(np.tile(bow_local * scale, (len(angles), 1))) + initial[:3]
            scores = np.zeros(len(angles))
            for t in [47.7, 48]:
                camera = cams[t]
                projected = ((bw - camera[:3, 3]) @ camera[:3, :3]) @ K.T
                scores += np.linalg.norm(projected[:, :2] / projected[:, 2:] - bows[t], axis=1) ** 2
            initial[3:] = rotations[int(np.argmin(scores))].as_quat()
        if seed > 1 and not (a.mirror_settled or a.align_bow_settled):
            initial[:3] += rng.uniform(-0.045, 0.045, 3)
            initial[3:] = (Rotation.from_rotvec(rng.normal(0, 0.35, 3)) * Rotation.from_quat(initial[3:])).as_quat()
        mujoco.mj_resetData(m, d)
        d.qpos[ha : ha + 3] = initial[:3]
        d.qpos[ha + 3 : ha + 7] = initial[3:][[3, 0, 1, 2]]
        mujoco.mj_forward(m, d)
        tail = []
        trace = [{"time_s": 0.0, "qpos_wxyz": d.qpos[ha : ha + 7].tolist()}]
        for k in range(round(7 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if k % 25 == 0:
                trace.append({"time_s": float(d.time), "qpos_wxyz": d.qpos[ha : ha + 7].tolist()})
            if d.time > 6 and k % 10 == 0:
                tail.append(d.qpos[ha : ha + 7].copy())
        q = d.qpos[ha : ha + 7].copy()
        pose = q[[0, 1, 2, 4, 5, 6, 3]]
        tail = np.array(tail)
        contacts = [c for c in d.contact if {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} == {hb, rb}]
        world = Rotation.from_quat(pose[3:]).apply(v) + pose[:3]
        bw = Rotation.from_quat(pose[3:]).apply(bow_local * scale) + pose[:3]
        errors = {}
        for t, camera in cams.items():
            uv = ((world - camera[:3, 3]) @ camera[:3, :3]) @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            poly = uv[ConvexHull(uv).vertices]
            points = np.array(obs[t])
            delta = np.roll(poly, -1, axis=0) - poly
            u = np.clip(((points[:, None] - poly) * delta).sum(2) / (delta * delta).sum(1), 0, 1)
            err = np.linalg.norm(points[:, None] - (poly[None] + u[:, :, None] * delta), axis=2).min(1)
            b = K @ ((bw - camera[:3, 3]) @ camera[:3, :3])
            errors[str(t)] = {
                "median_px": float(np.median(err)),
                "max_px": float(err.max()),
                "bow_error_px": float(np.linalg.norm(b[:2] / b[2] - bows[t])),
            }
        angles = (
            Rotation.from_quat(tail[:, [4, 5, 6, 3]]) * Rotation.from_quat(tail[0, [4, 5, 6, 3]]).inv()
        ).magnitude()
        record = {
            "seed": seed,
            "initial_pose_xyzw": initial.tolist(),
            "final_pose_xyzw": pose.tolist(),
            "rack_contacts": len(contacts),
            "tail_translation_range_m": float(np.linalg.norm(np.ptp(tail[:, :3], axis=0))),
            "tail_rotation_range_rad": float(angles.max()),
            "source_errors": errors,
            "trace": trace,
        }
        records.append(record)
        print({"seed": seed, "rack_contacts": len(contacts), "holdout": errors["48.3"]}, flush=True)
    suffix = (
        "_axis_shape"
        if a.axis_shape
        else ("_bow_aligned" if a.align_bow_settled else ("_mirrored" if a.mirror_settled else ""))
    )
    (a.output or review / f"03_04_05_hat_support_basin_family{suffix}.json").write_text(
        json.dumps(
            {
                "status": "multi_contact_native_support_diagnostic",
                "model": str(model_path),
                "scale_xyz": scale.tolist(),
                "bow_local_m": bow_local.tolist(),
                "pose_fit": str(a.pose_fit) if a.pose_fit else None,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
