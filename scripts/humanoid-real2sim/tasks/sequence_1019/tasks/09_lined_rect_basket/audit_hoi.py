"""Audit hand mesh clearance, palm direction and preserved body motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields
from fit_hoi import (
    CONTACT_BEGIN, CONTACT_END, FPS, MODEL, ORDER, START, TASK,
    basket_sdf, collision_points, frame,
)

WALLS = (
    ((0, 0, .004), (.19, .13, .004)),
    ((.186, 0, .074), (.004, .13, .066)),
    ((-.186, 0, .074), (.004, .13, .066)),
    ((0, .126, .074), (.182, .004, .066)),
    ((0, -.126, .074), (.182, .004, .066)),
)


def vertices(model: mujoco.MjModel, geom: int) -> np.ndarray:
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        begin = model.mesh_vertadr[mesh]
        return model.mesh_vert[begin:begin + model.mesh_vertnum[mesh]:12]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        x, y, z = model.geom_size[geom]
        return np.asarray([[a, b, c] for a in (-x, x) for b in (-y, y) for c in (-z, z)])
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return collision_points(model, geom)
    raise ValueError(f"unsupported hand geom: {geom}")


def penetration(local: np.ndarray) -> float:
    depth = 0.0
    for midpoint, half in WALLS:
        inside = np.asarray(half) - np.abs(local - midpoint)
        hits = np.all(inside > 0, axis=1)
        if np.any(hits):
            depth = max(depth, float(np.max(np.min(inside[hits], axis=1))))
    return depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path)
    args = parser.parse_args()
    prior = MotionFields.load(TASK / "body_prior.npz", ORDER)
    motion = MotionFields.load(args.candidate / "motion.npz", ORDER)
    with np.load(args.candidate / "object_estimate.npz", allow_pickle=False) as obj:
        centers = obj["position_w"].astype(float)
        rotations = Rotation.from_quat(obj["quaternion_xyzw"]).as_matrix()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    joint_addr = np.asarray([model.joint(name).qposadr[0] for name in motion.joint_names])
    geometry = {}
    for side in ("left", "right"):
        wrist = model.geom(f"{side}_wrist_collision").id
        forearm = model.geom(f"{side}_elbow_yaw_collision").id
        geometry[side] = [
            (i, vertices(model, i)) for i in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
            and name.endswith("_collision")
            and (name.startswith(f"{side}_hand_") or i in (wrist, forearm))
        ]
    per_frame = []
    for k in range(frame(START), frame(CONTACT_END) + 1):
        data.qpos[:3] = motion.body_pos_w[k, 0]
        data.qpos[3:7] = motion.body_quat_w[k, 0]
        data.qpos[joint_addr] = motion.joint_pos[k]
        mujoco.mj_forward(model, data)
        R, center = rotations[k], centers[k]
        row = {"time_s": START + k / FPS}
        for side, sign in (("left", 1), ("right", -1)):
            palm_id = model.geom(f"{side}_hand_palm_collision").id
            face = (-1 if side == "left" else 1) * data.geom_xmat[palm_id].reshape(3, 3)[:, 0]
            palm = R.T @ (data.geom_xpos[palm_id] - center)
            tips = np.stack([
                data.xpos[model.body(f"{side}_hand_{part}_finger_tip").id]
                for part in ("thumb", "index", "middle")
            ])
            tip_local = (R.T @ (tips - center).T).T
            worst = (0.0, "")
            nearest = float("inf")
            inside_basket = 0
            band_y = []
            wrist_band = 0
            forearm_band = 0
            wrist_depth = 0.0
            forearm_depth = 0.0
            wrist_outer_intrusion = 0.0
            forearm_outer_intrusion = 0.0
            for geom, points in geometry[side]:
                world = points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                local = (world - center) @ R
                depth = penetration(local)
                nearest = min(nearest, float(np.min(basket_sdf(local))))
                inside_basket += int(np.count_nonzero(
                    (np.abs(local[:, 0]) < .182) & (np.abs(local[:, 1]) < .122)
                    & (local[:, 2] > .008) & (local[:, 2] < .14)
                ))
                if geom in (model.geom(f"{side}_wrist_collision").id,
                            model.geom(f"{side}_elbow_yaw_collision").id):
                    side_x = sign * local[:, 0]
                    wall_height = (np.abs(local[:, 1]) <= .13) & (
                        (local[:, 2] >= .015) & (local[:, 2] <= .14))
                    outer_intrusion = (max(0.0, .19 - float(np.min(side_x[wall_height])))
                                       if np.any(wall_height) else 0.0)
                    band = (np.abs(side_x - .19) <= .006) & (np.abs(local[:, 1]) <= .13) & (
                        (local[:, 2] >= .015) & (local[:, 2] <= .14))
                    band_y.extend(local[band, 1].tolist())
                    if geom == model.geom(f"{side}_wrist_collision").id:
                        wrist_band = int(np.count_nonzero(band))
                        wrist_depth = depth
                        wrist_outer_intrusion = outer_intrusion
                    else:
                        forearm_band = int(np.count_nonzero(band))
                        forearm_depth = depth
                        forearm_outer_intrusion = outer_intrusion
                if depth > worst[0]:
                    worst = (depth, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
            row[side] = {
                "palm_local_m": palm.tolist(), "tips_local_m": tip_local.tolist(),
                "palm_down_error_deg": float(np.rad2deg(np.arccos(np.clip(-face[2], -1, 1)))),
                "penetration_m": worst[0], "worst_geom": worst[1],
                "inside_basket_samples": inside_basket,
                "nearest_mesh_surface_m": nearest,
                "wrist_band_samples": wrist_band,
                "forearm_band_samples": forearm_band,
                "wrist_penetration_m": wrist_depth,
                "forearm_penetration_m": forearm_depth,
                "wrist_outer_intrusion_m": wrist_outer_intrusion,
                "forearm_outer_intrusion_m": forearm_outer_intrusion,
                "contact_band_span_y_m": float(np.ptp(band_y)) if band_y else 0.0,
                "wrist_local_m": (R.T @ (data.geom_xpos[model.geom(f"{side}_wrist_collision").id] - center)).tolist(),
                "forearm_local_m": (R.T @ (data.geom_xpos[model.geom(f"{side}_elbow_yaw_collision").id] - center)).tolist(),
                "outer_wall_gap_m": float(abs(palm[0]) - .19),
            }
        per_frame.append(row)
    joint_limits = np.asarray([model.joint(name).range for name in motion.joint_names])
    violation = np.maximum(joint_limits[:, 0] - motion.joint_pos, 0) + np.maximum(
        motion.joint_pos - joint_limits[:, 1], 0)
    v = np.gradient(motion.joint_pos.astype(float), 1 / FPS, axis=0)
    a = np.gradient(v, 1 / FPS, axis=0)
    other = [i for i, name in enumerate(motion.joint_names) if not name.startswith(
        ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hand", "right_hand"))]
    summary = {
        "sample_window_s": [START, CONTACT_END], "sample_rate_hz": FPS,
        "joint_limit_max_violation_rad": float(np.max(violation)),
        "peak_joint_velocity_rad_s": float(np.max(np.abs(v))),
        "peak_joint_acceleration_rad_s2": float(np.max(np.abs(a))),
        "unchanged_root_max_error_m": float(np.max(np.abs(motion.body_pos_w[:, 0] - prior.body_pos_w[:, 0]))),
        "unchanged_other_joint_max_error_rad": float(np.max(np.abs(motion.joint_pos[:, other] - prior.joint_pos[:, other]))),
        "sides": {},
    }
    for side in ("left", "right"):
        approach_rows = [item[side] for item in per_frame if item["time_s"] <= CONTACT_BEGIN + 1e-8]
        start = np.asarray(approach_rows[0]["palm_local_m"])
        end = np.asarray(approach_rows[-1]["palm_local_m"])
        travel = end - start
        final_start = next(item for item in per_frame if abs(item["time_s"] - 98.6) < 1e-8)
        final_travel = end - np.asarray(final_start[side]["palm_local_m"])
        sign = 1 if side == "left" else -1
        contact_rows = [item for item in per_frame if item["time_s"] >= CONTACT_BEGIN - 1e-8]
        rows = [item[side] for item in contact_rows]
        depth = np.asarray([item["penetration_m"] for item in rows])
        precontact = [item[side] for item in per_frame if item["time_s"] < CONTACT_BEGIN]
        down = np.asarray([item["palm_down_error_deg"] for item in rows])
        gap = np.asarray([item["outer_wall_gap_m"] for item in rows])
        surface = np.asarray([item["nearest_mesh_surface_m"] for item in rows])
        wrist_band = np.asarray([item["wrist_band_samples"] for item in rows])
        forearm_band = np.asarray([item["forearm_band_samples"] for item in rows])
        wrist_depth = np.asarray([item["wrist_penetration_m"] for item in rows])
        forearm_depth = np.asarray([item["forearm_penetration_m"] for item in rows])
        wrist_outer = np.asarray([item["wrist_outer_intrusion_m"] for item in rows])
        forearm_outer = np.asarray([item["forearm_outer_intrusion_m"] for item in rows])
        patch_span = np.asarray([item["contact_band_span_y_m"] for item in rows])
        wrist_height = np.asarray([item["wrist_local_m"][2] for item in rows])
        worst = int(np.argmax(depth))
        summary["sides"][side] = {
            "mesh_penetration_max_mm": float(depth.max() * 1000),
            "precontact_penetration_max_mm": float(max(item["penetration_m"] for item in precontact) * 1000),
            "precontact_inside_basket_samples_max": max(item["inside_basket_samples"] for item in precontact),
            "frames_over_1mm": int(np.count_nonzero(depth > .001)),
            "worst_time_s": contact_rows[worst]["time_s"], "worst_geom": rows[worst]["worst_geom"],
            "palm_down_median_deg": float(np.median(down)), "palm_down_max_deg": float(down.max()),
            "palm_outer_wall_gap_median_mm": float(np.median(gap) * 1000),
            "nearest_mesh_surface_median_mm": float(np.median(surface) * 1000),
            "nearest_mesh_surface_max_mm": float(np.max(surface) * 1000),
            "wrist_height_median_mm_above_base": float(np.median(wrist_height) * 1000),
            "wrist_band_frame_fraction": float(np.mean(wrist_band > 0)),
            "forearm_band_frame_fraction": float(np.mean(forearm_band > 0)),
            "wrist_shallow_penetration_frame_fraction": float(np.mean(wrist_depth > 0)),
            "forearm_shallow_penetration_frame_fraction": float(np.mean(forearm_depth > 0)),
            "wrist_penetration_max_mm": float(wrist_depth.max() * 1000),
            "forearm_penetration_max_mm": float(forearm_depth.max() * 1000),
            "wrist_outer_intrusion_median_mm": float(np.median(wrist_outer) * 1000),
            "wrist_outer_intrusion_max_mm": float(wrist_outer.max() * 1000),
            "forearm_outer_intrusion_median_mm": float(np.median(forearm_outer) * 1000),
            "forearm_outer_intrusion_max_mm": float(forearm_outer.max() * 1000),
            "contact_band_span_y_median_mm": float(np.median(patch_span) * 1000),
            "approach_inward_mm": float(-sign * travel[0] * 1000),
            "approach_normal_alignment_cos": float(-sign * travel[0] / max(np.linalg.norm(travel), 1e-9)),
            "approach_tangent_drift_mm": float(np.linalg.norm(travel[1:]) * 1000),
            "final_approach_98_6_98_8_normal_alignment_cos": float(
                -sign * final_travel[0] / max(np.linalg.norm(final_travel), 1e-9)),
            "approach_penetration_max_mm": float(max(x["penetration_m"] for x in approach_rows) * 1000),
        }
    (args.audit_out or args.candidate / "audit.json").write_text(
        json.dumps({"summary": summary, "frames": per_frame}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
