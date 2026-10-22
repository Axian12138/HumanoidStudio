"""Fit a three-finger hook over the armchair back and a longer tableward push.

This is a kinematic candidate. The chair trajectory remains an estimate rather
than a measured 6D object track or a dynamic simulation result.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

from fit_hoi import HERE, MODEL, ORDER, ROOT, SIDES, chair_path, frame, smoothstep


OUT = HERE / "hoi_v2_backrest_hook"
CHAIR_FORWARD_M = 0.20
ROOT_FORWARD_M = 0.08
TABLE_NEAR_EDGE_Y = -2.11905405


def progress(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return smoothstep((t - start) / (end - start))


def fit() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    prior = MotionFields.load(HERE / "body_prior.npz", ORDER)
    with np.load(HERE / "human.npz", allow_pickle=False) as source:
        human = np.asarray(source["keypoints_w"], dtype=np.float64)
        foot_contact = np.asarray(source["foot_contact"], dtype=bool)
    chair_pos, chair_quat, _ = chair_path(human)
    t = 75 + np.arange(len(chair_pos)) / 50
    chair_pos[:, 1] -= CHAIR_FORWARD_M * progress(t, 79.0, 83.4)
    root_offset = np.zeros((len(t), 3), dtype=np.float64)
    root_offset[:, 1] = -ROOT_FORWARD_M * progress(t, 79.4, 83.3) * (
        1 - progress(t, 83.4, 83.9)
    )

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(n).qposadr[0] for n in prior.joint_names])
    joints = prior.joint_pos.astype(np.float64).copy()
    root = prior.body_pos_w[:, 0].astype(np.float64) + root_offset
    base_quat = prior.body_quat_w[:, 0]

    # Each planted ankle keeps the offset of its most recent touchdown. Swing
    # ankles move to a new landing with the translated root. Solve the six leg
    # joints against these per-foot FK targets, including foot orientation.
    foot_bodies = [model.body(f"{side}_ankle_roll_link").id for side in SIDES]
    baseline_foot_pos = np.empty((len(t), 2, 3))
    baseline_foot_quat = np.empty((len(t), 2, 4))
    for k in range(frame(79.3), frame(84.6)):
        data.qpos[:3] = prior.body_pos_w[k, 0]
        data.qpos[3:7] = base_quat[k]
        data.qpos[addresses] = prior.joint_pos[k]
        mujoco.mj_forward(model, data)
        baseline_foot_pos[k] = data.xpos[foot_bodies]
        baseline_foot_quat[k] = data.xquat[foot_bodies]

    foot_errors = {}
    for side_i, side in enumerate(SIDES):
        names = [f"{side}_{part}_joint" for part in (
            "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"
        )]
        cols = np.asarray([prior.joint_names.index(n) for n in names])
        addr = addresses[cols]
        lo = np.asarray([model.joint(n).range[0] + 1e-5 for n in names])
        hi = np.asarray([model.joint(n).range[1] - 1e-5 for n in names])
        touchdown_offset = np.zeros(3)
        swing_start = frame(79.4)
        swing_origin = np.zeros(3)
        errors = []
        for k in range(frame(79.4), frame(84.55)):
            if foot_contact[k, side_i] and not foot_contact[k - 1, side_i]:
                touchdown_offset = root_offset[k].copy()
            if not foot_contact[k, side_i] and foot_contact[k - 1, side_i]:
                swing_start = k
                swing_origin = touchdown_offset.copy()
            if foot_contact[k, side_i]:
                target_offset = touchdown_offset
            else:
                next_land = next((j for j in range(k + 1, len(t)) if foot_contact[j, side_i]), len(t) - 1)
                blend_end = min(next_land, frame(84.54))
                alpha = smoothstep((k - swing_start) / max(1, blend_end - swing_start))
                target_offset = (1 - alpha) * swing_origin + alpha * root_offset[k]
            target_pos = baseline_foot_pos[k, side_i] + target_offset
            target_rot = Rotation.from_quat(baseline_foot_quat[k, side_i, [1, 2, 3, 0]])
            baseline = prior.joint_pos[k, cols].astype(np.float64)
            data.qpos[:3] = root[k]
            data.qpos[3:7] = base_quat[k]
            data.qpos[addresses] = joints[k]
            foot = foot_bodies[side_i]

            def objective(q):
                data.qpos[addr] = q
                mujoco.mj_forward(model, data)
                got_rot = Rotation.from_quat(data.xquat[foot, [1, 2, 3, 0]])
                angle = (got_rot * target_rot.inv()).as_rotvec()
                return np.r_[28 * (data.xpos[foot] - target_pos), 3 * angle, .5 * (q - baseline)]

            seed = np.clip(baseline, lo, hi)
            result = least_squares(objective, seed, bounds=(lo, hi), max_nfev=16)
            joints[k, cols] = result.x
            data.qpos[addr] = result.x
            mujoco.mj_forward(model, data)
            errors.append(np.linalg.norm(data.xpos[foot] - target_pos))
        foot_errors[side] = {"mean_m": float(np.mean(errors)), "max_m": float(max(errors))}

    # Contact triplet in the chair frame. Index and middle fingertips sit in
    # front of the backrest face; the thumb comes over the upper edge. Unlike
    # the v1 palm-center target, these targets leave space for the chair mesh.
    contact_targets = {
        "thumb": (.150, .275, .780),
        "index": (.150, .120, .700),
        "middle": (.180, .120, .700),
    }
    contact_errors = {}
    for side, sign in (("left", 1), ("right", -1)):
        names = [f"{side}_{part}_joint" for part in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        )]
        names += [n for n in prior.joint_names if n.startswith(f"{side}_hand_")]
        cols = np.asarray([prior.joint_names.index(n) for n in names])
        addr = addresses[cols]
        lo = np.asarray([model.joint(n).range[0] + 1e-5 for n in names])
        hi = np.asarray([model.joint(n).range[1] - 1e-5 for n in names])
        tips = [model.body(f"{side}_hand_{part}_finger_tip").id
                for part in ("thumb", "index", "middle")]
        finger_geoms = {
            part: [model.geom(f"{side}_hand_{part}_{segment}_collision").id
                   for segment in (("1", "2") if part == "thumb" else ("0", "1"))]
            for part in ("thumb", "index", "middle")
        }
        wrist = model.body(f"{side}_wrist_yaw_link").id
        target = np.asarray([[sign * x, y, z] for x, y, z in contact_targets.values()])
        keys = np.arange(frame(79.15), frame(83.4) + 1, 10)
        offsets = []
        errors = []
        previous = None
        for k in keys:
            baseline = prior.joint_pos[k, cols].astype(np.float64)
            data.qpos[:3] = root[k]
            data.qpos[3:7] = base_quat[k]
            data.qpos[addresses] = joints[k]
            rot = Rotation.from_quat(chair_quat[k]).as_matrix()
            chair = chair_pos[k]
            seed = np.clip(baseline if previous is None else previous, lo, hi)

            def objective(q):
                data.qpos[addr] = q
                mujoco.mj_forward(model, data)
                actual = (rot.T @ (data.xpos[tips] - chair).T).T
                wrist_local = rot.T @ (data.xpos[wrist] - chair)
                index_middle_centers = (rot.T @ (
                    data.geom_xpos[finger_geoms["index"] + finger_geoms["middle"]] - chair
                ).T).T
                thumb_centers = (rot.T @ (
                    data.geom_xpos[finger_geoms["thumb"]] - chair
                ).T).T
                return np.r_[
                    22 * (actual - target).ravel(),
                    20 * np.maximum(0, index_middle_centers[:, 1] - .135),
                    20 * np.maximum(0, np.minimum(.27 - thumb_centers[:, 1], .80 - thumb_centers[:, 2])),
                    .9 * (q[:7] - baseline[:7]),
                    .2 * (q[7:] - baseline[7:]),
                    1.2 * (q - previous) if previous is not None else np.zeros(len(q)),
                    4 * max(0, .77 - wrist_local[2]),
                ]

            result = least_squares(objective, seed, bounds=(lo, hi), max_nfev=85)
            previous = result.x.copy()
            offsets.append(result.x - baseline)
            data.qpos[addr] = result.x
            mujoco.mj_forward(model, data)
            actual = (rot.T @ (data.xpos[tips] - chair).T).T
            errors.append(np.linalg.norm(actual - target, axis=1))
        spline_frames = np.r_[frame(78.35), keys, frame(84.45)]
        correction = CubicSpline(
            spline_frames, np.vstack((np.zeros(len(cols)), offsets, np.zeros(len(cols)))),
            axis=0, bc_type="clamped",
        )
        active = slice(frame(78.35), frame(84.45) + 1)
        joints[active, cols] += correction(np.arange(active.start, active.stop))
        joints[active, cols] = np.clip(joints[active, cols], lo, hi)
        contact_errors[side] = {
            "thumb_index_middle_mean_m": np.mean(errors, axis=0).tolist(),
            "thumb_index_middle_max_m": np.max(errors, axis=0).tolist(),
        }

    seed_motion = replace(prior, joint_pos=joints.astype(np.float32), body_pos_w=np.concatenate((
        root[:, None, :].astype(np.float32), prior.body_pos_w[:, 1:]
    ), axis=1))
    candidate = refresh_fk(seed_motion, MODEL)
    candidate.save(OUT / "motion.npz")
    np.savez_compressed(
        OUT / "object_estimate.npz", position_w=chair_pos, quaternion_xyzw=chair_quat,
        fps=np.asarray(50), source_range_s=np.asarray([75.0, 91.0]),
        trajectory_source=np.asarray("human_wrist_fit_plus_table_clearance_kinematic_estimate"),
    )
    with np.load(ROOT / "retarget/ground_support_27_end/sidecars/human_motion_world.npz", allow_pickle=False) as source:
        begin, end = (round((x - 27) * 50) for x in (75, 91))
        overlay = {name: (source[name][begin:end] if source[name].ndim and source[name].shape[0] == 5104 else source[name])
                   for name in source.files}
    overlay["source_range_s"] = np.asarray([75.0, 91.0])
    np.savez_compressed(OUT / "human_motion_world.npz", **overlay)
    shutil.copy2(HERE / "source_sync.npz", OUT / "source_sync.npz")
    review = {
        "stage": "3.2_kinematic_candidate_pending_visual_review",
        "replaces": "hoi_v1 (palm and fingers penetrated chair; chair stopped short of table)",
        "contact": "index and middle fingertips on front of upper backrest; thumb over top edge",
        "contact_targets_chair_local_m": contact_targets,
        "fit_errors_m": contact_errors,
        "foot_target_errors_m": foot_errors,
        "chair_extra_tableward_displacement_m": CHAIR_FORWARD_M,
        "maximum_root_tableward_adjustment_m": ROOT_FORWARD_M,
        "chair_final_position_m": chair_pos[-1].tolist(),
        "table_near_edge_y_m": TABLE_NEAR_EDGE_Y,
        "source_hashes_sha256": {name: sha256(HERE / name) for name in ("body_prior.npz", "human.npz", "source_sync.npz")},
        "limitations": ["Object path and finger contacts are kinematic estimates; verify in 3D and dynamics."],
    }
    (OUT / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    fit()
