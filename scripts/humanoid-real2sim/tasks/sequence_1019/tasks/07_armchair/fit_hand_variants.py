"""Create independent thumb-hook and palm-push hand revisions of HOI v2."""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
import shutil

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

from fit_hoi import HERE, MODEL, ORDER, SIDES, frame


SOURCE = HERE / "hoi_v2_backrest_hook"
MODES = {"thumb": HERE / "hoi_v3_thumb_contact", "palm": HERE / "hoi_v3_palm_push"}


def fit_mode(mode: str, base: MotionFields, chair_pos: np.ndarray, chair_quat: np.ndarray) -> None:
    out = MODES[mode]
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(n).qposadr[0] for n in base.joint_names])
    joints = base.joint_pos.astype(np.float64).copy()
    fit_errors = {}
    for side, sign in (("left", 1), ("right", -1)):
        names = [f"{side}_{part}_joint" for part in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        )]
        names += [n for n in base.joint_names if n.startswith(f"{side}_hand_")]
        cols = np.asarray([base.joint_names.index(n) for n in names])
        addr = addresses[cols]
        lower = np.asarray([model.joint(n).range[0] + 1e-5 for n in names])
        upper = np.asarray([model.joint(n).range[1] - 1e-5 for n in names])
        if mode == "thumb":
            thumb1 = names.index(f"{side}_hand_thumb_1_joint")
            if side == "left":
                lower[thumb1] = max(lower[thumb1], -.4)
            else:
                upper[thumb1] = min(upper[thumb1], .4)
        tips = [model.body(f"{side}_hand_{part}_finger_tip").id
                for part in ("thumb", "index", "middle")]
        thumb_distal = model.geom(f"{side}_hand_thumb_2_collision").id
        palm_geom = model.geom(f"{side}_hand_palm_collision").id
        # The palm contact face is opposite local X on the left Dex3 and
        # along local X on the right Dex3.
        palm_face_sign = -1 if side == "left" else 1
        keys = np.arange(frame(79.2), frame(83.4) + 1, 10)
        offsets = []
        errors = []
        previous = None
        for k in keys:
            data.qpos[:3] = base.body_pos_w[k, 0]
            data.qpos[3:7] = base.body_quat_w[k, 0]
            data.qpos[addresses] = base.joint_pos[k]
            q0 = base.joint_pos[k, cols].astype(np.float64)
            rotation = Rotation.from_quat(chair_quat[k]).as_matrix()
            position = chair_pos[k]
            seed = np.clip(q0 if previous is None else previous, lower, upper)

            def objective(q):
                data.qpos[addr] = q
                mujoco.mj_forward(model, data)
                tip = (rotation.T @ (data.xpos[tips] - position).T).T
                thumb_pad = rotation.T @ (data.geom_xpos[thumb_distal] - position)
                palm = rotation.T @ (data.geom_xpos[palm_geom] - position)
                palm_normal = rotation.T @ (
                    data.geom_xmat[palm_geom].reshape(3, 3)[:, 0] * palm_face_sign
                )
                continuity = 1.2 * (q - previous) if previous is not None else np.zeros(len(q))
                if mode == "thumb":
                    desired_tips = np.asarray([
                        [sign * .145, .230, .750],
                        [sign * .172, .155, .705],
                        [sign * .190, .135, .683],
                    ])
                    return np.r_[
                        30 * (tip - desired_tips).ravel(),
                        18 * (thumb_pad - np.array([sign * .16, .25, .755])),
                        .65 * (q[:7] - q0[:7]), .25 * (q[7:] - q0[7:]), continuity,
                    ]
                return np.r_[
                    30 * (palm - np.array([sign * .15, .255, .720])),
                    8 * (palm_normal - np.array([0, -1, 0])),
                    10 * np.maximum(0, tip[:, 1] - .16),
                    .6 * (q[:7] - q0[:7]), .25 * (q[7:] - q0[7:]), continuity,
                ]

            result = least_squares(objective, seed, bounds=(lower, upper), max_nfev=110)
            previous = result.x.copy()
            offsets.append(result.x - q0)
            errors.append(float(np.linalg.norm(result.fun)))
        spline_time = np.r_[frame(78.3), keys, frame(84.45)]
        correction = CubicSpline(spline_time, np.vstack((np.zeros(len(cols)), offsets, np.zeros(len(cols)))),
                                 axis=0, bc_type="clamped")
        active = slice(frame(78.3), frame(84.45) + 1)
        joints[active, cols] += correction(np.arange(active.start, active.stop))
        joints[active, cols] = np.clip(joints[active, cols], lower if mode == "palm" else
                                       np.asarray([model.joint(n).range[0] + 1e-5 for n in names]),
                                       upper if mode == "palm" else
                                       np.asarray([model.joint(n).range[1] - 1e-5 for n in names]))
        fit_errors[side] = {"mean_objective_norm": float(np.mean(errors)), "max_objective_norm": float(max(errors))}

    motion = refresh_fk(replace(base, joint_pos=joints.astype(np.float32)), MODEL)
    motion.save(out / "motion.npz")
    for name in ("object_estimate.npz", "human_motion_world.npz", "source_sync.npz"):
        shutil.copy2(SOURCE / name, out / name)
    mesh = trimesh.load_mesh(HERE / "grey_armchair_visual.obj")
    camera_ref = np.load(HERE / "hand_camera_reference.npz", allow_pickle=False)
    geometry = {side: {"contact_distance_m": [], "palm_forward_cos": [], "thumb1_joint_rad": []}
                for side in SIDES}
    for k in range(frame(79.2), frame(83.4) + 1, 5):
        data.qpos[:3] = motion.body_pos_w[k, 0]
        data.qpos[3:7] = motion.body_quat_w[k, 0]
        data.qpos[addresses] = motion.joint_pos[k]
        mujoco.mj_forward(model, data)
        rot = Rotation.from_quat(chair_quat[k]).as_matrix()
        for side in SIDES:
            geom = model.geom(f"{side}_hand_thumb_2_collision" if mode == "thumb" else
                              f"{side}_hand_palm_collision").id
            local = rot.T @ (data.geom_xpos[geom] - chair_pos[k])
            _, dist, _ = trimesh.proximity.closest_point(mesh, local[None])
            geometry[side]["contact_distance_m"].append(float(dist[0]))
            if mode == "palm":
                sign = -1 if side == "left" else 1
                normal = rot.T @ (data.geom_xmat[geom].reshape(3, 3)[:, 0] * sign)
                geometry[side]["palm_forward_cos"].append(float(-normal[1]))
            else:
                col = base.joint_names.index(f"{side}_hand_thumb_1_joint")
                geometry[side]["thumb1_joint_rad"].append(float(motion.joint_pos[k, col]))
    for side in SIDES:
        geometry[side] = {
            "mesh_distance_mean_m": float(np.mean(geometry[side]["contact_distance_m"])),
            "mesh_distance_max_m": float(max(geometry[side]["contact_distance_m"])),
            "palm_forward_cos_mean": float(np.mean(geometry[side]["palm_forward_cos"]))
            if mode == "palm" else None,
            "thumb1_joint_range_rad": [float(min(geometry[side]["thumb1_joint_rad"])),
                                        float(max(geometry[side]["thumb1_joint_rad"]))]
            if mode == "thumb" else None,
        }
    # The rigid-camera transform is diagnostic only in this fit. Marker-to-hand
    # mounting and human/robot morphology differ; neither camera-aligned pose
    # enters the optimization above.
    right_body = model.body("right_wrist_yaw_link").id
    camera_rot_errors = []
    camera_pos_errors = []
    for k in range(frame(79.2), frame(83.4) + 1, 5):
        data.qpos[:3] = motion.body_pos_w[k, 0]
        data.qpos[3:7] = motion.body_quat_w[k, 0]
        data.qpos[addresses] = motion.joint_pos[k]
        mujoco.mj_forward(model, data)
        robot_rotation = Rotation.from_matrix(data.xmat[right_body].reshape(3, 3))
        source_rotation = Rotation.from_quat(camera_ref["right_wrist_camera_aligned_quat_xyzw"][k])
        camera_rot_errors.append(float(np.degrees((robot_rotation.inv() * source_rotation).magnitude())))
        camera_pos_errors.append(float(np.linalg.norm(data.xpos[right_body] - camera_ref["right_wrist_camera_aligned_pos_w"][k])))
    speed = np.gradient(motion.joint_pos, .02, axis=0)
    report = {
        "status": "stage3_2_hand_variant_pending_visual_review",
        "mode": mode,
        "source": str(SOURCE.relative_to(HERE)),
        "fit": fit_errors,
        "geometry": geometry,
        "right_camera_aligned_wrist_reference": {
            "mean_orientation_difference_deg": float(np.mean(camera_rot_errors)),
            "mean_position_difference_m": float(np.mean(camera_pos_errors)),
            "interpretation": "diagnostic only; human marker mount and human-robot morphology prevent direct equality",
        },
        "max_joint_speed_rad_s": float(np.max(np.abs(speed))),
        "root_and_object_unchanged_from_v2": True,
        "limitations": ["Geometric proximity does not prove sustained contact force or dynamic pushing."],
    }
    (out / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(out), "geometry": geometry,
                      "camera": report["right_camera_aligned_wrist_reference"],
                      "max_joint_speed_rad_s": report["max_joint_speed_rad_s"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("thumb", "palm", "both"), default="both")
    args = parser.parse_args()
    base = MotionFields.load(SOURCE / "motion.npz", ORDER)
    with np.load(SOURCE / "object_estimate.npz", allow_pickle=False) as obj:
        chair_pos = obj["position_w"]
        chair_quat = obj["quaternion_xyzw"]
    for mode in MODES if args.mode == "both" else (args.mode,):
        fit_mode(mode, base, chair_pos, chair_quat)


if __name__ == "__main__":
    main()
