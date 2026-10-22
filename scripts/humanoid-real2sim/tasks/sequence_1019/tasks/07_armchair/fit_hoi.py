"""Fit bilateral palm contact while the human pushes the armchair under the table.

The chair path is a ground-supported kinematic estimate from the human wrists.
This script does not run dynamics or assert physical grasp/push success.
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
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


HERE = example_dir(__file__)
ROOT = HERE.parents[1]
OUT = HERE / "hoi_v1"
ORDER = ROOT / "retarget/ground_support_27_end/sidecars/motion_order.json"
MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
    "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)
FPS = 50
START = 75.0
CHAIR_BACK_CONTACT_Y = 0.18
CHAIR_CONTACT_Z = 0.685
SIDES = ("left", "right")


def frame(seconds: float) -> int:
    return round((seconds - START) * FPS)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def chair_path(human: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrists = savgol_filter(human[:, [13, 14], :2].astype(np.float64), 21, 3, axis=0)
    mid = wrists.mean(axis=1)
    span = wrists[:, 0] - wrists[:, 1]
    yaw = np.unwrap(np.arctan2(span[:, 1], span[:, 0]))
    yaw = savgol_filter(yaw, 21, 3)
    raw_xy = mid + np.column_stack((np.sin(yaw), -np.cos(yaw))) * CHAIR_BACK_CONTACT_Y
    first, last = frame(79.0), frame(83.4)
    xy = raw_xy.copy()
    xy[:first] = raw_xy[first]
    yaw[:first] = yaw[first]
    end_xy, end_yaw = raw_xy[last].copy(), float(yaw[last])
    xy[last:] = end_xy
    yaw[last:] = end_yaw
    for k in range(first, frame(79.4)):
        w = smoothstep((k - first) / (frame(79.4) - first))
        xy[k] = raw_xy[first] + w * (raw_xy[k] - raw_xy[first])
        yaw[k] = yaw[first] + w * (yaw[k] - yaw[first])
    for k in range(frame(83.0), last):
        w = smoothstep((last - k) / (last - frame(83.0)))
        xy[k] = end_xy + w * (raw_xy[k] - end_xy)
        yaw[k] = end_yaw + w * (yaw[k] - end_yaw)
    pos = np.column_stack((xy, np.full(len(xy), 0.001)))
    quat = Rotation.from_euler("z", yaw[:, None]).as_quat().astype(np.float32)
    return pos.astype(np.float32), quat, wrists


def contact_target(k: int, side: str, chair_pos: np.ndarray, chair_quat: np.ndarray,
                   human_wrists: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rot = Rotation.from_quat(chair_quat[k]).as_matrix()
    source = human_wrists[k, 0 if side == "left" else 1]
    local_x = np.clip((rot[:2, :2].T @ (source - chair_pos[k, :2]))[0], -0.23, 0.23)
    target = chair_pos[k].astype(np.float64) + rot @ np.array([local_x, CHAIR_BACK_CONTACT_Y, CHAIR_CONTACT_Z])
    direction = rot @ np.array([0.0, -0.72, -0.69])
    direction /= np.linalg.norm(direction)
    return target, direction, float(local_x)


def finger_factor(t: np.ndarray) -> np.ndarray:
    opening = smoothstep((t - 77.8) / 0.9)
    contact = smoothstep((t - 79.2) / 0.35)
    release = smoothstep((t - 83.4) / 0.85)
    return 1.0 - 0.86 * opening + 0.31 * contact + 0.55 * release


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"candidate exists: {OUT}")
    OUT.mkdir(parents=True)
    prior = MotionFields.load(HERE / "body_prior.npz", ORDER)
    with np.load(HERE / "human.npz", allow_pickle=False) as source:
        human = np.asarray(source["keypoints_w"], dtype=np.float64)
    if len(prior.joint_pos) != 800 or human.shape != (800, 20, 3):
        raise ValueError("armchair task must cover 75–91 s at 50 Hz")
    chair_pos, chair_quat, smooth_wrists = chair_path(human)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(name).qposadr[0] for name in prior.joint_names])
    joints = prior.joint_pos.astype(np.float64).copy()
    joint_time = START + np.arange(len(joints)) / FPS
    changed = {}
    residuals = {}
    for side in SIDES:
        names = tuple(f"{side}_{part}_joint" for part in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        ))
        cols = np.asarray([prior.joint_names.index(name) for name in names])
        arm_addr = addresses[cols]
        wrist = model.body(f"{side}_wrist_yaw_link").id
        lower = np.asarray([model.joint(name).range[0] for name in names]) + 1e-5
        upper = np.asarray([model.joint(name).range[1] for name in names]) - 1e-5
        wrist_limit = np.deg2rad(35)
        lower[4:] = np.maximum(lower[4:], -wrist_limit)
        upper[4:] = np.minimum(upper[4:], wrist_limit)
        keys = np.arange(frame(79.0), frame(83.4) + 1, 10)
        offsets = []
        errors = []
        previous = None
        for k in keys:
            baseline = prior.joint_pos[k, cols].astype(np.float64)
            data.qpos[:3] = prior.body_pos_w[k, 0]
            data.qpos[3:7] = prior.body_quat_w[k, 0]
            data.qpos[addresses] = prior.joint_pos[k]
            target, direction, _ = contact_target(k, side, chair_pos, chair_quat, smooth_wrists)
            lo = np.maximum(lower, baseline - np.array([0.45, 0.45, 0.45, 0.55, 0.30, 0.30, 0.30]))
            hi = np.minimum(upper, baseline + np.array([0.45, 0.45, 0.45, 0.55, 0.30, 0.30, 0.30]))
            seed = np.clip(baseline if previous is None else previous, lo + 1e-6, hi - 1e-6)

            def objective(q):
                data.qpos[arm_addr] = q
                mujoco.mj_forward(model, data)
                mat = data.xmat[wrist].reshape(3, 3)
                palm = data.xpos[wrist] + mat @ np.array([0.084, 0.0, 0.0])
                return np.r_[
                    45 * (palm - target), 2.3 * (mat[:, 0] - direction),
                    1.4 * (q - baseline),
                    0.3 * (q - previous) if previous is not None else np.zeros(7),
                ]

            solution = least_squares(objective, seed, bounds=(lo, hi), max_nfev=35)
            previous = solution.x.copy()
            offsets.append(solution.x - baseline)
            data.qpos[arm_addr] = solution.x
            mujoco.mj_forward(model, data)
            mat = data.xmat[wrist].reshape(3, 3)
            palm = data.xpos[wrist] + mat @ np.array([0.084, 0.0, 0.0])
            errors.append(float(np.linalg.norm(palm - target)))
        spline_frames = np.r_[frame(78.4), keys, frame(84.1)]
        spline_offsets = np.vstack((np.zeros(7), offsets, np.zeros(7)))
        correction = CubicSpline(spline_frames, spline_offsets, axis=0, bc_type="clamped")
        window = slice(frame(78.4), frame(84.1) + 1)
        joints[window, cols] += correction(np.arange(window.start, window.stop))
        joints[window, cols] = np.clip(joints[window, cols], lower, upper)
        changed[side] = float(np.max(np.abs(joints[:, cols] - prior.joint_pos[:, cols])))
        residuals[side] = {"max_m": float(max(errors)), "mean_m": float(np.mean(errors))}

    factor = finger_factor(joint_time)
    finger_cols = [i for i, name in enumerate(prior.joint_names) if "hand_" in name]
    joints[:, finger_cols] *= factor[:, None]
    velocity = np.gradient(joints, 1 / FPS, axis=0).astype(np.float32)
    candidate = replace(prior, joint_pos=joints.astype(np.float32), joint_vel=velocity)
    candidate = refresh_fk(candidate, MODEL)
    candidate.save(OUT / "motion.npz")
    np.savez_compressed(
        OUT / "object_estimate.npz", position_w=chair_pos, quaternion_xyzw=chair_quat,
        fps=np.asarray(FPS), source_range_s=np.asarray([75.0, 91.0]),
        trajectory_source=np.asarray("human_wrist_contact_fit_kinematic_no_physics"),
    )
    with np.load(ROOT / "retarget/ground_support_27_end/sidecars/human_motion_world.npz", allow_pickle=False) as data:
        begin, end = (round((t - 27.0) * FPS) for t in (75.0, 91.0))
        overlay = {key: (data[key][begin:end] if data[key].ndim and data[key].shape[0] == 5104 else data[key])
                   for key in data.files}
    overlay["source_range_s"] = np.asarray([75.0, 91.0])
    np.savez_compressed(OUT / "human_motion_world.npz", **overlay)
    shutil.copy2(HERE / "source_sync.npz", OUT / "source_sync.npz")
    review = {
        "stage": "3.2_kinematic_candidate_pending_visual_review",
        "source_window_s": [75.0, 91.0],
        "contact_type": "bilateral_nonprehensile_palm_push_on_outer_backrest",
        "source_video_observation": {
            "approach_s": [77.0, 79.0], "first_contact_s_estimate": [78.8, 79.3],
            "chair_move_s": [79.0, 83.4], "release_s": [83.3, 84.1],
            "right_table_support_preserved_s": [84.5, 88.8],
        },
        "chair_path_source": "smoothed original human wrists; fixed ground height; no observed object 6D track",
        "chair_initial_position_m": chair_pos[0].tolist(),
        "chair_final_position_m": chair_pos[-1].tolist(),
        "chair_initial_quaternion_xyzw": chair_quat[0].tolist(),
        "chair_final_quaternion_xyzw": chair_quat[-1].tolist(),
        "scene_reference_position_m": [-0.649999917, -1.25, 0.001],
        "scene_placement_difference_m": float(np.linalg.norm(chair_pos[0] - [-0.649999917, -1.25, 0.001])),
        "palm_contact_knot_residual_m": residuals,
        "max_arm_joint_change_rad": changed,
        "hand_command": {
            "open_start_s": 77.8, "mostly_open_s": 78.7,
            "partial_wrap_start_s": 79.2, "partial_wrap_complete_s": 79.55,
            "release_start_s": 83.4, "return_to_prior_s": 84.25,
            "status": "designed_reference_not_measured_actuator_response",
        },
        "source_hashes_sha256": {
            "body_prior.npz": sha256(HERE / "body_prior.npz"),
            "human.npz": sha256(HERE / "human.npz"),
            "source_sync.npz": sha256(HERE / "source_sync.npz"),
            "grey_armchair_v2.usd": sha256(Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/objects/1012/grey_armchair_v2.usd")),
        },
        "limitations": [
            "Initial chair placement is contact-fitted and needs calibrated scene/video review.",
            "Estimated chair motion is not measured or physically simulated.",
            "Palm contact and finger response require dynamics validation.",
        ],
    }
    (OUT / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "fit_residuals": residuals, "initial": chair_pos[0].tolist(),
                      "final": chair_pos[-1].tolist()}, indent=2))


if __name__ == "__main__":
    main()
