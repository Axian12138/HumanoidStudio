"""Append the new brush candidate to the already reviewed 1019 bin push.

The old 0–9.46 s prefix is copied verbatim in joint/root coordinates. A short
position/velocity correction is applied only to the new, non-contact suffix.
All object paths remain labeled kinematic estimates.
"""

from dataclasses import replace
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


ROOT = example_dir(__file__)
PUSH = ROOT.parent / "push_bin_1019"
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
NEW = ROOT / "retarget/brush_sweep_kinematic_v19"
OUT = ROOT / "retarget/push_to_brush_kinematic_v2"
NEW_START = 74  # 8.00 s + 74/50 = 9.48 s, after push frame 473 at 9.46 s
FADE_FRAMES = 30
DT = 1 / 50


def hermite_correction(position_delta, velocity_delta, count):
    s = np.arange(count) / (count - 1)
    duration = (count - 1) * DT
    h00 = 1 - 3 * s**2 + 2 * s**3
    h10 = s - 2 * s**2 + s**3
    return h00[:, None] * position_delta + h10[:, None] * duration * velocity_delta


def main():
    prefix_path = PUSH / "retarget/hoi_candidate_scene.npz"
    suffix_path = NEW / "motion.npz"
    prefix = MotionFields.load(prefix_path, ORDER)
    suffix = MotionFields.load(suffix_path, ORDER)
    if prefix.joint_names != suffix.joint_names or prefix.body_names != suffix.body_names:
        raise ValueError("Joint/body order mismatch")
    if len(prefix.joint_pos) != 474 or len(suffix.joint_pos) != 999:
        raise ValueError("Source frame counts changed")
    joints = suffix.joint_pos[NEW_START:].astype(float).copy()
    roots = suffix.body_pos_w[NEW_START:, 0].astype(float).copy()
    quats = suffix.body_quat_w[NEW_START:, 0].astype(float).copy()
    old_root = prefix.body_pos_w[:, 0].astype(float)
    old_joint = prefix.joint_pos.astype(float)
    old_root_velocity = (old_root[-1] - old_root[-2]) / DT
    old_joint_velocity = (old_joint[-1] - old_joint[-2]) / DT
    new_root_velocity = (roots[1] - roots[0]) / DT
    new_joint_velocity = (joints[1] - joints[0]) / DT
    roots[:FADE_FRAMES] += hermite_correction(old_root[-1] + DT * old_root_velocity - roots[0],
                                               old_root_velocity - new_root_velocity, FADE_FRAMES)
    joints[:FADE_FRAMES] += hermite_correction(old_joint[-1] + DT * old_joint_velocity - joints[0],
                                                old_joint_velocity - new_joint_velocity, FADE_FRAMES)
    old_rotation = Rotation.from_quat(prefix.body_quat_w[-1, 0, [1, 2, 3, 0]])
    old_previous = Rotation.from_quat(prefix.body_quat_w[-2, 0, [1, 2, 3, 0]])
    predicted_rotation = old_rotation * (old_previous.inv() * old_rotation)
    initial_new_rotation = Rotation.from_quat(quats[0, [1, 2, 3, 0]])
    correction = (initial_new_rotation.inv() * predicted_rotation).as_rotvec()
    for k in range(FADE_FRAMES):
        s = k / (FADE_FRAMES - 1)
        weight = 1 - 3 * s**2 + 2 * s**3
        rotation = Rotation.from_quat(quats[k, [1, 2, 3, 0]]) * Rotation.from_rotvec(correction * weight)
        quats[k] = rotation.as_quat()[[3, 0, 1, 2]]

    joined = replace(
        suffix,
        joint_pos=np.concatenate([prefix.joint_pos, joints]),
        joint_vel=np.concatenate([prefix.joint_vel, suffix.joint_vel[NEW_START:]]),
        body_pos_w=np.concatenate([prefix.body_pos_w, suffix.body_pos_w[NEW_START:]]),
        body_quat_w=np.concatenate([prefix.body_quat_w, suffix.body_quat_w[NEW_START:]]),
        body_lin_vel_w=np.concatenate([prefix.body_lin_vel_w, suffix.body_lin_vel_w[NEW_START:]]),
        body_ang_vel_w=np.concatenate([prefix.body_ang_vel_w, suffix.body_ang_vel_w[NEW_START:]]),
    )
    joined.body_pos_w[len(prefix.joint_pos):, 0] = roots
    joined.body_quat_w[len(prefix.joint_pos):, 0] = quats
    joined = refresh_fk(joined, MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    joined.save(OUT / "motion.npz")

    with np.load(PUSH / "sidecars/white_waste_bin_estimated.npz") as old_bin:
        bin_pos = np.concatenate([old_bin["position_w"], np.repeat(old_bin["position_w"][-1:], len(joints), axis=0)])
        bin_quat = np.concatenate([old_bin["quaternion_wxyz"], np.repeat(old_bin["quaternion_wxyz"][-1:], len(joints), axis=0)])
    with np.load(NEW / "object_estimate.npz") as new_brush:
        brush_pos = np.concatenate([np.repeat(new_brush["position_w"][:1], len(prefix.joint_pos), axis=0),
                                    new_brush["position_w"][NEW_START:]])
        brush_quat = np.concatenate([np.repeat(new_brush["quaternion_xyzw"][:1], len(prefix.joint_pos), axis=0),
                                     new_brush["quaternion_xyzw"][NEW_START:]])
    np.savez_compressed(OUT / "objects_estimated.npz", white_waste_bin_position_w=bin_pos,
                        white_waste_bin_quaternion_wxyz=bin_quat,
                        green_hand_brush_position_w=brush_pos,
                        green_hand_brush_quaternion_xyzw=brush_quat,
                        status=np.array("kinematic_estimates_not_physics"))
    with np.load(PUSH / "sidecars/source_sync.npz") as old_sync, np.load(ROOT / "source/source_sync_8_28.npz") as new_sync:
        shared = sorted(set(old_sync.files) & set(new_sync.files))
        sync = {name: np.concatenate([old_sync[name], new_sync[name][NEW_START:]]) for name in shared}
    sync["motion_time_s"] = np.arange(len(joined.joint_pos)) / 50
    np.savez_compressed(OUT / "source_sync.npz", **sync)
    boundary = len(prefix.joint_pos)
    root_step = float(np.linalg.norm(joined.body_pos_w[boundary, 0] - joined.body_pos_w[boundary - 1, 0]))
    joint_step = float(np.max(abs(joined.joint_pos[boundary] - joined.joint_pos[boundary - 1])))
    rotation_step = Rotation.from_quat(joined.body_quat_w[boundary - 1, 0, [1, 2, 3, 0]]).inv() * Rotation.from_quat(joined.body_quat_w[boundary, 0, [1, 2, 3, 0]])
    report = {
        "status": "kinematic_candidate_unreviewed",
        "frames": len(joined.joint_pos), "fps": 50, "source_window_s_half_open": [0, len(joined.joint_pos) / 50],
        "splice_frame": boundary, "splice_time_s": boundary / 50,
        "source_push_sha256": sha256(prefix_path), "source_brush_sha256": sha256(suffix_path),
        "combined_motion_sha256": sha256(OUT / "motion.npz"),
        "unmodified_push_joint_root_frames": boundary,
        "new_suffix_fade_frames": FADE_FRAMES,
        "splice_root_step_m": root_step,
        "splice_max_joint_step_rad": joint_step,
        "splice_root_angle_step_deg": float(np.degrees(rotation_step.magnitude())),
        "bin_track_status": "previous kinematic estimate held constant after push",
        "brush_track_status": "rigid kinematic estimate until release; not physical grasp",
    }
    (OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
