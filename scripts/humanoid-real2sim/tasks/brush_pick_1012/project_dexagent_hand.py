"""Project a hand-only brush proposal onto the existing seven-joint right arm.

Projection is for RSS/ScaleBridge validation. It does not optimize full-body qpos.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

ROOT = example_dir(__file__)
SOURCE = ROOT / "retarget/handle_kinematic_v1/motion.npz"
HAND = ROOT / "retarget/dexagent_hand_only_v1/hand_targets.npz"
OUT = ROOT / "retarget/dexagent_hand_only_v1"
MODEL = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/robot.xml")
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
KEYS = (450, 480, 500, 518, 525, 545, 560, 575, 600, 625, 660)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def main():
    motion = MotionFields.load(SOURCE, ORDER)
    targets = np.load(HAND)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    all_addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    arm_cols = np.array([motion.joint_names.index("right_" + n + "_joint") for n in ARM])
    arm_addr = all_addr[arm_cols]
    lo = np.array([model.joint(motion.joint_names[i]).range[0] for i in arm_cols])
    hi = np.array([model.joint(motion.joint_names[i]).range[1] for i in arm_cols])
    wrist = model.body("right_wrist_yaw_link").id
    key_delta = []
    residuals = []
    last = None
    for frame in KEYS:
        index = frame - 450
        data.qpos[:3] = motion.body_pos_w[frame, 0]
        data.qpos[3:7] = motion.body_quat_w[frame, 0]
        data.qpos[all_addr] = motion.joint_pos[frame]
        prior = motion.joint_pos[frame, arm_cols].astype(float)
        p_target = targets["wrist_position_w"][index]
        r_target = Rotation.from_quat(targets["wrist_quaternion_xyzw"][index])

        def residual(q):
            data.qpos[arm_addr] = q
            mujoco.mj_forward(model, data)
            r = Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))
            return np.r_[100 * (data.xpos[wrist] - p_target),
                         3 * (r_target * r.inv()).as_rotvec(), 0.15 * (q - prior)]

        seeds = [prior] + ([last] if last is not None else [])
        results = [least_squares(residual, np.clip(s, lo + 1e-5, hi - 1e-5),
                                 bounds=(lo + 1e-5, hi - 1e-5), max_nfev=150) for s in seeds]
        best = min(results, key=lambda s: np.linalg.norm(s.fun))
        data.qpos[arm_addr] = best.x
        mujoco.mj_forward(model, data)
        residuals.append({"frame": frame, "wrist_position_error_m": float(np.linalg.norm(data.xpos[wrist] - p_target)),
                          "wrist_angle_error_rad": float(np.linalg.norm((r_target * Rotation.from_matrix(data.xmat[wrist].reshape(3, 3)).inv()).as_rotvec()))})
        key_delta.append(best.x - prior)
        last = best.x
    joint = motion.joint_pos.astype(float).copy()
    frames = np.arange(450, 661)
    delta = PchipInterpolator(KEYS, np.stack(key_delta), axis=0)(frames)
    joint[np.ix_(frames, arm_cols)] += delta
    finger_names = [str(n) for n in targets["finger_joint_names"]]
    finger_cols = np.array([motion.joint_names.index(n) for n in finger_names])
    hand_q = targets["finger_joint_q"]
    hand_return = smooth((frames - 600) / 60)
    joint[np.ix_(frames, finger_cols)] = hand_q * (1 - hand_return[:, None]) + motion.joint_pos[np.ix_(frames, finger_cols)] * hand_return[:, None]
    candidate = refresh_fk(replace(motion, joint_pos=joint), MODEL)
    candidate.save(OUT / "projected_motion.npz")
    speed = np.abs(np.diff(joint, axis=0)) * motion.fps
    report = {"status": "projected_motion_unreviewed", "source_motion_sha256": sha256(SOURCE),
              "hand_targets_sha256": sha256(HAND), "motion_sha256": sha256(OUT / "projected_motion.npz"),
              "projected_keyframes": residuals, "max_wrist_position_error_m": max(r["wrist_position_error_m"] for r in residuals),
              "max_wrist_angle_error_rad": max(r["wrist_angle_error_rad"] for r in residuals),
              "max_changed_joint_speed_rad_s": float(np.max(speed[449:661, np.r_[arm_cols, finger_cols]])),
              "root_legs_torso_left_arm_unchanged": True,
              "physics_grasp_verified": False}
    (OUT / "projection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "max_wrist_position_error_m", "max_wrist_angle_error_rad", "max_changed_joint_speed_rad_s")}, indent=2))


if __name__ == "__main__":
    main()
