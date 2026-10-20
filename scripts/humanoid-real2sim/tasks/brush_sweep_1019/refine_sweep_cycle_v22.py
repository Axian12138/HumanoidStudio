"""Recover the human lift/far placement/table stroke/lift brush cycle.

Only the right arm over global 18.0–23.2 s changes. The accepted palm-side
grasp, knee/feet support, object attachment and return are inherited from v21.
"""

from dataclasses import replace
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


ROOT = example_dir(__file__)
SOURCE = ROOT / "retarget/brush_sweep_kinematic_v21_knee"
OUT = ROOT / "retarget/brush_sweep_kinematic_v22_cycle"
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
HUMAN = ROOT / "retarget/push_to_brush_kinematic_v3/human_motion_world.npz"
BRISTLES = ROOT.parent / "brush_pick_1012/assets/green_hand_brush/collision_hulls"
FPS = 50
GLOBAL_START, GLOBAL_STOP = 18.0, 23.2
LOCAL_START, LOCAL_STOP = round((GLOBAL_START - 8) * FPS), round((GLOBAL_STOP - 8) * FPS)


def smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def main():
    original = MotionFields.load(SOURCE / "motion.npz", ORDER)
    with np.load(SOURCE / "object_estimate.npz", allow_pickle=False) as file:
        old_object = {name: file[name] for name in file.files}
    with np.load(HUMAN, allow_pickle=False) as file:
        human_names = list(file["joint_names"])
        human_hand = file["positions_w"][:, human_names.index("right_hand_aux")].astype(float)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(name).qposadr[0] for name in original.joint_names])
    arm_cols = np.asarray([original.joint_names.index("right_" + part + "_joint") for part in
                           ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                            "wrist_roll", "wrist_pitch", "wrist_yaw")])
    arm_addresses = addresses[arm_cols]
    bounds = np.asarray([model.joint(original.joint_names[i]).range for i in arm_cols]).T
    bounds[0, 4:] = np.maximum(bounds[0, 4:], -np.deg2rad(35))
    bounds[1, 4:] = np.minimum(bounds[1, 4:], np.deg2rad(35))
    wrist_id = model.body("right_wrist_yaw_link").id
    wrist_motion = original.body_names.index("right_wrist_yaw_link")
    wrist_rot = Rotation.from_quat(original.body_quat_w[:, wrist_motion][:, [1, 2, 3, 0]])
    anchor = round((18.0 - 8) * FPS)
    local_pos = wrist_rot[anchor].inv().apply(
        old_object["position_w"][anchor] - original.body_pos_w[anchor, wrist_motion])
    local_rot = wrist_rot[anchor].inv() * Rotation.from_quat(old_object["quaternion_xyzw"][anchor])
    vertices = np.concatenate([trimesh.load(path, force="mesh").vertices
                               for path in sorted(BRISTLES.glob("Hull_bristles*.obj"))])
    hand = gaussian_filter1d(human_hand, sigma=2, axis=0)
    xy_offset = old_object["position_w"][anchor, :2] - hand[round(18.0 * FPS), :2]
    # The human hand at the near stroke (20.8–21.0 s) defines contact height.
    low_hand_z = float(np.median(hand[round(20.8 * FPS):round(21.0 * FPS), 2]))
    frames = np.unique(np.r_[LOCAL_START, np.arange(LOCAL_START + 5, LOCAL_STOP, 5), LOCAL_STOP])
    solved = []
    frame_reports = []
    last = None
    for k in frames:
        source_frame = k + 400
        target_xy = hand[source_frame, :2] + xy_offset
        clearance = float(np.clip(0.006 + hand[source_frame, 2] - low_hand_z, 0.006, 0.19))
        t = source_frame / FPS
        table_stroke = smooth((t - 19.65) / .35) * smooth((21.2 - t) / .2)
        clearance = (1 - table_stroke) * clearance + table_stroke * .004
        weight = smooth((k - LOCAL_START) / 12) * smooth((LOCAL_STOP - k) / 70)
        prior = original.joint_pos[k, arm_cols].astype(float)
        data.qpos[:3] = original.body_pos_w[k, 0]
        data.qpos[3:7] = original.body_quat_w[k, 0]
        data.qpos[addresses] = original.joint_pos[k]
        target_rot = wrist_rot[k]

        def measure(angles):
            data.qpos[arm_addresses] = angles
            mujoco.mj_forward(model, data)
            rot = Rotation.from_matrix(data.xmat[wrist_id].reshape(3, 3))
            pos = data.xpos[wrist_id] + rot.apply(local_pos)
            brush_rot = rot * local_rot
            bristle_world = brush_rot.apply(vertices) + pos
            return pos, rot, float(bristle_world[:, 2].min())

        def residual(angles):
            pos, rot, min_z = measure(angles)
            return np.r_[weight * 70 * (pos[:2] - target_xy),
                         weight * 100 * (min_z - (0.401 + clearance)),
                         1.0 * (target_rot.inv() * rot).as_rotvec(),
                         (.08 + 2.0 * (1 - weight)) * (angles - prior),
                         .60 * (angles - last) if last is not None else np.zeros(7)]

        if weight < 1e-4:
            answer = prior
        else:
            local_bounds = bounds.copy()
            if last is not None:
                local_bounds[0] = np.maximum(local_bounds[0], last - .42)
                local_bounds[1] = np.minimum(local_bounds[1], last + .42)
            seeds = [np.clip(prior, local_bounds[0] + 1e-6, local_bounds[1] - 1e-6)]
            if last is not None:
                seeds.insert(0, np.clip(last, local_bounds[0] + 1e-6, local_bounds[1] - 1e-6))
            fits = [least_squares(residual, seed, bounds=local_bounds, max_nfev=90) for seed in seeds]
            answer = min(fits, key=lambda fit: np.linalg.norm(fit.fun)).x
        pos, _, min_z = measure(answer)
        solved.append(answer - prior)
        last = answer
        frame_reports.append({"global_s": source_frame / FPS, "target_xy_m": target_xy.tolist(),
                              "brush_xy_m": pos[:2].tolist(), "target_bristle_clearance_m": clearance,
                              "bristle_clearance_m": min_z - .401})
    joints = original.joint_pos.astype(float).copy()
    indices = np.arange(LOCAL_START, LOCAL_STOP + 1)
    correction = PchipInterpolator(frames, np.asarray(solved), axis=0)(indices)
    filtered = gaussian_filter1d(correction, sigma=8, axis=0)
    filter_weight = smooth((indices - round((22.0 - 8) * FPS)) / 20)
    correction = correction * (1 - filter_weight[:, None]) + filtered * filter_weight[:, None]
    correction *= (smooth((indices - LOCAL_START) / 12)
                   * smooth((LOCAL_STOP - indices) / 25))[:, None]
    joints[np.ix_(indices, arm_cols)] += correction
    joints[np.ix_(indices, arm_cols)] = np.clip(joints[np.ix_(indices, arm_cols)], bounds[0], bounds[1])
    # Project the interpolated arm onto a per-frame velocity band while the
    # unchanged poses at both window boundaries remain fixed.
    for col in arm_cols:
        values = joints[indices, col].copy()
        for _ in range(6):
            for i in range(1, len(values) - 1):
                values[i] = np.clip(values[i], values[i - 1] - .119, values[i - 1] + .119)
            for i in range(len(values) - 2, 0, -1):
                values[i] = np.clip(values[i], values[i + 1] - .119, values[i + 1] + .119)
        joints[indices, col] = values
    candidate = refresh_fk(replace(original, joint_pos=joints), MODEL)
    position = old_object["position_w"].copy()
    orientation = old_object["quaternion_xyzw"].copy()
    new_rot = Rotation.from_quat(candidate.body_quat_w[indices, wrist_motion][:, [1, 2, 3, 0]])
    position[indices] = candidate.body_pos_w[indices, wrist_motion] + new_rot.apply(local_pos)
    orientation[indices] = (new_rot * local_rot).as_quat()
    OUT.mkdir(parents=True, exist_ok=True)
    candidate.save(OUT / "motion.npz")
    extra = {name: value for name, value in old_object.items() if name not in ("position_w", "quaternion_xyzw")}
    np.savez_compressed(OUT / "object_estimate.npz", position_w=position,
                        quaternion_xyzw=orientation, **extra)
    speed = float(np.max(np.abs(np.diff(joints[:, arm_cols], axis=0)) * FPS))
    report = {"status": "kinematic_cycle_candidate", "source_motion_sha256": sha256(SOURCE / "motion.npz"),
              "motion_sha256": sha256(OUT / "motion.npz"), "window_global_s": [GLOBAL_START, GLOBAL_STOP],
              "human_hand_xy_to_brush_xy_offset_m": xy_offset.tolist(),
              "human_near_stroke_hand_z_m": low_hand_z, "max_arm_joint_speed_rad_s": speed,
              "sampled_fit": frame_reports, "stage4_stage5": "skipped_by_user"}
    (OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "sampled_fit"}, indent=2))


if __name__ == "__main__":
    main()
