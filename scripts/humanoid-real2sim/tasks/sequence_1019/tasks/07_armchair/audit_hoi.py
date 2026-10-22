"""Audit the 75–91 s armchair HOI kinematic candidate and its source alignment."""

from __future__ import annotations

import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

from fit_hoi import HERE, MODEL, ORDER, OUT, chair_path, contact_target, frame


def obj_vertices(path):
    return np.asarray([
        [float(value) for value in line.split()[1:4]]
        for line in path.open() if line.startswith("v ")
    ])


def main() -> None:
    prior = MotionFields.load(HERE / "body_prior.npz", ORDER)
    motion = MotionFields.load(OUT / "motion.npz", ORDER)
    with np.load(HERE / "human.npz", allow_pickle=False) as data:
        human = data["keypoints_w"]
    with np.load(OUT / "human_motion_world.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["positions_w"], human)
    with np.load(OUT / "source_sync.npz", allow_pickle=False) as data:
        recording_time = data["recording_time_s"]
        assert len(recording_time) == 800
        np.testing.assert_allclose(recording_time, 75.0 + np.arange(800) / 50, atol=0.02)
    with np.load(OUT / "object_estimate.npz", allow_pickle=False) as data:
        chair_pos = data["position_w"]
        chair_quat = data["quaternion_xyzw"]
    _, _, wrists = chair_path(human)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    state = mujoco.MjData(model)
    addresses = np.asarray([model.joint(name).qposadr[0] for name in motion.joint_names])
    contact = {}
    for side in ("left", "right"):
        body = model.body(f"{side}_wrist_yaw_link").id
        distances, angles = [], []
        for k in range(frame(79.2), frame(83.2) + 1):
            state.qpos[:3] = motion.body_pos_w[k, 0]
            state.qpos[3:7] = motion.body_quat_w[k, 0]
            state.qpos[addresses] = motion.joint_pos[k]
            mujoco.mj_forward(model, state)
            matrix = state.xmat[body].reshape(3, 3)
            palm = state.xpos[body] + matrix @ np.array([0.084, 0.0, 0.0])
            target, direction, _ = contact_target(k, side, chair_pos, chair_quat, wrists)
            distances.append(float(np.linalg.norm(palm - target)))
            angles.append(float(np.degrees(np.arccos(np.clip(matrix[:, 0] @ direction, -1, 1)))))
        contact[side] = {
            "palm_to_back_patch_m_max": max(distances),
            "palm_to_back_patch_m_mean": float(np.mean(distances)),
            "palm_axis_error_deg_max": max(angles),
        }
    difference = motion.joint_pos - prior.joint_pos
    assert np.max(np.abs(difference[:frame(77.8)])) == 0
    assert np.max(np.abs(difference[frame(84.5):])) == 0
    protected = [i for i, name in enumerate(motion.joint_names)
                 if all(part not in name for part in ("shoulder", "elbow", "wrist", "hand_"))]
    assert np.max(np.abs(difference[:, protected])) == 0
    limits = np.asarray([model.joint(name).range for name in motion.joint_names])
    violation = np.maximum(limits[:, 0] - motion.joint_pos, motion.joint_pos - limits[:, 1])
    max_violation = max(0.0, float(np.max(violation)))
    assert max_violation < 1e-4
    speed = np.gradient(motion.joint_pos, 1 / 50, axis=0)
    acceleration = np.gradient(speed, 1 / 50, axis=0)
    yaw = Rotation.from_quat(chair_quat).as_euler("xyz")[:, 2]
    chair_speed = np.linalg.norm(np.diff(chair_pos[:, :2], axis=0) * 50, axis=1)
    chair_yaw_speed = np.abs(np.diff(yaw)) * 50
    assert np.max(np.abs(chair_pos[:, 2] - 0.001)) < 1e-6
    vertices = obj_vertices(HERE / "grey_armchair_visual.obj")
    table_min = np.array([-1.03, -2.97068505])
    table_max = np.array([0.57000405, -2.11905405])
    elevated_table_overlap = 0
    min_chair_z = np.inf
    for k in range(0, 800, 5):
        world = Rotation.from_quat(chair_quat[k]).apply(vertices) + chair_pos[k]
        min_chair_z = min(min_chair_z, float(world[:, 2].min()))
        inside_xy = np.all((world[:, :2] >= table_min) & (world[:, :2] <= table_max), axis=1)
        elevated_table_overlap += int(np.count_nonzero(inside_xy & (world[:, 2] >= 0.748)))
    assert min_chair_z >= 0
    audit = {
        "status": "kinematic_candidate_pending_visual_and_physical_review",
        "frames": 800,
        "fps": 50,
        "source_range_s": [75.0, 91.0],
        "human_overlay_matches_task_human": True,
        "source_sync_matches_recording_clock": True,
        "contact_window_s": [79.2, 83.2],
        "contact": contact,
        "max_joint_limit_violation_rad": max_violation,
        "max_joint_speed_rad_s": float(np.max(np.abs(speed))),
        "max_joint_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "max_arm_joint_change_rad": {
            side: float(np.max(np.abs(difference[:, [i for i, name in enumerate(motion.joint_names)
                                                    if name.startswith(f"{side}_") and
                                                    any(part in name for part in ("shoulder", "elbow", "wrist"))]])))
            for side in ("left", "right")
        },
        "root_legs_waist_unchanged": True,
        "right_table_support_84_5_to_88_8_unchanged": True,
        "chair_motion": {
            "initial_position_m": chair_pos[0].tolist(),
            "final_position_m": chair_pos[-1].tolist(),
            "horizontal_displacement_m": float(np.linalg.norm(chair_pos[-1, :2] - chair_pos[0, :2])),
            "max_horizontal_speed_m_s": float(chair_speed.max()),
            "initial_yaw_deg": float(np.degrees(yaw[0])),
            "final_yaw_deg": float(np.degrees(yaw[-1])),
            "max_yaw_rate_deg_s": float(np.degrees(chair_yaw_speed.max())),
            "minimum_visual_mesh_height_m": min_chair_z,
            "visual_vertices_at_tabletop_height_inside_table_xy_proxy": elevated_table_overlap,
            "trajectory_kind": "human_wrist_fitted_ground_supported_estimate",
        },
        "limitations": [
            "Table overlap uses XY bounds and tabletop height, not continuous mesh collision.",
            "Fitted chair placement and path need source-calibrated visual confirmation.",
            "Kinematic palm proximity is not measured force or dynamic chair movement.",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
