"""Archive the approved Stage 3 kinematic push without physics rollout data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
import yaml


TASK = example_dir(__file__)
WORKSPACE = TASK.parents[3]
SCALETRACK = WORKSPACE / "ScaleTrack"
ARCHIVE_REL = "data/1019test/push-bin-kinematic-v1"
SOURCE_MOTION = TASK / "retarget/hoi_candidate_scene.npz"
SOURCE_OBJECT = TASK / "sidecars/white_waste_bin_estimated.npz"
SOURCE_SYNC = TASK / "sidecars/source_sync.npz"
SOURCE_CONTACT = TASK / "sidecars/contact_schedule.json"
ORDER = SCALETRACK / "retarget3/assets/robot/motion_order.json"
ROBOT_FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w", "fps",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"archive already exists: {destination}")
    order = json.loads(ORDER.read_text())
    with np.load(SOURCE_MOTION, allow_pickle=False) as data:
        if set(data.files) != set(ROBOT_FIELDS):
            raise ValueError(f"unexpected robot motion fields: {data.files}")
        robot = {name: data[name].copy() for name in ROBOT_FIELDS}
    with np.load(SOURCE_OBJECT, allow_pickle=False) as data:
        object_track = {name: data[name].copy() for name in data.files}
    with np.load(SOURCE_SYNC, allow_pickle=False) as data:
        source_time = data["motion_time_s"].copy()

    count = len(robot["joint_pos"])
    fps = int(robot["fps"])
    if fps != 50 or count != 474:
        raise ValueError(f"expected 474 frames at 50 Hz, got {count} at {fps} Hz")
    if robot["joint_pos"].shape != (count, len(order["robot_joint_names"])):
        raise ValueError("joint motion and named order differ")
    if robot["body_pos_w"].shape != (count, len(order["robot_body_names"]), 3):
        raise ValueError("body motion and named order differ")
    if any(not np.isfinite(robot[name]).all() for name in ROBOT_FIELDS if name != "fps"):
        raise ValueError("robot motion contains non-finite values")
    if not np.allclose(np.linalg.norm(robot["body_quat_w"], axis=-1), 1, atol=2e-3):
        raise ValueError("robot body quaternion norm is invalid")
    if str(object_track["object_name"].item()) != "white_waste_bin":
        raise ValueError("wrong target object")
    if not bool(object_track["pose_is_estimated"].item()):
        raise ValueError("the kinematic trajectory must be explicitly marked estimated")
    for key, width in (
        ("position_w", 3), ("quaternion_wxyz", 4),
        ("linear_velocity_w", 3), ("angular_velocity_w", 3),
    ):
        if object_track[key].shape != (count, width) or not np.isfinite(object_track[key]).all():
            raise ValueError(f"invalid object {key}")
    if not np.allclose(np.linalg.norm(object_track["quaternion_wxyz"], axis=-1), 1, atol=2e-3):
        raise ValueError("object quaternion norm is invalid")
    timestamps = np.arange(count, dtype=np.float64) / fps
    if not np.allclose(object_track["timestamps"], timestamps) or not np.allclose(source_time, timestamps):
        raise ValueError("object/source timestamps do not align with motion")

    destination.mkdir(parents=True)
    metadata = destination / "metadata"
    metadata.mkdir()
    shutil.copy2(ORDER, metadata / "motion_order.json")
    shutil.copy2(SOURCE_SYNC, metadata / "source_sync.npz")
    shutil.copy2(SOURCE_CONTACT, metadata / "contact_schedule.json")
    object_arrays = {
        "object_names": np.asarray(["white_waste_bin"]),
        "object_pos_w": object_track["position_w"][:, None, :],
        "object_quat_w": object_track["quaternion_wxyz"][:, None, :],
        "object_lin_vel_w": object_track["linear_velocity_w"][:, None, :],
        "object_ang_vel_w": object_track["angular_velocity_w"][:, None, :],
        "timestamps": timestamps,
        "source_motion_frame": np.arange(count, dtype=np.int64),
        "interacting_object_names": np.asarray(["white_waste_bin"]),
        "object_trajectory_source": np.asarray(object_track["trajectory_source"]),
        "object_trajectory_measured": np.asarray(False),
    }
    np.savez_compressed(destination / "motion.npz", **robot, **object_arrays)
    np.savez_compressed(
        destination / "white_waste_bin_trajectory.npz",
        object_name=np.asarray("white_waste_bin"),
        fps=np.asarray(fps),
        timestamps=timestamps,
        source_motion_frame=np.arange(count, dtype=np.int64),
        position_w=object_track["position_w"],
        quaternion_wxyz=object_track["quaternion_wxyz"],
        linear_velocity_w=object_track["linear_velocity_w"],
        angular_velocity_w=object_track["angular_velocity_w"],
        kinematic_pose_defined=np.ones(count, dtype=bool),
        measured_gt_valid=np.zeros(count, dtype=bool),
        trajectory_source=np.asarray(object_track["trajectory_source"]),
    )

    base = ARCHIVE_REL
    bin_usd = "source/scaletrack/scaletrack/assets/objects/1012/white_waste_bin_v2.usd"
    room_usd = (
        "source/scaletrack/scaletrack/assets/scenes/tianze_sim_data/"
        "IKEA_LivingRoom_Mesh_1012/IKEA_LivingRoom_Mesh_1012_drawer_renderable_v4.usd"
    )
    scene_setup = {
        "version": 1,
        "layout_version": "v4",
        "status": "stage3_hoi_kinematic_reference",
        "runtime_ready": False,
        "runtime_note": "Kinematic reference only; tracker and object contact dynamics are not accepted.",
        "recording_id": "1-2026-10-19-16-31-26",
        "scene_placement_file": "source/scaletrack/scaletrack/assets/scenes/tianze_sim_data/IKEA_LivingRoom_Mesh_1012/initial_poses_v4.json",
        "canonical_assets_file": "source/scaletrack/scaletrack/assets/scenes/tianze_sim_data/IKEA_LivingRoom_Mesh_1012/canonical_assets_v4.json",
        "scene": {
            "usd_path": room_usd,
            "blend_path": "source/scaletrack/scaletrack/assets/scenes/tianze_sim_data/IKEA_LivingRoom_Mesh_1012/LivingRoom_1012_v4.blend",
            "coordinate_frame": "IKEA_1012_world",
            "world_translation_m": [0.0, 0.0, 0.0],
            "world_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "coffee_table_prim_path": "/World/MeasuredFurniture/CoffeeTable",
            "coffee_table_role": "static blocking surface at its authored room pose",
        },
        "object": {
            "id": "white_waste_bin",
            "type": "white_waste_bin",
            "asset_kind": "rigid_object",
            "physics_mode": "dynamic",
            "usd_path": bin_usd,
            "blend_path": "source/scaletrack/scaletrack/assets/objects/1012/white_waste_bin_v2.blend",
            "scale": [1.0, 1.0, 1.0],
            "initial_position_m": object_track["position_w"][0].tolist(),
            "initial_position_frame": "world",
            "initial_orientation_wxyz": object_track["quaternion_wxyz"][0].tolist(),
            "canonical_frame": "native_asset_default_prim",
            "scene_field": "white_waste_bin",
            "prim_name": "white_waste_bin",
            "trajectory_kind": "estimated_kinematic_reference",
        },
    }
    entry = {
        "path": f"{base}/motion.npz",
        "weight": 1.0,
        "source_recording_id": "1-2026-10-19-16-31-26",
        "motion_fps": float(fps),
        "source_frame_count": count,
        "duration_seconds": count / fps,
        "scene_setup": scene_setup,
        "accepted": False,
        "training_data_ready": False,
        "visual_gt": "not_generated",
        "trajectory_kind": "estimated_kinematic_reference",
        "original_source_frame_count": 1141,
        "slice_interval_seconds": [0.0, 9.5],
        "object_trajectories": {
            "white_waste_bin": {
                "path": f"{base}/white_waste_bin_trajectory.npz",
                "npz_object_index": 0,
                "interacts_in_this_clip": True,
                "coordinate_frame": "IKEA_1012_world",
                "quaternion_order": "wxyz",
                "measured": False,
                "source": str(object_track["trajectory_source"].item()),
            }
        },
        "interaction_events": [
            {
                "object": "white_waste_bin",
                "effector": "left_hand_thumb_index_web",
                "approach_seconds": [2.0, 3.5],
                "contact_seconds_estimated": [3.5, 7.5],
                "release_seconds_estimated": [7.5, 8.5],
                "timing_source": f"{base}/metadata/contact_schedule.json",
            }
        ],
    }
    (destination / "motions.yaml").write_text(
        yaml.safe_dump({"hoi1019_push_bin_kinematic_v1": entry}, sort_keys=False, allow_unicode=True)
    )

    provenance = {
        "status": "kinematic_archive_only",
        "source_recording_id": "1-2026-10-19-16-31-26",
        "source_mcap": "data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap",
        "source_motion": str(SOURCE_MOTION),
        "source_object_trajectory": str(SOURCE_OBJECT),
        "source_sync": str(SOURCE_SYNC),
        "source_contact_schedule": str(SOURCE_CONTACT),
        "source_motion_sha256": sha256(SOURCE_MOTION),
        "source_object_trajectory_sha256": sha256(SOURCE_OBJECT),
        "source_sync_sha256": sha256(SOURCE_SYNC),
        "source_contact_schedule_sha256": sha256(SOURCE_CONTACT),
        "notes": [
            "Robot motion is the Stage 3 retarget/HOI kinematic candidate, not a tracker rollout.",
            "Object trajectory is estimated from source video and left-wrist support prior, not physics or measured ground truth.",
            "Room and coffee table use the existing IKEA 1012 v4 USD; the canonical bin USD is referenced, not modified.",
        ],
    }
    outputs = (destination / "motion.npz", destination / "white_waste_bin_trajectory.npz", destination / "motions.yaml")
    provenance["output_sha256"] = {path.name: sha256(path) for path in outputs}
    (metadata / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (metadata / "validation.json").write_text(json.dumps({
        "valid": True,
        "frames": count,
        "fps": fps,
        "joint_count": len(order["robot_joint_names"]),
        "body_count": len(order["robot_body_names"]),
        "object_count": 1,
        "object_pose_is_estimated": True,
        "visual_gt_generated": False,
        "physics_rollout_used": False,
        "robot_quaternion_order": "wxyz",
        "object_quaternion_order": "wxyz",
        "world_frame": "IKEA_1012_world",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination.resolve())
