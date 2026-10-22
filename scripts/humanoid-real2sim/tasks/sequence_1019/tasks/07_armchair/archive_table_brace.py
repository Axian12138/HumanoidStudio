"""Archive the visually reviewed tabletop brace as a standalone kinematic clip."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


REPO = Path(__file__).resolve().parents[6]
SCALETRACK = Path("/home/galbot/workspace/ScaleTrack")
RUN = REPO / (
    "outputs/humanoid-real2sim/2026-10-20/sequence_1019__armchair_split_feedback/"
    "runs/16-57-32.904219+0800"
)
SOURCE = RUN / "table_support"
TASK = REPO / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot"
PUSH = SCALETRACK / "data/1019test/push-bin-kinematic-v1"
FULL_SCENE = SCALETRACK / "data/1012test/1012-1/stage7-visual-gt/motions.yaml"
PACKAGE = "brace-dining-table-kinematic-v1"
PACKAGE_ID = "hoi1019_brace_dining_table_kinematic_v1"
OBJECT = "grey_armchair"
START_S, END_S = 83.5, 91.0
FPS = 50
FRAME_COUNT = 375


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-path", default=f"data/1019test/{PACKAGE}")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"output already exists: {out}")
    rel = args.relative_path.rstrip("/")
    out.mkdir(parents=True)

    with np.load(SOURCE / "motion.npz", allow_pickle=False) as source:
        motion = {key: source[key].copy() for key in source.files}
    with np.load(SOURCE / "object_estimate.npz", allow_pickle=False) as source:
        chair_pos = source["position_w"].astype(np.float64)
        chair_quat_xyzw = source["quaternion_xyzw"].astype(np.float64)
    n = len(motion["joint_pos"])
    assert n == FRAME_COUNT == len(chair_pos) == len(chair_quat_xyzw)
    assert int(motion["fps"]) == FPS
    timestamps = np.arange(n, dtype=np.float64) / FPS
    chair_quat_wxyz = chair_quat_xyzw[:, [3, 0, 1, 2]]
    chair_lin = np.gradient(chair_pos, 1 / FPS, axis=0)
    delta_rot = (
        Rotation.from_quat(chair_quat_xyzw[:-1]).inv()
        * Rotation.from_quat(chair_quat_xyzw[1:])
    ).as_rotvec() * FPS
    chair_ang = np.vstack((delta_rot, delta_rot[-1]))
    trajectory = {
        "object_name": np.asarray(OBJECT),
        "fps": np.asarray(FPS),
        "timestamps": timestamps,
        "recording_time_s": START_S + timestamps,
        "position_w": chair_pos,
        "quaternion_wxyz": chair_quat_wxyz,
        "linear_velocity_w": chair_lin,
        "angular_velocity_w": chair_ang,
        "kinematic_pose_defined": np.ones(n, dtype=bool),
        "measured_gt_valid": np.zeros(n, dtype=bool),
        "trajectory_source": np.asarray("human_wrist_fitted_chair_pose_held_after_push"),
    }
    motion.update(
        object_names=np.asarray([OBJECT]),
        object_pos_w=chair_pos[:, None, :],
        object_quat_w=chair_quat_wxyz[:, None, :],
        object_lin_vel_w=chair_lin[:, None, :],
        object_ang_vel_w=chair_ang[:, None, :],
        timestamps=timestamps,
        recording_time_s=START_S + timestamps,
        source_motion_frame=np.arange(n, dtype=np.int64),
        interacting_object_names=np.asarray([], dtype="<U32"),
        object_trajectory_source=np.asarray("kinematic_chair_pose_estimate_no_physics"),
        object_trajectory_measured=np.asarray(False),
    )
    np.savez_compressed(out / "motion.npz", **motion)
    object_path = out / "metadata/object_trajectories/grey_armchair.npz"
    object_path.parent.mkdir(parents=True)
    np.savez_compressed(object_path, **trajectory)

    copies = {
        SOURCE / "motion.npz": "metadata/source_motion.npz",
        SOURCE / "human_motion_world.npz": "metadata/human_motion_world.npz",
        SOURCE / "human.npz": "metadata/human.npz",
        SOURCE / "review.json": "metadata/review/table_brace_geometry.json",
        TASK / "retarget/ground_support_27_end/sidecars/motion_order.json": "metadata/motion_order.json",
        TASK / "segments.json": "metadata/source_segments.json",
        REPO / "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/scene.py":
            "metadata/rss_scene.py",
        PUSH / "metadata/camera_calibration.json": "metadata/camera_calibration.json",
    }
    for source, destination in copies.items():
        copy(source, out / destination)
    with np.load(SOURCE / "source_sync.npz", allow_pickle=False) as source:
        sync = {key: source[key].copy() for key in source.files}
    assert np.allclose(sync["recording_time_s"], START_S + timestamps)
    sync["source_motion_time_s"] = sync["motion_time_s"].copy()
    sync["motion_time_s"] = timestamps
    np.savez_compressed(out / "metadata/source_sync.npz", **sync)

    with np.load(TASK / "tasks/07_armchair/hand_camera_reference.npz", allow_pickle=False) as source:
        rigid = {
            key: source[key][425:].copy() if source[key].ndim
            and source[key].shape[0] == 800 else source[key].copy()
            for key in source.files
        }
    assert len(rigid["right_rigid_pos_w"]) == n
    np.savez_compressed(out / "metadata/hand_rigid_reference.npz", **rigid)

    template = yaml.safe_load((PUSH / "motions.yaml").read_text())
    config = next(iter(template.values()))
    config["path"] = f"{rel}/motion.npz"
    config["source_frame_count"] = n
    config["duration_seconds"] = n / FPS
    config["slice_interval_seconds"] = [START_S, END_S]
    config["accepted"] = False
    config["training_data_ready"] = False
    config["kinematic_reference_accepted_by_user"] = True
    config["archive_status"] = "stage3_2_kinematic_user_visual_ok_no_stage4_or_5"
    config["trajectory_kind"] = "estimated_kinematic_reference"
    config["visual_gt"] = "not_generated"
    config["visual_dataset_status"] = "not_generated_for_table_brace"
    config.pop("original_source_frame_count", None)
    config["object_trajectories"] = {
        OBJECT: {
            "path": f"{rel}/metadata/object_trajectories/{OBJECT}.npz",
            "npz_object_index": 0,
            "interacts_in_this_clip": False,
            "coordinate_frame": "IKEA_1012_world",
            "quaternion_order": "wxyz",
            "measured": False,
            "source": "human_wrist_fitted_chair_pose_held_after_push",
        },
    }
    config["interaction_events"] = [{
        "effector": "right_closed_fist_and_forearm",
        "surface": "room_dining_table",
        "source_recording_time_s": [84.5, 88.8],
        "clip_time_s": [1.0, 5.3],
        "status": "kinematic_contact_geometry_only",
    }]
    scene = config["scene_setup"]
    scene["status"] = "stage3_2_kinematic_archive_no_physics"
    scene["runtime_ready"] = False
    scene["runtime_note"] = "User accepted visual fist-on-table pose; physical support not tested."
    scene["visual_data_source"] = "not_generated"
    scene["live_rgb_use_semantic_mask"] = False
    for key in ("mask_semantics", "mask_semantic_timeline", "mask_field_compatibility",
                "hoi_contact_schedule", "hoi_object_trajectories_estimated"):
        scene.pop(key, None)
    original_scene = next(iter(yaml.safe_load(FULL_SCENE.read_text()).values()))["scene_setup"]
    chair = next(item for item in original_scene["objects"] if item["id"] == OBJECT)
    chair["initial_position_m"] = chair_pos[0].tolist()
    chair["initial_orientation_wxyz"] = chair_quat_wxyz[0].tolist()
    chair["trajectory_kind"] = "estimated_kinematic_reference"
    scene["objects"] = [chair]
    scene["object"] = chair
    scene["camera"]["calibration_file"] = f"{rel}/metadata/camera_calibration.json"
    scene["camera"]["calibration_mode"] = "calibration_file"
    scene["camera"]["calibration_note"] = (
        "Inherited for optional future rendering; not used to produce this kinematic "
        "motion or its geometry audit. The right-hand rigid/camera diagnostic "
        "uses its own extrinsics stored in metadata/hand_rigid_reference.npz."
    )
    (out / "motions.yaml").write_text(
        yaml.safe_dump({PACKAGE_ID: config}, sort_keys=False, allow_unicode=True)
    )

    all_numeric = [motion[key] for key in (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
        "body_ang_vel_w", "object_pos_w", "object_quat_w", "object_lin_vel_w",
        "object_ang_vel_w",
    )]
    assert all(np.isfinite(value).all() for value in all_numeric)
    assert np.max(np.abs(np.linalg.norm(motion["body_quat_w"], axis=-1) - 1)) < 1e-4
    assert np.max(np.abs(np.linalg.norm(chair_quat_wxyz, axis=-1) - 1)) < 1e-5
    assert np.max(np.abs(chair_pos - chair_pos[0])) == 0
    for path in (scene["scene"]["usd_path"], chair["usd_path"],
                 scene["camera"]["robot_usd"]):
        assert (SCALETRACK / path).is_file(), path
    validation = {
        "status": "kinematic_archive_structurally_valid_not_physics_validated",
        "frames": n,
        "fps": FPS,
        "source_recording_interval_s": [START_S, END_S],
        "last_recording_frame_s": START_S + (n - 1) / FPS,
        "chair_pose_constant": True,
        "object_sidecar_matches_main_motion": True,
        "scene_usd_references_exist": True,
        "tabletop_height_m": 0.754,
        "tracker_rollout": False,
        "physical_support_verified": False,
        "visual_gt_generated": False,
        "training_data_ready": False,
    }
    (out / "metadata/validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (out / "README.md").write_text(
        "# 1019 right fist and forearm brace on dining table\n\n"
        "Standalone [83.5, 91.0) s recording slice, 375 frames at 50 Hz. The first "
        "1.0 s overlaps the chair-push clip. The chair is held at its estimated "
        "post-push pose, and the dining table belongs to the room scene. The user "
        "said the closed-fist support looks okay in the kinematic viewer. The "
        "motion, room, chair, source mapping and hand rigid diagnostic are archived "
        "together. No physical support, tracker rollout, or new visual GT is claimed. "
        "`accepted` and `training_data_ready` remain false.\n"
    )
    provenance = {
        "source_recording_id": "1-2026-10-19-16-31-26",
        "source_recording_interval_s": [START_S, END_S],
        "source_motion": str(SOURCE / "motion.npz"),
        "source_motion_sha256": sha(SOURCE / "motion.npz"),
        "source_object_trajectory": str(SOURCE / "object_estimate.npz"),
        "source_object_trajectory_sha256": sha(SOURCE / "object_estimate.npz"),
        "source_scene": str(REPO / "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/scene.py"),
        "source_stage_ledger": str(TASK / "stage_ledger.json"),
        "visual_review": "user_says_closed_fist_table_support_looks_ok_2026-10-20",
        "physics_status": "not_started",
        "object_trajectory_kind": "estimated_kinematic_chair_pose_held_constant",
        "files_sha256": {
            str(path.relative_to(out)): sha(path)
            for path in sorted(out.rglob("*")) if path.is_file()
        },
    }
    (out / "metadata/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
