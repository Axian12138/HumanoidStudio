"""Package the lined-basket v21 kinematic reference for ScaleTrack."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


REPO = Path(__file__).resolve().parents[6]
SCALETRACK = Path("/home/galbot/workspace/ScaleTrack")
SNAPSHOT = REPO / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot"
TASK = SNAPSHOT / "tasks/09_lined_rect_basket"
CANDIDATE = REPO / (
    "outputs/humanoid-real2sim/2026-10-20/sequence_1019__09_lined_rect_basket/"
    "runs/20-11-03.205681+0800/hoi_v21"
)
RIGID = REPO / (
    "outputs/humanoid-real2sim/2026-10-20/sequence_1019__09_lined_rect_basket/"
    "runs/17-29-01.627805+0800/rigid/wrist_reference.npz"
)
PACKAGE = "lined-rect-basket-kinematic-v1"
OBJECT = "lined_rect_basket"
FPS = 50
FRAMES = 550
START_S = 98.0
ROBOT_FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w", "fps",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path, relative_path: str) -> None:
    if output.exists():
        raise FileExistsError(f"archive already exists: {output}")
    relative_path = relative_path.rstrip("/")
    with np.load(CANDIDATE / "motion.npz", allow_pickle=False) as data:
        if set(data.files) != set(ROBOT_FIELDS):
            raise ValueError(f"unexpected robot fields: {data.files}")
        motion = {key: data[key].copy() for key in ROBOT_FIELDS}
    with np.load(CANDIDATE / "object_estimate.npz", allow_pickle=False) as data:
        position = data["position_w"].astype(np.float64)
        quat_xyzw = data["quaternion_xyzw"].astype(np.float64)
        trajectory_source = str(data["trajectory_source"].item())
        object_fps = int(data["fps"])
    with np.load(TASK / "source_sync.npz", allow_pickle=False) as data:
        recording_time = data["recording_time_s"].astype(np.float64)
        source_motion_frame = np.rint(data["motion_time_s"] * FPS).astype(np.int64)
    order_path = SNAPSHOT / "retarget/ground_support_27_end/sidecars/motion_order.json"
    order = json.loads(order_path.read_text())
    if (int(motion["fps"]), object_fps, len(position)) != (FPS, FPS, FRAMES):
        raise ValueError("robot and basket must have 550 frames at 50 Hz")
    if motion["joint_pos"].shape != (FRAMES, len(order["robot_joint_names"])):
        raise ValueError("joint order does not match motion")
    if motion["body_pos_w"].shape != (FRAMES, len(order["robot_body_names"]), 3):
        raise ValueError("body order does not match motion")
    if position.shape != (FRAMES, 3) or quat_xyzw.shape != (FRAMES, 4):
        raise ValueError("basket trajectory does not match motion")
    timestamps = np.arange(FRAMES, dtype=np.float64) / FPS
    if not np.allclose(recording_time, START_S + timestamps, atol=1e-6):
        raise ValueError("recording time is not the 98–109 s window")
    if not np.array_equal(source_motion_frame, np.arange(3550, 4100)):
        raise ValueError("source frame mapping is not the reviewed slice")
    if any(not np.isfinite(motion[key]).all() for key in ROBOT_FIELDS if key != "fps"):
        raise ValueError("robot motion contains non-finite values")
    if not np.isfinite(position).all() or not np.isfinite(quat_xyzw).all():
        raise ValueError("basket estimate contains non-finite values")
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    linear_velocity = np.gradient(position, 1 / FPS, axis=0)
    delta_rotation = (
        Rotation.from_quat(quat_xyzw[:-1]).inv()
        * Rotation.from_quat(quat_xyzw[1:])
    ).as_rotvec() * FPS
    angular_velocity = np.vstack((delta_rotation, delta_rotation[-1]))
    sidecar = {
        "object_name": np.asarray(OBJECT),
        "fps": np.asarray(FPS),
        "timestamps": timestamps,
        "recording_time_s": recording_time,
        "source_motion_frame": source_motion_frame,
        "position_w": position,
        "quaternion_wxyz": quat_wxyz,
        "linear_velocity_w": linear_velocity,
        "angular_velocity_w": angular_velocity,
        "kinematic_pose_defined": np.ones(FRAMES, dtype=bool),
        "measured_gt_valid": np.zeros(FRAMES, dtype=bool),
        "trajectory_source": np.asarray(trajectory_source),
    }
    object_arrays = {
        "object_names": np.asarray([OBJECT]),
        "object_pos_w": position[:, None, :],
        "object_quat_w": quat_wxyz[:, None, :],
        "object_lin_vel_w": linear_velocity[:, None, :],
        "object_ang_vel_w": angular_velocity[:, None, :],
        "timestamps": timestamps,
        "recording_time_s": recording_time,
        "source_motion_frame": source_motion_frame,
        "interacting_object_names": np.asarray([OBJECT]),
        "object_trajectory_source": np.asarray(trajectory_source),
        "object_trajectory_measured": np.asarray(False),
    }

    output.mkdir(parents=True)
    metadata = output / "metadata"
    metadata.mkdir()
    np.savez_compressed(output / "motion.npz", **motion, **object_arrays)
    np.savez_compressed(output / f"{OBJECT}_trajectory.npz", **sidecar)
    copies = {
        CANDIDATE / "motion.npz": "metadata/source_motion.npz",
        CANDIDATE / "object_estimate.npz": "metadata/source_object_estimate.npz",
        CANDIDATE / "fit.json": "metadata/fit.json",
        CANDIDATE / "audit.json": "metadata/audit.json",
        CANDIDATE / "human_motion_world.npz": "metadata/human_motion_world.npz",
        TASK / "source_sync.npz": "metadata/source_sync.npz",
        TASK / "human.npz": "metadata/human.npz",
        RIGID: "metadata/wrist_reference.npz",
        order_path: "metadata/motion_order.json",
        SNAPSHOT / "stage_ledger.json": "metadata/source_stage_ledger.json",
        REPO / "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/09_lined_rect_basket/scene_stage4.py":
            "metadata/rss_scene_stage4.py",
    }
    for source, target in copies.items():
        target_path = output / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)

    template = next(iter(yaml.safe_load((
        SCALETRACK / "data/1019test/round-basket-foot-push-kinematic-v1/motions.yaml"
    ).read_text()).values()))
    scene = copy.deepcopy(template["scene_setup"])
    base = next(iter(yaml.safe_load((SCALETRACK / "data/1012test/ikea1012.yaml").read_text()).values()))
    basket = copy.deepcopy(next(item for item in base["scene_setup"]["objects"] if item["id"] == OBJECT))
    basket["initial_position_m"] = position[0].tolist()
    basket["initial_orientation_wxyz"] = quat_wxyz[0].tolist()
    basket["trajectory_kind"] = "estimated_kinematic_reference"
    basket["physics_parameter_status"] = "not_finalized_in_ScaleTrack_USD"
    scene["object"] = basket
    scene["objects"] = [basket]
    scene["status"] = "stage3_2_kinematic_candidate_physics_pending"
    scene["runtime_ready"] = False
    scene["runtime_note"] = (
        "Robot and basket poses are kinematic; Stage 4 grip and 1.2 RSS friction "
        "have not been validated or authored into the ScaleTrack USD."
    )
    scene["visual_data_source"] = "not_generated"
    scene["live_rgb_use_semantic_mask"] = False
    for key in ("mask_semantics", "mask_semantic_timeline", "mask_field_compatibility"):
        scene.pop(key, None)
    scene["camera"]["calibration_file"] = f"{relative_path}/metadata/camera_calibration.json"
    scene["camera"]["calibration_note"] = "Existing camera calibration; no new visual GT."
    camera_source = (SCALETRACK / "data/1019test/push-bin-kinematic-v1/metadata/camera_calibration.json")
    shutil.copy2(camera_source, metadata / "camera_calibration.json")
    config = {
        "path": f"{relative_path}/motion.npz",
        "weight": 1.0,
        "source_recording_id": "1-2026-10-19-16-31-26",
        "motion_fps": float(FPS),
        "source_frame_count": FRAMES,
        "duration_seconds": FRAMES / FPS,
        "slice_interval_seconds": [98.0, 109.0],
        "accepted": False,
        "training_data_ready": False,
        "archive_status": "stage3_2_kinematic_v21_pending_stage4_physics_review",
        "trajectory_kind": "estimated_kinematic_reference",
        "visual_gt": "not_generated",
        "scene_setup": scene,
        "interacting_objects": [OBJECT],
        "object_trajectories": {OBJECT: {
            "path": f"{relative_path}/{OBJECT}_trajectory.npz",
            "npz_object_index": 0,
            "interacts_in_this_clip": True,
            "coordinate_frame": "IKEA_1012_world",
            "quaternion_order": "wxyz",
            "measured": False,
            "source": trajectory_source,
        }},
        "interaction_events": [{
            "effectors": ["left_wrist_and_forearm", "right_wrist_and_forearm"],
            "source_recording_time_s": [98.8, 106.4],
            "status": "kinematic_approximately_30mm_wall_overlap_physics_pending",
        }],
    }
    (output / "motions.yaml").write_text(yaml.safe_dump(
        {"hoi1019_lined_rect_basket_kinematic_v1": config},
        sort_keys=False, allow_unicode=True,
    ))
    (output / "README.md").write_text(
        "# 1019 lined rectangular basket: v21 kinematic candidate\n\n"
        "Recording interval [98, 109) s; 550 frames at 50 Hz. Robot motion is "
        "the v21 two-hand retarget. The basket trajectory is estimated from "
        "human wrist rigid markers and contact geometry; it is not measured "
        "object ground truth or an RSS physical rollout. v21 allows up to about "
        "30 mm side-wall overlap during carry; precontact remains collision-free "
        "in the sampled geometry audit. The Stage 4 RSS friction estimate is 1.2, "
        "but no final ScaleTrack USD physics variant was authored. Grip force, "
        "object lift, release, visual GT, and training suitability are pending. "
        "The seven robot motion fields, aligned basket estimate, source mapping, "
        "kinematic audit, and provenance are packaged here.\n"
    )
    provenance = {
        "source_motion": str(CANDIDATE / "motion.npz"),
        "source_object_estimate": str(CANDIDATE / "object_estimate.npz"),
        "source_recording_interval_s": [98.0, 109.0],
        "source_mcap_sha256": json.loads((SNAPSHOT / "stage_ledger.json").read_text())[
            "source_mcap_sha256"
        ],
        "source_files_sha256": {str(source): sha256(source) for source in copies},
        "rss_stage4_friction_estimate": 1.2,
        "physics_status": "not_accepted",
        "files_sha256": {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    (metadata / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    validate(output, relative_path)


def validate(output: Path, relative_path: str) -> None:
    config = next(iter(yaml.safe_load((output / "motions.yaml").read_text()).values()))
    with np.load(output / "motion.npz", allow_pickle=False) as motion, np.load(
        output / f"{OBJECT}_trajectory.npz", allow_pickle=False
    ) as sidecar:
        checks = {
            "robot_fields_and_order": set(ROBOT_FIELDS).issubset(motion.files)
            and motion["joint_pos"].shape == (FRAMES, 43)
            and motion["body_pos_w"].shape == (FRAMES, 50, 3),
            "frame_and_time_alignment": np.allclose(motion["timestamps"], np.arange(FRAMES) / FPS)
            and np.allclose(motion["recording_time_s"], START_S + np.arange(FRAMES) / FPS),
            "source_frame_mapping": np.array_equal(motion["source_motion_frame"], np.arange(3550, 4100)),
            "object_sidecar_matches_motion": all(np.array_equal(motion[main][:, 0], sidecar[other])
                                                  for main, other in (
                                                      ("object_pos_w", "position_w"),
                                                      ("object_quat_w", "quaternion_wxyz"),
                                                      ("object_lin_vel_w", "linear_velocity_w"),
                                                      ("object_ang_vel_w", "angular_velocity_w"),
                                                  )),
            "finite": all(np.isfinite(motion[key]).all() for key in ROBOT_FIELDS if key != "fps")
            and all(np.isfinite(sidecar[key]).all() for key in (
                "position_w", "quaternion_wxyz", "linear_velocity_w", "angular_velocity_w")),
            "quaternion_norm": np.max(np.abs(np.linalg.norm(motion["body_quat_w"], axis=-1) - 1)) < 2e-3
            and np.max(np.abs(np.linalg.norm(sidecar["quaternion_wxyz"], axis=-1) - 1)) < 2e-3,
            "yaml_paths": config["path"] == f"{relative_path}/motion.npz"
            and config["object_trajectories"][OBJECT]["path"]
            == f"{relative_path}/{OBJECT}_trajectory.npz",
            "scene_asset_paths": all((SCALETRACK / path).is_file() for path in (
                config["scene_setup"]["scene"]["usd_path"],
                config["scene_setup"]["object"]["usd_path"],
                config["scene_setup"]["camera"]["robot_usd"],
            )),
            "review_status": config["accepted"] is False and config["training_data_ready"] is False
            and config["visual_gt"] == "not_generated",
        }
    if not all(checks.values()):
        raise ValueError(f"archive checks failed: {[name for name, ok in checks.items() if not ok]}")
    (output / "metadata/validation.json").write_text(json.dumps({
        "status": "kinematic_archive_structurally_valid_pending_physics_review",
        "frames": FRAMES,
        "fps": FPS,
        "source_recording_interval_s": [START_S, START_S + FRAMES / FPS],
        "checks": {name: bool(ok) for name, ok in checks.items()},
        "tracker_rollout": False,
        "physical_contact_verified": False,
        "visual_gt_generated": False,
        "training_data_ready": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-path", default=f"data/1019test/{PACKAGE}")
    args = parser.parse_args()
    build(args.output.resolve(), args.relative_path)
    print(args.output.resolve())
