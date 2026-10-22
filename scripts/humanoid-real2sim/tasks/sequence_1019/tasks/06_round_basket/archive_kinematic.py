"""Archive the 59–76 s round-basket Stage 3.5 candidate without claiming physics success."""

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
TASK = SNAPSHOT / "tasks/06_round_basket"
CANDIDATE = TASK / "hoi_v5_kneel_59_76"
REVIEW = TASK / "review_59_76"
PACKAGE = "round-basket-foot-push-kinematic-v1"
OBJECT = "round_laundry_basket"
FPS = 50
START_S, END_S = 59.0, 76.0
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


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build(output: Path, relative_path: str) -> None:
    if output.exists():
        raise FileExistsError(f"archive already exists: {output}")
    relative_path = relative_path.rstrip("/")
    order_path = SNAPSHOT / "retarget/ground_support_27_end/sidecars/motion_order.json"
    order = json.loads(order_path.read_text())
    with np.load(CANDIDATE / "motion.npz", allow_pickle=False) as data:
        if set(data.files) != set(ROBOT_FIELDS):
            raise ValueError(f"unexpected robot motion fields: {data.files}")
        motion = {key: data[key].copy() for key in ROBOT_FIELDS}
    with np.load(CANDIDATE / "object_estimate.npz", allow_pickle=False) as data:
        basket_pos = data["position_w"].astype(np.float64)
        basket_quat_xyzw = data["quaternion_xyzw"].astype(np.float64)
        estimate_status = str(data["status"].item())
        object_fps = int(data["fps"])
    with np.load(TASK / "source_sync.npz", allow_pickle=False) as data:
        sync = {key: data[key].copy() for key in data.files}

    frames = len(motion["joint_pos"])
    fps = int(motion["fps"])
    timestamps = np.arange(frames, dtype=np.float64) / fps
    recording_time = START_S + timestamps
    source_frame = np.rint(sync["motion_time_s"] * fps).astype(np.int64)
    if (frames, fps, object_fps) != (850, FPS, FPS):
        raise ValueError("expected 850 robot/object frames at 50 Hz")
    if motion["joint_pos"].shape != (frames, len(order["robot_joint_names"])):
        raise ValueError("joint order does not match the robot motion")
    if motion["body_pos_w"].shape != (frames, len(order["robot_body_names"]), 3):
        raise ValueError("body order does not match the robot motion")
    if basket_pos.shape != (frames, 3) or basket_quat_xyzw.shape != (frames, 4):
        raise ValueError("basket pose is not aligned to robot frames")
    if not np.allclose(sync["recording_time_s"], recording_time, atol=1e-6):
        raise ValueError("source recording time is not the 59–76 s clip")
    if not np.array_equal(source_frame, np.arange(1600, 2450)):
        raise ValueError("source frame mapping is not the expected 1600–2449 slice")
    if any(not np.isfinite(motion[key]).all() for key in ROBOT_FIELDS if key != "fps"):
        raise ValueError("robot motion contains non-finite values")
    if not np.isfinite(basket_pos).all() or not np.isfinite(basket_quat_xyzw).all():
        raise ValueError("basket estimate contains non-finite values")
    if np.max(np.abs(np.linalg.norm(motion["body_quat_w"], axis=-1) - 1)) > 2e-3:
        raise ValueError("robot quaternion norm is invalid")
    if np.max(np.abs(np.linalg.norm(basket_quat_xyzw, axis=-1) - 1)) > 2e-3:
        raise ValueError("basket quaternion norm is invalid")

    basket_quat_wxyz = basket_quat_xyzw[:, [3, 0, 1, 2]]
    basket_lin_vel = np.gradient(basket_pos, 1 / fps, axis=0)
    delta_rot = (
        Rotation.from_quat(basket_quat_xyzw[:-1]).inv()
        * Rotation.from_quat(basket_quat_xyzw[1:])
    ).as_rotvec() * fps
    basket_ang_vel = np.vstack((delta_rot, delta_rot[-1]))
    object_arrays = {
        "object_names": np.asarray([OBJECT]),
        "object_pos_w": basket_pos[:, None, :],
        "object_quat_w": basket_quat_wxyz[:, None, :],
        "object_lin_vel_w": basket_lin_vel[:, None, :],
        "object_ang_vel_w": basket_ang_vel[:, None, :],
        "timestamps": timestamps,
        "recording_time_s": recording_time,
        "source_motion_frame": source_frame,
        "interacting_object_names": np.asarray([OBJECT]),
        "object_trajectory_source": np.asarray(estimate_status),
        "object_trajectory_measured": np.asarray(False),
    }
    sidecar = {
        "object_name": np.asarray(OBJECT),
        "fps": np.asarray(fps),
        "timestamps": timestamps,
        "recording_time_s": recording_time,
        "source_motion_frame": source_frame,
        "position_w": basket_pos,
        "quaternion_wxyz": basket_quat_wxyz,
        "linear_velocity_w": basket_lin_vel,
        "angular_velocity_w": basket_ang_vel,
        "kinematic_pose_defined": np.ones(frames, dtype=bool),
        "measured_gt_valid": np.zeros(frames, dtype=bool),
        "trajectory_source": np.asarray(estimate_status),
    }

    output.mkdir(parents=True)
    metadata = output / "metadata"
    metadata.mkdir()
    np.savez_compressed(output / "motion.npz", **motion, **object_arrays)
    np.savez_compressed(output / f"{OBJECT}_trajectory.npz", **sidecar)
    copies = {
        CANDIDATE / "motion.npz": "metadata/source_motion.npz",
        CANDIDATE / "object_estimate.npz": "metadata/source_object_estimate.npz",
        CANDIDATE / "kneel_review.json": "metadata/review/kneel_review.json",
        TASK / "hoi_v4_59_76/review.json": "metadata/review/foot_basket_review.json",
        TASK / "human.npz": "metadata/human.npz",
        TASK / "source_sync.npz": "metadata/source_sync.npz",
        REVIEW / "human_motion_world.npz": "metadata/human_motion_world.npz",
        REVIEW / "rigid_wrist_track.npz": "metadata/rigid_wrist_track.npz",
        SNAPSHOT / "stage_ledger.json": "metadata/source_stage_ledger.json",
        order_path: "metadata/motion_order.json",
        SNAPSHOT / "assets/morbylanga_table_1012.obj": "metadata/assets/morbylanga_table_1012.obj",
        TASK / "round_laundry_basket_visual.obj": "metadata/assets/round_laundry_basket_visual.obj",
        TASK / "round_laundry_basket_collision.obj": "metadata/assets/round_laundry_basket_collision.obj",
        REPO / "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/scene.py":
            "metadata/rss_scene.py",
        REPO / "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/scene_rigid_review.py":
            "metadata/rss_scene_review.py",
    }
    for source, target in copies.items():
        copy_file(source, output / target)

    template_path = SCALETRACK / "data/1012test/1012-1/stage7-visual-gt/motions.yaml"
    template = next(iter(yaml.safe_load(template_path.read_text()).values()))
    config = {
        "path": f"{relative_path}/motion.npz",
        "weight": 1.0,
        "source_recording_id": "1-2026-10-19-16-31-26",
        "motion_fps": float(fps),
        "source_frame_count": frames,
        "duration_seconds": frames / fps,
        "slice_interval_seconds": [START_S, END_S],
        "accepted": False,
        "training_data_ready": False,
        "archive_status": "stage3_5_kinematic_candidate_pending_contact_and_visual_review",
        "trajectory_kind": "estimated_kinematic_reference",
        "visual_gt": "not_generated",
        "scene_setup": copy.deepcopy(template["scene_setup"]),
        "interacting_objects": [OBJECT],
        "object_trajectories": {
            OBJECT: {
                "path": f"{relative_path}/{OBJECT}_trajectory.npz",
                "npz_object_index": 0,
                "interacts_in_this_clip": True,
                "coordinate_frame": "IKEA_1012_world",
                "quaternion_order": "wxyz",
                "measured": False,
                "source": estimate_status,
            }
        },
        "interaction_events": [
            {"effector": "left_foot_and_shin", "source_recording_time_s": [59.0, 67.0],
             "status": "contact_conditioned_kinematic_candidate"},
            {"effector": "left_knee_and_toe", "source_recording_time_s": [69.0, 71.0],
             "status": "kinematic_ground_geometry_checked"},
            {"effector": "right_forearm", "source_recording_time_s": [71.0, 76.0],
             "status": "candidate_reach_residual_pending"},
        ],
    }
    scene = config["scene_setup"]
    scene["status"] = "stage3_5_kinematic_candidate_no_physics"
    scene["runtime_ready"] = False
    scene["runtime_note"] = "Basket force direction and right forearm reach remain unresolved."
    scene["visual_data_source"] = "not_generated"
    scene["live_rgb_use_semantic_mask"] = False
    for key in ("mask_semantics", "mask_semantic_timeline", "mask_field_compatibility"):
        scene.pop(key, None)
    basket = copy.deepcopy(next(item for item in scene["objects"] if item["id"] == OBJECT))
    basket["initial_position_m"] = basket_pos[0].tolist()
    basket["initial_orientation_wxyz"] = basket_quat_wxyz[0].tolist()
    basket["trajectory_kind"] = "estimated_kinematic_reference"
    scene["object"] = basket
    scene["objects"] = [basket]
    scene["camera"]["calibration_file"] = f"{relative_path}/metadata/camera_calibration.json"
    scene["camera"]["calibration_mode"] = "calibration_file"
    scene["camera"]["calibration_note"] = "Inherited camera setup; no new visual GT was generated."
    copy_file(
        SCALETRACK / "data/1019test/push-bin-kinematic-v1/metadata/camera_calibration.json",
        metadata / "camera_calibration.json",
    )
    (output / "motions.yaml").write_text(
        yaml.safe_dump({"hoi1019_round_basket_foot_push_kinematic_v1": config},
                       sort_keys=False, allow_unicode=True)
    )
    (output / "README.md").write_text(
        "# 1019 round basket foot push: kinematic candidate\n\n"
        "Recording interval [59, 76) s; 850 frames at 50 Hz. The robot motion is "
        "Stage 3.5 candidate `hoi_v5_kneel_59_76`. The round-basket trajectory is "
        "an estimated contact-conditioned pose, not measured ground truth or a physics rollout. "
        "The foot/shin force direction and right forearm reach failed the current review; "
        "user visual review, tracker, physical contact, and visual GT are pending. "
        "`accepted` and `training_data_ready` are false. Source motion, object estimate, "
        "human/rigid overlays, source mapping, scene and review files are in `metadata/`.\n"
    )
    provenance = {
        "source_recording_id": "1-2026-10-19-16-31-26",
        "source_recording_interval_s": [START_S, END_S],
        "source_motion": str(CANDIDATE / "motion.npz"),
        "source_object_estimate": str(CANDIDATE / "object_estimate.npz"),
        "source_stage_ledger": str(SNAPSHOT / "stage_ledger.json"),
        "source_mcap_sha256": json.loads((SNAPSHOT / "stage_ledger.json").read_text())["source_mcap_sha256"],
        "source_files_sha256": {str(path): sha256(path) for path in copies},
        "source_motion_sha256": sha256(CANDIDATE / "motion.npz"),
        "source_object_estimate_sha256": sha256(CANDIDATE / "object_estimate.npz"),
        "source_frame_range_in_27s_motion": [int(source_frame[0]), int(source_frame[-1])],
        "object_trajectory_kind": estimate_status,
        "physics_status": "not_started",
        "visual_review_status": "pending",
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
        frames = len(motion["joint_pos"])
        checks = {
            "robot_fields_and_order": set(ROBOT_FIELDS).issubset(motion.files)
            and motion["joint_pos"].shape == (850, 43)
            and motion["body_pos_w"].shape == (850, 50, 3),
            "frame_and_time_alignment": frames == 850
            and np.allclose(motion["timestamps"], np.arange(frames) / FPS)
            and np.allclose(motion["recording_time_s"], 59 + np.arange(frames) / FPS),
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
            "source_frame_mapping": np.array_equal(motion["source_motion_frame"], np.arange(1600, 2450)),
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
    checks = {name: bool(passed) for name, passed in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"archive checks failed: {[key for key, passed in checks.items() if not passed]}")
    (output / "metadata/validation.json").write_text(json.dumps({
        "status": "kinematic_archive_structurally_valid_pending_hoi_review",
        "frames": 850,
        "fps": FPS,
        "source_recording_interval_s": [START_S, END_S],
        "checks": checks,
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
