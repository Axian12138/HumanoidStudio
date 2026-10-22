"""Archive both visually accepted armchair-push kinematic hand variants."""

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
SOURCE_ROOT = REPO / "outputs/humanoid-real2sim/2026-10-20"
SNAPSHOT = SOURCE_ROOT / "sequence_1019/curated_snapshot"
TEMPLATE = SCALETRACK / "data/1019test/brace-dining-table-kinematic-v1/motions.yaml"
CAMERA = SCALETRACK / "data/1019test/push-bin-kinematic-v1/metadata/camera_calibration.json"
TASK_SCRIPT = REPO / "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/scene.py"
FPS = 50
FRAMES = 475
START_S = 75.0
END_S = 84.5
OBJECT = "grey_armchair"
VARIANTS = {
    "index_middle": {
        "package": "push-armchair-index-middle-kinematic-v1",
        "id": "hoi1019_push_armchair_index_middle_kinematic_v1",
        "source": SOURCE_ROOT / (
            "sequence_1019__armchair_split_feedback/runs/17-54-03.563689+0800/"
            "chair_palm_surface"
        ),
        "contact": "index and middle fingers push the upper backrest; thumb locates it",
        "review": "user praised index/middle push after surface-contact review",
    },
    "thumb": {
        "package": "push-armchair-thumb-kinematic-v1",
        "id": "hoi1019_push_armchair_thumb_kinematic_v1",
        "source": SOURCE_ROOT / (
            "sequence_1019__armchair_finger_entry/runs/19-20-24.732142+0800/"
            "chair_thumb_backface_flip_open"
        ),
        "contact": (
            "straight thumb side pushes against the rear backrest face; "
            "index and middle fingers open before contact, then curl to locate"
        ),
        "review": "user accepted the thumb orientation and open-then-curl finger entry",
    },
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def archive_one(key: str, spec: dict, root: Path) -> Path:
    source = spec["source"]
    out = root / spec["package"]
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / "metadata").mkdir()
    rel = f"data/1019test/{spec['package']}"

    with np.load(source / "motion.npz", allow_pickle=False) as data:
        motion = {name: data[name].copy() for name in data.files}
    with np.load(source / "object_estimate.npz", allow_pickle=False) as data:
        chair_pos = data["position_w"].astype(np.float64)
        chair_quat_xyzw = data["quaternion_xyzw"].astype(np.float64)
        object_source = str(data["trajectory_source"])
    with np.load(source / "source_sync.npz", allow_pickle=False) as data:
        sync = {name: data[name].copy() for name in data.files}
    assert len(motion["joint_pos"]) == len(chair_pos) == len(chair_quat_xyzw) == FRAMES
    assert int(motion["fps"]) == int(FPS)
    assert np.allclose(sync["recording_time_s"], START_S + np.arange(FRAMES) / FPS)
    timestamps = np.arange(FRAMES, dtype=np.float64) / FPS
    source_motion_time_s = sync["motion_time_s"].copy()
    sync["source_motion_time_s"] = source_motion_time_s
    sync["motion_time_s"] = timestamps
    np.savez_compressed(out / "metadata/source_sync.npz", **sync)

    chair_quat_wxyz = chair_quat_xyzw[:, [3, 0, 1, 2]]
    chair_lin = np.gradient(chair_pos, 1 / FPS, axis=0)
    chair_delta_world = (
        Rotation.from_quat(chair_quat_xyzw[1:])
        * Rotation.from_quat(chair_quat_xyzw[:-1]).inv()
    ).as_rotvec() * FPS
    chair_ang = np.vstack((chair_delta_world, chair_delta_world[-1]))
    sidecar = {
        "object_name": np.asarray(OBJECT),
        "fps": np.asarray(FPS),
        "timestamps": timestamps,
        "recording_time_s": START_S + timestamps,
        "position_w": chair_pos,
        "quaternion_wxyz": chair_quat_wxyz,
        "linear_velocity_w": chair_lin,
        "angular_velocity_w": chair_ang,
        "kinematic_pose_defined": np.ones(FRAMES, dtype=bool),
        "measured_gt_valid": np.zeros(FRAMES, dtype=bool),
        "trajectory_source": np.asarray(object_source),
    }
    trajectory_path = out / "metadata/object_trajectories/grey_armchair.npz"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory_path, **sidecar)
    motion.update(
        object_names=np.asarray([OBJECT]),
        object_pos_w=chair_pos[:, None, :],
        object_quat_w=chair_quat_wxyz[:, None, :],
        object_lin_vel_w=chair_lin[:, None, :],
        object_ang_vel_w=chair_ang[:, None, :],
        timestamps=timestamps,
        recording_time_s=START_S + timestamps,
        source_motion_frame=np.rint(source_motion_time_s * FPS).astype(np.int64),
        interacting_object_names=np.asarray([OBJECT]),
        object_trajectory_source=np.asarray("estimated_kinematic_reference_no_physics"),
        object_trajectory_measured=np.asarray(False),
    )
    np.savez_compressed(out / "motion.npz", **motion)

    copies = {
        source / "motion.npz": "metadata/source_motion.npz",
        source / "object_estimate.npz": "metadata/source_object_estimate.npz",
        source / "human_motion_world.npz": "metadata/human_motion_world.npz",
        source / "human.npz": "metadata/human.npz",
        SNAPSHOT / "retarget/ground_support_27_end/sidecars/motion_order.json":
            "metadata/motion_order.json",
        SNAPSHOT / "segments.json": "metadata/source_segments.json",
        TASK_SCRIPT: "metadata/rss_kinematic_scene.py",
        CAMERA: "metadata/camera_calibration.json",
    }
    evidence = source / ("review.json" if key == "index_middle" else "finger_entry.json")
    copies[evidence] = "metadata/variant_review.json"
    for src, dst in copies.items():
        copy_file(src, out / dst)
    with np.load(SNAPSHOT / "tasks/07_armchair/hand_camera_reference.npz", allow_pickle=False) as data:
        rigid = {
            name: data[name][:FRAMES].copy()
            if data[name].ndim and data[name].shape[0] == 800 else data[name].copy()
            for name in data.files
        }
    np.savez_compressed(out / "metadata/hand_rigid_reference.npz", **rigid)

    template = next(iter(yaml.safe_load(TEMPLATE.read_text()).values()))
    config = copy.deepcopy(template)
    config["path"] = f"{rel}/motion.npz"
    config["source_frame_count"] = FRAMES
    config["duration_seconds"] = FRAMES / FPS
    config["slice_interval_seconds"] = [START_S, END_S]
    config["accepted"] = False
    config["training_data_ready"] = False
    config["kinematic_reference_accepted_by_user"] = True
    config["archive_status"] = "stage3_2_kinematic_user_visual_ok_stage4_physics_pending"
    config["trajectory_kind"] = "estimated_kinematic_reference"
    config["visual_dataset_status"] = "not_generated"
    config["visual_gt"] = "not_generated"
    config["object_trajectories"] = {
        OBJECT: {
            "path": f"{rel}/metadata/object_trajectories/{OBJECT}.npz",
            "npz_object_index": 0,
            "interacts_in_this_clip": True,
            "coordinate_frame": "IKEA_1012_world",
            "quaternion_order": "wxyz",
            "measured": False,
            "source": object_source,
        },
    }
    config["interaction_events"] = [{
        "object": OBJECT,
        "effector": spec["contact"],
        "surface": "upper chair backrest",
        "contact_seconds_estimated": [4.04, 8.4],
        "source_recording_time_s": [79.04, 83.4],
        "status": "kinematic_contact_geometry_only",
    }]
    scene = config["scene_setup"]
    scene["status"] = "stage3_2_kinematic_archive_no_physics"
    scene["runtime_ready"] = False
    scene["runtime_note"] = (
        "User visually accepted this hand variant. The table is fixed room scenery; "
        "chair trajectory is an estimate, not a physics rollout."
    )
    scene["visual_data_source"] = "not_generated"
    scene["live_rgb_use_semantic_mask"] = False
    scene["camera"]["calibration_file"] = f"{rel}/metadata/camera_calibration.json"
    scene["camera"]["calibration_mode"] = "calibration_file"
    scene["camera"]["calibration_note"] = "Inherited for optional future rendering; no visual GT in this archive."
    scene["scene"]["dining_table_role"] = "fixed room asset; no table trajectory"
    chair = copy.deepcopy(scene["object"])
    chair["initial_position_m"] = chair_pos[0].tolist()
    chair["initial_orientation_wxyz"] = chair_quat_wxyz[0].tolist()
    chair["trajectory_kind"] = "estimated_kinematic_reference"
    scene["object"] = chair
    scene["objects"] = [copy.deepcopy(chair)]
    (out / "motions.yaml").write_text(
        yaml.safe_dump({spec["id"]: config}, sort_keys=False, allow_unicode=True)
    )

    for name in (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
        "body_ang_vel_w", "object_pos_w", "object_quat_w", "object_lin_vel_w",
        "object_ang_vel_w",
    ):
        assert np.isfinite(motion[name]).all(), name
    assert np.max(np.abs(np.linalg.norm(motion["body_quat_w"], axis=-1) - 1)) < 1e-4
    assert np.max(np.abs(np.linalg.norm(chair_quat_wxyz, axis=-1) - 1)) < 1e-5
    assert np.array_equal(motion["object_pos_w"][:, 0], sidecar["position_w"])
    assert np.array_equal(motion["object_quat_w"][:, 0], sidecar["quaternion_wxyz"])
    assert np.all(np.diff(timestamps) > 0)
    for path in (scene["scene"]["usd_path"], chair["usd_path"],
                 scene["camera"]["robot_usd"]):
        assert (SCALETRACK / path).is_file(), path
    validation = {
        "status": "kinematic_archive_structurally_valid_physics_not_validated",
        "frames": FRAMES,
        "fps": FPS,
        "source_recording_interval_s": [START_S, END_S],
        "last_recording_frame_s": START_S + (FRAMES - 1) / FPS,
        "object_sidecar_matches_main_motion": True,
        "scene_usd_references_exist": True,
        "fixed_dining_table": True,
        "stage4_physics_included": False,
        "visual_gt_generated": False,
        "training_data_ready": False,
    }
    (out / "metadata/validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (out / "README.md").write_text(
        f"# 1019 armchair push: {key}\n\n"
        f"[75.0, 84.5) s of recording 1-2026-10-19-16-31-26; 475 frames "
        f"at 50 Hz. {spec['contact']}. The scene's dining table is fixed. "
        "This is a visually accepted kinematic reference with an estimated "
        "chair trajectory; it contains no Stage 4 physical rollout or visual GT. "
        "`accepted` and `training_data_ready` remain false.\n"
    )
    provenance = {
        "source_recording_id": "1-2026-10-19-16-31-26",
        "source_recording_interval_s": [START_S, END_S],
        "variant": key,
        "source_motion": str(source / "motion.npz"),
        "source_motion_sha256": sha(source / "motion.npz"),
        "source_object_trajectory": str(source / "object_estimate.npz"),
        "source_object_trajectory_sha256": sha(source / "object_estimate.npz"),
        "source_stage_ledger": str(SNAPSHOT / "stage_ledger.json"),
        "visual_review": spec["review"],
        "physics_status": "stage4_open_for_interactive_repair_not_included",
        "object_trajectory_kind": "estimated_kinematic_chair_pose",
        "files_sha256": {
            str(path.relative_to(out)): sha(path)
            for path in sorted(out.rglob("*")) if path.is_file()
        },
    }
    (out / "metadata/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for spec in VARIANTS.values():
        if (root / spec["package"]).exists():
            parser.error(f"output already exists: {root / spec['package']}")
    for key, spec in VARIANTS.items():
        print(archive_one(key, spec, root))


if __name__ == "__main__":
    main()
