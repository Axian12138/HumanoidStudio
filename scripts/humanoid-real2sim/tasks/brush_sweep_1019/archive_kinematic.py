"""Archive the reviewed 1019 kinematic candidate without stage 4/5 claims.

Run with the ScaleTrack Python environment, which provides PyYAML and SciPy.
The output directory must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import SCRIPTS, example_dir

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


HERE = example_dir(__file__)
SCALETRACK = Path("/home/galbot/workspace/ScaleTrack")
PUSH = SCALETRACK / "data/1019test/push-bin-kinematic-v1"
BRUSH = HERE / "retarget/push_to_brush_kinematic_v3"
NAME = "hoi1019_push_bin_brush_sweep_kinematic_v1"
REL = "data/1019test/push-bin-brush-sweep-kinematic-v1"
OBJECTS = ("white_waste_bin", "white_storage_box", "green_hand_brush")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wxyz_from_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def velocities(pos: np.ndarray, quat_wxyz: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    linear = np.gradient(pos, 1.0 / fps, axis=0)
    xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(xyzw)
    delta = (rot[:-1].inv() * rot[1:]).as_rotvec() * fps
    angular = np.empty_like(pos)
    angular[:-1] = delta
    angular[-1] = delta[-1]
    return linear, angular


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined", type=Path, default=BRUSH,
                        help="reviewed combined motion directory")
    parser.add_argument("--package-id", default=NAME)
    parser.add_argument("--relative-path", default=REL,
                        help="path under ScaleTrack used in motions.yaml")
    parser.add_argument("--kinematic-reference-accepted", action=argparse.BooleanOptionalAction,
                        default=True, help="whether the user accepted this exact kinematic candidate")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="first frame of the combined motion to include; 425 gives 0.98 s before the brush splice")
    args = parser.parse_args()
    brush = args.combined.resolve()
    package_id = args.package_id
    relative_path = args.relative_path
    out = args.output.resolve()
    if out.exists():
        parser.error(f"output already exists: {out}")
    (out / "metadata/object_trajectories").mkdir(parents=True)

    with np.load(brush / "motion.npz", allow_pickle=False) as motion:
        full_n = len(motion["joint_pos"])
        if not 0 <= args.start_frame < full_n:
            parser.error(f"start-frame must be in [0, {full_n})")
        robot = {
            key: (motion[key][args.start_frame:] if motion[key].ndim and
                  motion[key].shape[0] == full_n else motion[key])
            for key in motion.files
        }
    with np.load(brush / "objects_estimated.npz", allow_pickle=False) as objects:
        raw = {key: objects[key] for key in objects.files}
    n = len(robot["joint_pos"])
    fps = float(robot["fps"])
    assert full_n == 1399 and fps == 50.0
    timestamps = np.arange(n, dtype=np.float64) / fps
    source_frames = np.arange(args.start_frame, full_n, dtype=np.int64)

    positions = [
        np.asarray(raw["white_waste_bin_position_w"][args.start_frame:], dtype=np.float64),
        np.broadcast_to(np.array([0.944999933, -0.569999993, 0.400999993]), (n, 3)).copy(),
        np.asarray(raw["green_hand_brush_position_w"][args.start_frame:], dtype=np.float64),
    ]
    quats = [
        np.asarray(raw["white_waste_bin_quaternion_wxyz"][args.start_frame:], dtype=np.float64),
        np.broadcast_to(np.array([0.7071067812, 0.0, 0.0, 0.7071067812]), (n, 4)).copy(),
        wxyz_from_xyzw(np.asarray(raw["green_hand_brush_quaternion_xyzw"][args.start_frame:], dtype=np.float64)),
    ]
    linear, angular = zip(*(velocities(p, q, fps) for p, q in zip(positions, quats)))
    object_pos = np.stack(positions, axis=1)
    object_quat = np.stack(quats, axis=1)
    object_lin = np.stack(linear, axis=1)
    object_ang = np.stack(angular, axis=1)
    fields = dict(robot)
    fields.update(
        object_names=np.asarray(OBJECTS),
        object_pos_w=object_pos,
        object_quat_w=object_quat,
        object_lin_vel_w=object_lin,
        object_ang_vel_w=object_ang,
        timestamps=timestamps,
        source_motion_frame=source_frames,
        interacting_object_names=np.asarray(("green_hand_brush",) if args.start_frame else
                                            ("white_waste_bin", "green_hand_brush")),
        object_trajectory_source=np.asarray("kinematic_estimate_no_physics"),
        object_trajectory_measured=np.asarray(False),
    )
    np.savez_compressed(out / "motion.npz", **fields)

    for index, name in enumerate(OBJECTS):
        np.savez_compressed(
            out / "metadata/object_trajectories" / f"{name}.npz",
            object_name=np.asarray(name),
            fps=np.asarray(int(fps)),
            timestamps=timestamps,
            source_motion_frame=source_frames,
            position_w=object_pos[:, index],
            quaternion_wxyz=object_quat[:, index],
            linear_velocity_w=object_lin[:, index],
            angular_velocity_w=object_ang[:, index],
            kinematic_pose_defined=np.ones(n, dtype=bool),
            measured_gt_valid=np.zeros(n, dtype=bool),
            trajectory_source=np.asarray("kinematic_estimate_no_physics"),
        )

    copies = {
        HERE / "source/source_manifest.json": "metadata/source_manifest.json",
        brush / "source_sync.npz": "metadata/source_sync.npz",
        (brush / "human_motion_world.npz" if (brush / "human_motion_world.npz").exists()
         else BRUSH / "human_motion_world.npz"): "metadata/human_motion_world.npz",
        HERE / "sidecars/contact_schedule.json": "metadata/contact_schedule.json",
        HERE / "stage_ledger.json": "metadata/source_stage_ledger.json",
        SCRIPTS / "tasks/brush_sweep_1019/scene.py": "metadata/rss_scene.py",
        HERE / "review/pick_audit_v20.json": "metadata/review/pick_audit_v20.json",
        HERE / "review/hand_audit_v20.json": "metadata/review/hand_audit_v20.json",
        HERE / "review/kinematic_audit.json": "metadata/review/kinematic_audit.json",
        brush / "review.json": "metadata/review/splice_review.json",
        PUSH / "metadata/motion_order.json": "metadata/motion_order.json",
    }
    if brush != BRUSH.resolve():
        copies.update({
            HERE / "review/knee_audit_v21.json": "metadata/review/knee_audit_v21.json",
            HERE / "retarget/brush_sweep_kinematic_v21_knee/review.json":
                "metadata/review/knee_hoi_composition_v21.json",
            HERE / "retarget/right_knee_ground_v3/review/contact_schedule.json":
                "metadata/review/right_knee_contact_schedule.json",
        })
    if "cycle" in brush.name:
        copies.update({
            HERE / "review/sweep_cycle_audit_v22.json": "metadata/review/sweep_cycle_audit_v22.json",
            HERE / "review/knee_audit_v22.json": "metadata/review/knee_audit_v22.json",
            HERE / "retarget/brush_sweep_kinematic_v22_cycle/review.json":
                "metadata/review/sweep_cycle_fit_v22.json",
        })
    for source, rel in copies.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if args.start_frame:
        with np.load(brush / "source_sync.npz", allow_pickle=False) as sync:
            sync_fields = {
                key: (sync[key][args.start_frame:] if sync[key].ndim and
                      sync[key].shape[0] == full_n else sync[key])
                for key in sync.files
            }
        sync_fields["source_combined_motion_time_s"] = sync_fields["motion_time_s"].copy()
        sync_fields["motion_time_s"] = timestamps
        sync_fields["source_combined_motion_frame"] = source_frames
        np.savez_compressed(out / "metadata/source_sync.npz", **sync_fields)
        human_source = brush / "human_motion_world.npz"
        if not human_source.exists():
            human_source = BRUSH / "human_motion_world.npz"
        with np.load(human_source, allow_pickle=False) as human:
            human_fields = {
                key: (human[key][args.start_frame:] if human[key].ndim and
                      human[key].shape[0] == full_n else human[key])
                for key in human.files
            }
        np.savez_compressed(out / "metadata/human_motion_world.npz", **human_fields)
        schedule_path = out / "metadata/contact_schedule.json"
        schedule = json.loads(schedule_path.read_text())
        offset_s = args.start_frame / fps
        schedule["clip_start_source_time_s"] = offset_s
        for event in schedule["observed_events"]:
            event["clip_time_s"] = [round(value - offset_s, 6) for value in event["source_time_s"]]
            for key in ("source_start", "source_stop"):
                event[key]["clip_motion_frame_50hz"] = (
                    event[key]["motion_frame_global_50hz"] - args.start_frame
                )
        commands = schedule["designed_dex3_commands_s"]
        schedule["designed_dex3_commands_source_s"] = commands.copy()
        schedule["designed_dex3_commands_s"] = {
            key: round(value - offset_s, 6) for key, value in commands.items()
        }
        interval = schedule["estimated_object_attachment_interval_s"]
        schedule["estimated_object_attachment_interval_source_s"] = interval.copy()
        schedule["estimated_object_attachment_interval_s"] = [
            round(value - offset_s, 6) for value in interval
        ]
        schedule_path.write_text(json.dumps(schedule, indent=2, ensure_ascii=False) + "\n")

    with (PUSH / "motions.yaml").open() as stream:
        config = yaml.safe_load(stream)["hoi1019_push_bin_kinematic_v1"]
    with (SCALETRACK / "data/1012test/1012-1/stage7-visual-gt/motions.yaml").open() as stream:
        older = next(iter(yaml.safe_load(stream).values()))
    config["path"] = f"{relative_path}/motion.npz"
    config["source_frame_count"] = n
    config["duration_seconds"] = n / fps
    config["source_combined_frame_interval"] = [args.start_frame, full_n]
    config["source_combined_time_interval_seconds"] = [args.start_frame / fps, full_n / fps]
    config["accepted"] = False
    config["training_data_ready"] = False
    config["archive_status"] = "kinematic_only_user_requested_skip_stages_4_5"
    config["kinematic_reference_accepted_by_user"] = args.kinematic_reference_accepted
    config["trajectory_kind"] = "estimated_kinematic_reference"
    config["visual_dataset_status"] = "not_generated_for_combined_motion"
    config["visual_gt"] = "not_generated"
    config.pop("original_source_frame_count", None)
    config["slice_interval_seconds"] = [0.0, n / fps]
    scene = config["scene_setup"]
    scene["status"] = "stage3_kinematic_archive_no_stage4_or_stage5"
    scene["runtime_ready"] = False
    scene["runtime_note"] = "No tracker rollout, physical contact validation, or new visual GT."
    scene["visual_data_source"] = "not_generated"
    scene["live_rgb_use_semantic_mask"] = False
    scene.pop("mask_semantic_timeline", None)
    scene.pop("mask_field_compatibility", None)
    scene["hoi_contact_schedule"] = f"{relative_path}/metadata/contact_schedule.json"
    scene.pop("hoi_object_trajectories_estimated", None)
    template_objects = {item["id"]: item for item in older["scene_setup"]["objects"]}
    by_name = {item["id"]: item for item in scene["objects"]}
    for name in OBJECTS[1:]:
        by_name[name] = dict(template_objects[name])
        by_name[name]["trajectory_kind"] = "estimated_kinematic_reference"
    for index, name in enumerate(OBJECTS):
        item = by_name[name]
        item["initial_position_m"] = object_pos[0, index].tolist()
        item["initial_orientation_wxyz"] = object_quat[0, index].tolist()
    scene["objects"] = [by_name[name] for name in OBJECTS]
    scene["object"] = by_name["green_hand_brush" if args.start_frame else "white_waste_bin"]
    config["object_trajectories"] = {
        name: {
            "path": f"{relative_path}/metadata/object_trajectories/{name}.npz",
            "npz_object_index": index,
            "interacts_in_this_clip": name == "green_hand_brush" if args.start_frame else
                                      name != "white_storage_box",
            "coordinate_frame": "IKEA_1012_world",
            "quaternion_order": "wxyz",
            "measured": False,
            "source": "kinematic_estimate_no_physics",
        }
        for index, name in enumerate(OBJECTS)
    }
    config["interaction_events"] = [
        {"object": "green_hand_brush", "timing_source": f"{relative_path}/metadata/contact_schedule.json"},
    ] if args.start_frame else [
        {"object": "white_waste_bin", "timing_source": "source_push_kinematic_prior"},
        {"object": "green_hand_brush", "timing_source": f"{relative_path}/metadata/contact_schedule.json"},
    ]
    if args.start_frame:
        config["visual_dataset_status"] = "not_generated_for_brush_motion"
        config["brush_splice_frame_in_clip"] = 474 - args.start_frame
        config["pre_splice_overlap_seconds"] = (474 - args.start_frame) / fps
    (out / "motions.yaml").write_text(yaml.safe_dump({package_id: config}, sort_keys=False, allow_unicode=True))

    assert all(np.isfinite(value).all() for value in (
        robot["joint_pos"], robot["joint_vel"], robot["body_pos_w"], robot["body_quat_w"],
        object_pos, object_quat, object_lin, object_ang,
    ))
    assert np.max(np.abs(np.linalg.norm(robot["body_quat_w"], axis=-1) - 1)) < 1e-3
    assert np.max(np.abs(np.linalg.norm(object_quat, axis=-1) - 1)) < 1e-5
    for item in scene["objects"]:
        assert (SCALETRACK / item["usd_path"]).is_file(), item["usd_path"]
    validation = {
        "status": "kinematic_archive_structurally_valid_not_physics_validated",
        "frames": n,
        "fps": fps,
        "source_combined_frame_interval": [args.start_frame, full_n],
        "robot_joint_count": robot["joint_pos"].shape[1],
        "robot_body_count": robot["body_pos_w"].shape[1],
        "objects": list(OBJECTS),
        "finite_values": True,
        "quaternion_norms_valid": True,
        "usd_references_exist": True,
        "object_sidecars_match_main_motion": True,
        "tracker_rollout": False,
        "physical_grasp_or_sweep_verified": False,
        "visual_gt_generated": False,
        "training_data_ready": False,
    }
    (out / "metadata/validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (out / "README.md").write_text(
        ("# 1019 pick brush → sweep table → return brush\n\n" if args.start_frame else
         "# 1019 push bin → pick brush → sweep table → return brush\n\n") +
        f"This is a {n}-frame, 50 Hz **kinematic reference archive**. The user requested "
        "archival directly after retarget and skipped stages 4 and 5. `motion.npz` contains "
        "the seven ScaleTrack robot fields plus three estimated object trajectories. " +
        (f"The first {(474 - args.start_frame) / fps:.2f} s overlaps the end of the bin motion; "
         "the bin is stationary in this clip. " if args.start_frame else "") +
        "The brush path is a kinematic attachment estimate; no tracker rollout, physical "
        "grasp, sweep contact, release validation, or new visual GT is claimed. "
        f"The clip starts at source frame {args.start_frame} ({args.start_frame / fps:.2f} s). "
        "`accepted` and `training_data_ready` remain false. See `metadata/provenance.json`, "
        "`metadata/validation.json`, and `motions.yaml`.\n"
    )

    provenance = {
        "archive_status": config["archive_status"],
        "stages_completed": ["1_input", "2_scene_reuse", "3_body_prior_and_hoi"],
        "stages_skipped_by_user": ["4_tracker", "5_physical_object_review"],
        "accepted": False,
        "training_data_ready": False,
        "visual_gt": "not_generated",
        "source_recording": str(HERE / "source/source_manifest.json"),
        "source_motion": str(brush / "motion.npz"),
        "source_motion_sha256": sha(brush / "motion.npz"),
        "source_combined_frame_interval": [args.start_frame, full_n],
        "source_combined_time_interval_seconds": [args.start_frame / fps, full_n / fps],
        "pre_splice_overlap_seconds": (474 - args.start_frame) / fps if args.start_frame else None,
        "scene_source": str(SCRIPTS / "tasks/brush_sweep_1019/scene.py"),
        "object_trajectory_source": "estimated kinematic sidecar; no physical grasp/contact claim",
        "files_sha256": {
            str(path.relative_to(out)): sha(path)
            for path in sorted(out.rglob("*")) if path.is_file()
        },
    }
    (out / "metadata/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
