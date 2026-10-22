"""Build a replacement ScaleTrack package from Stage 3 robot and basket references."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[6]
SCALETRACK = Path("/home/galbot/workspace/ScaleTrack")
PACKAGE_REL = "data/1019test/round-basket-foot-push-kinematic-v1"
SOURCE = SCALETRACK / PACKAGE_REL
RUN = REPO / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/stage4_round_basket_refit_minus_x_5cm_01"
BASKET = "round_laundry_basket"
FPS = 50
ROBOT_FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "fps")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_usd(output: Path) -> dict:
    source_usd = SCALETRACK / "source/scaletrack/scaletrack/assets/objects/1012/round_laundry_basket_v2.usd"
    target_usd = output / "assets/round_laundry_basket_stage5_v1.usd"
    target_usd.parent.mkdir(parents=True)
    shutil.copy2(source_usd, target_usd)
    editor = Path(__file__).with_name("set_basket_usd_physics.py")
    result = subprocess.run(
        ["blender", "-b", "--python", str(editor), "--", str(target_usd)], capture_output=True, text=True, check=True
    )
    report_path = output / "metadata/basket_usd_physics.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    if report is None:
        # The Blender helper writes beside the USD so it can also be used alone.
        report_path = target_usd.with_suffix(".physics.json")
        report = json.loads(report_path.read_text())
        shutil.move(report_path, output / "metadata/basket_usd_physics.json")
    if not report["verified"]:
        raise ValueError(f"USD physics verification failed: {result.stdout[-1500:]}")
    return report


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    metadata = output / "metadata"
    metadata.mkdir()
    with np.load(SOURCE / "metadata/source_motion.npz", allow_pickle=False) as source:
        robot = {key: source[key].copy() for key in ROBOT_FIELDS}
    with np.load(SOURCE / "motion.npz", allow_pickle=False) as old:
        for key in ROBOT_FIELDS:
            if not np.array_equal(robot[key], old[key]):
                raise ValueError(f"original archived robot field differs from Stage 3 source: {key}")
        sync = {key: old[key].copy() for key in ("timestamps", "recording_time_s", "source_motion_frame")}
    with np.load(RUN / "retarget/object_estimate.npz", allow_pickle=False) as track:
        position = track["position_w"].astype(np.float64)
        quat_xyzw = track["quaternion_xyzw"].astype(np.float64)
        source_kind = str(track["status"].item())
    if position.shape != (850, 3) or quat_xyzw.shape != (850, 4) or int(robot["fps"]) != FPS:
        raise ValueError("Stage 3 robot/object alignment is not 850 frames at 50 Hz")
    from scipy.spatial.transform import Rotation

    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    linear = np.gradient(position, 1 / FPS, axis=0)
    delta_rot = (Rotation.from_quat(quat_xyzw[:-1]).inv() * Rotation.from_quat(quat_xyzw[1:])).as_rotvec() * FPS
    angular = np.vstack((delta_rot, delta_rot[-1]))
    objects = {
        "object_names": np.asarray([BASKET]),
        "object_pos_w": position[:, None, :],
        "object_quat_w": quat_wxyz[:, None, :],
        "object_lin_vel_w": linear[:, None, :],
        "object_ang_vel_w": angular[:, None, :],
        **sync,
        "interacting_object_names": np.asarray([BASKET]),
        "object_trajectory_source": np.asarray(source_kind),
        "object_trajectory_measured": np.asarray(False),
    }
    np.savez_compressed(output / "motion.npz", **robot, **objects)
    np.savez_compressed(
        output / f"{BASKET}_trajectory.npz",
        object_name=np.asarray(BASKET),
        fps=np.asarray(FPS),
        **sync,
        position_w=position,
        quaternion_wxyz=quat_wxyz,
        linear_velocity_w=linear,
        angular_velocity_w=angular,
        kinematic_pose_defined=np.ones(850, dtype=bool),
        measured_gt_valid=np.zeros(850, dtype=bool),
        trajectory_source=np.asarray(source_kind),
    )
    copies = {
        SOURCE / "metadata/source_motion.npz": "metadata/source_motion.npz",
        RUN / "retarget/object_estimate.npz": "metadata/source_object_estimate.npz",
        RUN / "retarget/kinematic_review.json": "metadata/review/kinematic_basket_refit.json",
        SOURCE / "metadata/motion_order.json": "metadata/motion_order.json",
        SOURCE / "metadata/source_sync.npz": "metadata/source_sync.npz",
        SOURCE / "metadata/human.npz": "metadata/human.npz",
        SOURCE / "metadata/human_motion_world.npz": "metadata/human_motion_world.npz",
        SOURCE / "metadata/rigid_wrist_track.npz": "metadata/rigid_wrist_track.npz",
        SOURCE / "metadata/camera_calibration.json": "metadata/camera_calibration.json",
        SOURCE / "metadata/review/foot_basket_review.json": "metadata/review/foot_basket_review.json",
        SOURCE / "metadata/review/kneel_review.json": "metadata/review/kneel_review.json",
        SOURCE / "metadata/assets/morbylanga_table_1012.obj": "metadata/assets/morbylanga_table_1012.obj",
        SOURCE / "metadata/assets/round_laundry_basket_visual.obj": "metadata/assets/round_laundry_basket_visual.obj",
        SOURCE
        / "metadata/assets/round_laundry_basket_collision.obj": "metadata/assets/round_laundry_basket_collision.obj",
        REPO
        / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/stage_ledger.json": "metadata/source_stage_ledger.json",
    }
    for source, relative in copies.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    physics = make_usd(output)
    config = copy.deepcopy(next(iter(yaml.safe_load((SOURCE / "motions.yaml").read_text()).values())))
    config["path"] = f"{PACKAGE_REL}/motion.npz"
    config["source_frame_count"] = 850
    config["duration_seconds"] = 17.0
    config["archive_status"] = "stage3_kinematic_basket_refit_with_stage5_object_parameters"
    config["trajectory_kind"] = "estimated_kinematic_reference"
    config["visual_gt"] = "not_generated"
    config["visual_dataset_status"] = "invalidated_previous_basket_visual_features_need_regeneration"
    config["accepted"] = False
    config["training_data_ready"] = False
    config["object_trajectories"][BASKET]["path"] = f"{PACKAGE_REL}/{BASKET}_trajectory.npz"
    config["object_trajectories"][BASKET]["source"] = source_kind
    scene = config["scene_setup"]
    scene["status"] = "stage3_kinematic_reference_with_stage5_object_parameters"
    scene["visual_data_source"] = "not_generated"
    scene["runtime_ready"] = False
    for key in ("mask_semantics", "mask_semantic_timeline", "mask_field_compatibility"):
        scene.pop(key, None)
    scene["runtime_note"] = (
        "Object pose and physics revised; Stage 4 contact still did not follow the kinematic basket path."
    )
    scene["object"]["initial_position_m"] = position[0].tolist()
    scene["objects"][0]["initial_position_m"] = position[0].tolist()
    for basket in (scene["object"], scene["objects"][0]):
        basket["usd_path"] = f"{PACKAGE_REL}/assets/round_laundry_basket_stage5_v1.usd"
        basket["mass_kg"] = 1.6
        basket["center_of_mass_local_m"] = [0.0, 0.0, 0.08]
        basket["static_friction"] = 0.8
        basket["dynamic_friction"] = 0.8
        basket["physics_parameter_status"] = "stage5_candidate_from_RSS_free_rest_and_tracking_review"
        basket["trajectory_kind"] = "estimated_kinematic_reference"
    (output / "motions.yaml").write_text(
        yaml.safe_dump({"hoi1019_round_basket_foot_push_kinematic_v1": config}, sort_keys=False, allow_unicode=True)
    )
    (output / "README.md").write_text(
        "# 1019 round basket foot push: Stage 3 kinematic package\n\n"
        "Recording interval [59, 76) s; 850 frames at 50 Hz. The seven robot motion fields are exact copies "
        "of the original Stage 3 candidate. The basket initial world X position is 5 cm lower and its entire "
        "kinematic path was refit from the unchanged foot/shin motion. The basket USD variant has mass 1.6 kg, "
        "local center of mass (0, 0, 0.08) m, and static/dynamic friction 0.8. These physics parameters are "
        "Stage 5 tuning candidates. This package does not contain the Stage 4 physical rollout. Older RGB, "
        "mask, and DINO arrays were removed because they correspond to the superseded basket path; visual GT "
        "requires regeneration. The kinematic object path is estimated, not measured. Contact tracking and "
        "training acceptance remain pending.\n"
    )
    provenance = {
        "source_stage3_robot_motion": str(SOURCE / "metadata/source_motion.npz"),
        "source_stage3_robot_motion_sha256": sha256(SOURCE / "metadata/source_motion.npz"),
        "source_refit_object_track": str(RUN / "retarget/object_estimate.npz"),
        "source_refit_object_track_sha256": sha256(RUN / "retarget/object_estimate.npz"),
        "source_original_object_usd": str(
            SCALETRACK / "source/scaletrack/scaletrack/assets/objects/1012/round_laundry_basket_v2.usd"
        ),
        "source_original_object_usd_sha256": sha256(
            SCALETRACK / "source/scaletrack/scaletrack/assets/objects/1012/round_laundry_basket_v2.usd"
        ),
        "stage4_diagnostic_only": str(RUN / "review/trajectory_comparison.json"),
        "stage4_rollout_included": False,
        "robot_fields_unchanged": list(ROBOT_FIELDS),
        "removed_stale_visual_arrays": True,
        "physics_usd_verification": physics,
        "files_sha256": {
            str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    (metadata / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    validate(output)


def validate(output: Path) -> None:
    with (
        np.load(output / "motion.npz", allow_pickle=False) as data,
        np.load(output / f"{BASKET}_trajectory.npz", allow_pickle=False) as track,
        np.load(output / "metadata/source_motion.npz", allow_pickle=False) as robot,
    ):
        checks = {
            "robot_arrays_exact": all(np.array_equal(data[key], robot[key]) for key in ROBOT_FIELDS),
            "robot_order_and_shape": data["joint_pos"].shape == (850, 43) and data["body_pos_w"].shape == (850, 50, 3),
            "kinematic_basket_sidecar_matches": all(
                np.array_equal(data[main][:, 0], track[side])
                for main, side in (
                    ("object_pos_w", "position_w"),
                    ("object_quat_w", "quaternion_wxyz"),
                    ("object_lin_vel_w", "linear_velocity_w"),
                    ("object_ang_vel_w", "angular_velocity_w"),
                )
            ),
            "new_initial_x": abs(float(track["position_w"][0, 0]) - 0.07475284934043884) < 1e-6,
            "source_frame_mapping": np.array_equal(data["source_motion_frame"], np.arange(1600, 2450)),
            "no_stale_visual_arrays": not any(
                key.startswith(("rgb_", "dino_", "apple_mask_", "semantic_mask_")) for key in data.files
            ),
            "quaternion_norms": np.max(np.abs(np.linalg.norm(track["quaternion_wxyz"], axis=-1) - 1)) < 2e-3,
            "finite": all(np.isfinite(data[key]).all() for key in ROBOT_FIELDS if key != "fps")
            and np.isfinite(track["position_w"]).all(),
        }
    config = next(iter(yaml.safe_load((output / "motions.yaml").read_text()).values()))
    basket = config["scene_setup"]["object"]
    checks["yaml_points_to_new_usd"] = basket["usd_path"] == f"{PACKAGE_REL}/assets/round_laundry_basket_stage5_v1.usd"
    checks["yaml_and_usd_physics_match"] = (
        basket["mass_kg"] == 1.6
        and basket["center_of_mass_local_m"] == [0.0, 0.0, 0.08]
        and basket["static_friction"] == 0.8
        and basket["dynamic_friction"] == 0.8
        and json.loads((output / "metadata/basket_usd_physics.json").read_text())["verified"]
    )
    checks["no_stage4_rollout"] = not any(path.name == "rollout.npz" for path in output.rglob("*.npz"))
    checks["visual_gt_invalidated"] = (
        config["visual_gt"] == "not_generated" and not (output / "metadata/kinematic_capture").exists()
    )
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"archive validation failed: {[key for key, ok in checks.items() if not ok]}")
    (output / "metadata/validation.json").write_text(
        json.dumps(
            {
                "status": "stage3_kinematic_archive_replaced_with_stage5_object_parameters",
                "checks": checks,
                "frames": 850,
                "fps": FPS,
                "stage4_rollout_included": False,
                "visual_gt_generated_for_new_trajectory": False,
                "training_data_ready": False,
                "motion_npz_sha256": sha256(output / "motion.npz"),
                "basket_trajectory_sha256": sha256(output / f"{BASKET}_trajectory.npz"),
                "basket_usd_sha256": sha256(output / "assets/round_laundry_basket_stage5_v1.usd"),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validate(args.out) if args.validate_only else build(args.out)
