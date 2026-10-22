"""Align task 09 OptiTrack hand rigid poses and the calibrated wrist camera.

The right camera chain is measured. The left rigid pose is retained as a
low-weight observation because this recording has no left rigid-to-camera
extrinsic in the checked calibration sources.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import sha256


TASK = example_dir(__file__)
SNAPSHOT = TASK.parents[1]
RECORDING = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
RETARGET2 = Path("/home/galbot/workspace/ScaleTrack/retarget2")
CAMERA_USD = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
    "g1_29dof_dex3/g1_29dof_dex3_cam_mat.usda"
)
SIDES = ("left", "right")


def transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def robot_wrist_camera_optical(side: str) -> np.ndarray:
    block = CAMERA_USD.read_text().split(f'def Camera "camera_hand_{side}"', 1)[1].split("}", 1)[0]
    quat_wxyz = np.asarray([float(x) for x in re.search(r"xformOp:orient = \(([^)]*)\)", block).group(1).split(",")])
    position = np.asarray([float(x) for x in re.search(r"xformOp:translate = \(([^)]*)\)", block).group(1).split(",")])
    optical = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix() @ np.diag([-1, 1, -1])
    return transform(optical, position)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(RETARGET2 / "src"), str(RETARGET2 / "scripts")]
    from extract_right_camera_targets import T_RIGHT_RIGID_CAMERA, parse_rigid_frame
    from mcap_skeleton_to_bvh import McapReader
    from processors import SOURCE_TO_CANONICAL

    with np.load(TASK / "source_sync.npz", allow_pickle=False) as sync:
        numbers = np.asarray(sync["source_motive_frame_number"], dtype=np.int64)
        source_time = np.asarray(sync["source_receive_timestamp_ns"], dtype=np.float64) / 1e9
        target_time = np.asarray(sync["receive_timestamp_ns"], dtype=np.float64) / 1e9
        recording_time = np.asarray(sync["recording_time_s"], dtype=np.float64)
    with np.load(SNAPSHOT / "tasks/07_armchair/source_sync.npz", allow_pickle=False) as sync:
        overlap_number = int(sync["source_motive_frame_number"][0])
    with np.load(SNAPSHOT / "tasks/07_armchair/hand_camera_reference.npz", allow_pickle=False) as reference:
        overlap_world = {side: np.asarray(reference[f"{side}_rigid_pos_w"][0], dtype=float) for side in SIDES}

    topics = {f"/optitrack/{side}_hand_rigid" for side in SIDES}
    reader = McapReader(RECORDING)
    for topic in topics:
        reader.require_topic(topic, "optitrack_msgs/msg/RigidBodyFrame")
    tracks = {side: {} for side in SIDES}
    first, last = int(numbers[0]), int(numbers[-1])
    for topic, sequence, log_time, _, payload in reader.iter_messages(topics):
        frame = parse_rigid_frame(sequence, log_time, payload)
        if first <= frame.motive_frame_number <= last or frame.motive_frame_number == overlap_number:
            side = "left" if topic.endswith("left_hand_rigid") else "right"
            tracks[side][frame.motive_frame_number] = frame
    missing = {side: sorted(set(map(int, numbers)) - set(tracks[side])) for side in SIDES}
    if any(missing.values()):
        raise ValueError(f"missing rigid source frames: {[len(missing[side]) for side in SIDES]}")

    source_to_world = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ SOURCE_TO_CANONICAL
    ground = []
    for side in SIDES:
        ground.append(float((source_to_world @ tracks[side][overlap_number].position)[2] - overlap_world[side][2]))
    if abs(ground[0] - ground[1]) > 0.01:
        raise ValueError(f"ground calibration disagrees between hands: {ground}")
    ground_height = float(np.mean(ground))

    result: dict[str, np.ndarray] = {}
    stats = {}
    for side in SIDES:
        selected = [tracks[side][int(n)] for n in numbers]
        xyz = np.asarray([frame.position for frame in selected]) @ source_to_world.T
        xyz[:, 2] -= ground_height
        rotation = Rotation.from_matrix(source_to_world) * Rotation.from_quat(
            np.asarray([frame.orientation_xyzw for frame in selected])
        )
        valid_source = np.asarray([frame.resolved and frame.present and frame.tracking_valid for frame in selected])
        right = np.searchsorted(source_time, target_time).clip(1, len(source_time) - 1)
        valid_target = valid_source[right - 1] & valid_source[right]
        pos = np.column_stack([np.interp(target_time, source_time, xyz[:, axis]) for axis in range(3)])
        quat = Slerp(source_time, rotation)(np.clip(target_time, source_time[0], source_time[-1])).as_quat()
        result[f"{side}_rigid_pos_w"] = pos.astype(np.float32)
        result[f"{side}_rigid_quat_xyzw"] = quat.astype(np.float32)
        result[f"{side}_valid"] = valid_target
        stats[side] = {"source_valid_fraction": float(valid_source.mean()), "target_valid_fraction": float(valid_target.mean())}

    real_camera = np.asarray([
        transform(R, p) @ T_RIGHT_RIGID_CAMERA
        for R, p in zip(Rotation.from_quat(result["right_rigid_quat_xyzw"]).as_matrix(), result["right_rigid_pos_w"])
    ])
    right_wrist = real_camera @ np.linalg.inv(robot_wrist_camera_optical("right"))
    result["right_camera_aligned_wrist_pos_w"] = right_wrist[:, :3, 3].astype(np.float32)
    result["right_camera_aligned_wrist_quat_xyzw"] = Rotation.from_matrix(right_wrist[:, :3, :3]).as_quat().astype(np.float32)
    result["T_right_rigid_real_camera_optical"] = T_RIGHT_RIGID_CAMERA
    for side in SIDES:
        result[f"T_robot_{side}_wrist_camera_optical"] = robot_wrist_camera_optical(side)
    result["recording_time_s"] = recording_time
    result["source_range_s"] = np.asarray([98.0, 109.0])
    result["ground_height_m"] = np.asarray(ground_height)
    np.savez_compressed(args.out / "wrist_reference.npz", **result)
    report = {
        "source_range_s": [98, 109], "frames": len(target_time), "tracking": stats,
        "ground_height_m": ground_height, "ground_disagreement_m": abs(ground[0] - ground[1]),
        "right_camera_chain": "T_world_right_rigid @ T_right_rigid_real_camera_optical @ inv(T_robot_right_wrist_camera_optical)",
        "left_camera_chain": "not_applied: real left rigid-to-camera extrinsic unavailable; retain raw left rigid 6D pose",
        "source_sha256": {str(path): sha256(path) for path in (RECORDING, CAMERA_USD, TASK / "source_sync.npz")},
    }
    (args.out / "wrist_reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.out / "wrist_reference.npz"), **report}, indent=2))


if __name__ == "__main__":
    main()
