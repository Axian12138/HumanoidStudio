"""Build a 59–76 s diagnostic motion and independently tracked wrist overlays."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields


HERE = example_dir(__file__)
ROOT = HERE.parents[1]
OUTPUT = HERE / "review_59_76"
SOURCE = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
RETARGET2 = Path("/home/galbot/workspace/ScaleTrack/retarget2")
START, STOP, FPS = 59.0, 76.0, 50
SIDES = ("left", "right")


def read_rigid_tracks(numbers: np.ndarray) -> dict[str, dict[int, object]]:
    sys.path.insert(0, str(RETARGET2 / "src"))
    sys.path.insert(0, str(RETARGET2 / "scripts"))
    from extract_right_camera_targets import parse_rigid_frame
    from mcap_skeleton_to_bvh import McapReader

    reader = McapReader(SOURCE)
    topics = {f"/optitrack/{side}_hand_rigid" for side in SIDES}
    for topic in topics:
        reader.require_topic(topic, "optitrack_msgs/msg/RigidBodyFrame")
    tracks = {side: {} for side in SIDES}
    first, last = int(numbers[0]), int(numbers[-1])
    for topic, sequence, log_time, _, payload in reader.iter_messages(topics):
        frame = parse_rigid_frame(sequence, log_time, payload)
        if first <= frame.motive_frame_number <= last:
            side = "left" if topic.endswith("left_hand_rigid") else "right"
            tracks[side][frame.motive_frame_number] = frame
    if any(set(map(int, numbers)) - set(tracks[side]) for side in SIDES):
        raise ValueError("Motive rigid tracker frames are missing")
    return tracks


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    order = ROOT / "retarget/ground_support_27_end/sidecars/motion_order.json"
    prior = MotionFields.load(ROOT / "retarget/ground_support_27_end/supported_motion.npz", order)
    hoi = MotionFields.load(HERE / "hoi_v3_nonprehensile/motion.npz", order)
    begin, split, end = (round((time - 27.0) * FPS) for time in (START, 69.0, STOP))
    if len(hoi.joint_pos) != end - split:
        raise ValueError("69–76 s HOI candidate length changed")
    fields = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    values = {}
    for key in fields:
        full = getattr(prior, key)[begin:end].copy()
        full[split - begin:] = getattr(hoi, key)
        values[key] = full
    from dataclasses import replace

    motion = replace(prior, **values)
    motion.save(OUTPUT / "motion.npz")
    with np.load(ROOT / "retarget/ground_support_27_end/sidecars/human_motion_world.npz", allow_pickle=False) as data:
        human = np.asarray(data["positions_w"][begin:end], dtype=np.float32)
        names = np.asarray(data["joint_names"])
        edges = np.asarray(data["edges"])
    np.savez_compressed(
        OUTPUT / "human_motion_world.npz", positions_w=human, joint_names=names, edges=edges,
        fps=np.asarray(FPS), source_range_s=np.asarray([START, STOP]),
    )
    with np.load(ROOT / "retarget/body_prior_27_end/sidecars/human.npz", allow_pickle=False) as data:
        original_human = np.asarray(data["keypoints_w"][begin:end], dtype=np.float32)
    if not np.allclose(original_human, human, atol=1e-6):
        raise ValueError("59–76 s skeleton differs from the task's original human motion")
    np.savez_compressed(OUTPUT / "human_reference.npz", keypoints_w=original_human, fps=np.asarray(FPS))
    source_begin, source_end = (round((time - 27.0) * 120) for time in (START, STOP))
    with np.load(ROOT / "retarget/body_prior_27_end/sidecars/timing.npz", allow_pickle=False) as data:
        numbers = np.asarray(data["source_motive_frame_number"][source_begin:source_end], dtype=np.int64)
        source_time = np.asarray(data["source_receive_timestamp_ns"][source_begin:source_end], dtype=np.float64) / 1e9
        target_time = np.asarray(data["receive_timestamp_ns"][begin:end], dtype=np.float64) / 1e9
        recording_time = np.asarray(data["motion_time_s"][begin:end], dtype=np.float64) + 27.0
    np.savez_compressed(OUTPUT / "source_sync.npz", recording_time_s=recording_time)
    if len(numbers) != 2040 or len(target_time) != 850 or np.any(np.diff(numbers) != 1):
        raise ValueError("59–76 s source time mapping is not contiguous")
    with np.load(ROOT / "retarget/body_prior_27_end/intermediate/canonical.npz", allow_pickle=False) as data:
        ground = float(data["source_ground_height"])
    # Motive: X left, Y up, Z forward. Retarget world adds 90 degrees about Z.
    source_to_canonical = np.asarray([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    transform = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ source_to_canonical
    tracks = read_rigid_tracks(numbers)
    positions, quaternions, valid, proxy = [], [], [], []
    offsets, stats = {}, {}
    fit_window = (target_time >= target_time[0]) & (target_time < target_time[0] + 5)
    for side in SIDES:
        frames = [tracks[side][int(number)] for number in numbers]
        source_position = np.stack([frame.position for frame in frames]) @ transform.T
        source_position[:, 2] -= ground
        source_rotation = Rotation.from_matrix(transform) * Rotation.from_quat(
            np.stack([frame.orientation_xyzw for frame in frames])
        )
        position = np.column_stack([
            np.interp(target_time, source_time, source_position[:, axis]) for axis in range(3)
        ])
        rotation = Slerp(source_time, source_rotation)(np.clip(target_time, source_time[0], source_time[-1]))
        good = np.asarray([frame.resolved and frame.present and frame.tracking_valid for frame in frames])
        right = np.searchsorted(source_time, target_time).clip(1, len(source_time) - 1)
        interpolated_good = good[right - 1] & good[right]
        wrist = human[:, list(map(str, names)).index(f"{side}_wrist")]
        matrix = rotation.as_matrix()
        fit = fit_window & interpolated_good
        offset = np.linalg.lstsq(matrix[fit].reshape(-1, 3), (wrist[fit] - position[fit]).reshape(-1), rcond=None)[0]
        wrist_proxy = position + matrix @ offset
        residual = np.linalg.norm(wrist_proxy - wrist, axis=1)
        offsets[side] = offset.tolist()
        stats[side] = {
            "source_valid_fraction": float(good.mean()),
            "target_valid_fraction": float(interpolated_good.mean()),
            "offset_local_m": offset.tolist(),
            "fit_59_64_median_residual_m": float(np.median(residual[fit])),
            "review_69_76_median_residual_m": float(np.median(residual[(target_time >= target_time[0] + 10) & interpolated_good])),
        }
        positions.append(position)
        quaternions.append(rotation.as_quat())
        valid.append(interpolated_good)
        proxy.append(wrist_proxy)
    np.savez_compressed(
        OUTPUT / "rigid_wrist_track.npz",
        position_w=np.stack(positions, axis=1).astype(np.float32),
        quaternion_xyzw=np.stack(quaternions, axis=1).astype(np.float32),
        wrist_proxy_w=np.stack(proxy, axis=1).astype(np.float32),
        tracking_valid=np.stack(valid, axis=1),
        names=np.asarray(SIDES), fps=np.asarray(FPS), source_range_s=np.asarray([START, STOP]),
        motive_frame_number=numbers, offset_local_m=np.asarray([offsets[side] for side in SIDES]),
        source_topic=np.asarray([f"/optitrack/{side}_hand_rigid" for side in SIDES]),
    )
    # Only a visual hypothesis before 69 s; preserve the existing 69–76 estimate.
    time = START + np.arange(end - begin) / FPS
    knots = [59.0, 60.0, 62.0, 64.0, 66.0, 69.0]
    xy = [[0.16, -1.92], [0.16, -1.92], [0.19, -1.96], [0.22, -1.91], [0.25, -1.78], [0.25, -1.78]]
    estimated = np.column_stack((PchipInterpolator(knots, xy, axis=0)(time[:split - begin]),
                                 np.full(split - begin, 0.001)))
    with np.load(HERE / "hoi_v3_nonprehensile/object_estimate.npz", allow_pickle=False) as data:
        object_position = np.concatenate((estimated, data["position_w"]), axis=0)
        object_quaternion = np.concatenate((np.repeat([[0, 0, 0, 1]], len(estimated), axis=0),
                                            data["quaternion_xyzw"]), axis=0)
    np.savez_compressed(
        OUTPUT / "object_estimate.npz", position_w=object_position.astype(np.float32),
        quaternion_xyzw=object_quaternion.astype(np.float32), fps=np.asarray(FPS),
        status=np.asarray("pre69_video_phase_interpolation_hypothesis; post69_hoi_v3_estimate"),
    )
    report = {
        "source_window_s": [START, STOP],
        "motion_before_69_s": "body prior only; foot-to-basket HOI not yet retargeted",
        "motion_from_69_s": "hoi_v3_nonprehensile pending user review",
        "object_before_69_s": "provisional video-phase interpolation, not measured object tracking",
        "rigid_source": str(SOURCE), "rigid_transform": "Rz(90 deg) @ SOURCE_TO_CANONICAL; z -= source_ground_height",
        "rigid_offset_fit_window_s": [59, 64], "rigid_stats": stats,
    }
    (OUTPUT / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
