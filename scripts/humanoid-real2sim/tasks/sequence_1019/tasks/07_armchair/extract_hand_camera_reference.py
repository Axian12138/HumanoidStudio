"""Extract 75–91 s hand rigid poses and a camera-aligned right wrist prior."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from fit_hoi import HERE, MODEL


RECORDING = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
RETARGET2 = Path("/home/galbot/workspace/ScaleTrack/retarget2")
ROBOT_USD = MODEL.with_name("g1_29dof_dex3_offsethand_cam.usda")
OUT = HERE / "hand_camera_reference.npz"


def transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def right_robot_camera_optical() -> np.ndarray:
    text = ROBOT_USD.read_text()
    block = text.split('def Camera "camera_hand_right"', 1)[1].split("}", 1)[0]
    q = np.asarray([float(x) for x in re.search(r"xformOp:orient = \(([^)]*)\)", block).group(1).split(",")])
    xyz = np.asarray([float(x) for x in re.search(r"xformOp:translate = \(([^)]*)\)", block).group(1).split(",")])
    # The robot's USD camera orientation differs from the calibrated optical
    # camera by the documented axis signs. The resulting matrix agrees with
    # camera_calibration.json within approximately 1.3 degrees.
    optical = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix() @ np.diag([-1, 1, -1])
    return transform(optical, xyz)


def main() -> None:
    sys.path.insert(0, str(RETARGET2 / "src"))
    sys.path.insert(0, str(RETARGET2 / "scripts"))
    from extract_right_camera_targets import T_RIGHT_RIGID_CAMERA, parse_rigid_frame
    from mcap_skeleton_to_bvh import McapReader
    from processors import SOURCE_TO_CANONICAL

    with np.load(HERE / "source_sync.npz", allow_pickle=False) as source:
        numbers = source["source_motive_frame_number"]
        source_time = source["source_receive_timestamp_ns"].astype(np.float64) / 1e9
        target_time = source["receive_timestamp_ns"].astype(np.float64) / 1e9
    source_to_world = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ SOURCE_TO_CANONICAL
    topics = {f"/optitrack/{side}_hand_rigid" for side in ("left", "right")}
    reader = McapReader(RECORDING)
    for topic in topics:
        reader.require_topic(topic, "optitrack_msgs/msg/RigidBodyFrame")
    tracks = {side: {} for side in ("left", "right")}
    first, last = int(numbers[0]), int(numbers[-1])
    for topic, sequence, log_time, _, payload in reader.iter_messages(topics):
        rigid = parse_rigid_frame(sequence, log_time, payload)
        if first <= rigid.motive_frame_number <= last:
            side = "left" if topic.endswith("left_hand_rigid") else "right"
            tracks[side][rigid.motive_frame_number] = rigid

    # The canonical intermediate is no longer in this checkout. The earlier
    # 59–76 s rigid diagnostic used the same MCAP, world transform and source
    # ground height; recover that scalar from its overlapping 75 s sample.
    overlap = HERE.parent / "06_round_basket/review_59_76/rigid_wrist_track.npz"
    with np.load(overlap, allow_pickle=False) as prior_rigid:
        old_pos = prior_rigid["position_w"][round((75 - 59) * 50)]
    grounds = []
    for side_i, side in enumerate(("left", "right")):
        raw_z = (source_to_world @ tracks[side][int(numbers[0])].position)[2]
        grounds.append(raw_z - old_pos[side_i, 2])
    if abs(grounds[0] - grounds[1]) > .01:
        raise ValueError(f"inconsistent recovered ground heights: {grounds}")
    ground = float(np.mean(grounds))

    result = {}
    for side in ("left", "right"):
        selected = [tracks[side].get(int(n)) for n in numbers]
        valid = np.asarray([p is not None and p.resolved and p.present and p.tracking_valid for p in selected])
        if not valid.all():
            raise ValueError(f"{side} rigid missing or invalid: {np.count_nonzero(~valid)} of {len(valid)}")
        xyz = np.asarray([p.position for p in selected]) @ source_to_world.T
        xyz[:, 2] -= ground
        rotation = Rotation.from_matrix(source_to_world) * Rotation.from_quat(
            np.asarray([p.orientation_xyzw for p in selected])
        )
        position_50 = np.column_stack([np.interp(target_time, source_time, xyz[:, axis]) for axis in range(3)])
        rotation_50 = Slerp(source_time, rotation)(np.clip(target_time, source_time[0], source_time[-1]))
        result[f"{side}_rigid_pos_w"] = position_50.astype(np.float32)
        result[f"{side}_rigid_quat_xyzw"] = rotation_50.as_quat().astype(np.float32)
    camera = np.asarray([transform(R, p) @ T_RIGHT_RIGID_CAMERA
                         for R, p in zip(Rotation.from_quat(result["right_rigid_quat_xyzw"]).as_matrix(),
                                         result["right_rigid_pos_w"])])
    wrist_from_camera = np.linalg.inv(right_robot_camera_optical())
    wrist_proxy = camera @ wrist_from_camera
    result["right_camera_pos_w"] = camera[:, :3, 3].astype(np.float32)
    result["right_camera_quat_xyzw"] = Rotation.from_matrix(camera[:, :3, :3]).as_quat().astype(np.float32)
    result["right_wrist_camera_aligned_pos_w"] = wrist_proxy[:, :3, 3].astype(np.float32)
    result["right_wrist_camera_aligned_quat_xyzw"] = Rotation.from_matrix(wrist_proxy[:, :3, :3]).as_quat().astype(np.float32)
    result["T_right_rigid_real_camera_optical"] = T_RIGHT_RIGID_CAMERA
    result["T_robot_wrist_robot_camera_optical"] = right_robot_camera_optical()
    result["tracking_valid"] = np.ones((len(target_time), 2), dtype=bool)
    result["recording_time_s"] = np.arange(len(target_time)) / 50 + 75
    result["source_mcap"] = np.asarray(str(RECORDING))
    result["robot_camera_usd"] = np.asarray(str(ROBOT_USD))
    np.savez_compressed(OUT, **result)
    print(json.dumps({"output": str(OUT), "frames": len(target_time),
                      "right_wrist_camera_aligned_at_81_s": wrist_proxy[round((81 - 75) * 50), :3, 3].tolist()}, indent=2))


if __name__ == "__main__":
    main()
