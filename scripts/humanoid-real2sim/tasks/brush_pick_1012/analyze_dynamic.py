"""Summarize actual RSS brush motion and hand proximity in a rollout."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.rollout)
    names = (list(data["body_names"]) if "body_names" in data
             else json.loads(args.rollout.with_name("scene_labels.json").read_text())["body_names"])
    brush = names.index("green_hand_brush")
    right_fingers = [
        index for index, name in enumerate(names)
        if name.rsplit("/", 1)[-1] in (
            "right_hand_thumb_2_link", "right_hand_middle_1_link", "right_hand_index_1_link"
        )
    ]
    pose = data["achieved_body_q"]
    position = pose[:, brush, :3]
    grip = position + Rotation.from_quat(pose[:, brush, 3:7]).apply(
        np.broadcast_to(np.array([-0.065, 0.0, 0.0609027427]), position.shape)
    )
    distances = np.linalg.norm(pose[:, right_fingers, :3] - grip[:, None, :], axis=-1)
    # Each executed source frame k is stored at sample k+1.
    start = 501
    contact = 526
    lift = slice(546, 601)
    height = position[:, 2]
    sustained = np.convolve((height[lift] >= height[start] + 0.05).astype(int), np.ones(20, int), "valid")
    report = {
        "rollout": str(args.rollout.resolve()),
        "sample_count": len(position),
        "actual_brush_position_m": {
            str(frame): position[frame + 1].tolist() for frame in (500, 525, 550, 575, 599)
        },
        "brush_lift_at_12s_m": float(height[600] - height[start]),
        "max_lift_after_10p9s_m": float(np.max(height[lift]) - height[start]),
        "sustained_5cm_lift_for_0p4s": bool(np.any(sustained == 20)),
        "brush_displacement_10p5_to_12s_m": float(np.linalg.norm(position[600] - position[contact])),
        "finger_link_origin_min_distance_to_grip_10to11s_m": {
            names[index].rsplit("/", 1)[-1]: float(np.min(distances[501:551, column]))
            for column, index in enumerate(right_fingers)
        },
        "contact_force_available": False,
        "finger_joint_state_available": False,
        "grasp_verified": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
