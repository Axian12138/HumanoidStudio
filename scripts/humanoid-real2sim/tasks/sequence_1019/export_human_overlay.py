"""Expose the existing 27 s-end, world-aligned Motive skeleton to Stage 3.5."""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np


ROOT = example_dir(__file__)
SOURCE = ROOT / "retarget/body_prior_27_end/sidecars/human.npz"
OUTPUT = ROOT / "retarget/ground_support_27_end/sidecars/human_motion_world.npz"
NAMES = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_sole", "right_sole",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand_aux", "right_hand_aux",
    "pelvis_aux", "left_elbow_twist_aux", "right_elbow_twist_aux",
)
EDGE_NAMES = (
    ("pelvis", "left_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("left_ankle", "left_sole"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"), ("right_ankle", "right_sole"),
    ("pelvis", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("left_wrist", "left_hand_aux"),
    ("pelvis", "right_shoulder"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("right_wrist", "right_hand_aux"),
    ("left_shoulder", "right_shoulder"),
)


def main() -> None:
    with np.load(SOURCE, allow_pickle=False) as source:
        positions = np.asarray(source["keypoints_w"], dtype=np.float32)
        fps = float(source["fps"])
    with np.load(ROOT / "retarget/ground_support_27_end/supported_motion.npz", allow_pickle=False) as motion:
        robot_root = np.asarray(motion["body_pos_w"][:, 0], dtype=np.float32)
        motion_fps = float(motion["fps"])
    if positions.shape != (len(robot_root), len(NAMES), 3) or fps != motion_fps:
        raise ValueError("human and robot time axes do not match")
    if not np.isfinite(positions).all():
        raise ValueError("non-finite human positions")
    root_xy_error = np.linalg.norm(positions[:, 0, :2] - robot_root[:, :2], axis=1)
    if np.percentile(root_xy_error, 95) > 0.1:
        raise ValueError("human and robot world XY do not align")
    edges = np.asarray([(NAMES.index(a), NAMES.index(b)) for a, b in EDGE_NAMES], dtype=np.int32)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT, positions_w=positions, joint_names=np.asarray(NAMES), edges=edges,
        fps=np.asarray(fps), source=np.asarray(str(SOURCE)),
        alignment=np.asarray("retarget3 canonical_to_world_yaw_deg=90, translation=(0,0,0); no additional transform or limb scaling"),
    )
    print(f"{OUTPUT}: {len(positions)} frames, root XY p95 {np.percentile(root_xy_error, 95):.3f} m")


if __name__ == "__main__":
    main()
