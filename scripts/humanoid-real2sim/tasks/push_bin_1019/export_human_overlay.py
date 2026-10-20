"""Export the 1019 BVH human skeleton into the Stage 3 world frame at 50 Hz.

Use ScaleBridge's Python environment with Torch and the local retarget2 vendor
directory on PYTHONPATH. The original human joint positions are only rotated
and translated in XY to align with the existing retarget root trajectory;
human limb lengths and heights are not rescaled.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from scipy.interpolate import interp1d

from preprocess_bvh import load_source_profiles
from preprocess_common import build_canonical_motion
from processors import get_processor


ROOT = example_dir(__file__)
BVH = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization/mocap.bvh")
PROFILES = Path("/home/galbot/workspace/ScaleTrack/retarget3/vendor/retarget2/source_profiles.json")
EDGES = (
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_sole"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_sole"),
    ("pelvis", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand_aux"), ("pelvis", "right_shoulder"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("right_wrist", "right_hand_aux"), ("left_shoulder", "right_shoulder"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, default=BVH)
    parser.add_argument("--motion", type=Path, default=ROOT / "retarget/hoi_candidate_scene.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "sidecars/human_motion_world.npz")
    args = parser.parse_args()
    with np.load(args.motion) as motion:
        robot_root = motion["body_pos_w"][:, 0, :2].astype(np.float64)
        fps = float(motion["fps"])
    lines = args.bvh.read_text().splitlines(keepends=True)
    marker = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    source_fps = 1.0 / float(lines[marker + 2].split(":", 1)[1])
    required = math.ceil((len(robot_root) - 1) / fps * source_fps) + 3
    lines[marker + 1] = f"Frames: {required}\n"
    profile = load_source_profiles(PROFILES)["optitrack_data"]
    with tempfile.TemporaryDirectory(prefix="rss_human_overlay_") as directory:
        clip = Path(directory) / "human_clip.bvh"
        clip.write_text("".join(lines[: marker + 3 + required]))
        canonical = build_canonical_motion(
            clip, profile_name="optitrack_data", profile=profile, spec=get_processor("optitrack_v1")
        )
    source_positions = np.asarray(canonical["positions"], dtype=np.float64)
    canonical_fps = float(canonical["fps"])
    timeline = np.arange(len(robot_root)) / fps
    human = interp1d(
        np.arange(len(source_positions)) / canonical_fps,
        source_positions, axis=0, bounds_error=False, fill_value="extrapolate",
    )(timeline)
    names = list(map(str, canonical["keypoint_names"]))
    human_root = human[:, names.index("pelvis"), :2]
    h0 = human_root - human_root.mean(axis=0)
    r0 = robot_root - robot_root.mean(axis=0)
    u, _, vt = np.linalg.svd(h0.T @ r0)
    rotation = u @ np.diag([1.0, np.linalg.det(u @ vt)]) @ vt
    translation = robot_root.mean(axis=0) - human_root.mean(axis=0) @ rotation
    human[:, :, :2] = human[:, :, :2] @ rotation + translation
    aligned_root = human[:, names.index("pelvis"), :2]
    rmse = float(np.sqrt(np.mean(np.sum((aligned_root - robot_root) ** 2, axis=1))))
    if rmse > 0.06:
        raise ValueError(f"human/robot planar alignment RMSE {rmse:.3f} m exceeds 60 mm")
    edges = np.asarray([(names.index(a), names.index(b)) for a, b in EDGES], dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        positions_w=human.astype(np.float32), joint_names=np.asarray(names), edges=edges,
        fps=np.asarray(fps), source_bvh=np.asarray(str(args.bvh)),
        source_fps=np.asarray(source_fps), canonical_fps=np.asarray(canonical_fps),
        xy_rotation=rotation, xy_translation_m=translation,
        root_alignment_rmse_m=np.asarray(rmse),
        note=np.asarray("Original human geometry; rigid XY world alignment only, no limb scaling"),
    )
    print(f"saved {args.out}: {len(human)} frames, root XY RMSE {rmse * 100:.1f} cm")


if __name__ == "__main__":
    main()
