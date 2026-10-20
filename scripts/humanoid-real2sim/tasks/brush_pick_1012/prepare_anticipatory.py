"""Advance the body approach without changing the fixed Dex3 schedule."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

ROOT = example_dir(__file__)
SCALETRACK = Path("/home/galbot/workspace/ScaleTrack")
ORDER = SCALETRACK / "retarget3/assets/robot/motion_order.json"
MODEL = SCALETRACK / "retarget3/assets/robot/robot.xml"
SOURCE = ROOT / "retarget/kinematic_motion.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--advance-frames", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = MotionFields.load(SOURCE, ORDER)
    n = len(source.joint_pos)
    frames = np.arange(n, dtype=float)
    source_frame = frames.copy()
    start, stop = 400, 525
    phase = (frames[start:stop + 1] - start) / (stop - start)
    source_frame[start:stop + 1] += args.advance_frames * np.sin(np.pi * phase) ** 2
    # Preserve the fixed right-hand opening/closing commands at original times.
    fingers = [i for i, name in enumerate(source.joint_names) if name.startswith("right_hand_")]
    joints = np.stack([
        np.interp(source_frame, frames, source.joint_pos[:, i])
        for i in range(len(source.joint_names))
    ], axis=1)
    joints[:, fingers] = source.joint_pos[:, fingers]
    root_pos = np.stack([
        np.interp(source_frame, frames, source.body_pos_w[:, 0, i]) for i in range(3)
    ], axis=1)
    rotations = Rotation.from_quat(source.body_quat_w[:, 0, [1, 2, 3, 0]])
    root_quat = Slerp(frames, rotations)(source_frame).as_quat()[:, [3, 0, 1, 2]]
    body_pos = source.body_pos_w.copy()
    body_quat = source.body_quat_w.copy()
    body_pos[:, 0] = root_pos
    body_quat[:, 0] = root_quat
    base = replace(source, joint_pos=joints, body_pos_w=body_pos, body_quat_w=body_quat)
    candidate = refresh_fk(base, MODEL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    candidate.save(args.out)
    change = np.max(np.abs(candidate.joint_pos - source.joint_pos), axis=0)
    report = {
        "source": str(SOURCE.resolve()), "source_sha256": sha256(SOURCE),
        "output": str(args.out.resolve()), "output_sha256": sha256(args.out),
        "advance_peak_frames": args.advance_frames,
        "window_frames": [start, stop],
        "right_hand_schedule_unchanged": bool(np.array_equal(candidate.joint_pos[:, fingers], source.joint_pos[:, fingers])),
        "max_changed_joint_rad": float(change.max()),
        "max_joint_speed_400_525_rad_s": float(np.max(np.abs(candidate.joint_vel[start:stop + 1]))),
        "max_root_speed_400_525_m_s": float(np.max(np.linalg.norm(candidate.body_lin_vel_w[start:stop + 1, 0], axis=1))),
        "source_frame_map": source_frame[start:stop + 1].tolist(),
        "status": "unreviewed_kinematic_candidate",
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "source_frame_map"}, indent=2))


if __name__ == "__main__":
    main()
