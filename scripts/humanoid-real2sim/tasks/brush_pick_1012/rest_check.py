"""Measure free brush and basket settling in RSS MuJoCo Warp."""

import argparse
import json
from pathlib import Path

import numpy as np

from scene import BRUSH_POSITION, BRUSH_QUAT_XYZW, build_rest_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--brush-y", type=float, default=BRUSH_POSITION[1])
    parser.add_argument("--brush-x", type=float, default=BRUSH_POSITION[0])
    parser.add_argument("--brush-z", type=float, default=BRUSH_POSITION[2])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--stage4-quat", action="store_true")
    parser.add_argument("--pitch-deg", type=float)
    args = parser.parse_args()
    initial = (args.brush_x, args.brush_y, args.brush_z)
    quat = (0.0, 0.16524102652752787, 0.0, 0.9862532145205555) if args.stage4_quat else BRUSH_QUAT_XYZW
    if args.pitch_deg is not None:
        angle = np.deg2rad(args.pitch_deg) / 2
        quat = (0.0, float(np.sin(angle)), 0.0, float(np.cos(angle)))
    sim = build_rest_sim(brush_position=initial, brush_quat=quat)
    samples = []
    for step in range(args.steps + 1):
        if step % 10 == 0:
            samples.append({
                "step": step,
                "brush_pose_xyzw": sim["green_hand_brush"].pose.tolist(),
                "brush_velocity": sim["green_hand_brush"].velocity.tolist(),
                "basket_pose_xyzw": sim["white_storage_box"].pose.tolist(),
            })
        if step < args.steps:
            sim.run(1).sync()
    positions = np.array([sample["brush_pose_xyzw"][:3] for sample in samples])
    window = positions[-6:]
    report = {
        "physics": "RSS MUJOCO_500HZ free rigid bodies",
        "initial_position_m": initial,
        "duration_s": args.steps / 50,
        "initial_pose_xyzw": samples[0]["brush_pose_xyzw"],
        "final_pose_xyzw": samples[-1]["brush_pose_xyzw"],
        "final_velocity": samples[-1]["brush_velocity"],
        "last_1s_position_range_m": np.ptp(window, axis=0).tolist(),
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
