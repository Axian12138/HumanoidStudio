"""Replay the projected hand proposal with the actual ScaleBridge tracker and RSS physics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir, SCRIPTS

import numpy as np

sys.path.insert(0, str(SCRIPTS))
from robosimstudio.humanoid_real2sim.track import launcher

from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=610)
    args = parser.parse_args()
    meta = json.loads(launcher.DEFAULT_METADATA.read_text())
    motion = ReferenceMotion.load(args.motion, joint_names=tuple(meta["joint_names"]),
                                  body_names=tuple(meta["selected_body_names"]),
                                  motion_order=launcher.DEFAULT_ORDER)
    temporary, asset = launcher._position_asset(launcher.DEFAULT_ROBOT, meta)
    tracker = None
    try:
        sim = launcher._load_scene_file(Path(__file__).with_name("scene.py"), asset, motion)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "scene_labels.json").write_text(json.dumps({
            "body_names": [str(n) for n in sim.model.body_label],
            "joint_names": [str(n) for n in sim.model.joint_label],
            "joint_q_start": sim.model.joint_q_start.numpy().astype(int).tolist(),
        }, indent=2) + "\n")
        if args.frames == 0:
            return
        tracker = BFMTracker(python=launcher.SCALEBRIDGE / ".venv/bin/python",
                             scalebridge=launcher.SCALEBRIDGE,
                             checkpoint=launcher.DEFAULT_CHECKPOINT,
                             metadata=launcher.DEFAULT_METADATA)
        runtime = RepairRuntime(sim, motion, tracker, original_model=launcher.DEFAULT_FK_MODEL,
                                motion_order=launcher.DEFAULT_ORDER)
        for frame in range(min(args.frames, motion.frames)):
            runtime.prepare()
            sim.step_teleop()
            runtime.record_after_step()
            if (frame + 1) % 50 == 0:
                print(f"physical replay {frame + 1}/{args.frames}", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        runtime.save(args.out)
        print(args.out / "rollout.npz", flush=True)
    finally:
        if tracker is not None:
            tracker.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
