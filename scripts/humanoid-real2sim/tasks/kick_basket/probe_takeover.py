"""Probe one bounded left-ankle takeover without exporting an accepted motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import baseline
import numpy as np

from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker
from robosimstudio.scene.entity import label_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=60)
    parser.add_argument("--takeover-steps", type=int, default=15)
    parser.add_argument("--direction", choices=("-X", "-Y", "+Z"), default="-X")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    launcher = baseline.launcher
    metadata_path = example_dir(__file__) / "tracker_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    motion = ReferenceMotion.load(
        Path("/home/galbot/workspace/ScaleTrack/data/1012test/kick-basket-v1/motion.npz"),
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
        motion_order=launcher.DEFAULT_ORDER,
    )
    temporary, asset = launcher._position_asset(launcher.DEFAULT_ROBOT, metadata)
    tracker = None
    try:
        sim = launcher._load_scene_file(Path(__file__).with_name("scene.py"), asset, motion)
        tracker = BFMTracker(
            python=launcher.SCALETRACK / ".venv/bin/python",
            scalebridge=launcher.SCALEBRIDGE,
            checkpoint=Path(metadata["checkpoint"]),
            metadata=metadata_path,
        )
        runtime = RepairRuntime(sim, motion, tracker)
        basket_index = label_index(list(sim.model.body_label), "round_laundry_basket")
        basket_poses = [sim.pipeline.snapshot.body_q[basket_index].copy()]

        def advance() -> None:
            runtime.prepare()
            sim.step_teleop()
            runtime.record_after_step()
            basket_poses.append(sim.pipeline.snapshot.body_q[basket_index].copy())

        for _ in range(args.start):
            advance()
        runtime.begin_takeover()
        direction = {"-X": (-1, 0, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1)}[args.direction]
        runtime.timeline.set_velocity("left_ankle_roll_link", np.array(direction), 0.2)
        for _ in range(args.takeover_steps):
            advance()
        runtime.begin_recovery(runtime.timeline.source_frame)
        for _ in range(80):
            advance()
            if runtime.timeline.phase == "reference":
                break
        while runtime.timeline.source_frame < motion.frames - 1 and len(runtime.records) < 260:
            advance()
        basket = np.asarray(basket_poses)
        root = np.stack([record.body_q[runtime.root_body, :3] for record in runtime.records])
        report = {
            "start": args.start,
            "direction": args.direction,
            "takeover_steps": args.takeover_steps,
            "end_phase": runtime.timeline.phase,
            "source_frame": runtime.timeline.source_frame,
            "simulated_frames": len(runtime.records),
            "basket_xy_displacement_m": float(np.linalg.norm(basket[-1, :2] - basket[0, :2])),
            "pelvis_min_z_m": float(root[:, 2].min()),
        }
        (args.out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if tracker is not None:
            tracker.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
