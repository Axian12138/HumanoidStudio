"""Replay a fixed reference through RSS/ScaleBridge for Stage 5 object trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir, SCRIPTS

import numpy as np

sys.path.insert(0, str(SCRIPTS))
from robosimstudio.humanoid_real2sim.track import launcher

TASK_METADATA = launcher.DEFAULT_METADATA
TASK_CHECKPOINT = launcher.DEFAULT_CHECKPOINT
TASK_WORKER = launcher.SCALEBRIDGE / ".venv/bin/python"

from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker
from robosimstudio.scene.entity import label_index


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=example_dir(__file__) / "retarget" / "hoi_candidate_scene.npz")
    parser.add_argument("--scene-file", type=Path, default=Path(__file__).with_name("scene_stage5_coffee_mu_0p15.py"))
    parser.add_argument("--metadata", type=Path, default=TASK_METADATA)
    parser.add_argument("--checkpoint", type=Path, default=TASK_CHECKPOINT)
    parser.add_argument("--worker-python", type=Path, default=TASK_WORKER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.metadata.read_text())
    motion = ReferenceMotion.load(
        args.motion,
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
        motion_order=launcher.DEFAULT_ORDER,
    )
    temporary, asset = launcher._position_asset(launcher.DEFAULT_ROBOT, metadata)
    tracker = None
    try:
        sim = launcher._load_scene_file(args.scene_file, asset, motion)
        bin_index = label_index(list(sim.model.body_label), "white_waste_bin")
        bin_shapes = np.array([
            i for i, label in enumerate(sim.model.shape_label)
            if str(label).startswith("white_waste_bin_collision")
        ])
        table_shapes = np.array([
            i for i, label in enumerate(sim.model.shape_label)
            if str(label).startswith("coffee_table_collision")
        ])
        print(f"bin mass={sim.model.body_mass.numpy()[bin_index]:.3f} kg, "
              f"mu={np.median(sim.model.shape_material_mu.numpy()[bin_shapes]):.3f}, "
              f"COM={sim.model.body_com.numpy()[bin_index].tolist()}, "
              f"table collision parts={len(table_shapes)}", flush=True)
        tracker = BFMTracker(
            python=args.worker_python,
            scalebridge=launcher.SCALEBRIDGE,
            checkpoint=args.checkpoint,
            metadata=args.metadata,
        )
        runtime = RepairRuntime(
            sim, motion, tracker,
            original_model=launcher.DEFAULT_FK_MODEL,
            motion_order=launcher.DEFAULT_ORDER,
        )
        poses = [sim.pipeline.snapshot.body_q[bin_index].copy()]
        for frame in range(motion.frames):
            runtime.prepare()
            sim.step_teleop()
            runtime.record_after_step()
            poses.append(sim.pipeline.snapshot.body_q[bin_index].copy())
            if frame % 50 == 49 or frame == motion.frames - 1:
                print(f"Stage 5 physical replay {frame + 1}/{motion.frames}", flush=True)
        runtime.save(args.out)
        np.save(args.out / "bin_pose_xyzw.npy", np.asarray(poses))
        (args.out / "inputs.json").write_text(json.dumps({
            "motion": str(args.motion.resolve()),
            "motion_sha256": sha256(args.motion),
            "scene": str(args.scene_file.resolve()),
            "scene_sha256": sha256(args.scene_file),
            "coffee_table_source_usd": str(Path(__file__).with_name("export_coffee_table.py")),
            "coffee_table_visual_sha256": sha256(example_dir(__file__) / "assets" / "coffee_table_visual.obj"),
            "coffee_table_collision_sha256": [sha256(p) for p in sorted(example_dir(__file__) / "assets".glob("coffee_table_collision_*.obj"))],
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "metadata": str(args.metadata),
            "metadata_sha256": sha256(args.metadata),
            "frames": motion.frames,
            "body_labels": list(map(str, sim.model.body_label)),
            "bin_body_index": bin_index,
            "table_body_index": None,
            "bin_mass_kg": float(sim.model.body_mass.numpy()[bin_index]),
            "bin_com_local_m": sim.model.body_com.numpy()[bin_index].tolist(),
            "bin_rss_friction": float(np.median(sim.model.shape_material_mu.numpy()[bin_shapes])),
            "table_collision_parts": len(table_shapes),
        }, indent=2) + "\n")
    finally:
        if tracker is not None:
            tracker.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
