"""Run the unedited kick-basket reference through RSS and ScaleBridge BFM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir, SCRIPTS

import numpy as np

sys.path.insert(0, str(SCRIPTS))
from robosimstudio.humanoid_real2sim.track import launcher

from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker
from robosimstudio.scene.entity import label_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--scene-file", type=Path, default=Path(__file__).with_name("scene.py"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = example_dir(__file__) / "tracker_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    checkpoint = Path(metadata["checkpoint"])
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
        tracker = BFMTracker(
            python=launcher.SCALETRACK / ".venv/bin/python",
            scalebridge=launcher.SCALEBRIDGE,
            checkpoint=checkpoint,
            metadata=metadata_path,
        )
        runtime = RepairRuntime(
            sim, motion, tracker,
            original_model=launcher.DEFAULT_FK_MODEL,
            motion_order=launcher.DEFAULT_ORDER,
        )
        basket_index = label_index(list(sim.model.body_label), "round_laundry_basket")
        basket_mass = float(sim.model.body_mass.numpy()[basket_index])
        basket_com = np.asarray(sim.model.body_com.numpy()[basket_index]).tolist()
        basket_shapes = [i for i, label in enumerate(sim.model.shape_label) if str(label).startswith("round_laundry_basket_collision")]
        basket_friction = float(np.median(sim.model.shape_material_mu.numpy()[basket_shapes]))
        print(f"basket mass={basket_mass:.3f} kg, friction={basket_friction:.3f}, COM={basket_com}", flush=True)
        basket_pose = [sim.pipeline.snapshot.body_q[basket_index].copy()]
        count = min(args.frames, motion.frames)
        for frame in range(count):
            runtime.prepare()
            sim.step_teleop()
            runtime.record_after_step()
            basket_pose.append(sim.pipeline.snapshot.body_q[basket_index].copy())
            if frame % 25 == 24 or frame == count - 1:
                print(f"RSS baseline {frame + 1}/{count}", flush=True)
        runtime.save(args.out)
        np.save(args.out / "basket_pose_xyzw.npy", np.asarray(basket_pose))
        np.save(args.out / "selected_body_pos_w.npy", np.stack([
            record.body_q[runtime.body_indices, :3] for record in runtime.records
        ]))
        (args.out / "inputs.json").write_text(json.dumps({
            "motion": str(args.motion.resolve()),
            "motion_sha256": hashlib.sha256(args.motion.read_bytes()).hexdigest(),
            "scene": str(args.scene_file.resolve()),
            "scene_sha256": hashlib.sha256(args.scene_file.read_bytes()).hexdigest(),
            "shared_scene_sha256": hashlib.sha256(Path(__file__).with_name("scene.py").read_bytes()).hexdigest(),
            "basket_asset": str(example_dir(__file__) / "assets" / "round_laundry_basket_v2.obj"),
            "basket_asset_sha256": hashlib.sha256((example_dir(__file__) / "assets" / "round_laundry_basket_v2.obj").read_bytes()).hexdigest(),
            "basket_collision_hulls": 65,
            "basket_collision_sha256": hashlib.sha256(b"".join(
                path.name.encode() + b"\0" + path.read_bytes()
                for path in sorted((example_dir(__file__) / "assets" / "collision_hulls").glob("Hull_*.obj"))
            )).hexdigest(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "frames": count,
            "basket_mass_kg": basket_mass,
            "basket_friction": basket_friction,
            "basket_com_local_m": basket_com,
            "scene_scope": "local contact scene; full living room not imported",
            "diagnostic_gravcomp": os.environ.get("RSS_KICK_DIAGNOSTIC_GRAVCOMP") == "1",
            "diagnostic_gain_scale": float(os.environ.get("RSS_KICK_DIAGNOSTIC_GAIN_SCALE", "1")),
        }, indent=2) + "\n")
        print(f"saved {args.out}")
    finally:
        if tracker is not None:
            tracker.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
