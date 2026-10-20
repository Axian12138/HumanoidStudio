"""Diagnostic: apply reference joint positions directly in the same RSS scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir, SCRIPTS

import numpy as np

sys.path.insert(0, str(SCRIPTS))
from robosimstudio.humanoid_real2sim.track import launcher  # noqa: E402

from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion  # noqa: E402
from robosimstudio.scene.entity import label_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hold-first", action="store_true", help="hold frame-zero joints for all 150 steps")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(launcher.DEFAULT_METADATA.read_text())
    motion = ReferenceMotion.load(
        args.motion,
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
        motion_order=launcher.DEFAULT_ORDER,
    )
    temporary, asset = launcher._position_asset(launcher.DEFAULT_ROBOT, metadata)
    try:
        sim = launcher._load_scene_file(Path(__file__).with_name("scene.py"), asset, motion)
        q_start = sim.model.joint_q_start.numpy()
        labels = list(sim.model.joint_label)
        controlled = np.array([q_start[label_index(labels, name)] for name in motion.joint_names])
        hands = np.array([q_start[label_index(labels, name)] for name in motion.all_joint_names if "_hand_" in name])
        hand_columns = [i for i, name in enumerate(motion.all_joint_names) if "_hand_" in name]
        body_indices = np.array([label_index(list(sim.model.body_label), name) for name in motion.body_names])
        basket_index = label_index(list(sim.model.body_label), "round_laundry_basket")
        bodies = []
        basket = []
        for frame in range(motion.frames):
            source_frame = 0 if args.hold_first else frame
            target = sim.pipeline.snapshot.joint_target_q.astype(np.float32).copy()
            target[controlled] = motion.joint_pos[source_frame]
            target[hands] = motion.all_joint_pos[source_frame, hand_columns]
            sim.control.joint_target_q.assign(target)
            sim.step_teleop()
            bodies.append(sim.pipeline.snapshot.body_q[body_indices, :3].copy())
            basket.append(sim.pipeline.snapshot.body_q[basket_index].copy())
        np.save(args.out / "selected_body_pos_w.npy", np.asarray(bodies))
        np.save(args.out / "basket_pose_xyzw.npy", np.asarray(basket))
        (args.out / "body_names.json").write_text(json.dumps(motion.body_names, indent=2) + "\n")
        print(args.out)
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    main()
