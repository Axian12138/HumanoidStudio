"""Physical 6–12 s brush pickup probe with a separate rollout output."""

import argparse
import json
import sys
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import SCRIPTS, example_dir

sys.path.insert(0, str(SCRIPTS))
from robosimstudio.humanoid_real2sim.track import launcher

from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker

ROOT = example_dir(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direct-arm", action="store_true")
    parser.add_argument("--hand-kp", type=float, default=2.0)
    parser.add_argument("--hand-kv", type=float, default=0.1)
    args = parser.parse_args()
    metadata = json.loads(launcher.DEFAULT_METADATA.read_text())
    motion = ReferenceMotion.load(
        args.motion,
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
        motion_order=launcher.DEFAULT_ORDER,
    )
    temporary, asset = launcher._position_asset(
        launcher.DEFAULT_ROBOT, metadata, hand_kp=args.hand_kp, hand_kv=args.hand_kv,
    )
    tracker = None
    try:
        sim = launcher._load_scene_file(SCRIPTS / "tasks/brush_pick_1012/scene.py", asset, motion)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "scene_labels.json").write_text(json.dumps({
            "body_names": [str(name) for name in sim.model.body_label],
            "joint_names": [str(name) for name in sim.model.joint_label],
            "joint_q_start": sim.model.joint_q_start.numpy().astype(int).tolist(),
        }, indent=2) + "\n")
        tracker = BFMTracker(
            python=launcher.SCALEBRIDGE / ".venv/bin/python",
            scalebridge=launcher.SCALEBRIDGE,
            checkpoint=launcher.DEFAULT_CHECKPOINT,
            metadata=launcher.DEFAULT_METADATA,
        )
        runtime = RepairRuntime(
            sim, motion, tracker,
            original_model=launcher.DEFAULT_FK_MODEL,
            motion_order=launcher.DEFAULT_ORDER,
            reference_pid_wrists=("right",),
            reference_pid_right_arm=args.direct_arm,
            hand_open_frame=490,
            hand_close_frame=560,
        )
        runtime.seek_reference(300)
        for frame in range(300, 600):
            runtime.prepare()
            sim.step_teleop()
            runtime.record_after_step()
            if (frame + 1) % 50 == 0:
                print(f"physical replay {frame + 1}/600", flush=True)
        runtime.save(args.out)
        print(args.out / "rollout.npz", flush=True)
    finally:
        if tracker is not None:
            tracker.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
