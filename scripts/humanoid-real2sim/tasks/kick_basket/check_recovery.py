"""Replay a saved physical state into a candidate reference recovery frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir, SCRIPTS
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(SCRIPTS))
from robosimstudio.humanoid_real2sim.track import launcher

from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker
from robosimstudio.scene.entity import label_index


def errors(runtime: RepairRuntime) -> dict[str, float]:
    snap = runtime.sim.pipeline.snapshot
    frame = runtime.timeline.source_frame
    actual = snap.body_q[runtime.body_indices]
    reference = runtime.motion.body_pos_w[frame]
    distances = np.linalg.norm(actual[:, :3] - reference, axis=-1)
    quat = actual[:, [6, 3, 4, 5]]
    ref_quat = runtime.motion.body_quat_w[frame]
    dots = np.clip(np.abs(np.sum(quat * ref_quat, axis=-1)), 0, 1)
    joints = np.abs(snap.joint_q[runtime.joint_coords] - runtime.motion.joint_pos[frame])
    return {
        "source_frame": int(frame),
        "mean_body_error_m": float(distances.mean()),
        "max_body_error_m": float(distances.max()),
        "max_body_angle_deg": float(np.degrees(2 * np.arccos(dots)).max()),
        "max_joint_error_rad": float(joints.max()),
        "pelvis_z_m": float(actual[0, 2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output-index", type=int, required=True)
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = example_dir(__file__) / "tracker_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    motion = ReferenceMotion.load(
        launcher.SCALETRACK / "data/1012test/kick-basket-v1/motion.npz",
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
        motion_order=launcher.DEFAULT_ORDER,
    )
    temporary, asset = launcher._position_asset(launcher.DEFAULT_ROBOT, metadata)
    tracker = None
    try:
        sim = launcher._load_scene_file(Path(__file__).with_name("scene_stage5_v2.py"), asset, motion)
        tracker = BFMTracker(
            python=launcher.SCALETRACK / ".venv/bin/python",
            scalebridge=launcher.SCALEBRIDGE,
            checkpoint=Path(metadata["checkpoint"]),
            metadata=metadata_path,
        )
        runtime = RepairRuntime(sim, motion, tracker, original_model=launcher.DEFAULT_FK_MODEL)
        basket = label_index(list(sim.model.body_label), "round_laundry_basket")
        with np.load(args.rollout, allow_pickle=False) as saved:
            index = args.output_index
            if not 0 <= index < len(saved["source_frame"]):
                raise ValueError("output-index outside saved rollout")
            state = {
                "joint_q": saved["achieved_joint_q"][index].copy(),
                "joint_qd": saved["achieved_joint_qd"][index].copy(),
                "body_q": saved["achieved_body_q"][index].copy(),
                "body_qd": saved["achieved_body_qd"][index].copy(),
                "particle_q": saved["achieved_particle_q"][index].copy(),
                "particle_qd": saved["achieved_particle_qd"][index].copy(),
            }
            runtime._restore_physics(state, saved["commanded_joint_target_q"][index], index + 1)
            runtime.sim.pipeline.reset_solvers(flags=0)
            runtime.history.clear()
            for prior in range(max(0, index - 2), index):
                record = SimpleNamespace(
                    joint_q=saved["achieved_joint_q"][prior],
                    joint_qd=saved["achieved_joint_qd"][prior],
                    body_q=saved["achieved_body_q"][prior],
                    body_qd=saved["achieved_body_qd"][prior],
                )
                runtime.history.append(runtime._state_from_record(record))
            runtime.action_history.clear()
            for prior in range(max(0, index - 2), index + 1):
                runtime.action_history.append(saved["tracker_action"][prior].copy())
            source_frame = int(saved["source_frame"][index])
            initial_basket = state["body_q"][basket, :3].copy()
        achieved = runtime._snapshot()
        runtime.timeline.take_over(achieved["body_pos"], achieved["body_quat"])
        runtime.begin_recovery(args.reference_frame, duration_frames=25)
        before = errors(runtime)
        samples = []
        for step in range(args.steps):
            runtime.prepare()
            sim.step_teleop()
            runtime.record_after_step()
            row = errors(runtime)
            row["step"] = step + 1
            row["phase"] = runtime.timeline.phase
            row["basket_xy_from_start_m"] = float(
                np.linalg.norm(sim.pipeline.snapshot.body_q[basket, :2] - initial_basket[:2])
            )
            samples.append(row)
            if (step + 1) % 25 == 0 or runtime.timeline.phase == "reference":
                print(f"recovery {step + 1}/{args.steps}: {row}", flush=True)
            if runtime.timeline.phase == "reference":
                break
        report = {
            "saved_rollout": str(args.rollout.resolve()),
            "output_index": args.output_index,
            "source_frame_before": source_frame,
            "reference_frame": args.reference_frame,
            "before": before,
            "interpolation_complete": runtime.timeline.phase == "reference",
            "physical_match_at_end": bool(
                samples
                and samples[-1]["max_body_error_m"] <= 0.05
                and samples[-1]["max_body_angle_deg"] <= float(np.degrees(0.25))
                and samples[-1]["max_joint_error_rad"] <= 0.2
            ),
            "steps_run": len(samples),
            "samples": samples,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"saved {args.out}", flush=True)
    finally:
        if tracker is not None:
            tracker.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
