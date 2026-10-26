"""Record a headless, unmodified RSS repair rollout for cross-engine comparison."""

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio.humanoid_real2sim.track.launcher import (
    DEFAULT_ROBOT,
    SCALEBRIDGE,
    SCALETRACK,
    _align_start_pose,
    _position_asset,
)
from robosimstudio.humanoid_real2sim.track.package_scene import build_package_sim
from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--isaac-hand-gains", action="store_true")
    parser.add_argument("--ls-iterations", type=int)
    args = parser.parse_args()
    rss.MUJOCO_500HZ = replace(rss.MUJOCO_500HZ, substeps=args.substeps)
    if args.ls_iterations is not None:
        original_backend = rss.backend
        rss.backend = lambda name, **kw: original_backend(name, **dict(kw, ls_iterations=args.ls_iterations))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    meta = json.loads(
        (root / "outputs/humanoid-real2sim/2026-10-24/refined_tracker_1019/runs/ckpt_010000/metadata.json").read_text()
    )
    meta["checkpoint"] = str(args.checkpoint.resolve())
    metadata = args.out / "metadata.json"
    metadata.write_text(json.dumps(meta, indent=2))
    package = SCALETRACK / "data/1019test/brace-dining-table-wipes-kinematic-v1"
    order = package / "metadata/motion_order.json"
    motion = ReferenceMotion.load(
        package / "motion.npz",
        joint_names=tuple(meta["joint_names"]),
        body_names=tuple(meta["selected_body_names"]),
        motion_order=order,
    )
    temporary, asset = _position_asset(DEFAULT_ROBOT, meta)
    if args.isaac_hand_gains:
        tree = ET.parse(asset)
        for actuator in tree.getroot().findall("./actuator/position"):
            name = actuator.get("joint", "")
            if "_hand_" in name and "thumb_0" not in name:
                actuator.set("kp", "0.5")
        tree.write(asset, encoding="unicode")
    tracker = None
    try:
        sim = build_package_sim(asset, motion, package=package, scaletrack=SCALETRACK, out=args.out / "scene")
        _align_start_pose(sim, motion)
        tracker = BFMTracker(
            python=SCALETRACK / ".venv/bin/python",
            scalebridge=SCALEBRIDGE,
            checkpoint=args.checkpoint,
            metadata=metadata,
        )
        runtime = RepairRuntime(sim, motion, tracker, max_edit_m=1.0, original_model=DEFAULT_ROBOT, motion_order=order)
        labels = list(sim.model.joint_label)
        starts = sim.model.joint_q_start.numpy()
        vstarts = sim.model.joint_qd_start.numpy()
        joint_ids = [
            next(i for i, n in enumerate(labels) if n.split("/")[-1] == name) for name in motion.all_joint_names
        ]
        coord = starts[joint_ids]
        dof = vstarts[joint_ids]
        body_names = [str(n).split("/")[-1] for n in sim.model.body_label]
        selected = [body_names.index(n) for n in motion.body_names]
        rows = []

        def capture(step, action):
            snap = sim.pipeline.snapshot
            rows.append(
                {
                    "step": step,
                    "joint_pos": snap.joint_q[coord].copy(),
                    "joint_vel": snap.joint_qd[dof].copy(),
                    "body_pos": snap.body_q[selected, :3].copy(),
                    "body_quat": snap.body_q[selected, 3:][:, [3, 0, 1, 2]].copy(),
                    "body_vel": snap.body_qd[selected].copy(),
                    "reference_body_pos": motion.body_pos_w[min(step, motion.frames - 1)].copy(),
                    "action": action.copy(),
                }
            )

        capture(0, np.zeros(29))
        for step in range(args.steps):
            runtime.prepare()
            sim.step_teleop()
            runtime.record_after_step()
            capture(step + 1, runtime.records[-1].action)
        np.savez_compressed(args.out / "states.npz", **{key: np.asarray([row[key] for row in rows]) for key in rows[0]})
        (args.out / "contract.json").write_text(
            json.dumps(
                {
                    "engine": "RSS/Newton/MuJoCo",
                    "checkpoint": str(args.checkpoint),
                    "body_names": motion.body_names,
                    "joint_names": motion.all_joint_names,
                    "physics_dt": 0.02 / args.substeps,
                    "isaac_hand_gains": args.isaac_hand_gains,
                    "ls_iterations": int(sim.pipeline.solver.mj_model.opt.ls_iterations),
                    "control_dt": 0.02,
                    "hand_kp": 2.0,
                    "hand_kd": 0.1,
                },
                indent=2,
            )
        )
        print("COMPLETE", args.out, flush=True)
    finally:
        if tracker is not None:
            tracker.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
