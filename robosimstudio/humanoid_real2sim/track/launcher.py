"""Pause, rewind and locally repair a G1 reference with RSS physics.

The ScaleBridge Python process runs BFM inference only. All stepping and all
object/deformable contact use RoboSimStudio's configured physics arrangement.
Torch-TensorRT checkpoints require CUDA; raw ScaleTrack checkpoints also run
with pure Torch for contract debugging.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/robosimstudio_humanoid_real2sim_warp")

from robosimstudio.humanoid_real2sim.track.repair_studio import RepairStudio

import robosimstudio as rss
from robosimstudio.humanoid_real2sim.track.runtime import RepairRuntime
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.track.tracker import BFMTracker
from robosimstudio.scene.entity import label_index

WORKSPACE = Path(__file__).resolve().parents[4]
SCALEBRIDGE = WORKSPACE / "ScaleBridge"
SCALETRACK = WORKSPACE / "ScaleTrack"
DEFAULT_METADATA = (
    SCALEBRIDGE / "scalebridge/data/model/bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt_metadata.json"
)
DEFAULT_CHECKPOINT = SCALEBRIDGE / "scalebridge/data/model/bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt.pt"
DEFAULT_MOTION = SCALEBRIDGE / "scalebridge/data/motion/cartwheel.npz"
DEFAULT_ORDER = SCALETRACK / "retarget3/assets/robot/motion_order.json"
DEFAULT_ROBOT = SCALETRACK / "source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
DEFAULT_FK_MODEL = DEFAULT_ROBOT


def _position_asset(
    source: Path, metadata: dict, *, hand_kp: float = 2.0, hand_kv: float = 0.1,
) -> tuple[tempfile.TemporaryDirectory, Path]:
    if not (np.isfinite(hand_kp) and hand_kp > 0 and np.isfinite(hand_kv) and hand_kv >= 0):
        raise ValueError("hand position gains must be finite and nonnegative")
    tree = ET.parse(source)
    root = tree.getroot()
    joints = {node.get("name"): node for node in root.findall(".//worldbody//joint")}
    gains = {
        name: (float(metadata["stiffness"][i]), float(metadata["damping"][i]))
        for i, name in enumerate(metadata["joint_names"])
    }
    motors = root.findall("./actuator/motor")
    if not motors:
        raise ValueError(f"no motors in {source}")
    for motor in motors:
        name = motor.get("joint")
        joint = joints[name]
        torque_range = motor.get("ctrlrange")
        kp, kv = gains.get(name, (hand_kp, hand_kv) if "_hand_" in name else (2.0, 0.1))
        motor.tag = "position"
        motor.attrib.clear()
        motor.set("name", name)
        motor.set("joint", name)
        motor.set("kp", str(kp))
        motor.set("kv", str(kv))
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", joint.get("range"))
        if torque_range:
            motor.set("forcelimited", "true")
            motor.set("forcerange", torque_range)
    # Some G1 MJCF variants omit the ankle-roll visual meshes. Keep their
    # contact shapes and add each mesh only when the source omits it.
    for side in ("left", "right"):
        foot = root.find(f".//body[@name='{side}_ankle_roll_link']")
        if foot is None:
            raise ValueError(f"missing {side} foot body")
        if not any(geom.get("mesh") == f"{side}_ankle_roll_link" for geom in foot.findall("geom")):
            ET.SubElement(
                foot,
                "geom",
                {
                    "name": f"{side}_foot_visual",
                    "type": "mesh",
                    "mesh": f"{side}_ankle_roll_link",
                    "group": "1",
                    "density": "0",
                    "contype": "0",
                    "conaffinity": "0",
                    "rgba": "0.2 0.2 0.2 1",
                },
            )
    temporary = tempfile.TemporaryDirectory(prefix="rss_g1_repair_")
    folder = Path(temporary.name)
    (folder / "meshes").symlink_to(source.parent / "meshes", target_is_directory=True)
    output = folder / "g1_position.xml"
    tree.write(output, encoding="unicode")
    return temporary, output


def _align_start_pose(sim, motion: ReferenceMotion) -> None:
    """Make reset return to the first retargeted full-body reference pose."""
    q = sim.pipeline.snapshot.joint_q.astype(np.float32).copy()
    labels = list(sim.model.joint_label)
    starts = sim.model.joint_q_start.numpy()
    root = int(starts[label_index(labels, "floating_base")])
    q[root : root + 3] = motion.body_pos_w[0, 0]
    q[root + 3 : root + 7] = motion.body_quat_w[0, 0, [1, 2, 3, 0]]
    for name, value in zip(motion.joint_names, motion.joint_pos[0], strict=True):
        q[int(starts[label_index(labels, name)])] = value
    if motion.all_joint_pos is not None:
        for name in motion.all_joint_names:
            if "_hand_" in name:
                q[int(starts[label_index(labels, name)])] = motion.all_joint_pos[0, motion.all_joint_names.index(name)]
    sim._start_joint_q = q
    sim.reset()


def _build_scene(asset: Path, motion: ReferenceMotion, *, soft: bool, soft_solver: str = "newton"):
    body_names = motion.joint_names
    g1 = rss.EmbodimentSpec(
        name="g1_repair",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=body_names,
                ee_body_name="right_wrist_yaw_link",
                home_q=tuple(float(v) for v in motion.joint_pos[0]),
            ),
        ),
    )
    if soft and soft_solver == "superdex":
        physics = rss.replace(
            rss.GENERIC.with_backend("soft", rss.backend("superdex", substeps=2, feedback_dv_max=20.0)),
            control_hz=50.0,
        )
    elif soft:
        physics = rss.replace(
            rss.GENERIC,
            backends=tuple(entry for entry in rss.GENERIC.backends if entry[0] != "soft"),
            control_hz=50.0,
        )
    else:
        physics = rss.MUJOCO_500HZ
    scene = rss.Scene(cell="void", name="g1_motion_repair", physics=physics)
    scene.add(rss.robot(g1))
    focus = motion.body_pos_w[0, 0]
    scene.add(
        rss.CameraSpec(
            name="review",
            width=960,
            height=540,
            eye=(float(focus[0] + 2.4), float(focus[1] - 2.8), float(focus[2] + 0.95)),
            look_at=tuple(float(v) for v in focus),
        )
    )
    if soft:
        scene.add(
            rss.sponge(name="sponge", size=(0.12, 0.07, 0.04), at=(float(focus[0] + 0.35), float(focus[1] - 0.2), 0.0))
        )
    sim = scene.build(control_mode="joint_position")
    _align_start_pose(sim, motion)
    return sim


def _load_scene_file(path: Path, asset: Path, motion: ReferenceMotion):
    spec = importlib.util.spec_from_file_location("rss_user_repair_scene", path.resolve())
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load scene module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "build_sim", None)
    if not callable(factory):
        raise ValueError(f"{path} must define build_sim(asset, motion)")
    sim = factory(asset, motion)
    if not hasattr(sim, "pipeline"):
        raise TypeError("build_sim must return a built RoboSimStudio Sim")
    _align_start_pose(sim, motion)
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-order", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--fk-model", type=Path, default=DEFAULT_FK_MODEL)
    parser.add_argument(
        "--scalebridge-python",
        type=Path,
        help="Python for the inference worker; raw ScaleTrack checkpoints use ScaleTrack's venv",
    )
    parser.add_argument("--soft", action="store_true", help="include a RoboSimStudio soft sponge")
    parser.add_argument("--soft-solver", choices=("newton", "superdex"), default="newton")
    parser.add_argument(
        "--scene-file", type=Path, help="Python module defining build_sim(asset, motion) for a reconstructed RSS scene"
    )
    parser.add_argument("--hand-open-frame", type=int)
    parser.add_argument("--hand-close-frame", type=int)
    parser.add_argument("--hand-kp", type=float, default=2.0)
    parser.add_argument("--hand-kv", type=float, default=0.1)
    parser.add_argument(
        "--reference-pid-wrists", choices=("none", "left", "right", "both"), default="none",
        help="send selected wrist roll/pitch/yaw refmotion positions directly to RSS position actuators",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="initial source frame for physical playback")
    parser.add_argument("--stop-frame", type=int, help="exclusive source frame at which playback pauses")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--out", type=Path, default=Path("outputs/humanoid-real2sim"))
    parser.add_argument("--resume-rollout", type=Path, help="restore a saved physical repair cache")
    parser.add_argument("--rewind-resume", action="store_true", help="open a restored physical cache at its first slot")
    parser.add_argument("--resume-link", help="link previously repaired in a resumed cache")
    parser.add_argument("--resume-hand-left", type=float, help="restore the left-hand manual closure amount")
    parser.add_argument("--resume-hand-right", type=float, help="restore the right-hand manual closure amount")
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument("--show-reference", dest="show_reference", action="store_true", default=True,
                                 help="show the retarget reference robot ghost (Stage 4 default)")
    reference_group.add_argument("--hide-reference", dest="show_reference", action="store_false",
                                 help="hide the retarget reference robot ghost")
    parser.add_argument("--reference-video", type=Path, help="source human video synchronized to the motion clock")
    parser.add_argument("--review-only", action="store_true", help="show cached physics and source motion without repair controls")
    args = parser.parse_args()
    if args.reference_video is not None and not args.reference_video.is_file():
        parser.error(f"reference video does not exist: {args.reference_video}")
    metadata = json.loads(args.metadata.read_text())
    worker_python = args.scalebridge_python or (
        SCALETRACK / ".venv/bin/python"
        if metadata.get("policy_kind") == "raw_scaletrack_bfm"
        else SCALEBRIDGE / ".venv/bin/python"
    )
    motion = ReferenceMotion.load(
        args.motion,
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
        motion_order=args.motion_order,
    )
    if not 0 <= args.start_frame < motion.frames:
        parser.error("--start-frame is outside the motion")
    if args.stop_frame is not None and not args.start_frame < args.stop_frame <= motion.frames:
        parser.error("--stop-frame must be after --start-frame and within the motion")
    temporary, asset = _position_asset(args.robot, metadata, hand_kp=args.hand_kp, hand_kv=args.hand_kv)
    tracker = None
    try:
        sim = (
            _load_scene_file(args.scene_file, asset, motion)
            if args.scene_file is not None
            else _build_scene(asset, motion, soft=args.soft, soft_solver=args.soft_solver)
        )
        tracker = BFMTracker(
            python=worker_python,
            scalebridge=SCALEBRIDGE,
            checkpoint=args.checkpoint,
            metadata=args.metadata,
        )
        runtime = RepairRuntime(
            sim,
            motion,
            tracker,
            max_edit_m=1.0,
            hand_open_frame=args.hand_open_frame,
            hand_close_frame=args.hand_close_frame,
            original_model=args.fk_model,
            motion_order=args.motion_order,
            reference_pid_wrists=("left", "right") if args.reference_pid_wrists == "both" else (
                () if args.reference_pid_wrists == "none" else (args.reference_pid_wrists,)
            ),
        )
        if args.start_frame and args.resume_rollout is None:
            runtime.seek_reference(args.start_frame)
        if args.resume_rollout is not None:
            runtime.load_saved_rollout(args.resume_rollout, repaired_link=args.resume_link)
            if args.rewind_resume:
                runtime.seek_cached(0)
            for side, amount in (("left", args.resume_hand_left), ("right", args.resume_hand_right)):
                if amount is not None:
                    runtime.set_hand_amount(side, amount)
                    runtime.records[-1].hand_override_after[side] = amount
        RepairStudio(
            sim, runtime, out=args.out, port=args.port, stop_frame=args.stop_frame,
            show_reference=args.show_reference, reference_video=args.reference_video,
            review_only=args.review_only,
        ).run()
    finally:
        if tracker is not None:
            tracker.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
