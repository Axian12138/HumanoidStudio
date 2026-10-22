"""Launch the established human + humanoid kinematic Viser review page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/robosimstudio_humanoid_real2sim_warp")

from robosimstudio.humanoid_real2sim.track.launcher import (
    DEFAULT_METADATA,
    DEFAULT_MOTION,
    DEFAULT_ORDER,
    DEFAULT_ROBOT,
    _build_scene,
    _load_scene_file,
    _position_asset,
)
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.visualization.alignment import validate_human_alignment
from robosimstudio.humanoid_real2sim.visualization.kinematic import KinematicStudio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-order", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--scene-file", type=Path)
    parser.add_argument("--human-motion", type=Path, help="frame-aligned world-space human skeleton NPZ")
    parser.add_argument(
        "--human-reference", type=Path, help="task-local human.npz used to verify the overlay's source window"
    )
    parser.add_argument("--source-sync", type=Path, help="task-local source_sync.npz with recording_time_s")
    parser.add_argument("--object-track", type=Path, help="optional estimated object poses for visual review")
    parser.add_argument(
        "--tracks-manifest", type=Path, help="source-aligned multi-object and scalar mechanism joint tracks JSON"
    )
    parser.add_argument("--context-mesh", type=Path, help="world-frame OBJ mesh drawn in the 3D review scene")
    parser.add_argument(
        "--rigid-track", type=Path, help="independent left/right OptiTrack hand rigid poses and wrist proxies"
    )
    parser.add_argument("--object-name", help="free-joint name in the scene")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int, help="exclusive; defaults to motion length")
    parser.add_argument("--source-offset-s", type=float, default=0.0, help="source recording time of motion frame zero")
    parser.add_argument("--segments-json", type=Path, help="optional task split manifest for timeline jumps")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--camera-eye-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(1.8, -2.4, 1.2),
        help="initial camera offset from robot root in world metres",
    )
    parser.add_argument(
        "--camera-look-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(-0.2, -0.35, -0.25),
        help="initial look-at offset from robot root",
    )
    parser.add_argument(
        "--follow-root",
        action="store_true",
        help="move the review camera with the motion root while retaining manual orbit offsets",
    )
    parser.add_argument(
        "--dark-mode", action="store_true", help="dark Viser canvas for light robot and tracker overlays"
    )
    args = parser.parse_args()
    if args.human_motion is None:
        inferred = args.motion.resolve().parent.parent / "sidecars/human_motion_world.npz"
        if inferred.is_file():
            args.human_motion = inferred
        else:
            parser.error("Stage 3 visualization requires --human-motion (positions_w, joint_names, edges, fps)")
    metadata = json.loads(args.metadata.read_text())
    motion = ReferenceMotion.load(
        args.motion,
        joint_names=tuple(metadata["joint_names"]),
        body_names=tuple(metadata["selected_body_names"]),
        motion_order=args.motion_order,
    )
    validate_human_alignment(
        args.human_motion,
        frames=motion.frames,
        fps=motion.fps,
        source_offset_s=args.source_offset_s,
        reference=args.human_reference,
        source_sync=args.source_sync,
    )
    segments = []
    if args.segments_json:
        for task in json.loads(args.segments_json.read_text())["tasks"]:
            first = round((float(task["start_s"]) - args.source_offset_s) * motion.fps)
            last = round((float(task["end_s"]) - args.source_offset_s) * motion.fps)
            if last > 0 and first < motion.frames:
                segments.append({"id": task["id"], "start": max(0, first), "stop": min(motion.frames, last)})
    temporary, asset = _position_asset(args.robot, metadata)
    try:
        sim = (
            _load_scene_file(args.scene_file, asset, motion)
            if args.scene_file
            else _build_scene(asset, motion, soft=False)
        )
        KinematicStudio(
            sim,
            motion,
            start=args.start_frame,
            stop=motion.frames if args.stop_frame is None else args.stop_frame,
            port=args.port,
            human_motion=args.human_motion,
            object_track=args.object_track,
            tracks_manifest=args.tracks_manifest,
            context_mesh=args.context_mesh,
            rigid_track=args.rigid_track,
            object_name=args.object_name,
            source_offset_s=args.source_offset_s,
            segments=segments,
            camera_eye_offset=tuple(args.camera_eye_offset),
            camera_look_offset=tuple(args.camera_look_offset),
            follow_root=args.follow_root,
            dark_mode=args.dark_mode,
        ).run()
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    main()
