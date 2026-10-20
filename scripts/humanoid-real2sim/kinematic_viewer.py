"""Launch the established human + humanoid kinematic Viser review page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/robosimstudio_humanoid_real2sim_warp")

from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.visualization.kinematic import KinematicStudio
from robosimstudio.humanoid_real2sim.track.launcher import (
    DEFAULT_METADATA, DEFAULT_MOTION, DEFAULT_ORDER, DEFAULT_ROBOT,
    _build_scene, _load_scene_file, _position_asset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-order", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--scene-file", type=Path)
    parser.add_argument("--human-motion", type=Path, help="frame-aligned world-space human skeleton NPZ")
    parser.add_argument("--object-track", type=Path, help="optional estimated object poses for visual review")
    parser.add_argument("--object-name", help="free-joint name in the scene")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int, help="exclusive; defaults to motion length")
    parser.add_argument("--source-offset-s", type=float, default=0.0,
                        help="source recording time of motion frame zero")
    parser.add_argument("--segments-json", type=Path, help="optional task split manifest for timeline jumps")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if args.human_motion is None:
        inferred = args.motion.resolve().parent.parent / "sidecars/human_motion_world.npz"
        if inferred.is_file():
            args.human_motion = inferred
        else:
            parser.error("Stage 3 visualization requires --human-motion (positions_w, joint_names, edges, fps)")
    metadata = json.loads(args.metadata.read_text())
    motion = ReferenceMotion.load(args.motion, joint_names=tuple(metadata["joint_names"]),
                                  body_names=tuple(metadata["selected_body_names"]), motion_order=args.motion_order)
    segments = []
    if args.segments_json:
        for task in json.loads(args.segments_json.read_text())["tasks"]:
            first = round((float(task["start_s"]) - args.source_offset_s) * motion.fps)
            last = round((float(task["end_s"]) - args.source_offset_s) * motion.fps)
            if last > 0 and first < motion.frames:
                segments.append({"id": task["id"], "start": max(0, first),
                                 "stop": min(motion.frames, last)})
    temporary, asset = _position_asset(args.robot, metadata)
    try:
        sim = _load_scene_file(args.scene_file, asset, motion) if args.scene_file else _build_scene(asset, motion, soft=False)
        KinematicStudio(sim, motion, start=args.start_frame,
                        stop=motion.frames if args.stop_frame is None else args.stop_frame,
                        port=args.port, human_motion=args.human_motion,
                        object_track=args.object_track, object_name=args.object_name,
                        source_offset_s=args.source_offset_s, segments=segments).run()
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    main()
