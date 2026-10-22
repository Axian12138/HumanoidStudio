"""Extract independent head rigid poses at explicitly named recording times."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[110.0, 111.0, 112.0, 113.0])
    args = parser.parse_args()
    sys.path[:0] = [
        "/home/galbot/workspace/ScaleTrack/retarget2/src",
        "/home/galbot/workspace/ScaleTrack/retarget2/scripts",
    ]
    from extract_right_camera_targets import parse_rigid_frame
    from mcap_skeleton_to_bvh import McapReader
    from processors import SOURCE_TO_CANONICAL

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    sync = np.load(args.run / "retarget/body_prior/10_flower_bouquet/source_sync.npz")
    seconds = args.times
    targets = sync["receive_timestamp_ns"][[round((t - 108) * 50) for t in seconds]]
    world = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ SOURCE_TO_CANONICAL
    reader = McapReader(source)
    best = {}
    validity = {}
    for _, sequence, ts, _, payload in reader.iter_messages({"/optitrack/head_rigid"}):
        frame = parse_rigid_frame(sequence, ts, payload)
        for j, t in enumerate(targets):
            if j not in best or abs(ts - t) < best[j][0]:
                pos = world @ frame.position
                pos[2] += 0.08245383203029633
                quat = (Rotation.from_matrix(world) * Rotation.from_quat(frame.orientation_xyzw)).as_quat()
                best[j] = (int(abs(ts - t)), pos.tolist(), quat.tolist())
                validity[j] = {
                    "resolved": bool(frame.resolved),
                    "present": bool(frame.present),
                    "tracking_valid": bool(frame.tracking_valid),
                    "motive_frame_number": int(frame.motive_frame_number),
                    "rigid_name": frame.name,
                    "receive_timestamp_ns": int(ts),
                }
    out = args.run / "inputs/10_flower_bouquet"
    (out / "head_keyframes.json").write_text(json.dumps(best, indent=2))
    (out / "head_keyframes_metadata.json").write_text(
        json.dumps(
            {
                "recording_times_s": seconds,
                "validity": validity,
                "source": str(source),
                "tuple_fields": ["nearest_receive_error_ns", "position_w_m", "quaternion_xyzw"],
                "coordinate_chain": "Rz90 @ SOURCE_TO_CANONICAL; z += 0.08245383203029633",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
