"""Preserve exact original three-camera payload/header evidence around hat release."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    origin = 1792398696937749979
    out = a.run / "analysis/03_04_05/hat_release_dense"
    samples = []
    for camera in ["head", "hand_right", "hand_left"]:
        topic = f"/camera/{camera}/color/image_raw/compressed"
        for index, (_, _, log, _pub, data) in enumerate(McapReader(source).iter_messages({topic})):
            if index not in range(1428, 1433):
                continue
            c = CdrReader(data)
            sec, ns, frame = c.i32(), c.u32(), c.string()
            fmt = c.string()
            size = c.u32()
            payload = bytes(data[c.pos : c.pos + size])
            name = f"{camera}_mcap_{index}.jpg"
            (out / name).write_bytes(payload)
            samples.append(
                {
                    "camera": camera,
                    "video_frame_index": index,
                    "video_pts_s": index / 30,
                    "log_time_ns": log,
                    "header_time_ns": sec * 10**9 + ns,
                    "log_relative_s": (log - origin) / 1e9,
                    "header_relative_s": (sec * 10**9 + ns - origin) / 1e9,
                    "image": name,
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "frame_id": frame,
                    "format": fmt,
                }
            )
            if index == 1432:
                break
    (out / "exact_timestamps.json").write_text(json.dumps({"origin_ns": origin, "samples": samples}, indent=2) + "\n")
    print(samples)


if __name__ == "__main__":
    main()
