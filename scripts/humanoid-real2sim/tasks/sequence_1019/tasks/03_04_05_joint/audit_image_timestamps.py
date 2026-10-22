"""Preserve image header/log timestamp differences at calibration observations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[35.0, 39.0, 44.0, 47.0, 48.0, 51.0])
    parser.add_argument("--output-subdir", default="inputs/03_04_05")
    args = parser.parse_args()
    output = args.run / args.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    origin = 1792398696937749979
    requested = args.times
    best = {}
    images = {}
    reader = McapReader(source)
    for frame_index, (_, _, log, pub, data) in enumerate(
        reader.iter_messages({"/camera/head/color/image_raw/compressed"})
    ):
        for seconds in requested:
            distance = abs(log - (origin + round(seconds * 1e9)))
            if seconds in best and distance >= best[seconds]["target_distance_ns"]:
                continue
            c = CdrReader(data)
            sec, ns, frame = c.i32(), c.u32(), c.string()
            best[seconds] = {
                "target_distance_ns": distance,
                "video_frame_index": frame_index,
                "log_time_ns": log,
                "publish_time_ns": pub,
                "header_time_ns": sec * 10**9 + ns,
                "frame_id": frame,
                "header_to_log_ms": (log - sec * 10**9 - ns) / 1e6,
                "payload_sha256": hashlib.sha256(data).hexdigest(),
            }
            _image_format = c.string()
            size = c.u32()
            images[seconds] = bytes(data[c.pos : c.pos + size])
    for seconds, image_bytes in images.items():
        (output / f"head_{seconds:g}_header.jpg").write_bytes(image_bytes)
    (output / "image_header_timestamp_audit.json").write_text(
        json.dumps(
            {
                "origin_log_time_ns": origin,
                "samples": best,
                "interpretation": "Observed header/log differences only. No fitted or applied motion time shift.",
            },
            indent=2,
        )
    )
    print({t: row["header_to_log_ms"] for t, row in best.items()})


if __name__ == "__main__":
    main()
