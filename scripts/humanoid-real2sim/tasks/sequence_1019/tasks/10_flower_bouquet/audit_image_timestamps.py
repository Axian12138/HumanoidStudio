"""Preserve image header/log timestamp differences at calibration observations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    origin = 1792398696937749979
    requested = [90.0, 92.0, 96.0, 109.4, 110.0, 111.0, 112.0, 113.0, 114.0, 120.0, 123.4, 124.0, 124.6, 125.0, 128.0]
    best = {}
    payloads = {}
    reader = McapReader(source)
    for _, _, log, pub, data in reader.iter_messages({"/camera/head/color/image_raw/compressed"}):
        for seconds in requested:
            distance = abs(log - (origin + round(seconds * 1e9)))
            if seconds in best and distance >= best[seconds]["target_distance_ns"]:
                continue
            c = CdrReader(data)
            sec, ns, frame = c.i32(), c.u32(), c.string()
            c.string()  # Consume the encoded-image format before the payload length.
            size = c.u32()
            payloads[seconds] = bytes(c.data[c.pos : c.pos + size])
            best[seconds] = {
                "target_distance_ns": distance,
                "log_time_ns": log,
                "publish_time_ns": pub,
                "header_time_ns": sec * 10**9 + ns,
                "frame_id": frame,
                "header_to_log_ms": (log - sec * 10**9 - ns) / 1e6,
                "payload_sha256": hashlib.sha256(data).hexdigest(),
            }
    image_dir = args.run / "inputs/10_flower_bouquet/header_views"
    image_dir.mkdir(exist_ok=True)
    for seconds, payload in payloads.items():
        (image_dir / f"{seconds}.jpg").write_bytes(payload)
    (args.run / "inputs/10_flower_bouquet/image_header_timestamp_audit.json").write_text(
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
