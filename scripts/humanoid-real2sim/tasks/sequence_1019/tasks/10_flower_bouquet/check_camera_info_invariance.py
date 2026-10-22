"""Verify CameraInfo calibration fields stay unchanged throughout this recording."""

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
    reader = McapReader(source)
    topics = {c.topic for c in reader.channels.values() if c.topic.endswith("/camera_info")}
    variants = {t: {} for t in topics}
    for topic, _, timestamp, _, payload in reader.iter_messages(topics):
        c = CdrReader(payload)
        c.i32()
        c.u32()
        frame = c.string()
        height, width = c.u32(), c.u32()
        model = c.string()
        d = [c.f64() for _ in range(c.u32())]
        k = [c.f64() for _ in range(9)]
        rotation = [c.f64() for _ in range(9)]
        projection = [c.f64() for _ in range(12)]
        trailing = [c.u32() for _ in range(6)] + [c.boolean()]
        value = {
            "frame": frame,
            "height": height,
            "width": width,
            "distortion_model": model,
            "D": d,
            "K": k,
            "R": rotation,
            "P": projection,
            "binning_roi_rectify": trailing,
        }
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        row = variants[topic].setdefault(digest, {"count": 0, "first_log_time_ns": timestamp, "calibration": value})
        row["count"] += 1
        row["last_log_time_ns"] = timestamp
    target = a.run / "inputs/10_flower_bouquet/camera_info_invariance.json"
    target.write_text(
        json.dumps({"variants": variants, "invariant": all(len(v) == 1 for v in variants.values())}, indent=2)
    )
    print({k: [r["count"] for r in v.values()] for k, v in variants.items()})


if __name__ == "__main__":
    main()
