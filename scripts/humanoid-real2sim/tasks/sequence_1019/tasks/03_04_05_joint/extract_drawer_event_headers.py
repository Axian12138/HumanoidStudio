"""Preserve original camera payloads and Header times around drawer closure."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=1515)
    p.add_argument("--end-frame", type=int, default=1635)
    p.add_argument("--output-subdir", default="analysis/03_04_05/drawer_event_dense")
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()
    if a.stride < 1:
        p.error("--stride must be positive")
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    origin = 1792398696937749979
    out = a.run / a.output_subdir
    if (out / "exact_timestamps.json").exists():
        raise FileExistsError(f"Preserving prior extraction: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cameras = ["head", "hand_left", "hand_right"]
    topics = {f"/camera/{camera}/color/image_raw/compressed": camera for camera in cameras}
    samples = []
    # Iterate each topic separately: this reader returns no topic name in its tuple.
    for topic, camera in topics.items():
        for index, (_, _, log, _pub, data) in enumerate(McapReader(source).iter_messages({topic})):
            if index > a.end_frame:
                break
            if index < a.start_frame or (index - a.start_frame) % a.stride:
                continue
            c = CdrReader(data)
            sec, ns, frame = c.i32(), c.u32(), c.string()
            fmt, size = c.string(), c.u32()
            payload = bytes(data[c.pos : c.pos + size])
            name = f"{camera}_mcap_{index}.jpg"
            (out / name).write_bytes(payload)
            samples.append(
                {
                    "camera": camera,
                    "video_frame_index": index,
                    "video_pts_s": index / 30,
                    "header_relative_s": (sec * 10**9 + ns - origin) / 1e9,
                    "header_time_ns": sec * 10**9 + ns,
                    "log_time_ns": log,
                    "image": name,
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "frame_id": frame,
                    "format": fmt,
                }
            )
    (out / "exact_timestamps.json").write_text(
        json.dumps({"source": str(source), "origin_ns": origin, "samples": samples}, indent=2) + "\n"
    )
    print({"samples": len(samples), "output": str(out)})


if __name__ == "__main__":
    main()
