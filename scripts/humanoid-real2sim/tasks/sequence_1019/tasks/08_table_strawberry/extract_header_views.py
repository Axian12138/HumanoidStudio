"""Extract original head images and nearest Motive poses at image header times."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    out = a.run / "inputs/08_table_strawberry/header_views"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader, parse_frame_info

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    origin = 1792398696937749979
    times = [90.0, 90.5, 91.0, 91.5, 92.0, 96.0]
    best = {}
    payloads = {}
    reader = McapReader(source)
    for _, _, log, _, payload in reader.iter_messages({"/camera/head/color/image_raw/compressed"}):
        for t in times:
            distance = abs(log - (origin + round(t * 1e9)))
            if t in best and distance >= best[t]["target_error_ns"]:
                continue
            c = CdrReader(payload)
            sec, ns, frame = c.i32(), c.u32(), c.string()
            best[t] = {
                "target_error_ns": distance,
                "image_log_time_ns": log,
                "image_header_time_ns": sec * 10**9 + ns,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
            payloads[t] = payload[payload.index(b"\xff\xd8") :]
    print("images selected", flush=True)
    poses = {}
    for _, _, log, _, payload in reader.iter_messages({"/optitrack/head_rigid"}):
        for t in times:
            distance = abs(log - best[t]["image_header_time_ns"])
            if t in poses and distance >= poses[t]["target_error_ns"]:
                continue
            c = CdrReader(payload)
            frame = parse_frame_info(c)
            name, resolved, present, rid = c.string(), c.boolean(), c.boolean(), c.i32()
            pos = [c.f64() for _ in range(3)]
            quat = [c.f64() for _ in range(4)]
            error, params, valid, fused = c.f32(), c.i16(), c.boolean(), c.boolean()
            poses[t] = {
                "target_error_ns": distance,
                "log_time_ns": log,
                "motive_frame_number": frame.motive_frame_number,
                "name": name,
                "resolved": resolved,
                "present": present,
                "rigid_id": rid,
                "params": params,
                "position_raw": pos,
                "quaternion_xyzw_raw": quat,
                "mean_error": error,
                "valid": valid,
                "fused": fused,
            }
    for t, payload in payloads.items():
        (out / f"{t:.1f}.jpg").write_bytes(payload)
    (out / "metadata.json").write_text(json.dumps({"source": str(source), "images": best, "poses": poses}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
