"""Read raw pose, fusion flags and validity without applying coordinate assumptions."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--clock", choices=["receive", "image-header"], default="receive")
    args = parser.parse_args()
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader, parse_frame_info

    sync = np.load(args.run / "retarget/body_prior/10_flower_bouquet/source_sync.npz")
    origin = int(sync["receive_timestamp_ns"][0]) - round(float(sync["recording_time_s"][0]) * 1e9)
    times = [90.0, 92.0, 96.0, 109.4, 110.0, 111.0, 112.0, 113.0, 114.0, 120.0, 123.4, 124.0, 124.6, 125.0, 128.0]
    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    reader = McapReader(source)
    header_samples = json.loads((args.run / "inputs/10_flower_bouquet/image_header_timestamp_audit.json").read_text())[
        "samples"
    ]
    if args.clock == "image-header":
        times = [t for t in times if str(t) in header_samples]
    target_times = {
        t: (header_samples[str(t)]["header_time_ns"] if args.clock == "image-header" else origin + round(t * 1e9))
        for t in times
    }
    rows = {}
    for _, _, timestamp, _, payload in reader.iter_messages({"/optitrack/head_rigid"}):
        for time in times:
            error = abs(timestamp - target_times[time])
            if time in rows and error >= rows[time]["target_error_ns"]:
                continue
            c = CdrReader(payload)
            frame = parse_frame_info(c)
            name, resolved, present, rigid_id = c.string(), c.boolean(), c.boolean(), c.i32()
            position = [c.f64() for _ in range(3)]
            quaternion = [c.f64() for _ in range(4)]
            mean_error, params, valid, fused = c.f32(), c.i16(), c.boolean(), c.boolean()
            rows[time] = {
                "target_error_ns": error,
                "log_time_ns": timestamp,
                "motive_frame_number": frame.motive_frame_number,
                "name": name,
                "resolved": resolved,
                "present": present,
                "rigid_id": rigid_id,
                "position_raw": position,
                "quaternion_xyzw_raw": quaternion,
                "mean_error": mean_error,
                "params": params,
                "tracking_valid": valid,
                "sensor_fused": fused,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
    target = args.run / "inputs/10_flower_bouquet/head_raw_pose_audit.json"
    if args.clock == "image-header":
        target = target.with_stem(target.stem + "_header")
    target.write_text(json.dumps({"origin_ns": origin, "source": str(source), "rows": rows}, indent=2))
    print({t: (row["sensor_fused"], row["mean_error"], row["tracking_valid"]) for t, row in rows.items()})


if __name__ == "__main__":
    main()
