"""Extract measured hand marker positions for camera checks independent of room geometry."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--clock", choices=["receive", "image-header"], default="receive")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[51.5, 52.0, 52.5, 53.0, 53.5])
    parser.add_argument("--output-subdir", default="inputs/03_04_05/drawer_progress_holdout")
    args = parser.parse_args()
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader, parse_frame_info
    from processors import SOURCE_TO_CANONICAL

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    sync = np.load(args.run / "retarget/body_prior/10_flower_bouquet/source_sync.npz")
    origin = int(sync["receive_timestamp_ns"][0]) - round(float(sync["recording_time_s"][0]) * 1e9)
    times = args.times
    output = args.run / args.output_subdir
    world = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ SOURCE_TO_CANONICAL
    header_samples = json.loads(
        (output / "image_header_timestamp_audit.json").read_text()
    )["samples"]
    if args.clock == "image-header":
        times = [t for t in times if str(t) in header_samples]
    target_times = {
        t: (header_samples[str(t)]["header_time_ns"] if args.clock == "image-header" else origin + round(t * 1e9))
        for t in times
    }
    rows = {}
    reader = McapReader(source)
    for topic, _, timestamp, _, payload in reader.iter_messages({f"/optitrack/{args.side}_hand_rigid"}):
        for time in times:
            error = abs(timestamp - target_times[time])
            if time in rows and error >= rows[time]["target_error_ns"]:
                continue
            c = CdrReader(payload)
            info = parse_frame_info(c)
            name, resolved, present, rid = c.string(), c.boolean(), c.boolean(), c.i32()
            position = [c.f64() for _ in range(3)]
            quaternion = [c.f64() for _ in range(4)]
            mean_error, params, valid, fused = c.f32(), c.i16(), c.boolean(), c.boolean()
            markers = []
            for _ in range(c.u32()):
                marker_id, model_id, member_id = c.i32(), c.i32(), c.i32()
                raw = [c.f64() for _ in range(3)]
                xyz = world @ raw
                xyz[2] += 0.08245383203029633
                size, marker_params, residual = c.f32(), c.i16(), c.f32()
                names = [
                    "occluded",
                    "point_cloud_solved",
                    "model_solved",
                    "has_model",
                    "unlabeled",
                    "active",
                    "established",
                    "measurement",
                ]
                flags = {n: c.boolean() for n in names}
                markers.append(
                    {
                        "id": marker_id,
                        "model_id": model_id,
                        "member_id": member_id,
                        "position_raw": raw,
                        "position_world": xyz.tolist(),
                        "size": size,
                        "params": marker_params,
                        "residual": residual,
                        "flags": flags,
                    }
                )
            c.require_eof("RigidBodyFrame with markers")
            rows[time] = {
                "target_error_ns": error,
                "log_time_ns": timestamp,
                "source_motive_frame": info.motive_frame_number,
                "topic": topic,
                "name": name,
                "resolved": resolved,
                "present": present,
                "rigid_id": rid,
                "position_raw": position,
                "quaternion_xyzw_raw": quaternion,
                "mean_error": mean_error,
                "params": params,
                "valid": valid,
                "fused": fused,
                "markers": markers,
            }
    target = output / f"{args.side}_hand_marker_audit.json"
    if args.clock == "image-header":
        target = target.with_stem(target.stem + "_header")
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps({"source": str(source), "rows": rows}, indent=2))
    print({t: len(v["markers"]) for t, v in rows.items()})


if __name__ == "__main__":
    main()
