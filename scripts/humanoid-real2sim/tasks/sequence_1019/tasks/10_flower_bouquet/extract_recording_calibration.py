"""Read calibration carried by this MCAP rather than inherited dataset metadata."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    reader = McapReader(source)
    topics = {
        c.topic
        for c in reader.channels.values()
        if "camera_info" in c.topic or "T_head_tracker_to_head_camera" in c.topic
    }
    records = {}
    counts = {}
    for topic, _, ts, _, payload in reader.iter_messages(topics):
        counts[topic] = counts.get(topic, 0) + 1
        if topic in records:
            continue
        raw = args.run / "inputs/10_flower_bouquet/calibration_payloads"
        raw.mkdir(exist_ok=True)
        raw_path = raw / (topic.strip("/").replace("/", "_") + ".cdr")
        raw_path.write_bytes(payload)
        c = CdrReader(payload)
        if "camera_info" in topic:
            sec, ns, frame = c.i32(), c.u32(), c.string()
            height, width = c.u32(), c.u32()
            distortion = c.string()
            d = [c.f64() for _ in range(c.u32())]
            k = [c.f64() for _ in range(9)]
            rot = [c.f64() for _ in range(9)]
            projection = [c.f64() for _ in range(12)]
            bx, by = c.u32(), c.u32()
            roi = [c.u32() for _ in range(4)]
            rectify = c.boolean()
            assert not any(payload[c.pos :]), "nonzero trailing CameraInfo bytes"
            records[topic] = {
                "log_time_ns": ts,
                "stamp": [sec, ns],
                "frame_id": frame,
                "height": height,
                "width": width,
                "distortion_model": distortion,
                "D": d,
                "K": k,
                "R": rot,
                "P": projection,
                "binning": [bx, by],
                "roi": roi,
                "do_rectify": rectify,
            }
        else:
            dims = [{"label": c.string(), "size": c.u32(), "stride": c.u32()} for _ in range(c.u32())]
            offset = c.u32()
            data = [c.f64() for _ in range(c.u32())]
            c.require_eof("Float64MultiArray")
            records[topic] = {"log_time_ns": ts, "layout": dims, "data_offset": offset, "data": data}
        records[topic]["raw_payload_path"] = str(raw_path)
        records[topic]["raw_payload_sha256"] = hashlib.sha256(payload).hexdigest()
        records[topic]["message_type"] = (
            "sensor_msgs/msg/CameraInfo" if "camera_info" in topic else "std_msgs/msg/Float64MultiArray"
        )
    out = args.run / "inputs/10_flower_bouquet/recording_calibration.json"
    out.write_text(json.dumps({"source": str(source), "counts": counts, "records": records}, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
