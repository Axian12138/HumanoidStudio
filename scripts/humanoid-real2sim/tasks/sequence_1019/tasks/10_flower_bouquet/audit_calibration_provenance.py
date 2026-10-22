"""Preserve recorder metadata and test calibration's tracker identity assumptions."""

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
    from mcap_skeleton_to_bvh import (
        OP_CHUNK,
        OP_SCHEMA,
        CdrReader,
        McapReader,
        McapScalarReader,
        iter_embedded_records,
        iter_top_records,
        parse_chunk,
    )

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    out = args.run / "inputs/10_flower_bouquet"
    raw = out / "calibration_payloads"
    raw.mkdir(exist_ok=True)
    reader = McapReader(source)
    records = {}
    topics = {"/daq/device_info", "/daq/data_system", "/daq/optitrack_capture_mode", "/daq/seat_id"}
    for topic, _, ts, pub, payload in reader.iter_messages(topics):
        if topic in records:
            continue
        path = raw / (topic.strip("/").replace("/", "_") + ".cdr")
        path.write_bytes(payload)
        value = CdrReader(payload).string()
        records[topic] = {
            "value": value,
            "log_time_ns": ts,
            "publish_time_ns": pub,
            "message_type": "std_msgs/msg/String",
            "raw_path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if len(records) == len(topics):
            break
    for opcode, payload in iter_top_records(source, {OP_SCHEMA, OP_CHUNK}):
        entries = (
            [(opcode, payload)] if opcode == OP_SCHEMA else iter_embedded_records(parse_chunk(payload, reader.zstd))
        )
        found = False
        for op, data in entries:
            if op != OP_SCHEMA:
                continue
            scalar = McapScalarReader(data)
            scalar.u16()
            name, _encoding, text = scalar.string(), scalar.string(), scalar.string()
            if name == "optitrack_msgs/msg/RigidBodyFrame":
                schema = out / "RigidBodyFrame.ros2msg"
                schema.write_text(text)
                found = True
                break
        if found:
            break
    report = {
        "source": str(source),
        "records": records,
        "rigid_schema": str(schema),
        "rigid_schema_sha256": hashlib.sha256(schema.read_bytes()).hexdigest(),
        "quaternion_field_order": "x,y,z,w, confirmed geometry_msgs/Quaternion schema",
        "recorded_topics": sorted(c.topic for c in reader.channels.values()),
        "finding": "device_info names OpenVR tracker_4 for head, tracker_2/3 for hands. Recording has OptiTrack rigid topics and no OpenVR pose topics. The tracker-to-camera calibration cannot be assumed to use Motive head_rigid local axes.",
        "status": "camera_info authoritative for recorded stream; extrinsic tracker identity unresolved; direct head_rigid @ T and inverse T both fail cabinet reprojection",
    }
    target = out / "calibration_provenance.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(target)


if __name__ == "__main__":
    main()
