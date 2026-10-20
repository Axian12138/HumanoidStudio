"""Map 50 Hz source times to Motive and camera frames from recorded timestamps."""

import csv
import hashlib
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np


ROOT = example_dir(__file__)
VIZ = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization")
START_S = 8.0
FRAMES = 999
FPS = 50.0


def nearest(sorted_times, requested):
    after = np.searchsorted(sorted_times, requested).clip(1, len(sorted_times) - 1)
    before = after - 1
    return np.where(abs(sorted_times[before] - requested) <= abs(sorted_times[after] - requested), before, after)


def main():
    source_times = []
    with (VIZ / "timestamps_table.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            source_times.append(float(row["/optitrack/op_skeleton"]))
    source_times = np.asarray(source_times)
    motion_time = START_S + np.arange(FRAMES) / FPS
    source_index = np.rint(motion_time * 120).astype(int)
    receive_times = source_times[source_index]
    arrays = {
        "motion_time_s": motion_time,
        "source_motive_frame_number": 656007 + source_index,
        "mocap_receive_timestamp_ns": np.rint(receive_times * 1e9).astype(np.int64),
    }
    for short, stem in (("head", "camera_head"), ("hand_left", "camera_hand_left"), ("hand_right", "camera_hand_right")):
        lines = (VIZ / f"{stem}_color_image_raw_compressed_frames.txt").read_text().splitlines()
        times = np.asarray([float(x) for x in lines[4:]])
        frames = nearest(times, receive_times)
        arrays[f"camera_{short}_frame_index"] = frames.astype(np.int32)
        arrays[f"camera_{short}_time_error_s"] = (times[frames] - receive_times).astype(np.float32)
    output = ROOT / "source/source_sync_8_28.npz"
    np.savez_compressed(output, **arrays)
    hashes = {}
    for name in ("mocap.bvh", "camera_head_color_image_raw_compressed.mp4", "camera_hand_right_color_image_raw_compressed.mp4"):
        digest = hashlib.sha256()
        with (VIZ / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    (ROOT / "source/source_manifest.json").write_text(json.dumps({
        "recording_id": "1-2026-10-19-16-31-26",
        "source_mcap": str(VIZ.parent / "1-2026-10-19-16-31-26_0.mcap"),
        "time_window_s_half_open": [START_S, START_S + FRAMES / FPS],
        "motive_frames_inclusive": [int(arrays["source_motive_frame_number"][0]), int(arrays["source_motive_frame_number"][-1])],
        "motion_fps": FPS,
        "source_to_motion": "Motive 120 Hz frame index rounded from absolute recording seconds; camera nearest by recorded Unix receive timestamp",
        "source_hashes_sha256": hashes,
    }, indent=2) + "\n")
    print(json.dumps({"frames": FRAMES, "motive": [int(arrays["source_motive_frame_number"][0]), int(arrays["source_motive_frame_number"][-1])], "max_camera_error_ms": {name: float(np.max(abs(arrays[f"camera_{name}_time_error_s"])) * 1000) for name in ("head", "hand_left", "hand_right")}}, indent=2))


if __name__ == "__main__":
    main()
