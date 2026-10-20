"""Slice the continuous prior without altering motion or boundary velocities."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np


ROOT = example_dir(__file__)
FULL = ROOT / "retarget/body_prior_27_end"
SUPPORTED = ROOT / "retarget/ground_support_27_end/supported_motion.npz"
VIZ = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization")
SOURCE_START_S = 27.0


def main():
    manifest = json.loads((ROOT / "segments.json").read_text())
    with np.load(SUPPORTED) as z:
        motion = {key: z[key] for key in z.files}
    with np.load(FULL / "sidecars/timing.npz") as z:
        timing = {key: z[key] for key in z.files}
    with np.load(FULL / "sidecars/human.npz") as z:
        human = {key: z[key] for key in z.files}
    fps = int(motion["fps"])
    camera_sync = {}
    output_receive_s = timing["receive_timestamp_ns"].astype(np.float64) / 1e9
    for label, stem in [("head", "camera_head"), ("left", "camera_hand_left"), ("right", "camera_hand_right")]:
        camera_times = np.asarray([float(line) for line in (VIZ / f"{stem}_color_image_raw_compressed_frames.txt").read_text().splitlines()[4:]])
        after = np.searchsorted(camera_times, output_receive_s).clip(1, len(camera_times) - 1)
        before = after - 1
        nearest = np.where(abs(camera_times[before]-output_receive_s) <= abs(camera_times[after]-output_receive_s), before, after)
        camera_sync[f"camera_{label}_frame_index"] = nearest.astype(np.int32)
        camera_sync[f"camera_{label}_time_error_s"] = (camera_times[nearest]-output_receive_s).astype(np.float32)
    for task in manifest["tasks"]:
        if task["status"] == "existing":
            continue
        begin = round((task["start_s"] - SOURCE_START_S) * fps)
        end = min(round((task["end_s"] - SOURCE_START_S) * fps), len(motion["joint_pos"]))
        source_begin = round((task["start_s"] - SOURCE_START_S) * 120)
        source_end = min(round((task["end_s"] - SOURCE_START_S) * 120), len(timing["source_motive_frame_number"]))
        if not 0 <= begin < end <= len(motion["joint_pos"]):
            raise ValueError((task["id"], begin, end))
        dest = ROOT / "tasks" / task["id"]
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "body_prior.npz", **{
            key: (value[begin:end] if value.ndim and len(value) == len(motion["joint_pos"]) else value)
            for key, value in motion.items()
        })
        np.savez_compressed(dest / "source_sync.npz", **{
            key: (
                value[begin:end] if value.ndim and len(value) == len(motion["joint_pos"])
                else value[source_begin:source_end] if value.ndim and len(value) == len(timing["source_motive_frame_number"])
                else value
            ) for key, value in timing.items()
        }, **{key: value[begin:end] for key, value in camera_sync.items()},
            recording_time_s=SOURCE_START_S + timing["motion_time_s"][begin:end])
        np.savez_compressed(dest / "human.npz", **{
            key: (value[begin:end] if value.ndim and len(value) == len(motion["joint_pos"]) else value)
            for key, value in human.items()
        })
        task["body_prior"] = str((dest / "body_prior.npz").relative_to(ROOT))
        task["motion_frames"] = end - begin
        task["motive_frames_inclusive"] = [
            int(timing["source_motive_frame_number"][source_begin]),
            int(timing["source_motive_frame_number"][source_end - 1]),
        ]
    (ROOT / "segments.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("Wrote", sum(t["status"] != "existing" for t in manifest["tasks"]), "task priors")


if __name__ == "__main__":
    main()
