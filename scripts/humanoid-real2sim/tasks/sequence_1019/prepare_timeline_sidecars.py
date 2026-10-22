"""Normalize selected source metadata and attach the full-world human reference."""

import argparse
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import sha256

ALIASES = {
    "receive_timestamp_ns": ("receive_timestamp_ns",),
    "camera_head_frame_index": ("camera_head_frame_index", "head_frame_index"),
    "camera_head_time_error_s": ("camera_head_time_error_s", "head_time_error_s"),
    "camera_left_frame_index": ("camera_left_frame_index", "hand_left_frame_index"),
    "camera_left_time_error_s": ("camera_left_time_error_s", "hand_left_time_error_s"),
    "camera_right_frame_index": ("camera_right_frame_index", "hand_right_frame_index"),
    "camera_right_time_error_s": ("camera_right_time_error_s", "hand_right_time_error_s"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--human-source-start-s", type=float, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assembly = json.loads((args.candidate / "assembly.json").read_text())
    if "spec_sha256" in assembly and sha256(args.spec) != assembly["spec_sha256"]:
        raise ValueError("selection specification changed after motion assembly")
    with np.load(args.candidate / "source_time.npz") as z:
        times = z["recording_time_s"]
    arrays = {key: [] for key in (*ALIASES, "motive_frame_number", "motive_time_error_s")}
    source_times, mappings = [], []
    for selection in assembly["selection"]:
        entry = spec["clips"][selection["clip"]]
        path = Path(entry["directory"]) / entry.get("metadata", ".") / "source_sync.npz"
        if sha256(path) != selection["sync_sha256"]:
            raise ValueError(f"source metadata changed after assembly: {path}")
        if "order_sha256" in selection and sha256(path.with_name("motion_order.json")) != selection["order_sha256"]:
            raise ValueError(f"source naming order changed after assembly: {path.parent}")
        first, stop = selection["source_interval_s"]
        mapping = {}
        with np.load(path) as z:
            t = z["recording_time_s"]
            ticks = np.rint(t * 50).astype(int)
            mask = (ticks >= round(first * 50)) & (ticks < round(stop * 50))
            source_times.append(t[mask])
            for key, aliases in ALIASES.items():
                present = [alias for alias in aliases if alias in z]
                if len(present) != 1 or z[present[0]].shape != t.shape:
                    raise ValueError(f"{path}: missing or ambiguous frame field {key}")
                arrays[key].append(z[present[0]][mask])
                mapping[key] = present[0]
            if "motive_frame_number" in z and "motive_time_error_s" in z:
                frames, errors = z["motive_frame_number"], z["motive_time_error_s"]
                if frames.shape != t.shape or errors.shape != t.shape:
                    raise ValueError(f"{path}: invalid per-frame Motive mapping")
                mapping["motive"] = "existing per-frame nearest receive-time mapping"
            else:
                source_ns = z["source_receive_timestamp_ns"]
                source_frames = z["source_motive_frame_number"]
                target_ns = z["receive_timestamp_ns"]
                if (
                    source_ns.ndim != 1
                    or len(source_ns) < 2
                    or source_frames.shape != source_ns.shape
                    or np.any(np.diff(source_ns) <= 0)
                ):
                    raise ValueError(f"{path}: invalid raw Motive receive-time mapping")
                indexes = np.searchsorted(source_ns, target_ns).clip(1, len(source_ns) - 1)
                indexes = np.where(
                    abs(source_ns[indexes - 1] - target_ns) <= abs(source_ns[indexes] - target_ns),
                    indexes - 1,
                    indexes,
                )
                frames = source_frames[indexes]
                errors = (source_ns[indexes] - target_ns) / 1e9
                mapping["motive"] = "nearest source receive timestamp; earlier sample wins ties"
            arrays["motive_frame_number"].append(frames[mask])
            arrays["motive_time_error_s"].append(errors[mask])
        mappings.append({"clip": selection["clip"], "fields": mapping})
    if not np.allclose(np.concatenate(source_times), times, atol=1e-7, rtol=0):
        raise ValueError("normalized source metadata does not match assembled motion")
    sync = {key: np.concatenate(chunks) for key, chunks in arrays.items()}
    sync.update(recording_time_s=times, motion_time_s=np.arange(len(times)) / 50)
    if np.any(np.diff(sync["receive_timestamp_ns"]) <= 0):
        raise ValueError("receive timestamps must increase across all seams")
    if np.any(np.diff(sync["motive_frame_number"]) <= 0):
        raise ValueError("Motive frame numbers must increase across all seams")
    for key, values in sync.items():
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {key}")
    with np.load(args.human) as z:
        if float(z["fps"]) != 50:
            raise ValueError("human reference must be 50 Hz")
        indexes = np.rint((times - args.human_source_start_s) * 50).astype(int)
        if indexes[0] < 0 or indexes[-1] >= len(z["positions_w"]):
            raise ValueError("human reference does not cover source window")
        human = {key: z[key] for key in z.files}
        human["positions_w"] = human["positions_w"][indexes]
    for name in ("source_sync.npz", "human_motion_world.npz", "sidecars.json"):
        if (args.candidate / name).exists():
            raise FileExistsError(args.candidate / name)
    np.savez_compressed(args.candidate / "source_sync.npz", **sync)
    np.savez_compressed(args.candidate / "human_motion_world.npz", **human)
    report = {
        "scope": "Source metadata and human reference only, no interaction acceptance.",
        "frames": len(times),
        "source_alias_mappings": mappings,
        "human_source": str(args.human),
        "human_source_sha256": sha256(args.human),
        "human_source_start_s": args.human_source_start_s,
        "max_motive_time_error_s": float(abs(sync["motive_time_error_s"]).max()),
        "max_camera_time_error_s": {
            key: float(np.abs(value).max())
            for key, value in sync.items()
            if key.startswith("camera_") and key.endswith("time_error_s")
        },
        "outputs_sha256": {
            name: sha256(args.candidate / name) for name in ("source_sync.npz", "human_motion_world.npz")
        },
    }
    (args.candidate / "sidecars.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "source_alias_mappings"}, indent=2))


if __name__ == "__main__":
    main()
