"""Audit source-aligned motion overlaps without assembling or accepting candidates."""

import argparse
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import sha256

FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def load(entry):
    folder = Path(entry["directory"])
    metadata = folder / entry.get("metadata", ".")
    motion_path = folder / entry.get("motion", "motion.npz")
    with np.load(motion_path) as z:
        arrays = {key: z[key] for key in FIELDS}
        if float(z["fps"]) != 50:
            raise ValueError(f"{folder}: expected 50 Hz")
    sync_path = metadata / "source_sync.npz"
    with np.load(sync_path) as z:
        times = z["recording_time_s"]
    ticks = np.rint(times * 50).astype(np.int64)
    if len(times) != len(arrays["joint_pos"]) or not np.allclose(times, ticks / 50, atol=1e-7, rtol=0):
        raise ValueError(f"{folder}: invalid source time grid")
    if np.any(np.diff(ticks) != 1):
        raise ValueError(f"{folder}: missing or duplicated frames")
    for key, values in arrays.items():
        if len(values) != len(times) or not np.isfinite(values).all():
            raise ValueError(f"{folder}: invalid {key}")
    order_path = metadata / "motion_order.json"
    order = json.loads(order_path.read_text())
    return (
        arrays,
        ticks,
        order,
        {
            "motion": str(motion_path),
            "motion_sha256": sha256(motion_path),
            "sync_sha256": sha256(sync_path),
            "order": str(order_path),
            "order_sha256": sha256(order_path),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    loaded = {name: load(entry) for name, entry in spec["clips"].items()}
    rows = []
    for pair in spec["pairs"]:
        left, lt, lo, _ = loaded[pair["left"]]
        right, rt, ro, _ = loaded[pair["right"]]
        for key in ("robot_joint_names", "robot_body_names"):
            if lo[key] != ro[key]:
                raise ValueError(f"{pair}: {key} mismatch")
        common, li, ri = np.intersect1d(lt, rt, return_indices=True)
        if not len(common):
            raise ValueError(f"{pair}: no source overlap")
        row = {"left": pair["left"], "right": pair["right"], "frames": len(common), "fields": {}}
        row["source_interval_inclusive_s"] = [float(common[0] / 50), float(common[-1] / 50)]
        for key in FIELDS:
            delta = np.abs(left[key][li] - right[key][ri])
            per_frame = delta.reshape(len(common), -1).max(axis=1)
            changed = per_frame > 1e-7
            row["fields"][key] = {
                "max_absolute_delta": float(delta.max()),
                "overlap_interior_max_delta": float(per_frame[1:-1].max()) if len(common) > 2 else None,
                "changed_frames_above_1e_7": int(changed.sum()),
                "changed_frames_above_1e_5": int((per_frame > 1e-5).sum()),
                "changed_source_interval_s": (
                    [float(common[changed][0] / 50), float(common[changed][-1] / 50)] if changed.any() else None
                ),
            }
        if "switch_s" in pair:
            tick = round(pair["switch_s"] * 50)
            a = int(np.flatnonzero(lt == tick - 1)[0])
            b = int(np.flatnonzero(rt == tick)[0])
            step = right["joint_pos"][b] - left["joint_pos"][a]
            row["switch"] = {
                "source_s": tick / 50,
                "largest_joint_step_rad": float(abs(step).max()),
                "largest_joint_finite_difference_rad_s": float(abs(step).max() * 50),
                "stored_joint_velocity_jump_rad_s": float(abs(right["joint_vel"][b] - left["joint_vel"][a]).max()),
                "root_step_m": float(np.linalg.norm(right["body_pos_w"][b, 0] - left["body_pos_w"][a, 0])),
                "scope": "Boundary readings only; not a joint-specific speed or acceleration gate.",
            }
        rows.append(row)
    result = {
        "scope": "Candidate overlap evidence; no candidate acceptance or final timeline generated.",
        "tolerance_note": "1e-7 and 1e-5 are reporting bins, not acceptance gates. Float32 FK and derivative "
        "rounding may exceed 1e-7; isolated slice endpoint derivatives need full-axis recomputation.",
        "inputs": {name: values[3] for name, values in loaded.items()},
        "pairs": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
