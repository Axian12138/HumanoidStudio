"""Assemble explicitly selected candidate intervals; recompute derivatives before slicing.

This prepares a diagnostic robot motion only. It does not accept the interactions,
create missing object states, or replace any original motion.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_timeline_overlaps import FIELDS, load

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    chunks, ticks, provenance = [], [], []
    order = None
    for selection in spec["selection"]:
        name = selection["clip"]
        data, source_ticks, clip_order, hashes = load(spec["clips"][name])
        if order is None:
            order = clip_order
        for key in ("robot_joint_names", "robot_body_names"):
            if order[key] != clip_order[key]:
                raise ValueError(f"{name}: incompatible {key}")
        first, last = [round(t * 50) for t in selection["source_interval_s"]]
        mask = (source_ticks >= first) & (source_ticks < last)
        selected = source_ticks[mask]
        if not np.array_equal(selected, np.arange(first, last)):
            raise ValueError(f"{name}: selection has missing source frames")
        chunks.append({key: data[key][mask] for key in FIELDS})
        ticks.append(selected)
        provenance.append({"clip": name, "source_interval_s": [first / 50, last / 50], **hashes})
    ticks = np.concatenate(ticks)
    if np.any(np.diff(ticks) != 1):
        raise ValueError("selection must form one contiguous non-overlapping time axis")
    values = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in FIELDS}
    motion = MotionFields(tuple(order["robot_joint_names"]), tuple(order["robot_body_names"]), **values, fps=50)
    refreshed = refresh_fk(motion, args.model)
    # Verify that FK refresh does not hide a pose/root inconsistency in source slices.
    pose_delta = float(abs(refreshed.body_pos_w - motion.body_pos_w).max())
    quat_delta = np.minimum(
        np.linalg.norm(refreshed.body_quat_w - motion.body_quat_w, axis=-1),
        np.linalg.norm(refreshed.body_quat_w + motion.body_quat_w, axis=-1),
    )
    if pose_delta > 1e-5 or quat_delta.max() > 1e-5:
        raise ValueError(f"source FK inconsistency: position {pose_delta}, quaternion {quat_delta.max()}")
    args.output.mkdir(parents=True, exist_ok=False)
    refreshed.save(args.output / "motion.npz")
    (args.output / "motion_order.json").write_text(json.dumps(order, indent=2) + "\n")
    np.savez_compressed(args.output / "source_time.npz", recording_time_s=ticks / 50, fps=50)
    report = {
        "status": "diagnostic_robot_only_not_accepted",
        "spec_sha256": sha256(args.spec),
        "assembly_script_sha256": sha256(Path(__file__)),
        "selection": provenance,
        "frames": len(ticks),
        "source_interval_s": [ticks[0] / 50, (ticks[-1] + 1) / 50],
        "model_sha256": sha256(args.model),
        "motion_sha256": sha256(args.output / "motion.npz"),
        "order_sha256": sha256(args.output / "motion_order.json"),
        "joint_positions_changed": False,
        "fk_refresh_max_position_delta_m": pose_delta,
        "fk_refresh_max_quaternion_sign_invariant_delta": float(quat_delta.max()),
        "derivatives": "Full-axis central joint/linear differences; forward world rotational differences.",
        "pending": [
            "All source candidates retain their own unresolved acceptance items.",
            "Synchronized human and full source metadata.",
            "All free objects and articulated state continuity.",
            "Unified scene, final slice export and whole-motion review.",
            "Whole-axis speed, acceleration, support and collision acceptance.",
        ],
    }
    (args.output / "assembly.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
