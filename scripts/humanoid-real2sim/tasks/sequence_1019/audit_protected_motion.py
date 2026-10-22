"""Independently compare protected coordinates against a source-aligned baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--allow-arm", choices=("left", "right"), action="append", default=[])
    parser.add_argument("--allow-hand", choices=("left", "right"), action="append", default=[])
    parser.add_argument("--source-offset-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = MotionFields.load(args.baseline, args.order)
    candidate = MotionFields.load(args.candidate, args.order)
    if baseline.joint_pos.shape != candidate.joint_pos.shape or baseline.fps != candidate.fps:
        raise ValueError("baseline and candidate must cover the identical source window")
    arm_suffixes = (
        "shoulder_pitch_joint",
        "shoulder_roll_joint",
        "shoulder_yaw_joint",
        "elbow_joint",
        "wrist_roll_joint",
        "wrist_pitch_joint",
        "wrist_yaw_joint",
    )
    allowed = {f"{side}_{suffix}" for side in args.allow_arm for suffix in arm_suffixes}
    missing = allowed - set(candidate.joint_names)
    if missing:
        raise ValueError(f"arm joint names not found: {sorted(missing)}")
    for side in args.allow_hand:
        hand = [name for name in candidate.joint_names if name.startswith(f"{side}_hand_")]
        if len(hand) != 7:
            raise ValueError(f"expected seven {side} hand joints, found {len(hand)}")
        allowed.update(hand)
    protected = [i for i, name in enumerate(candidate.joint_names) if name not in allowed]
    delta = np.abs(candidate.joint_pos[:, protected] - baseline.joint_pos[:, protected])
    frame, coordinate = np.unravel_index(np.argmax(delta), delta.shape)
    root = candidate.body_names.index("pelvis")
    root_position_delta = np.abs(candidate.body_pos_w[:, root] - baseline.body_pos_w[:, root])
    # q and -q represent the same orientation.
    qa, qb = candidate.body_quat_w[:, root], baseline.body_quat_w[:, root]
    root_quat_delta = np.minimum(np.linalg.norm(qa - qb, axis=1), np.linalg.norm(qa + qb, axis=1))
    changed_frames = np.flatnonzero(np.max(delta, axis=1) > 1e-7)
    passed = not len(changed_frames) and np.max(root_position_delta) <= 1e-7 and np.max(root_quat_delta) <= 1e-7
    report = {
        "scope": "protected joint positions and root pose only; not whole-body collision, support or HOI acceptance",
        "passed": bool(passed),
        "inputs": {str(p.resolve()): digest(p) for p in (args.baseline, args.candidate, args.order)},
        "source_offset_s": args.source_offset_s,
        "frames": len(candidate.joint_pos),
        "fps": candidate.fps,
        "allowed_joint_names": sorted(allowed),
        "protected_joint_names": [candidate.joint_names[i] for i in protected],
        "max_protected_position_delta_rad": float(delta.max()),
        "worst_joint": candidate.joint_names[protected[coordinate]],
        "worst_source_time_s": args.source_offset_s + int(frame) / candidate.fps,
        "changed_frames": changed_frames.tolist(),
        "root_position_max_delta_m": float(root_position_delta.max()),
        "root_quaternion_max_sign_invariant_delta": float(root_quat_delta.max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ("passed", "max_protected_position_delta_rad", "root_position_max_delta_m")})
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
