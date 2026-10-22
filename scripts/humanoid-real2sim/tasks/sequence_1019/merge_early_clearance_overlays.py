"""Compose audited local repairs without replacing unrelated early-task motion."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("root_overlay", type=Path)
    parser.add_argument("leg_overlay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    def load(folder):
        return MotionFields.load(folder / "motion.npz", folder / "motion_order.json")

    root_review = json.loads((args.root_overlay / "review.json").read_text())
    leg_review = json.loads((args.leg_overlay / "review.json").read_text())
    root_base = Path(root_review["base"])
    leg_base = Path(leg_review["source_candidate"])
    base, root, leg, rb, lb = map(load, (args.base, args.root_overlay, args.leg_overlay, root_base, leg_base))
    for motion in (root, leg, rb, lb):
        if motion.joint_names != base.joint_names or motion.body_names != base.body_names:
            raise ValueError("Named model orders differ")
        if motion.joint_pos.shape != base.joint_pos.shape or motion.fps != base.fps:
            raise ValueError("Motion dimensions or sample rates differ")
    time = 27 + np.arange(len(base.joint_pos)) / base.fps
    root_window = root_review["window_s"]
    root_rows = np.flatnonzero((time >= root_window[0] - 1e-7) & (time <= root_window[1] + 1e-7))
    leg_rows = np.flatnonzero((time >= 60.14 - 1e-7) & (time <= 61.4 + 1e-7))
    if np.intersect1d(root_rows, leg_rows).size:
        raise ValueError("Repair windows overlap")
    names = ["left_" + n + "_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    cols = [base.joint_names.index(n) for n in names]
    finger_cols = [i for i, name in enumerate(base.joint_names) if "_hand_" in name]
    root_cols = [i for i in range(len(base.joint_names)) if i not in finger_cols]

    def close(a, b, label):
        error = float(np.max(np.abs(a.astype(float) - b.astype(float))))
        if error > 2e-7:
            raise ValueError(f"{label} changed since repair derivation: {error}")
        return error

    checks = {
        "root_window_base_q_error": close(
            base.joint_pos[np.ix_(root_rows, root_cols)],
            rb.joint_pos[np.ix_(root_rows, root_cols)],
            "Root-window non-finger joints",
        ),
        "root_overlay_fingers_unchanged": close(
            root.joint_pos[np.ix_(root_rows, finger_cols)],
            rb.joint_pos[np.ix_(root_rows, finger_cols)],
            "Root-overlay fingers",
        ),
        "root_window_base_root_error": close(
            base.body_pos_w[root_rows, 0], rb.body_pos_w[root_rows, 0], "Root-window root"
        ),
        "root_window_base_quat_error": close(
            base.body_quat_w[root_rows, 0], rb.body_quat_w[root_rows, 0], "Root-window quaternion"
        ),
        "leg_window_base_q_error": close(
            base.joint_pos[np.ix_(leg_rows, cols)], lb.joint_pos[np.ix_(leg_rows, cols)], "Leg-window joints"
        ),
        "leg_window_root_error": close(base.body_pos_w[leg_rows, 0], lb.body_pos_w[leg_rows, 0], "Leg-window root"),
        "leg_window_quat_error": close(
            base.body_quat_w[leg_rows, 0], lb.body_quat_w[leg_rows, 0], "Leg-window quaternion"
        ),
        "root_overlay_rotation_error": close(root.body_quat_w[:, 0], rb.body_quat_w[:, 0], "Root-overlay rotation"),
        "leg_overlay_root_error": close(leg.body_pos_w[:, 0], lb.body_pos_w[:, 0], "Leg-overlay root"),
    }
    q = base.joint_pos.astype(float).copy()
    root_position = base.body_pos_w.astype(float).copy()
    # The root repair preserves wrist world poses and does not solve fingers.
    # Keep newer grasp shapes while requiring every upstream arm, waist and leg
    # coordinate to match the derivation of the root repair.
    q[np.ix_(root_rows, root_cols)] = root.joint_pos[np.ix_(root_rows, root_cols)]
    root_position[root_rows, 0] = root.body_pos_w[root_rows, 0]
    q[np.ix_(leg_rows, cols)] = leg.joint_pos[np.ix_(leg_rows, cols)]
    outside = np.ones(len(q), dtype=bool)
    outside[np.r_[root_rows, leg_rows]] = False
    if not np.array_equal(q[outside], base.joint_pos[outside]):
        raise AssertionError("Composition changed joints outside authorized windows")
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(base, joint_pos=q, body_pos_w=root_position), robot)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "merge_snapshot.py")
    for name in (
        "motion_order.json",
        "source_sync.npz",
        "human_motion_world.npz",
        "body_prior.npz",
        "black_ribbon_hat.npz",
        "floor_strawberry.npz",
        "round_laundry_basket.npz",
        "drawer_slide.npz",
        "tracks_manifest.json",
    ):
        shutil.copy2(args.base / name, args.output / name)
    result.save(args.output / "motion.npz")
    paths = [p / "motion.npz" for p in (args.base, args.root_overlay, args.leg_overlay, root_base, leg_base)]
    report = {
        "status": "pending_composed_candidate_review",
        "base": str(args.base),
        "root_overlay": str(args.root_overlay),
        "leg_overlay": str(args.leg_overlay),
        "input_sha256": {str(p): sha256(p) for p in paths},
        "source_compatibility_checks": checks,
        "root_window_s": root_window,
        "leg_window_s": [60.14, 61.4],
        "outside_joint_positions_exact": True,
        "root_window_fingers_from_base": True,
        "object_tracks_source": str(args.base),
        "limitations": [
            "Composition does not accept upstream hat geometry or source motion.",
            "Requires independent full-body geometry and source review after composition.",
            "Hip-pelvis enabled proxy overlap remains measured separately from native visual shells.",
        ],
    }
    (args.output / "merge_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
