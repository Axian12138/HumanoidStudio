"""Measure a kick-basket RSS baseline against its 50 Hz source reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--motion-order", type=Path, required=True)
    args = parser.parse_args()
    folder = args.rollout
    with np.load(args.motion, allow_pickle=False) as reference:
        order = json.loads(args.motion_order.read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        measured = np.load(folder / "selected_body_pos_w.npy")
        basket = np.load(folder / "basket_pose_xyzw.npy")[:, :3]
        basket_ref = reference["object_pos_w"][:, list(reference["object_names"]).index("round_laundry_basket")]
        errors = {}
        for name in ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link"):
            actual = measured[:, manifest["selected_body_names"].index(name)]
            expected = reference["body_pos_w"][:, order["robot_body_names"].index(name)]
            distance = np.linalg.norm(actual - expected, axis=1)
            errors[name] = {
                "mean_m": float(distance.mean()),
                "contact_window_mean_m": float(distance[67:106].mean()),
                "max_m": float(distance.max()),
                "first_over_5cm_frame": next((int(i) for i, d in enumerate(distance) if d > 0.05), None),
                "first_over_20cm_frame": next((int(i) for i, d in enumerate(distance) if d > 0.2), None),
            }
        actual_delta = basket[-1, :2] - basket[0, :2]
        expected_delta = basket_ref[-1, :2] - basket_ref[0, :2]
        actual_distance = float(np.linalg.norm(actual_delta))
        expected_distance = float(np.linalg.norm(expected_delta))
        direction_dot = (
            float(np.dot(actual_delta, expected_delta) / (actual_distance * expected_distance))
            if actual_distance > 1e-9 and expected_distance > 1e-9 else None
        )
        pelvis_fell = float(measured[:, manifest["selected_body_names"].index("pelvis"), 2].min()) < 0.5
        basket_error = float(np.linalg.norm(basket[-1] - basket_ref[-1]))
        status = "stage4_baseline_fell" if pelvis_fell else (
            "stage4_baseline_contact_mismatch" if basket_error > 0.05 else "stage4_baseline_contact_matched"
        )
        report = {
            "status": status,
            "frames": int(len(measured)),
            "source_contact_window_frames": [67, 105],
            "body_position_errors": errors,
            "basket_actual_xy_displacement_m": actual_distance,
            "basket_reference_xy_displacement_m": expected_distance,
            "basket_motion_direction_dot": direction_dot,
            "basket_final_position_error_m": basket_error,
            "basket_first_25_frames_max_xy_drift_m": float(np.max(np.linalg.norm(basket[:26, :2] - basket[0, :2], axis=1))),
            "candidate_motion_exported": False,
            "reason": (
                "Pelvis dropped below 0.5 m during playback."
                if pelvis_fell else
                "Pelvis remained upright, but foot-to-basket contact did not reproduce reference displacement."
                if basket_error > 0.05 else
                "Basket displacement matches the reference within 5 cm; inspect pose and seam quality before export."
            ),
        }
    path = folder / "analysis.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
