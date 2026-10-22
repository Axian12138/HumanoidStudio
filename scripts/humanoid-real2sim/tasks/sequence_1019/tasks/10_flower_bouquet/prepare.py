"""Immutable source audit and scene evidence preparation for clip 10."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
SNAP = ROOT / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot"
SOURCE = SNAP / "tasks/10_flower_bouquet"
VIZ = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    dest = r / "retarget/body_prior/10_flower_bouquet"
    dest.mkdir(parents=True, exist_ok=True)
    for f in SOURCE.glob("*.npz"):
        shutil.copy2(f, dest / f.name)
    order = SNAP / "retarget/ground_support_27_end/sidecars/motion_order.json"
    shutil.copy2(order, dest / "motion_order.json")
    h = np.load(dest / "human.npz")
    s = np.load(dest / "source_sync.npz")
    m = np.load(dest / "body_prior.npz")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "overlay", ROOT / "scripts/humanoid-real2sim/tasks/sequence_1019/export_human_overlay.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    np.savez_compressed(
        dest / "human_motion_world.npz",
        positions_w=h["keypoints_w"],
        joint_names=np.array(mod.NAMES),
        edges=np.array([(mod.NAMES.index(x), mod.NAMES.index(y)) for x, y in mod.EDGE_NAMES]),
        fps=h["fps"],
        source_time_s=s["recording_time_s"],
    )
    analysis = r / "analysis/10_flower_bouquet"
    analysis.mkdir(parents=True, exist_ok=True)
    events = [
        ("right_open_approach", 112.0, 113.0, "flower_bouquet", "right", "approach"),
        ("right_grasp_wrapped_stems", 113.0, 114.5, "flower_bouquet", "right", "grasp"),
        ("right_hold_lift", 114.5, 121.0, "flower_bouquet", "right", "fixed_grasp_estimate"),
        ("left_draw_basket_out", 118.0, 120.5, "small_wicker_basket", "left", "rim_handle_pull"),
        ("left_release", 120.5, 121.0, "small_wicker_basket", "left", "release"),
        ("flower_insert", 121.0, 123.4, "flower_bouquet", "right", "place_leaning_against_cabinet"),
        ("flower_release", 123.4, 124.0, "flower_bouquet", "right", "release"),
        ("left_push_basket_back", 124.7, 127.6, "small_wicker_basket", "left", "unilateral_exterior_push"),
        ("withdraw", 127.6, 129.06, "small_wicker_basket", "left", "withdraw"),
    ]
    report = {
        "source_window_s": [108.0, 129.06],
        "frames": 1054,
        "fps": 50,
        "event_uncertainty_s": 0.35,
        "events": [dict(zip(["event", "start_s", "end_s", "object", "hand", "mode"], e, strict=True)) for e in events],
        "observation": "Head and both wrist synchronized source samples. Wrapped lower stems grasped, basket pulled at near rim, flower leaned against tall cabinet side. Last basket push is left hand; right already released.",
        "not_claimed": [
            "Full continuous visual review before dense sheets inspected",
            "metric camera alignment",
            "actual force closure",
            "physics interaction",
        ],
    }
    (analysis / "events.json").write_text(json.dumps(report, indent=2))
    audit = {
        "status": "pending_scene_and_visual_review",
        "shape_contract": {k: list(m[k].shape) for k in m.files},
        "finite": all(np.isfinite(m[k]).all() for k in m.files),
        "quat_norm_max_error": float(abs(np.linalg.norm(m["body_quat_w"], axis=-1) - 1).max()),
        "root_human_xy_max_m": float(
            np.linalg.norm(m["body_pos_w"][:, 0, :2] - h["keypoints_w"][:, 0, :2], axis=1).max()
        ),
        "sync": {},
        "input_sha256": {str(f): sha(f) for f in [*SOURCE.glob("*.npz"), order]},
    }
    for cam in ["head", "left", "right"]:
        err = s[f"camera_{cam}_time_error_s"]
        audit["sync"][cam] = {
            "max_abs_error_s": float(abs(err).max()),
            "repeated_video_indices": int((np.diff(s[f"camera_{cam}_frame_index"]) == 0).sum()),
            "frame_range": s[f"camera_{cam}_frame_index"][[0, -1]].tolist(),
        }
    (dest / "audit.json").write_text(json.dumps(audit, indent=2))
    print(dest)


if __name__ == "__main__":
    main()
