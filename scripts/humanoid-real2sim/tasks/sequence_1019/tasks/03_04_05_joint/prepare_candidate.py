"""Create a source-synchronous joint candidate and explicit geometric hypotheses.

This bootstrap never labels unreviewed object attachments as successful HOI.
It preserves the selected task 06 motion while exporting its basket context.
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

ROOT = Path(__file__).resolve().parents[6]
HISTORY = ROOT / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/historical_examples"
ST = Path("/home/galbot/workspace/ScaleTrack")
BASKET = ST / "data/1019test/round-basket-foot-push-kinematic-v1"
MODEL = ST / "source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def slice_npz(path, output, start, end, full_length):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k][start:end] if z[k].ndim and len(z[k]) == full_length else z[k] for k in z.files}
    np.savez_compressed(output, **arrays)


def main():
    import mujoco
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve() / "retarget/candidates/03_04_05_joint_v0"
    out.mkdir(parents=True, exist_ok=True)
    order = HISTORY / "retarget/ground_support_27_end/sidecars/motion_order.json"
    prior_file = HISTORY / "retarget/ground_support_27_end/supported_motion.npz"
    prior = MotionFields.load(prior_file, order)
    n = 2450
    fields = {
        k: getattr(prior, k)[:n].copy()
        for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    }
    combined = replace(prior, **fields)
    combined.save(out / "body_prior.npz")
    selected_z = np.load(BASKET / "motion.npz")
    for name, value in fields.items():
        value[1600:2450] = selected_z[name]
    combined = replace(prior, **fields)
    combined.save(out / "motion.npz")
    shutil.copy2(order, out / "motion_order.json")
    human_path = HISTORY / "retarget/ground_support_27_end/sidecars/human_motion_world.npz"
    slice_npz(human_path, out / "human_motion_world.npz", 0, n, len(prior.joint_pos))
    timing_path = HISTORY / "retarget/body_prior_27_end/sidecars/timing.npz"
    timing = np.load(timing_path)
    time = 27 + np.arange(n) / 50
    stamp = timing["receive_timestamp_ns"][:n]
    source_stamp = timing["source_receive_timestamp_ns"]
    mi = np.searchsorted(source_stamp, stamp).clip(1, len(source_stamp) - 1)
    mi = np.where(abs(source_stamp[mi - 1] - stamp) <= abs(source_stamp[mi] - stamp), mi - 1, mi)
    sync = {
        "recording_time_s": time,
        "receive_timestamp_ns": stamp,
        "motive_frame_number": timing["source_motive_frame_number"][mi],
        "motive_time_error_s": (source_stamp[mi] - stamp) / 1e9,
    }
    video = ST / "data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization"
    camera_errors = {}
    for camera in ["head", "hand_left", "hand_right"]:
        ts = np.loadtxt(video / f"camera_{camera}_color_image_raw_compressed_frames.txt", skiprows=4)
        k = np.searchsorted(ts, stamp / 1e9).clip(1, len(ts) - 1)
        k = np.where(abs(ts[k - 1] - stamp / 1e9) <= abs(ts[k] - stamp / 1e9), k - 1, k)
        sync[f"{camera}_frame_index"] = k
        sync[f"{camera}_time_error_s"] = ts[k] - stamp / 1e9
        camera_errors[camera] = float(np.max(abs(sync[f"{camera}_time_error_s"])))
    np.savez_compressed(out / "source_sync.npz", **sync)
    tracks = []

    def track(name, p, q, **metadata):
        np.savez_compressed(
            out / f"{name}.npz",
            position_w=p.astype(np.float32),
            quaternion_xyzw=q.astype(np.float32),
            fps=np.asarray(50),
            source_time_s=time,
            **metadata,
        )
        tracks.append({"name": name + "_free", "display_name": name, "path": f"{name}.npz"})

    basket = np.load(BASKET / "round_laundry_basket_trajectory.npz")
    bp = np.vstack([np.repeat(basket["position_w"][:1], 1600, axis=0), basket["position_w"]])
    bq = np.vstack([np.repeat(basket["quaternion_wxyz"][:1], 1600, axis=0), basket["quaternion_wxyz"]])[:, [1, 2, 3, 0]]
    track("round_laundry_basket", bp, bq)
    assumptions = {}
    for name, hand, initial, grasp, release in [
        ("floor_strawberry", "left", [0.65, -0.05, 0.06], 31.2, 66.7),
        ("black_ribbon_hat", "right", [2.005000114, 0.819999993, 0.443100005], 39.0, 48.0),
    ]:
        idx = combined.body_names.index(hand + "_wrist_yaw_link")
        wp = combined.body_pos_w[:, idx]
        wr = Rotation.from_quat(combined.body_quat_w[:, idx][:, [1, 2, 3, 0]])
        g, r = round((grasp - 27) * 50), round((release - 27) * 50)
        relative_p = wr[g].inv().apply(np.asarray(initial) - wp[g])
        relative_r = wr[g].inv()
        p = np.repeat(np.asarray(initial)[None], n, axis=0)
        q = np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]]), n, axis=0)
        p[g : r + 1] = wp[g : r + 1] + wr[g : r + 1].apply(relative_p)
        q[g : r + 1] = (wr[g : r + 1] * relative_r).as_quat()
        p[r + 1 :] = p[r]
        q[r + 1 :] = q[r]
        if name == "floor_strawberry":
            # Release target uses the selected basket's inner floor; transition is
            # an explicit kinematic hypothesis, not gravity or measured support.
            target = bp[r] + [0, 0, 0.085]
            stop = r + 20
            u = np.linspace(0, 1, stop - r + 1)
            s = u * u * (3 - 2 * u)
            p[r : stop + 1] = p[r] + s[:, None] * (target - p[r])
            p[stop + 1 :] = bp[stop + 1 :] + np.array([0, 0, 0.085])
        track(name, p, q, trajectory_kind=np.asarray("unvalidated_fixed_attachment_hypothesis"))
        assumptions[name] = {
            "grasp_s": grasp,
            "release_s": release,
            "wrist_relative_translation_m": relative_p.tolist(),
            "wrist_relative_rotation_xyzw": relative_r.as_quat().tolist(),
            "source_initial_position_estimated": True,
            "geometry_reviewed": False,
            "support_reviewed": False,
            "hand_timing_implemented": False,
        }
    u = np.clip((time - 51.5) / 2.5, 0, 1)
    drawer = 0.19 * (1 - u * u * (3 - 2 * u))
    np.savez_compressed(out / "drawer_slide.npz", position=drawer, fps=np.asarray(50), source_time_s=time)
    save_json(
        out / "tracks_manifest.json",
        {"objects": tracks, "joints": [{"name": "drawer_slide", "path": "drawer_slide.npz"}]},
    )
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    limits = np.asarray([model.joint(name).range for name in combined.joint_names])
    violations = (combined.joint_pos < limits[:, 0] - 1e-5) | (combined.joint_pos > limits[:, 1] + 1e-5)
    joins = {}
    for field in fields:
        joins[field] = float(np.max(abs(fields[field][1600] - fields[field][1599])))
    review = {
        "status": "bootstrap_candidate_not_passed",
        "stage_1": "partial_source_review",
        "stage_2": "geometry_exported_live_camera_and_free_rest_pending",
        "stage_3_1": "reuse_prior_review_pending_join",
        "stage_3_2": "not_passed",
        "agent_reviewed": False,
        "user_reviewed_this_run": False,
        "physics_interaction_verified": False,
        "training_data_ready": False,
        "source_window_s": [27, 76],
        "frames": n,
        "fps": 50,
        "quaternion_convention": "motion_wxyz_tracks_xyzw",
        "camera_max_abs_sync_error_s": camera_errors,
        "joint_limit_violation_samples": int(violations.sum()),
        "joint_limit_violation_frames": np.flatnonzero(np.any(violations, axis=1)).tolist(),
        "max_joint_speed_rad_s": float(abs(combined.joint_vel).max()),
        "max_joint_acceleration_rad_s2": float(abs(np.gradient(combined.joint_vel, 0.02, axis=0)).max()),
        "protected_06_max_joint_difference_rad": float(abs(combined.joint_pos[1600:] - selected_z["joint_pos"]).max()),
        "join_59s_max_component_differences": joins,
        "missing_checks": [
            "complete continuous source visual review",
            "camera calibration and same-view scene check",
            "free-object rest",
            "grasp and hand closure envelope",
            "hat rack support and release",
            "continuous collision sweep",
            "drawer palm contact and path",
            "full numerical body support audit",
            "interactive stage3 review",
        ],
        "object_assumptions": assumptions,
    }
    save_json(out / "review.json", review)
    save_json(
        out / "manifest.json",
        {
            "inputs": {
                str(p): sha(p)
                for p in [
                    prior_file,
                    order,
                    human_path,
                    timing_path,
                    BASKET / "motion.npz",
                    BASKET / "round_laundry_basket_trajectory.npz",
                    MODEL,
                ]
            },
            "outputs": {p.name: sha(p) for p in out.glob("*.npz")},
            "code": str(Path(__file__).resolve()),
            "code_sha256": sha(__file__),
            "status": "candidate_only",
        },
    )
    print(out)


if __name__ == "__main__":
    main()
