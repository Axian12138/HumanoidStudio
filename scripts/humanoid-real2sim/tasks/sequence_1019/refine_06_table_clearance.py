"""Candidate swing-foot detour before the protected basket contact begins."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--offset-y", type=float, default=0.06)
    parser.add_argument("--offset-z", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()
    model_path = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    motion = MotionFields.load(args.candidate / "motion.npz", args.candidate / "motion_order.json")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    names = [
        "left_" + name + "_joint" for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    cols = [motion.joint_names.index(name) for name in names]
    addresses = np.array([int(model.joint(name).qposadr[0]) for name in motion.joint_names])
    lower, upper = np.array([model.joint(name).range for name in names]).T
    body = model.body("left_ankle_roll_link").id
    q = motion.joint_pos.astype(float).copy()
    times = 27 + np.arange(len(q)) / motion.fps
    start, end = 60.14, 61.4
    rows = np.flatnonzero((times > start + 1e-7) & (times < end - 1e-7))
    human_path = Path(
        "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/retarget/ground_support_27_end/sidecars/human.npz"
    )
    with np.load(human_path) as human:
        contact = human["foot_contact"][rows]
    if contact[:, 0].any() or not contact[:, 1].all():
        raise ValueError("Detour must lie in left swing / right stance according to source contact labels")
    diagnostics = []
    for k in rows:
        data.qpos[:3] = motion.body_pos_w[k, 0]
        data.qpos[3:7] = motion.body_quat_w[k, 0]
        data.qpos[addresses] = motion.joint_pos[k]
        mujoco.mj_forward(model, data)
        origin = data.xpos[body].copy()
        target_rotation = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
        weight = np.sin(np.pi * (times[k] - start) / (end - start)) ** 2
        target_rotation = Rotation.from_rotvec([0, 0, args.yaw * weight]) * target_rotation
        target = origin + np.array([0, args.offset_y * weight, args.offset_z * weight])
        prior = motion.joint_pos[k, cols].astype(float)

        def residual(values, target=target, target_rotation=target_rotation, prior=prior):
            data.qpos[addresses[cols]] = values
            mujoco.mj_forward(model, data)
            rotation = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
            return np.r_[
                (data.xpos[body] - target) * 100,
                (rotation * target_rotation.inv()).as_rotvec() * 10,
                (values - prior) * 0.02,
            ]

        bent = prior.copy()
        bent[[0, 3, 4]] += np.array([-0.2, 0.4, -0.2]) * np.sqrt(weight)
        fits = [
            least_squares(
                residual,
                np.clip(seed, lower, upper),
                bounds=(lower, upper),
                max_nfev=100,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            for seed in (prior, bent)
        ]
        fit = min(fits, key=lambda item: item.cost)
        error = residual(fit.x)
        q[k, cols] = fit.x
        diagnostics.append(
            {
                "source_s": float(times[k]),
                "target_offset_y_m": float(args.offset_y * weight),
                "target_offset_z_m": float(args.offset_z * weight),
                "position_residual_m": float(np.linalg.norm(error[:3]) / 100),
                "orientation_residual_rad": float(np.linalg.norm(error[3:6]) / 10),
                "max_joint_delta_rad": float(abs(fit.x - prior).max()),
            }
        )
    args.output.mkdir(parents=True, exist_ok=False)
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
        shutil.copy2(args.candidate / name, args.output / name)
    result = refresh_fk(replace(motion, joint_pos=q), model_path)
    result.save(args.output / "motion.npz")
    report = {
        "status": "swing_detour_candidate_pending_collision_and_support_review",
        "source_candidate": str(args.candidate),
        "input_motion_sha256": sha256(args.candidate / "motion.npz"),
        "human_contact_sha256": sha256(human_path),
        "window_s": [start, end],
        "changed_joints": names,
        "target_offset_y_m": args.offset_y,
        "target_offset_z_m": args.offset_z,
        "target_yaw_rad": args.yaw,
        "scope": "Left swing leg only; root, right stance, object tracks and all states from61.4s retained. Original basket contact begins61.64s. Does not yet prove new obstacle clearance.",
        "fit": diagnostics,
    }
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print("max foot target error", max(row["position_residual_m"] for row in diagnostics))


if __name__ == "__main__":
    main()
