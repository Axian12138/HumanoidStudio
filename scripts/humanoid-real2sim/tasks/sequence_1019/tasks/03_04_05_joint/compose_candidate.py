"""Apply reviewed fixed strawberry geometry and a measured hip-clearance correction.

Preserves every leg joint and root frame, and writes numerical protection checks.
The result remains a candidate until approach, rack placement and drawer pass.
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)
ARM = tuple(
    f"left_{part}_joint"
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
)


def main():
    import mujoco
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    src = a.run / "retarget/candidates/03_04_05_joint_v5"
    out = a.run / "retarget/candidates/03_04_05_joint_v6"
    out.mkdir(parents=True, exist_ok=True)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    prior = MotionFields.load(a.run / "retarget/candidates/03_04_05_joint_v0/motion.npz", src / "motion_order.json")
    selected = tuple(
        n for n in motion.joint_names if n.startswith("left_") and any(k in n for k in ["shoulder", "elbow", "wrist"])
    )
    assert set(selected) == set(ARM) and len(selected) == 7
    time = 27 + np.arange(len(motion.joint_pos)) / 50
    q = motion.joint_pos.astype(float).copy()

    def blend(start, end):
        u = np.clip((time - start) / (end - start), 0, 1)
        return u * u * (3 - 2 * u)

    # Signed mesh queries at source35.5s: +0.10rad gives16.97mm clearance;
    # -0.10rad worsens penetration to15.61mm. Smooth finite window only.
    shoulder_delta = 0.1 * blend(34, 35) * (1 - blend(40.5, 41.5))
    q[:, motion.joint_names.index("left_shoulder_roll_joint")] += shoulder_delta
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "floor_strawberry.npz",
            "review.json",
            "manifest.json",
            "collision_diagnostic.json",
        ]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    geometry = json.loads((a.run / "review/stage3/03_04_05_joint_v2_floor_strawberry_grasp_geometry.json").read_text())[
        "records"
    ][0]
    center = np.array(geometry["center_local_m"])
    g, r = 210, 1985
    wi = motion.body_names.index("left_wrist_yaw_link")
    wp = result.body_pos_w[:, wi]
    wr = Rotation.from_quat(result.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    initial = wp[g] + wr[g].apply(center)
    initial[2] = 0.0367203
    center = wr[g].inv().apply(initial - wp[g])
    pos = np.repeat(initial[None], len(time), axis=0)
    quat = np.repeat([[0.0, 0.0, 0.0, 1.0]], len(time), axis=0)
    pos[g : r + 1] = wp[g : r + 1] + wr[g : r + 1].apply(center)
    quat[g : r + 1] = (wr[g : r + 1] * wr[g].inv()).as_quat()
    basket = np.load(src / "round_laundry_basket.npz")["position_w"]
    target = basket[r] + np.array([0, 0, 0.073])
    stop = r + 20
    u = np.linspace(0, 1, 21)
    s = u * u * (3 - 2 * u)
    pos[r : stop + 1] = pos[r] + s[:, None] * (target - pos[r])
    pos[stop + 1 :] = basket[stop + 1 :] + np.array([0, 0, 0.073])
    quat[r + 1 :] = quat[r]
    np.savez_compressed(
        out / "floor_strawberry.npz", position_w=pos, quaternion_xyzw=quat, fps=np.array(50), source_time_s=time
    )
    leg = [i for i, n in enumerate(result.joint_names) if any(s in n for s in ["hip_", "knee", "ankle"])]
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    limits = np.array([model.joint(n).range for n in result.joint_names])
    bad = (q < limits[:, 0] - 1e-5) | (q > limits[:, 1] + 1e-5)
    review = json.loads((src / "review.json").read_text())
    review.update(
        status="geometry_updated_candidate_not_passed",
        attachment_local_center_m=center.tolist(),
        floor_initial_position_m=initial.tolist(),
        floor_xy_change_from_estimate_m=(initial[:2] - [0.65, -0.05]).tolist(),
        held_hip_correction={
            "joint": "left_shoulder_roll_joint",
            "window_s": [34, 41.5],
            "max_rad": 0.1,
            "source_contact_probe": {"at_s": 35.5, "old_gap_m": -0.011387, "after_positive_point1_gap_m": 0.016974},
        },
        protection={
            "leg_joint_names": [result.joint_names[i] for i in leg],
            "max_leg_angle_change_from_joint_v0_rad": float(
                abs(result.joint_pos[:, leg] - prior.joint_pos[:, leg]).max()
            ),
            "max_root_position_change_m": float(abs(result.body_pos_w[:, 0] - prior.body_pos_w[:, 0]).max()),
            "max_root_quaternion_change": float(abs(result.body_quat_w[:, 0] - prior.body_quat_w[:, 0]).max()),
            "max_right_arm_change_after_59s_rad": float(
                abs(
                    result.joint_pos[1600:][:, [i for i, n in enumerate(result.joint_names) if n.startswith("right_")]]
                    - prior.joint_pos[1600:][:, [i for i, n in enumerate(result.joint_names) if n.startswith("right_")]]
                ).max()
            ),
        },
        joint_limit_violation_samples=int(bad.sum()),
        max_joint_speed_rad_s=float(abs(result.joint_vel).max()),
        max_joint_acceleration_rad_s2=float(abs(np.gradient(result.joint_vel, 0.02, axis=0)).max()),
        physics_interaction_verified=False,
    )
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob("*.npz")}
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "status": "candidate_not_passed",
                "source_motion": str(src / "motion.npz"),
                "source_sha256": hashlib.sha256((src / "motion.npz").read_bytes()).hexdigest(),
                "outputs": hashes,
                "code": str(Path(__file__).resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    print(review["protection"])


if __name__ == "__main__":
    main()
