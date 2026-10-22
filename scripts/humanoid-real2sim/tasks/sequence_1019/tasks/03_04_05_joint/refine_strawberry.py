"""Continuous arm-window candidate with an explicit large-object hand schedule.

The geometric hand target is a design hypothesis and remains pending collision
and visual review. Neither closed fingers nor fixed attachment imply grasp force.
"""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output-version", default="03_04_05_joint_v1")
    p.add_argument("--geometry", type=Path)
    p.add_argument("--contact-weight", type=float, default=80)
    p.add_argument("--orientation-weight", type=float, default=0)
    a = p.parse_args()
    src = a.run / "retarget/candidates/03_04_05_joint_v0"
    out = a.run / "retarget/candidates" / a.output_version
    out.mkdir(parents=True, exist_ok=True)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(model)
    arm = [
        i
        for i, n in enumerate(m.joint_names)
        if n.startswith("left_") and any(k in n for k in ["shoulder", "elbow", "wrist"])
    ]
    addr = np.array([model.joint(n).qposadr[0] for n in m.joint_names])
    w = model.body("left_wrist_yaw_link").id
    q = m.joint_pos.astype(float).copy()
    time = 27 + np.arange(len(q)) / 50
    chosen = json.loads(a.geometry.read_text())["records"][0] if a.geometry else None
    center_local = np.array(chosen["center_local_m"] if chosen else [0.14, -0.045, 0.0])
    thumb_angle = chosen["thumb_1_2_rad"] if chosen else 0.45
    finger_angle = chosen["index_middle_0_1_rad"] if chosen else -0.65
    floor = np.array([0.65, -0.05, 0.0367203])
    basket = np.load(src / "round_laundry_basket.npz")["position_w"]
    reports = []
    for begin, contact, end, target in [(29.0, 31.2, 33.0, floor), (65.0, 66.7, 68.2, basket[1985] + [0, 0, 0.49])]:
        rows = np.flatnonzero((time >= begin) & (time <= end))
        knots = np.array([begin, contact - 0.45, contact, contact + 0.35, end])
        reference = q[rows][:, arm].copy()
        t = time[rows]
        contact_weight = np.exp(-(((t - contact) / 0.23) ** 4))
        lower = np.array([model.joint(m.joint_names[c]).range[0] for c in arm])
        upper = np.array([model.joint(m.joint_names[c]).range[1] for c in arm])

        def unpack(x, reference=reference, knots=knots, t=t):
            return reference + CubicSpline(
                knots, np.vstack([np.zeros(7), x.reshape(3, 7), np.zeros(7)]), bc_type="clamped"
            )(t)

        contact_frame = round((contact - 27) * 50)
        desired_rotation = Rotation.from_quat(
            m.body_quat_w[contact_frame, m.body_names.index("left_wrist_yaw_link")][[1, 2, 3, 0]]
        ).as_matrix()

        def points(values, rows=rows, orientation=False, desired_rotation=desired_rotation):
            result = []
            for k, v in zip(rows, values, strict=True):
                d.qpos[:3] = m.body_pos_w[k, 0]
                d.qpos[3:7] = m.body_quat_w[k, 0]
                d.qpos[addr] = q[k]
                d.qpos[addr[arm]] = v
                mujoco.mj_forward(model, d)
                rotation = d.xmat[w].reshape(3, 3)
                result.append(
                    Rotation.from_matrix(desired_rotation.T @ rotation).as_rotvec()
                    if orientation
                    else d.xpos[w] + rotation @ center_local
                )
            return np.array(result)

        def residual(x, target=target, contact_weight=contact_weight, reference=reference, lower=lower, upper=upper):
            values = unpack(x)
            return np.concatenate(
                [
                    ((points(values) - target) * contact_weight[:, None] * a.contact_weight).ravel(),
                    (points(values, orientation=True) * contact_weight[:, None] * a.orientation_weight).ravel(),
                    ((values - reference) * 0.5).ravel(),
                    (np.diff(values - reference, n=2, axis=0) * 70).ravel(),
                    (np.maximum(lower - values, 0) * 100).ravel(),
                    (np.maximum(values - upper, 0) * 100).ravel(),
                ]
            )

        before = points(reference)
        fit = least_squares(residual, np.zeros(21), max_nfev=45, ftol=2e-5)
        values = unpack(fit.x)
        q[np.ix_(rows, arm)] = values
        after = points(values)
        j = np.argmin(abs(t - contact))
        reports.append(
            {
                "window_s": [begin, end],
                "contact_s": contact,
                "target_m": target.tolist(),
                "before_error_m": float(np.linalg.norm(before[j] - target)),
                "after_error_m": float(np.linalg.norm(after[j] - target)),
                "success": bool(fit.success),
                "nfev": fit.nfev,
                "contact_orientation_error_deg": float(
                    np.linalg.norm(points(values, orientation=True)[j]) * 180 / np.pi
                ),
                "max_arm_change_rad": float(abs(values - reference).max()),
            }
        )

    def blend(t0, t1):
        u = np.clip((time - t0) / (t1 - t0), 0, 1)
        return u * u * (3 - 2 * u)

    # Open before the floor approach, partially close around the large fruit,
    # release before withdrawal. Explicit design values, not optimized fingers.
    hand = [i for i, n in enumerate(m.joint_names) if n.startswith("left_hand")]
    openness = blend(28.5, 29.2)
    holding = blend(30.8, 31.2)
    released = blend(66.5, 67.0)
    for i in hand:
        n = m.joint_names[i]
        hold = 0.0 if "thumb_0" in n else (thumb_angle if "thumb" in n else finger_angle)
        q[:, i] = q[:, i] * (1 - openness) + hold * holding * (1 - released)
        # Return to original neutral hand only after release and withdrawal.
        reset = blend(68.0, 69.0)
        q[:, i] = q[:, i] * (1 - reset) + m.joint_pos[:, i] * reset
    candidate = refresh_fk(replace(m, joint_pos=q), MODEL)
    candidate.save(out / "motion.npz")
    for file in src.iterdir():
        if file.is_file() and file.name not in [
            "motion.npz",
            "floor_strawberry.npz",
            "review.json",
            "manifest.json",
            "collision_diagnostic.json",
        ]:
            shutil.copy2(file, out / file.name)
    idx = m.body_names.index("left_wrist_yaw_link")
    wp = candidate.body_pos_w[:, idx]
    wr = Rotation.from_quat(candidate.body_quat_w[:, idx][:, [1, 2, 3, 0]])
    g, r = 210, 1985
    rot0 = wr[g].inv()
    pos = np.repeat(floor[None], len(q), axis=0)
    quat = np.repeat([[0.0, 0.0, 0.0, 1.0]], len(q), axis=0)
    pos[g : r + 1] = wp[g : r + 1] + wr[g : r + 1].apply(center_local)
    quat[g : r + 1] = (wr[g : r + 1] * rot0).as_quat()
    target = basket[r] + [0, 0, 0.073]
    stop = r + 20
    u = np.linspace(0, 1, 21)
    s = u * u * (3 - 2 * u)
    pos[r : stop + 1] = pos[r] + s[:, None] * (target - pos[r])
    pos[stop + 1 :] = basket[stop + 1 :] + np.array([0, 0, 0.073])
    quat[r + 1 :] = quat[r]
    np.savez_compressed(
        out / "floor_strawberry.npz", position_w=pos, quaternion_xyzw=quat, fps=np.array(50), source_time_s=time
    )
    review = json.loads((src / "review.json").read_text())
    review.update(
        status="partial_strawberry_refinement_not_passed",
        arm_fit=reports,
        hand_schedule={
            "open_s": [28.5, 29.2],
            "close_s": [30.8, 31.2],
            "release_s": [66.5, 67.0],
            "reset_s": [68, 69],
            "held_angles_rad": {"thumb0": 0, "thumb1_2": thumb_angle, "index_middle_0_1": finger_angle},
        },
        attachment_local_center_m=center_local.tolist(),
        grasp_geometry_verified=False,
    )
    review["protected_06_nonleft_arm_hand_joint_difference_rad"] = float(
        np.max(
            abs(
                q[1600:][:, [i for i in range(43) if i not in arm + hand]]
                - m.joint_pos[1600:][:, [i for i in range(43) if i not in arm + hand]]
            )
        )
    )
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(reports)


if __name__ == "__main__":
    main()
