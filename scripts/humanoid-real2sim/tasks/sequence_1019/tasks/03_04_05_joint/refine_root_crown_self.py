"""Small torso translation with both feet and grasp wrists preserved in world space."""

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
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--base", default="03_04_05_joint_v64")
    p.add_argument("--output", default="03_04_05_joint_v70")
    p.add_argument("--arm-only", action="store_true")
    p.add_argument("--seed-candidate")
    p.add_argument("--smooth-return", action="store_true")
    a = p.parse_args()
    r = a.run
    c = r / "retarget/candidates" / a.base
    out = r / "retarget/candidates" / a.output
    out.mkdir(exist_ok=False)
    motion = MotionFields.load(c / "motion.npz", c / "motion_order.json")
    m = mujoco.MjModel.from_xml_path(str(r / "scene/03_04_05/robot_native_shoulder_hulls_diagnostic.xml"))
    d = mujoco.MjData(m)
    addr = np.array([m.joint(n).qposadr[0] for n in motion.joint_names])
    names = [
        side + "_" + j + "_joint"
        for side in ["left", "right"]
        for j in [
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
    ]
    if a.arm_only:
        names = [
            "right_" + j + "_joint"
            for j in [
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            ]
        ]
    cols = [motion.joint_names.index(n) for n in names]
    assert len(cols) == (7 if a.arm_only else 26)
    nr = 0 if a.arm_only else 3
    lo = np.r_[[-0.03] * nr, [m.joint(n).range[0] for n in names]]
    hi = np.r_[[0.03] * nr, [m.joint(n).range[1] for n in names]]
    bodies = [
        m.body(side + "_" + part).id for side in ["left", "right"] for part in ["ankle_roll_link", "wrist_yaw_link"]
    ]
    bodycols = [motion.body_names.index(m.body(b).name) for b in bodies]
    g1 = m.geom("native_hull_torso_link").id
    g2 = m.geom("native_hull_right_shoulder_yaw_link").id
    q = motion.joint_pos.copy().astype(float)
    bp = motion.body_pos_w.copy().astype(float)
    records = []
    seed = (
        np.load(r / "retarget/candidates" / a.seed_candidate / "motion.npz")["joint_pos"] if a.seed_candidate else None
    )
    previous_delta = np.zeros(len(cols))
    end_time = 40.9 if a.smooth_return else 40.8
    for k in range(round((40.2 - 27) * 50), round((end_time - 27) * 50) + 1):
        prior = q[k, cols].copy()
        root = motion.body_pos_w[k, 0]
        targetp = motion.body_pos_w[k, bodycols]
        targetr = Rotation.from_quat(motion.body_quat_w[k, bodycols][:, [1, 2, 3, 0]])
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = q[k]
        desired_delta = previous_delta
        if a.smooth_return and 27 + k / 50 >= 40.54:
            anchor = round((40.54 - 27) * 50)
            u = np.clip((27 + k / 50 - 40.54) / (end_time - 40.54), 0, 1)
            desired_delta = (seed[anchor, cols] - motion.joint_pos[anchor, cols]) * (
                1 - (10 * u**3 - 15 * u**4 + 6 * u**5)
            )

        def residual(x, root=root, targetp=targetp, targetr=targetr, prior=prior, desired_delta=desired_delta):
            d.qpos[:3] = root + (x[:3] if nr else 0)
            d.qpos[addr[cols]] = x[nr:]
            mujoco.mj_forward(m, d)
            rot = Rotation.from_matrix(d.xmat[bodies].reshape(-1, 3, 3))
            gap = mujoco.mj_geomDistance(m, d, g1, g2, 0.1, None)
            return np.r_[
                ((d.xpos[bodies] - targetp) * 20000).ravel(),
                ((rot * targetr.inv()).as_rotvec() * 3000).ravel(),
                max(0.001 - gap, 0) * 5000,
                x[:nr] * 80,
                (x[nr:] - prior) * 0.1,
                (x[nr:] - prior - desired_delta) * (10 if a.smooth_return else (1 if a.arm_only else 0)),
            ]

        initial = np.r_[np.zeros(nr), seed[k, cols] if seed is not None else prior]
        fit = least_squares(
            residual, np.clip(initial, lo + 1e-8, hi - 1e-8), bounds=(lo, hi), max_nfev=160, ftol=1e-9, diff_step=1e-4
        )
        residual(fit.x)
        q[k, cols] = fit.x[nr:]
        previous_delta = fit.x[nr:] - prior
        root_delta = fit.x[:3] if nr else np.zeros(3)
        bp[k, 0] = root + root_delta
        rec = {
            "source_s": 27 + k / 50,
            "root_delta_m": root_delta.tolist(),
            "max_world_position_error_m": float(np.linalg.norm(d.xpos[bodies] - targetp, axis=1).max()),
            "native_hull_gap_m": float(mujoco.mj_geomDistance(m, d, g1, g2, 0.1, None)),
            "q_delta_max": float(abs(fit.x[nr:] - prior).max()),
            "success": bool(fit.success),
        }
        records.append(rec)
        print(rec, flush=True)
    result = refresh_fk(replace(motion, joint_pos=q, body_pos_w=bp), MODEL)
    for f in c.iterdir():
        if (
            f.is_file()
            and f.suffix in [".npz", ".json", ".xml"]
            and f.name not in ["motion.npz", "review.json", "manifest.json"]
            and not f.name.startswith(("collision_", "grasp_"))
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    (out / "review.json").write_text(
        json.dumps(
            {
                "status": "root_crown_self_candidate_pending_support_speed_room_native_audit",
                "base": str(c),
                "window_s": [40.2, end_time],
                "records": records,
                "finger_object_tracks_unchanged": True,
                "arm_only": a.arm_only,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
