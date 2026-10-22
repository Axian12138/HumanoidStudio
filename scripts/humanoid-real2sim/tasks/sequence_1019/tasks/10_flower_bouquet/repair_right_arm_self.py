"""Track the existing right palm through a torso-clear redundant arm branch."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    for name in ["run", "motion", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--target-motion", type=Path)
    p.add_argument("--window", type=float, nargs=2, default=[111.5, 122.0])
    p.add_argument("--vmax", type=float, default=6.0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    seed = MotionFields.load(a.motion, order)
    target_motion = MotionFields.load(a.target_motion, order) if a.target_motion else seed
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(robot))
    data = mujoco.MjData(model)
    target_data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in seed.joint_names])
    cols = [
        i
        for i, n in enumerate(seed.joint_names)
        if n.startswith("right_") and any(x in n for x in ["shoulder", "elbow", "wrist"])
    ]
    aa = addr[cols]
    q = seed.joint_pos.copy()
    first, last = [round((time - 108) * 50) for time in a.window]
    previous = q[first, cols].astype(float)
    endpoint = q[last, cols].astype(float)
    lo = np.array([model.joint(seed.joint_names[i]).range[0] + 1e-7 for i in cols])
    hi = np.array([model.joint(seed.joint_names[i]).range[1] - 1e-7 for i in cols])
    palm = model.geom("right_hand_palm_collision").id
    pairs = [
        (model.geom("right_shoulder_yaw_collision").id, model.geom("torso_collision" + str(i)).id) for i in [1, 2, 3]
    ] + [(model.geom("right_elbow_yaw_collision").id, model.geom("torso_collision2").id)]
    records = []
    for k in range(first, last + 1):
        data.qpos[:3] = seed.body_pos_w[k, 0]
        data.qpos[3:7] = seed.body_quat_w[k, 0]
        data.qpos[addr] = seed.joint_pos[k]
        mujoco.mj_forward(model, data)
        target_data.qpos[:3] = target_motion.body_pos_w[k, 0]
        target_data.qpos[3:7] = target_motion.body_quat_w[k, 0]
        target_data.qpos[addr] = target_motion.joint_pos[k]
        mujoco.mj_forward(model, target_data)
        target = target_data.geom_xpos[palm].copy()
        targetR = target_data.geom_xmat[palm].reshape(3, 3).copy()
        baseline = q[k, cols].astype(float)

        def residual(x, target=target, targetR=targetR, baseline=baseline, previous=previous):
            data.qpos[aa] = x
            mujoco.mj_forward(model, data)
            gaps = np.array([mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None) for g1, g2 in pairs])
            return np.r_[
                40000 * (data.geom_xpos[palm] - target),
                4000 * (data.geom_xmat[palm].reshape(3, 3) - targetR).ravel(),
                40000 * np.maximum(0.003 - gaps, 0),
                10 * np.maximum(0.015 - gaps, 0),
                20 * (x - baseline),
                1 * (x - previous),
            ]

        lower = np.maximum.reduce([lo, previous - a.vmax / 50, endpoint - a.vmax * (last - k) / 50])
        upper = np.minimum.reduce([hi, previous + a.vmax / 50, endpoint + a.vmax * (last - k) / 50])
        if k in [first, last]:
            result = baseline
        else:
            eq = lower >= upper
            if np.any(eq):
                if np.max((lower - upper)[eq]) > 1e-6:
                    raise ValueError("infeasible speed bounds")
                mid = (lower[eq] + upper[eq]) / 2
                lower[eq] = mid - 1e-8
                upper[eq] = mid + 1e-8
            result = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper), max_nfev=100).x
        q[k, cols] = result
        previous = result.copy()
        residual(result)
        records.append(
            {
                "source_s": 108 + k / 50,
                "native_gaps_m": [float(mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None)) for g1, g2 in pairs],
                "palm_delta_m": float(np.linalg.norm(data.geom_xpos[palm] - target)),
                "palm_rotation_delta_deg": float(
                    np.rad2deg(Rotation.from_matrix(data.geom_xmat[palm].reshape(3, 3) @ targetR.T).magnitude())
                ),
                "joint_delta_rad": (result - baseline).tolist(),
            }
        )
        if k % 25 == 0:
            (a.out / "progress.json").write_text(json.dumps(records[-1]))
    refresh_fk(replace(seed, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "report.json").write_text(
        json.dumps(
            {
                "status": "unreviewed local candidate",
                "window_source_s": a.window,
                "velocity_bound_rad_s": a.vmax,
                "target_motion": str(a.target_motion or a.motion),
                "clearance_target_m": 0.003,
                "records": records,
            },
            indent=2,
        )
    )
    print(a.out)


if __name__ == "__main__":
    main()
