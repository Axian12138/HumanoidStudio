"""Locally clear the released left upper arm from the torso without moving the grasping arm."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    for name in ["run", "motion", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    seed = MotionFields.load(a.motion, order)
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(robot))
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in seed.joint_names])
    cols = [
        i
        for i, n in enumerate(seed.joint_names)
        if n.startswith("left_") and any(x in n for x in ["shoulder", "elbow", "wrist"])
    ]
    aa = addr[cols]
    q = seed.joint_pos.copy()
    first, last = 635, 665
    previous = q[first, cols].astype(float)
    endpoint = q[last, cols].astype(float)
    lo = np.array([model.joint(seed.joint_names[i]).range[0] + 1e-7 for i in cols])
    hi = np.array([model.joint(seed.joint_names[i]).range[1] - 1e-7 for i in cols])
    g1 = model.geom("left_shoulder_yaw_collision").id
    g2 = model.geom("torso_collision1").id
    palm = model.geom("left_hand_palm_collision").id
    records = []
    for k in range(first, last + 1):
        data.qpos[:3] = seed.body_pos_w[k, 0]
        data.qpos[3:7] = seed.body_quat_w[k, 0]
        data.qpos[addr] = seed.joint_pos[k]
        mujoco.mj_forward(model, data)
        target = data.geom_xpos[palm].copy()
        targetR = data.geom_xmat[palm].reshape(3, 3).copy()
        baseline = q[k, cols].astype(float)
        before = float(mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None))

        def residual(x, target=target, targetR=targetR, baseline=baseline):
            data.qpos[aa] = x
            mujoco.mj_forward(model, data)
            gap = mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None)
            return np.r_[
                100 * (data.geom_xpos[palm] - target),
                10 * (data.geom_xmat[palm].reshape(3, 3) - targetR).ravel(),
                20000 * max(0.003 - gap, 0),
                10 * (x - baseline),
            ]

        lower = np.maximum.reduce([lo, baseline - 0.3, previous - 6 / 50, endpoint - 6 * (last - k) / 50])
        upper = np.minimum.reduce([hi, baseline + 0.3, previous + 6 / 50, endpoint + 6 * (last - k) / 50])
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
                "native_gap_before_m": before,
                "native_gap_after_m": float(mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None)),
                "palm_delta_m": (data.geom_xpos[palm] - target).tolist(),
                "joint_delta_rad": (result - baseline).tolist(),
            }
        )
    refresh_fk(replace(seed, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "report.json").write_text(
        json.dumps(
            {
                "status": "unreviewed local candidate",
                "window_source_s": [120.7, 121.3],
                "clearance_target_m": 0.003,
                "records": records,
            },
            indent=2,
        )
    )
    print(a.out)


if __name__ == "__main__":
    main()
