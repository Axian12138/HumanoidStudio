"""Restore safe retreat branches and bring both hook pads close to the inner wall."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    parent = json.load(open(a.run / "inputs/parent.json"))
    order = parent["parent"]["order"]
    s = MotionFields.load(a.motion, order)
    old = MotionFields.load(parent["motion"], order)
    q = s.joint_pos.copy()
    time = 108 + np.arange(len(q)) / 50

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x**3 * (10 - 15 * x + 6 * x * x)

    for side, start, end in [("right", 124.0, 125.2), ("left", 120.8, 121.8)]:
        cols = [
            i
            for i, n in enumerate(s.joint_names)
            if n.startswith(side + "_") and any(x in n for x in ["shoulder", "elbow", "wrist"])
        ]
        w = smooth((time - start) / (end - start))
        q[:, cols] = (1 - w[:, None]) * q[:, cols] + w[:, None] * old.joint_pos[:, cols]
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(robot))
    d = mujoco.MjData(m)
    addr = np.array([m.joint(n).qposadr[0] for n in s.joint_names])
    cols = [
        i
        for i, n in enumerate(s.joint_names)
        if n.startswith("left_") and any(x in n for x in ["shoulder", "elbow", "wrist"])
    ]
    wb = m.body("left_wrist_yaw_link").id
    basket = np.load(Path(parent["parent"]["tracks"]) / "small_wicker_basket.npz")
    for k, t in enumerate(time):
        w = min(smooth((t - 118.2) / 0.5), smooth((120.9 - t) / 0.5))
        if w <= 0:
            continue
        d.qpos[:3] = s.body_pos_w[k, 0]
        d.qpos[3:7] = s.body_quat_w[k, 0]
        d.qpos[addr] = q[k]
        mujoco.mj_forward(m, d)
        rr = R.from_quat(basket["quaternion_xyzw"][k]).as_matrix()
        target = d.xpos[wb].copy() + rr @ np.array([0, -0.0033 * w, 0])
        rot = d.xmat[wb].reshape(3, 3).copy()
        initial = q[k, cols].astype(float)

        def fun(x, target=target, rot=rot, initial=initial):
            d.qpos[addr[cols]] = x
            mujoco.mj_forward(m, d)
            return np.r_[
                (d.xpos[wb] - target) * 1000,
                R.from_matrix(d.xmat[wb].reshape(3, 3) @ rot.T).as_rotvec() * 100,
                (x - initial),
            ]

        lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
        fit = least_squares(fun, initial, bounds=(lim[:, 0], lim[:, 1]), max_nfev=30)
        q[k, cols] = fit.x
    refresh_fk(replace(s, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "motion_order.json").write_bytes(Path(order).read_bytes())


if __name__ == "__main__":
    main()
