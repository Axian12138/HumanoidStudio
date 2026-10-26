"""Lift the two hook fingers over the rim and clear the web before full hand opening."""

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
    q = s.joint_pos.copy()
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(robot))
    d = mujoco.MjData(m)
    ad = np.array([m.joint(n).qposadr[0] for n in s.joint_names])

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x**3 * (10 - 15 * x + 6 * x * x)

    for side, begin, peak, end, height in [("left", 116.5, 117.5, 118.7, 0.13), ("left", 120.4, 120.85, 121.4, 0.045)]:
        cols = [
            s.joint_names.index(side + "_" + n + "_joint")
            for n in [
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            ]
        ]
        wb = m.body(side + "_wrist_yaw_link").id
        lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
        for k in range(round((begin - 108) * 50), round((end - 108) * 50) + 1):
            t = 108 + k / 50
            w = min(smooth((t - begin) / (peak - begin)), smooth((end - t) / (end - peak)))
            d.qpos[:3] = s.body_pos_w[k, 0]
            d.qpos[3:7] = s.body_quat_w[k, 0]
            d.qpos[ad] = q[k]
            mujoco.mj_forward(m, d)
            rot = d.xmat[wb].reshape(3, 3).copy()
            offset = np.array([0, 0, height * w]) if side == "left" else -rot[:, 1] * height * w
            target = d.xpos[wb].copy() + offset
            initial = q[k, cols].astype(float)

            def fun(x, initial=initial, rot=rot, target=target, cols=cols, wb=wb):
                d.qpos[ad[cols]] = x
                mujoco.mj_forward(m, d)
                return np.r_[
                    (d.xpos[wb] - target) * 1000,
                    R.from_matrix(d.xmat[wb].reshape(3, 3) @ rot.T).as_rotvec() * 100,
                    (x - initial) * 0.5,
                ]

            fit = least_squares(fun, initial, bounds=(lim[:, 0], lim[:, 1]), max_nfev=70)
            q[k, cols] = fit.x
    refresh_fk(replace(s, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "motion_order.json").write_bytes(Path(order).read_bytes())


if __name__ == "__main__":
    main()
