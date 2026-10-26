"""Fit the actual web center and cylinder axis, retaining the continuous arm branch."""

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
    cols = [
        s.joint_names.index("right_" + n + "_joint")
        for n in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    wb = m.body("right_wrist_yaw_link").id
    center = np.array(json.load(open(a.run / "review/grip_design.json"))[0]["center"])
    tr = np.load(Path(parent["parent"]["tracks"]) / "flower_bouquet.npz")
    lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
    lim[5] = [-0.46, 0.46]
    rows = []

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * x * (10 - 15 * x + 6 * x * x)

    for k in range(250, 776):
        t = 108 + k / 50
        w = min(smooth((t - 113) / 0.2), smooth((123.5 - t) / 0.5))
        d.qpos[:3] = s.body_pos_w[k, 0]
        d.qpos[3:7] = s.body_quat_w[k, 0]
        d.qpos[ad] = q[k]
        mujoco.mj_forward(m, d)
        wr = d.xmat[wb].reshape(3, 3)
        rr = R.from_quat(tr["quaternion_xyzw"][k]).as_matrix()
        target = (1 - w) * (d.xpos[wb] + wr @ center) + w * (tr["position_w"][k] + rr @ np.array([0, 0, 0.25]))
        axis = (1 - w) * wr[:, 2] + w * rr[:, 2]
        axis /= np.linalg.norm(axis)
        initial = q[k, cols].astype(float)

        def fun(x, initial=initial, target=target, axis=axis):
            d.qpos[ad[cols]] = x
            mujoco.mj_forward(m, d)
            wr = d.xmat[wb].reshape(3, 3)
            return np.r_[(d.xpos[wb] + wr @ center - target) * 10000, (wr[:, 2] - axis) * 1000, (x - initial) * 2]

        fit = least_squares(
            fun, np.clip(initial, lim[:, 0] + 1e-8, lim[:, 1] - 1e-8), bounds=(lim[:, 0], lim[:, 1]), max_nfev=70
        )
        q[k, cols] = fit.x
        q[k, s.joint_names.index("right_hand_middle_0_joint")] += 0.025 * w
        rows.append([t, float(np.linalg.norm(fun(fit.x)[:3]) / 10000)])
    refresh_fk(replace(s, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "motion_order.json").write_bytes(Path(order).read_bytes())
    (a.out / "axis_report.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
