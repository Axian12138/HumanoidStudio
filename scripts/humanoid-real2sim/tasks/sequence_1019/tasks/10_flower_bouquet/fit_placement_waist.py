"""Reproduce a task-specific flower HOI refinement step; see the run ledger for inputs."""


def main():

    import argparse
    import json
    from dataclasses import replace
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    r = parser.parse_args().run
    order = r / "retarget/web_hook_v9/motion_order.json"
    s = MotionFields.load(r / "retarget/web_hook_v9/motion.npz", order)
    v8 = MotionFields.load(r / "retarget/web_hook_v8/motion.npz", order)
    q = s.joint_pos.copy()
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(robot))
    d = mujoco.MjData(m)
    ad = np.array([m.joint(n).qposadr[0] for n in s.joint_names])
    arm = [
        s.joint_names.index("right_" + n + "_joint")
        for n in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    waist = s.joint_names.index("waist_pitch_joint")
    cols = [*arm, waist]
    wb = m.body("right_wrist_yaw_link").id
    des = json.load(open(r / "review/grip_design.json"))[0]
    center = np.array(des["center"])
    parent = json.load(open(r / "inputs/parent.json"))
    tr = np.load(Path(parent["parent"]["tracks"]) / "flower_bouquet.npz")
    lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
    lim[5] = [-0.46, 0.46]

    def sm(x):
        x = np.clip(x, 0, 1)
        return x**3 * (10 - 15 * x + 6 * x * x)

    rows = []
    for k in range(700, 761):
        t = 108 + k / 50
        w = sm((t - 122) / 0.5)
        d.qpos[:3] = s.body_pos_w[k, 0]
        d.qpos[3:7] = s.body_quat_w[k, 0]
        d.qpos[ad] = s.joint_pos[k]
        rr = R.from_quat(tr["quaternion_xyzw"][k]).as_matrix()
        target = tr["position_w"][k] + rr @ np.array([0, 0, 0.25])
        axis = rr[:, 2]
        initial = q[k, cols].astype(float)
        low = lim[:, 0].copy()
        high = lim[:, 1].copy()
        low[-1] = initial[-1] - 0.15 * w - 1e-8
        high[-1] = initial[-1] + 0.15 * w + 1e-8

        def fun(x, target=target, axis=axis, initial=initial):
            d.qpos[ad[cols]] = x
            mujoco.mj_forward(m, d)
            wr = d.xmat[wb].reshape(3, 3)
            return np.r_[
                (d.xpos[wb] + wr @ center - target) * 10000,
                (wr[:, 2] - axis) * 1000,
                (x[:-1] - initial[:-1]) * 2,
                (x[-1] - initial[-1]) * 25,
            ]

        fit = least_squares(fun, np.clip(initial, low + 1e-9, high - 1e-9), bounds=(low, high), max_nfev=100)
        q[k, cols] = fit.x
        rows.append([t, float(np.linalg.norm(fun(fit.x)[:3]) / 10000), float(fit.x[-1] - initial[-1])])
    # Return to the previously collision-checked retreat after the supported placement.
    delta = q[760, cols] - v8.joint_pos[760, cols]
    for k in range(761, 801):
        q[k, cols] = v8.joint_pos[k, cols] + (1 - sm((108 + k / 50 - 123.2) / 0.8)) * delta
    fcols = [s.joint_names.index("right_hand_" + n + "_joint") for n in des["names"]]
    grip = np.array(des["hand"])
    grip[5] += 0.025
    for k in range(750, 801):
        w = sm((108 + k / 50 - 123.2) / 0.8)
        q[k, fcols] = (1 - w) * grip + w * v8.joint_pos[800, fcols]
    out = r / "retarget/waist_v11"
    out.mkdir()
    refresh_fk(replace(s, joint_pos=q), robot).save(out / "motion.npz")
    (out / "motion_order.json").write_bytes(order.read_bytes())
    (r / "review/waist_v11.json").write_text(json.dumps(rows, indent=2))
    print("peak errors/waist", np.max(np.array(rows)[:, 1:], axis=0))


if __name__ == "__main__":
    main()
