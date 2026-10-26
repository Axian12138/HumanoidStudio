"""Compare right arm solutions while retaining the web grip point and cylinder axis."""


def main():

    import argparse
    import json
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.optimize import least_squares

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    r = parser.parse_args().run
    p = json.load(open(r / "inputs/parent.json"))
    s = MotionFields.load(p["motion"], p["parent"]["order"])
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    ad = np.array([m.joint(n).qposadr[0] for n in s.joint_names])
    cols = [
        s.joint_names.index("right_" + n + "_joint")
        for n in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    wb = m.body("right_wrist_yaw_link").id
    el = m.body("right_elbow_link").id
    sh = m.body("right_shoulder_roll_link").id
    lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
    lim[5] = [-0.47, 0.47]
    lim[6] = [-0.65, 0.65]
    out = []
    for t in [113, 114, 116, 118, 120]:
        k = round((t - 108) * 50)
        d.qpos[:3] = s.body_pos_w[k, 0]
        d.qpos[3:7] = s.body_quat_w[k, 0]
        d.qpos[ad] = s.joint_pos[k]
        mujoco.mj_forward(m, d)
        pos = d.xpos[wb].copy()
        rot = d.xmat[wb].reshape(3, 3).copy()
        ini = s.joint_pos[k, cols]
        center = np.array([0.12437059, 0.05823027, 0])
        target = pos + rot @ center
        axis = rot[:, 2]
        for branch in ["negative"]:
            bounds = lim.copy()
            bounds[3] = [-1.0472, -0.001] if branch == "negative" else [0.001, 2.0944]

            def fun(x, center=center, target=target, axis=axis, ini=ini):
                d.qpos[ad[cols]] = x
                mujoco.mj_forward(m, d)
                return np.r_[
                    (d.xpos[wb] + d.xmat[wb].reshape(3, 3) @ center - target) * 2000,
                    (d.xmat[wb].reshape(3, 3)[:, 2] - axis) * 300,
                    x[4:] * 2,
                    (x - ini) * 0.005,
                ]

            best = None
            for j in range(12):
                x = np.clip(ini + np.random.default_rng(j).normal(0, 1, 7), bounds[:, 0] + 1e-7, bounds[:, 1] - 1e-7)
                z = least_squares(fun, x, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=120)
                if best is None or np.linalg.norm(z.fun) < np.linalg.norm(best.fun):
                    best = z
            v = fun(best.x)
            row = {
                "t": t,
                "branch": branch,
                "pos": float(np.linalg.norm(v[:3]) / 1000),
                "rot": float(np.linalg.norm(v[3:6]) / 150),
                "q": best.x.tolist(),
                "elbow": (d.xpos[el] - d.xpos[sh]).tolist(),
            }
            out.append(row)
            print(row, flush=True)
    (r / "review/right_axis_negative.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
