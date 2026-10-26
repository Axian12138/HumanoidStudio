"""Raise the held flower and wrist together before crossing the basket rim."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np


def smooth(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 - 15 * x + 6 * x * x)


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--height", type=float, default=0.06)
    a = p.parse_args()
    info = json.loads((a.run / "inputs/parent.json").read_text())
    s = MotionFields.load(info["motion"], info["parent"]["order"])
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
    lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
    lim[5] = [-0.47, 0.47]
    lim[6] = [-0.65, 0.65]
    times = 108 + np.arange(len(q)) / 50
    dz = a.height * np.minimum(smooth((times - 120.2) / 0.65), smooth((122.65 - times) / 1.1))
    out = a.run / "retarget/lift_v3"
    out.mkdir(parents=True, exist_ok=False)
    tracks = a.run / "tracks_lift"
    tracks.mkdir(exist_ok=True)
    for f in Path(info["original_tracks"]).iterdir():
        if f.is_file():
            shutil.copy2(f, tracks / f.name)
    flower = dict(np.load(tracks / "flower_bouquet.npz"))
    flower["position_w"] = flower["position_w"].copy()
    flower["position_w"][:, 2] += dz
    np.savez(tracks / "flower_bouquet.npz", **flower)
    info["parent"]["tracks"] = str(tracks)
    (a.run / "inputs/parent.json").write_text(json.dumps(info, indent=2))
    previous = q[610, cols].astype(float)
    rows = []
    for k in range(610, 734):
        d.qpos[:3] = s.body_pos_w[k, 0]
        d.qpos[3:7] = s.body_quat_w[k, 0]
        d.qpos[ad] = s.joint_pos[k]
        mujoco.mj_forward(m, d)
        target = [*d.xpos[wb].copy(), 0, 0, dz[k]]
        rot = d.xmat[wb].reshape(3, 3).copy()
        base = q[k, cols].astype(float)

        def fun(x, target=target, rot=rot, base=base, previous=previous):
            d.qpos[ad[cols]] = x
            mujoco.mj_forward(m, d)
            return np.r_[
                (d.xpos[wb] - target) * 5000,
                R.from_matrix(d.xmat[wb].reshape(3, 3) @ rot.T).as_rotvec() * 500,
                (x - base) * 5,
                (x - previous) * 2,
            ]

        fit = least_squares(
            fun,
            np.clip(base, lim[:, 0] + 1e-8, lim[:, 1] - 1e-8),
            bounds=(lim[:, 0], lim[:, 1]),
            max_nfev=100,
            ftol=1e-9,
            gtol=1e-9,
            xtol=1e-9,
        )
        if np.linalg.norm(fit.fun[:3]) > 1.0:
            for initial in [base, base + np.random.default_rng(k).normal(0, 0.5, 7)]:
                other = least_squares(
                    fun,
                    np.clip(initial, lim[:, 0] + 1e-8, lim[:, 1] - 1e-8),
                    bounds=(lim[:, 0], lim[:, 1]),
                    max_nfev=150,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                )
                if other.cost < fit.cost:
                    fit = other
        q[k, cols] = fit.x
        previous = fit.x
        res = fun(fit.x)
        rows.append(
            {
                "time": times[k],
                "height": dz[k],
                "position_error_m": float(np.linalg.norm(res[:3]) / 5000),
                "rotation_error_rad": float(np.linalg.norm(res[3:6]) / 500),
            }
        )
    refresh_fk(replace(s, joint_pos=q), robot).save(out / "motion.npz")
    (out / "motion_order.json").write_bytes(Path(info["parent"]["order"]).read_bytes())
    (a.run / "review/lift_v3_fit.json").write_text(json.dumps(rows, indent=2))
    print("max errors", max(x["position_error_m"] for x in rows), max(x["rotation_error_rad"] for x in rows))


if __name__ == "__main__":
    main()
