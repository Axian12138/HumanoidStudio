"""Compare shoulder/elbow branches for the same deep rim hook end-effector pose."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--time", type=float, default=119)
    a = p.parse_args()
    parent = json.load(open(a.run / "inputs/parent.json"))
    s = MotionFields.load(parent["motion"], parent["parent"]["order"])
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(model))
    d = mujoco.MjData(m)
    ad = np.array([m.joint(n).qposadr[0] for n in s.joint_names])
    k = round((a.time - 108) * 50)
    d.qpos[:3] = s.body_pos_w[k, 0]
    d.qpos[3:7] = s.body_quat_w[k, 0]
    d.qpos[ad] = s.joint_pos[k]
    mujoco.mj_forward(m, d)
    cols = [
        s.joint_names.index("left_" + n + "_joint")
        for n in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    wb = m.body("left_wrist_yaw_link").id
    el = m.body("left_elbow_link").id
    shoulder = m.body("left_shoulder_roll_link").id
    design = json.load(open(a.run / "review/grip_design.json"))[0]
    b = np.load(Path(parent["parent"]["tracks"]) / "small_wicker_basket.npz")
    br = R.from_quat(b["quaternion_xyzw"][k]).as_matrix()
    wr = br[:, [1, 2, 0]]
    rim = b["position_w"][k] + br @ np.array([0, -0.16872634, 0.20841585])
    target = rim - wr @ np.array(design["rim_in_wrist"])
    pairs = [
        (m.geom("left_" + part + "_collision").id, m.geom("torso_collision" + str(i)).id)
        for part in ["shoulder_yaw", "elbow_yaw"]
        for i in [1, 2, 3]
    ]
    ini = s.joint_pos[k, cols].astype(float)
    lim = np.array([m.joint(s.joint_names[c]).range for c in cols])
    lim[5] = [-0.44, 0.44]
    results = []
    for branch in ["negative_elbow", "positive_elbow"]:
        bounds = lim.copy()
        bounds[3] = [-1.0472, -0.02] if branch == "negative_elbow" else [0.02, 2.0944]

        def fun(x):
            d.qpos[ad[cols]] = x
            mujoco.mj_forward(m, d)
            g = np.array([mujoco.mj_geomDistance(m, d, i, j, 0.03, None) for i, j in pairs])
            return np.r_[
                (d.xpos[wb] - target) * 1000,
                R.from_matrix(d.xmat[wb].reshape(3, 3) @ wr.T).as_rotvec() * 100,
                np.minimum(g - 0.002, 0) * 2000,
                x[5] * 0.5,
                (x - ini) * 0.02,
            ]

        best = None
        for j in range(16):
            x = ini + np.random.default_rng(j).normal(0, 0.8, 7)
            x[3] = -0.6 if branch == "negative_elbow" else 1.0
            x = np.clip(x, bounds[:, 0] + 1e-8, bounds[:, 1] - 1e-8)
            z = least_squares(fun, x, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=180)
            if best is None or np.linalg.norm(z.fun) < np.linalg.norm(best.fun):
                best = z
        res = fun(best.x)
        results.append(
            {
                "branch": branch,
                "score": float(np.linalg.norm(res)),
                "q": best.x.tolist(),
                "joint_names": [s.joint_names[c] for c in cols],
                "wrist_error_m": float(np.linalg.norm(res[:3]) / 1000),
                "rotation_error_rad": float(np.linalg.norm(res[3:6]) / 100),
                "elbow_relative_shoulder": (d.xpos[el] - d.xpos[shoulder]).tolist(),
                "target_wrist_position": target.tolist(),
                "target_wrist_rotation": wr.tolist(),
            }
        )
    (
        a.run / ("review/branch_comparison.json" if a.time == 119 else f"review/branch_comparison_{a.time}.json")
    ).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
