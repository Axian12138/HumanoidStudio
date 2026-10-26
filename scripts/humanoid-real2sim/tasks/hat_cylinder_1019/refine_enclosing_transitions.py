"""Keep the hat observed pose while the open hand enters and closes around it."""


def main():
    import argparse
    import json
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import mujoco
    import numpy as np
    from fit import ARM, MODEL, smooth
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    src = r / "retarget/enclosing_v2"
    out = r / "retarget/enclosing_v4"
    shutil.copytree(src, out)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = motion.joint_pos.astype(float).copy()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    cols = [motion.joint_names.index("right_" + x + "_joint") for x in ARM]
    w = model.body("right_wrist_yaw_link").id
    lo = np.array([model.joint(motion.joint_names[j]).range[0] for j in cols])
    hi = np.array([model.joint(motion.joint_names[j]).range[1] for j in cols])
    design = json.load(open(r / "review/grasp_design_enclosing_v1.json"))
    rel = R.from_quat(design["relative_hat_rotation_xyzw"])
    center = np.array(design["selected"]["center"])
    hat = np.load(src / "scene_tracks/black_ribbon_hat_free.npz")
    hr = R.from_quat(hat["quaternion_xyzw"])
    hp = hat["position_w"]
    records = []
    for i in range(len(q)):
        t = 35 + i / 50
        if not (38.1 <= t <= 39.3 or 47.5 <= t <= 48.6):
            continue
        alpha = smooth(t, 38.3, 39.05) * (1 - smooth(t, 47.64, 48.1))
        if 47.64 <= t <= 48.6:
            for part, value in zip(
                ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
                design["selected"]["hand"],
                strict=True,
            ):
                q[i, motion.joint_names.index("right_hand_" + part + "_joint")] = value * alpha
        c = center + np.array([0.02, 0.01, 0]) * np.sin(np.pi * alpha) ** 2
        tr = hr[i] * rel.inv()
        tp = hp[i] + hr[i].apply([0, 0, 0.11]) - tr.apply(c)
        d.qpos[:3] = motion.body_pos_w[i, 0]
        d.qpos[3:7] = motion.body_quat_w[i, 0]
        d.qpos[addr] = q[i]
        prior = q[i, cols].copy()

        def fun(x, prior=prior, tp=tp, tr=tr):
            d.qpos[addr[cols]] = x
            mujoco.mj_forward(model, d)
            wr = R.from_matrix(d.xmat[w].reshape(3, 3))
            return np.r_[(d.xpos[w] - tp) * 1000, (wr * tr.inv()).as_rotvec() * 100, (x - prior) * 0.05]

        fit = least_squares(fun, np.clip(prior, lo + 1e-7, hi - 1e-7), bounds=(lo, hi), max_nfev=70)
        weight = (
            smooth(t, 38.1, 38.28) * (1 - smooth(t, 39.16, 39.3))
            if t < 40
            else smooth(t, 47.5, 47.64) * (1 - smooth(t, 48.1, 48.4))
        )
        q[i, cols] += weight * (fit.x - prior)
        records.append([t, float(np.linalg.norm(fit.fun[:3]) / 1000)])
    res = refresh_fk(replace(motion, joint_pos=q), MODEL)
    res.save(out / "motion.npz")
    (out / "transition_alignment.json").write_text(
        json.dumps(
            {
                "center_path": "center + [0.02,0.01,0] sin(pi closure)^2",
                "object_unchanged": True,
                "fit_errors": records,
            },
            indent=2,
        )
    )
    print("max target error", max(x[1] for x in records))


if __name__ == "__main__":
    main()
