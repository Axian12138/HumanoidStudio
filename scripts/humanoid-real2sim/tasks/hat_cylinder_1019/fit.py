"""Fit fixed Dex3 hand designs to the authored 70 mm cylinder, then retarget the right arm."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)
PARTS = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def smooth(t, a, b):
    x = np.clip((t - a) / (b - a), 0, 1)
    return x * x * x * (10 + x * (-15 + 6 * x))


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--cylinder-yaw-deg", type=float, default=0.0)
    p.add_argument("--neutral-wrist-weight", type=float, default=0.0)
    p.add_argument("--grasp-roll-deg", type=float, default=0.0)
    p.add_argument("--joint-step-speed", type=float, default=0.0)
    p.add_argument("--multistart", action="store_true")
    a = p.parse_args()
    r = a.run
    src = r / "retarget/baseline"
    out = r / "retarget" / a.version
    out.mkdir(exist_ok=False)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = motion.joint_pos.astype(float).copy()
    n = len(q)
    time = 35 + np.arange(n) / 50
    m = mujoco.MjModel.from_xml_path(str(r / "scene/robot_props_cylinder_hat.xml"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    addr = np.array([m.joint(x).qposadr[0] for x in motion.joint_names])
    ac = np.array([motion.joint_names.index("right_" + x + "_joint") for x in ARM])
    hc = np.array([motion.joint_names.index("right_hand_" + x + "_joint") for x in PARTS])
    w = m.body("right_wrist_yaw_link").id
    oid = m.body("black_ribbon_hat").id
    free = m.joint("black_ribbon_hat_free").qposadr[0]
    groups = {}
    for name in ["right_wrist_yaw_link"] + ["right_hand_" + x + "_link" for x in PARTS]:
        bid = m.body(name).id
        groups[name] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g]]
    og = [g for g in range(m.ngeom) if m.geom_bodyid[g] == oid and m.geom_contype[g]]
    # Object +X lies along wrist -Z, and its upward crown points toward wrist -Y.
    relative = R.from_matrix(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.0]])) * R.from_euler(
        "z", a.cylinder_yaw_deg, degrees=True
    )
    relative = R.from_euler("z", a.grasp_roll_deg, degrees=True) * relative
    k = (39 - 35) * 50
    d.qpos[:3] = motion.body_pos_w[k, 0]
    d.qpos[3:7] = motion.body_quat_w[k, 0]
    d.qpos[addr] = q[k]
    records = []
    for thumb0 in [0.0]:
        for f in [0.35, 0.5, 0.65, 0.8, 1.0]:
            for th in [0.25, 0.4, 0.55, 0.7]:
                hand = np.array([thumb0, -th, -th, f, f, f, f])
                d.qpos[addr[hc]] = hand
                mujoco.mj_forward(m, d)
                wp = d.xpos[w].copy()
                wr = R.from_matrix(d.xmat[w].reshape(3, 3))
                hr = wr * relative

                def distances(c, wp=wp, wr=wr, hr=hr):
                    center = np.array([c[0], c[1], 0.0])
                    hp = wp + wr.apply(center) - hr.apply([0, 0, 0.11])
                    d.qpos[free : free + 3] = hp
                    d.qpos[free + 3 : free + 7] = hr.as_quat()[[3, 0, 1, 2]]
                    mujoco.mj_forward(m, d)
                    return np.array(
                        [
                            min(mujoco.mj_geomDistance(m, d, g, o, 0.2, None) for g in gs for o in og) if gs else 0.2
                            for gs in groups.values()
                        ]
                    )

                def fun(c, distances=distances):
                    ds = distances(c)
                    return np.r_[
                        (ds[[3, 5, 7]] - 0.001) * 150, np.minimum(ds - 0.0003, 0) * 600, (c - [0.105, 0.065]) * 0.2
                    ]

                fit = least_squares(
                    fun, [0.0898168, 0.098252], bounds=([0.06, 0.035], [0.16, 0.12]), diff_step=1e-4, max_nfev=55
                )
                ds = distances(fit.x)
                self_gap = min(
                    mujoco.mj_geomDistance(m, d, g, h, 0.1, None)
                    for g in groups["right_hand_thumb_2_link"]
                    for name in ["right_hand_middle_1_link", "right_hand_index_1_link"]
                    for h in groups[name]
                )
                score = np.linalg.norm(fun(fit.x)) + max(0.001 - self_gap, 0) * 10000
                records.append(
                    {
                        "hand": hand.tolist(),
                        "center": [float(fit.x[0]), float(fit.x[1]), 0.0],
                        "distances": dict(zip(groups, ds.tolist(), strict=True)),
                        "score": float(score),
                        "finger_self_gap_m": float(self_gap),
                    }
                )
    best = min(records, key=lambda x: x["score"])
    center = np.array(best["center"])
    closed = np.array(best["hand"])
    opened = np.array([0.0, -1.04, -0.7, 0, 0, 0, 0])
    (r / "review" / ("grasp_design_" + a.version + ".json")).write_text(
        json.dumps(
            {
                "selected": best,
                "candidates": records,
                "relative_hat_rotation_xyzw": relative.as_quat().tolist(),
                "method": "fixed declared finger families; fit cylinder axis center with signed MJCF collision distances",
            },
            indent=2,
        )
    )
    print("design", best, flush=True)
    hat = np.load(src / "scene_tracks/black_ribbon_hat_free.npz")
    hp = hat["position_w"].astype(float)
    hr = R.from_quat(hat["quaternion_xyzw"])
    desiredR = hr * relative.inv()
    desiredP = hp + hr.apply([0, 0, 0.11]) - desiredR.apply(center)
    wi = motion.body_names.index("right_wrist_yaw_link")
    oldP = motion.body_pos_w[:, wi]
    oldR = R.from_quat(motion.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    # Orient while outside, enter palm-first, close only once the cylinder has entered.
    g = round((39.05 - 35) * 50)
    release = round((47.64 - 35) * 50)
    # Seed prior at both ends; fixed target transform during the full transport.
    targetsP = []
    targetsR = []
    for i, t in enumerate(time):
        if t < 38.3:
            alpha = smooth(t, 35.5, 38.3)
            clearance = 0.10 * (1 - smooth(t, 37.2, 38.3))
            p0 = desiredP[i] - desiredR[i].apply([0, clearance, 0])
            tr = R.from_rotvec((desiredR[i] * oldR[i].inv()).as_rotvec() * alpha) * oldR[i]
            tp = oldP[i] + alpha * (p0 - oldP[i])
        elif t <= 47.64:
            tp = desiredP[i]
            tr = desiredR[i]
        else:
            alpha = 1 - smooth(t, 48.39, 50)
            clearance = 0.10 * smooth(t, 48.39, 49.0)
            p0 = desiredP[release] - desiredR[release].apply([0, clearance, 0])
            tp = oldP[i] + alpha * (p0 - oldP[i])
            tr = R.from_rotvec((desiredR[release] * oldR[i].inv()).as_rotvec() * alpha) * oldR[i]
        targetsP.append(tp)
        targetsR.append(tr)
    lo = np.array([m.joint(motion.joint_names[j]).range[0] for j in ac])
    hi = np.array([m.joint(motion.joint_names[j]).range[1] for j in ac])
    torso = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("torso_link").id and m.geom_contype[g]]
    elbow = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("right_elbow_link").id and m.geom_contype[g]]
    last = q[0, ac].copy()
    errors = []
    for i, t in enumerate(time):
        d.qpos[:3] = motion.body_pos_w[i, 0]
        d.qpos[3:7] = motion.body_quat_w[i, 0]
        d.qpos[addr] = q[i]
        prior = q[i, ac].copy()

        def fun(x, i=i, prior=prior, last=last):
            d.qpos[addr[ac]] = x
            mujoco.mj_forward(m, d)
            rot = R.from_matrix(d.xmat[w].reshape(3, 3))
            return np.r_[
                (d.xpos[w] - targetsP[i]) * 250,
                (rot * targetsR[i].inv()).as_rotvec() * 25,
                (x - prior) * 0.4,
                x[4:] * a.neutral_wrist_weight,
                (x - last) * 1.5,
                [min(mujoco.mj_geomDistance(m, d, g, h, 0.1, None) - 0.003, 0) * 1500 for g in elbow for h in torso],
            ]

        frame_lo = np.maximum(lo, last - a.joint_step_speed / 50) if a.joint_step_speed else lo
        frame_hi = np.minimum(hi, last + a.joint_step_speed / 50) if a.joint_step_speed else hi
        fit = least_squares(
            fun, np.clip(last, frame_lo + 1e-7, frame_hi - 1e-7), bounds=(frame_lo, frame_hi), max_nfev=55, ftol=1e-6
        )
        if a.multistart:
            seed = prior.copy()
            seed[4:] = 0
            alternate = least_squares(
                fun, np.clip(seed, frame_lo + 1e-7, frame_hi - 1e-7), bounds=(frame_lo, frame_hi), max_nfev=80
            )
            if np.linalg.norm(alternate.fun) < np.linalg.norm(fit.fun):
                fit = alternate
        q[i, ac] = fit.x
        last = fit.x.copy()
        res = fun(fit.x)
        errors.append([float(t), float(np.linalg.norm(res[:3]) / 250), float(np.linalg.norm(res[3:6]) / 25)])
        opening = smooth(t, 35, 36.3) * (1 - smooth(t, 48.8, 50))
        closure = smooth(t, 38.3, 39.05) * (1 - smooth(t, 47.64, 48.39))
        hand = opened + (closed - opened) * closure
        neutral = motion.joint_pos[0, hc] * (1 - smooth(t, 48.8, 50)) + motion.joint_pos[-1, hc] * smooth(t, 48.8, 50)
        q[i, hc] = (1 - opening) * neutral + opening * hand
        q[i, hc[0]] = 0.0  # User: no axial thumb-root rotation, including transitions.
        if i % 150 == 0:
            print("fit", errors[-1], flush=True)
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    result.save(out / "motion.npz")
    for name in ["motion_order.json", "human_motion_world.npz", "source_sync.npz"]:
        shutil.copy2(src / name, out / name)
    shutil.copytree(src / "scene_tracks", out / "scene_tracks")
    # Kinematic held estimate is rigid relative to actual FK, never a dynamic grasp claim.
    wr = R.from_quat(result.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    newR = wr * relative
    newP = result.body_pos_w[:, wi] + wr.apply(center) - newR.apply([0, 0, 0.11])
    pp = hp.copy()
    qq = hr.as_quat().copy()
    for i, t in enumerate(time):
        alpha = smooth(t, 38.3, 39.05) if i < g else 1
        if i <= release and t >= 38.3:
            pp[i] = hp[i] + alpha * (newP[i] - hp[i])
            qq[i] = (R.from_rotvec((newR[i] * hr[i].inv()).as_rotvec() * alpha) * hr[i]).as_quat()
        elif i > release:
            pp[i] = newP[release]
            qq[i] = newR[release].as_quat()
    data = {k: hat[k] for k in hat.files}
    data.update(position_w=pp, quaternion_xyzw=qq)
    np.savez_compressed(out / "scene_tracks/black_ribbon_hat_free.npz", **data)
    report = {
        "status": "stage3_2_candidate_pending_sweep_review",
        "source_interval_s": [35, 50],
        "grasp_command_s": [38.3, 39.05],
        "release_command_s": [47.64, 48.39],
        "fixed_hand_design": best,
        "fit_errors": errors,
        "object_deviation_max_m": float(np.linalg.norm(pp - hp, axis=1).max()),
        "release_deviation_m": float(np.linalg.norm(pp[release] - hp[release])),
        "protected_joint_columns_unchanged": bool(
            np.array_equal(
                q[:, [j for j in range(43) if j not in set(ac) | set(hc)]],
                motion.joint_pos[:, [j for j in range(43) if j not in set(ac) | set(hc)]],
            )
        ),
        "max_adjacent_speed_rad_s": dict(
            zip(
                [motion.joint_names[j] for j in ac],
                np.max(abs(np.diff(q[:, ac], axis=0)) * 50, axis=0).tolist(),
                strict=True,
            )
        ),
        "physics_verified": False,
    }
    (out / "review.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k not in ["fit_errors", "fixed_hand_design"]}, flush=True)


if __name__ == "__main__":
    main()
