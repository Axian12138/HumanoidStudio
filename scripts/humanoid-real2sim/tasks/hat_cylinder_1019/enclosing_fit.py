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
    q = MotionFields.load(r / "retarget/seed/motion.npz", src / "motion_order.json").joint_pos.astype(float).copy()
    n = len(q)
    time = 35 + np.arange(n) / 50
    m = mujoco.MjModel.from_xml_path(str(r / "scene/robot_props_cylinder_hat.xml"))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    addr = np.array([m.joint(x).qposadr[0] for x in motion.joint_names])
    ac = np.array([motion.joint_names.index("right_" + x + "_joint") for x in ARM])
    hc = np.array([motion.joint_names.index("right_hand_" + x + "_joint") for x in PARTS])
    w = m.body("right_wrist_yaw_link").id
    # Object +X lies along wrist -Z, and its upward crown points toward wrist -Y.
    relative = R.from_matrix(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.0]])) * R.from_euler(
        "z", a.cylinder_yaw_deg, degrees=True
    )
    relative = R.from_euler("z", a.grasp_roll_deg, degrees=True) * relative
    records = json.loads((r / "review/enclosing_grip_search.json").read_text())
    best = records[0]
    path_rows = json.loads((r / "review/enclosing_center_path.json").read_text())
    phases = np.array([x["alpha"] for x in path_rows])
    centers = np.array([x["center"] for x in path_rows])

    def phase_center(alpha):
        return np.array([np.interp(alpha, phases, centers[:, j]) for j in range(3)])

    center = np.array(best["center"])
    closed = np.array(best["hand"])
    opened = np.zeros(7)
    (r / "review" / ("grasp_design_" + a.version + ".json")).write_text(
        json.dumps(
            {
                "selected": best,
                "candidates": records,
                "relative_hat_rotation_xyzw": relative.as_quat().tolist(),
                "method": "declared open-to-closed fingers; opposing cylinder contacts and collision-cleared insertion center path",
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
    release = round((47.64 - 35) * 50)
    # Seed prior at both ends; fixed target transform during the full transport.
    targetsP = []
    targetsR = []
    for i, t in enumerate(time):
        closure = smooth(t, 38.3, 39.05) * (1 - smooth(t, 47.64, 48.39))
        offset = phase_center(closure) - center
        if t < 38.3:
            alpha = smooth(t, 35.5, 38.3)
            clearance = 0.10 * (1 - smooth(t, 37.2, 38.3))
            p0 = desiredP[i] - desiredR[i].apply([0, clearance, 0])
            tr = R.from_rotvec((desiredR[i] * oldR[i].inv()).as_rotvec() * alpha) * oldR[i]
            tp = oldP[i] + alpha * (p0 - oldP[i])
        elif t <= 47.64:
            tp = desiredP[i] - desiredR[i].apply(offset)
            tr = desiredR[i]
        else:
            alpha = 1 - smooth(t, 48.39, 50)
            clearance = 0.10 * smooth(t, 48.39, 49.0)
            p0 = desiredP[release] - desiredR[release].apply(np.array([0, clearance, 0]) + offset)
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
    from scipy.ndimage import gaussian_filter1d

    q[:, ac] = gaussian_filter1d(q[:, ac], 2, axis=0, mode="nearest")
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
        alpha = smooth(t, 39.05, 39.15) if t < 39.15 else 1
        if i <= release and t >= 39.05:
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
