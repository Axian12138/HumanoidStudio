"""Carry a collision-tested sofa grasp transform through the complete hat hold window."""

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


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--input-version", default="03_04_05_joint_v42")
    p.add_argument("--seed-candidate")
    p.add_argument("--family-json", default="03_04_05_sofa_grasp_family.json")
    p.add_argument("--output-version", default="03_04_05_joint_v45")
    a = p.parse_args()
    r = a.run
    src = r / "retarget/candidates" / a.input_version
    out = r / "retarget/candidates" / a.output_version
    out.mkdir(exist_ok=False)
    family = json.loads((r / "review/scene" / a.family_json).read_text())
    seed = next(x for x in family["records"] if x["seed"] == a.seed)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    hat = np.load(src / "black_ribbon_hat.npz")
    q = motion.joint_pos.astype(float).copy()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    addr = np.array([m.joint(n).qposadr[0] for n in motion.joint_names])
    cols = [motion.joint_names.index(n) for n in family["joints"][:7]]
    handcols = [motion.joint_names.index(n) for n in family["joints"][7:]]
    lo = np.array([m.joint(motion.joint_names[j]).range[0] for j in cols])
    hi = np.array([m.joint(motion.joint_names[j]).range[1] for j in cols])
    w = m.body("right_wrist_yaw_link").id
    wi = motion.body_names.index("right_wrist_yaw_link")
    g = 600
    if a.seed_candidate:
        seed_motion = MotionFields.load(
            r / "retarget/candidates" / a.seed_candidate / "motion.npz",
            r / "retarget/candidates" / a.seed_candidate / "motion_order.json",
        )
        assert seed_motion.joint_names == motion.joint_names
        seed = {"seed": "candidate39:" + a.seed_candidate, "q": seed_motion.joint_pos[g, cols + handcols].tolist()}
    d.qpos[:3] = motion.body_pos_w[g, 0]
    d.qpos[3:7] = motion.body_quat_w[g, 0]
    d.qpos[addr] = q[g]
    d.qpos[addr[cols]] = seed["q"][:7]
    mujoco.mj_forward(m, d)
    wR = Rotation.from_matrix(d.xmat[w].reshape(3, 3))
    center = wR.inv().apply(hat["position_w"][g] - d.xpos[w])
    relative = wR.inv() * Rotation.from_quat(hat["quaternion_xyzw"][g])
    time = 27 + np.arange(len(q)) / 50
    rows = np.flatnonzero((time >= 36.3) & (time <= 49.5))
    hR = Rotation.from_quat(hat["quaternion_xyzw"])
    desiredR = hR * relative.inv()
    desiredP = hat["position_w"] - desiredR.apply(center)
    oldR = Rotation.from_quat(motion.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    oldP = motion.body_pos_w[:, wi]
    release = 1032
    release_delta = desiredP[release] - oldP[release]
    release_rot = desiredR[release] * oldR[release].inv()
    grasp_start = round((38.3 - 27) * 50)
    approach_delta = desiredP[grasp_start] - oldP[grasp_start]
    approach_rotation = desiredR[grasp_start] * oldR[grasp_start].inv()
    errors = []
    lastq = q[rows[0], cols].copy()
    for k in rows:
        t = time[k]
        alpha = np.clip((t - 36.3) / 2, 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)
        if t < 38.3:
            targetP = oldP[k] + alpha * approach_delta
            targetR = Rotation.from_rotvec(approach_rotation.as_rotvec() * alpha) * oldR[k]
        elif t <= 47.64:
            targetP = oldP[k] + alpha * (desiredP[k] - oldP[k])
            targetR = Rotation.from_rotvec((desiredR[k] * oldR[k].inv()).as_rotvec() * alpha) * oldR[k]
        else:
            fade = np.clip((49.5 - t) / (49.5 - 47.64), 0, 1)
            fade = fade * fade * (3 - 2 * fade)
            targetP = oldP[k] + fade * release_delta
            targetR = Rotation.from_rotvec(release_rot.as_rotvec() * fade) * oldR[k]
        d.qpos[:3] = motion.body_pos_w[k, 0]
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = q[k]
        prior = q[k, cols].copy()

        def residual(x, targetP=targetP, targetR=targetR, prior=prior, lastq=lastq):
            d.qpos[addr[cols]] = x
            mujoco.mj_forward(m, d)
            rot = Rotation.from_matrix(d.xmat[w].reshape(3, 3))
            return np.r_[
                (d.xpos[w] - targetP) * 500,
                (rot * targetR.inv()).as_rotvec() * 100,
                (x - prior) * 0.08,
                (x - lastq) * 0.12,
            ]

        fit = least_squares(residual, np.clip(lastq, lo + 1e-7, hi - 1e-7), bounds=(lo, hi), max_nfev=35, ftol=1e-7)
        if t > 47.64 and np.linalg.norm(residual(fit.x)[:3]) / 500 > 0.003:
            alternate = least_squares(
                residual, np.clip(prior, lo + 1e-7, hi - 1e-7), bounds=(lo, hi), max_nfev=60, ftol=1e-8
            )
            if alternate.cost < fit.cost:
                fit = alternate
        q[k, cols] = fit.x
        lastq = fit.x.copy()
        residual(fit.x)
        errors.append(
            [
                t,
                float(np.linalg.norm(d.xpos[w] - targetP)),
                float((Rotation.from_matrix(d.xmat[w].reshape(3, 3)) * targetR.inv()).magnitude()),
            ]
        )
        hand_alpha = alpha if t <= 47.64 else 0
        if t <= 47.64:
            q[k, handcols] = (1 - hand_alpha) * q[k, handcols] + hand_alpha * np.array(seed["q"][7:])
        else:
            finger_fade = np.clip((48.3 - t) / (48.3 - 47.64), 0, 1)
            finger_fade = finger_fade * finger_fade * (3 - 2 * finger_fade)
            q[k, handcols] += finger_fade * (np.array(seed["q"][7:]) - motion.joint_pos[release, handcols])
        if k % 100 == 0:
            print({"source_s": t, "position_error_m": errors[-1][1]}, flush=True)
    q[rows[[0, -1]]] = motion.joint_pos[rows[[0, -1]]]
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in src.iterdir():
        if (
            f.is_file()
            and f.name not in ["motion.npz", "review.json", "manifest.json"]
            and not f.name.startswith("collision_")
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    review = json.loads((src / "review.json").read_text())
    review["status"] = "new_hat_grip_continuous_candidate_not_accepted"
    review["new_hat_grip"] = {
        "seed": seed["seed"],
        "input_version": a.input_version,
        "fixed_center_local_m": center.tolist(),
        "fixed_rotation_xyzw": relative.as_quat().tolist(),
        "window_s": [36.3, 49.5],
        "hold_window_s": [38.3, 47.64],
        "errors": errors,
        "source_geometry_unresolved": True,
        "all_collision_and_speed_checks_pending": True,
    }
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print({"max_position_error_m": max(x[1] for x in errors), "max_rotation_error_rad": max(x[2] for x in errors)})


if __name__ == "__main__":
    main()
