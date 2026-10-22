"""Retarget the 69–76 s basket interaction as foot pull and forearm push.

The object path is a ground-supported kinematic hypothesis, not a rollout.
The left-leg body prior supplies the hook timing; left-leg joints are locally
refined against the basket during the moving contact window. The right arm is
refined during the return push. Dex3 fingers are untouched.
"""

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

from export_aligned_human import main as export_human


HERE = example_dir(__file__)
ROOT = HERE.parents[1]
OUT = HERE / "hoi_v3_nonprehensile"
PRIOR = HERE / "hoi_v3_nonprehensile/body_prior_69_76.npz"
ORDER = ROOT / "retarget/ground_support_27_end/sidecars/motion_order.json"
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
             "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
FPS = 50
START = 69.0
ARM = tuple("right_" + name + "_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
))
LEG = tuple("left_" + name + "_joint" for name in (
    "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
))


def frame(seconds):
    return round((seconds - START) * FPS)


def object_path():
    # Provisional placement is contact-fitted against the left-shin path.
    # The foot advances the basket to the left; the forearm then drives it
    # diagonally back under the table. There is no vertical object motion.
    knots_t = np.array([69.0, 69.65, 69.85, 70.0, 70.3, 70.55,
                        71.9, 72.5, 73.2, 73.8, 76.0])
    knots_xy = np.array([
        [0.25, -1.78], [0.25, -1.78], [0.22, -1.77],
        [0.19, -1.76], [0.176, -1.76], [0.176, -1.76],
        [0.176, -1.76], [0.19, -1.80], [0.22, -1.85],
        [0.25, -1.88], [0.25, -1.88],
    ])
    times = START + np.arange(350) / FPS
    xy = PchipInterpolator(knots_t, knots_xy, axis=0)(times)
    return np.column_stack((xy, np.full(len(times), 0.001)))


def local_mesh(model, name):
    body = model.body(name).id
    vertices = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or model.geom_group[geom] == 3 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[geom]
        points = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        vertices.append(Rotation.from_quat(model.geom_quat[geom][[1,2,3,0]]).apply(points) + model.geom_pos[geom])
    return body, np.concatenate(vertices)


def main():
    prior = MotionFields.load(PRIOR, ORDER)
    if len(prior.joint_pos) != 350 or float(prior.fps) != FPS:
        raise ValueError("round basket prior must cover 69–76 s at 50 Hz")
    position = object_path()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(name).qposadr[0] for name in prior.joint_names])
    arm_cols = np.asarray([prior.joint_names.index(name) for name in ARM])
    arm_addr = addresses[arm_cols]
    leg_cols = np.asarray([prior.joint_names.index(name) for name in LEG])
    leg_addr = addresses[leg_cols]
    knee, knee_mesh = local_mesh(model, "left_knee_link")
    foot, foot_mesh = local_mesh(model, "left_ankle_roll_link")
    elbow = model.body("right_elbow_link").id
    wrist = model.body("right_wrist_yaw_link").id
    lower = np.asarray([model.joint(name).range[0] for name in ARM]) + 1e-5
    upper = np.asarray([model.joint(name).range[1] for name in ARM]) - 1e-5
    lower[4:] = np.maximum(lower[4:], -np.deg2rad(35))
    upper[4:] = np.minimum(upper[4:], np.deg2rad(35))
    leg_lower = np.asarray([model.joint(name).range[0] for name in LEG]) + 1e-5
    leg_upper = np.asarray([model.joint(name).range[1] for name in LEG]) - 1e-5
    joint = prior.joint_pos.astype(np.float64).copy()
    leg_knots = np.round(np.arange(69.8, 70.201, 0.1), 4)
    leg_offsets = []
    leg_log = []
    previous_leg = None
    for t in leg_knots:
        k = frame(t)
        baseline = prior.joint_pos[k, leg_cols].astype(np.float64)
        data.qpos[:3] = prior.body_pos_w[k, 0]
        data.qpos[3:7] = prior.body_quat_w[k, 0]
        data.qpos[addresses] = prior.joint_pos[k]
        mujoco.mj_forward(model, data)
        knee_world = knee_mesh @ data.xmat[knee].reshape(3,3).T + data.xpos[knee]
        foot_world = foot_mesh @ data.xmat[foot].reshape(3,3).T + data.xpos[foot]
        knee_index = np.argmin(np.linalg.norm(knee_world - np.array([0.35,-1.72,0.25]), axis=1))
        foot_index = np.argmin(foot_world[:,0])
        center = position[k]
        knee_target = np.array([center[0]+0.16, center[1]+0.015, knee_world[knee_index,2]])
        foot_target = np.array([center[0]+0.14, center[1]-0.025, 0.14])

        def leg_residual(q):
            data.qpos[leg_addr] = q
            mujoco.mj_forward(model, data)
            knee_point = knee_mesh[knee_index] @ data.xmat[knee].reshape(3,3).T + data.xpos[knee]
            foot_world = foot_mesh @ data.xmat[foot].reshape(3,3).T + data.xpos[foot]
            foot_point = foot_world[foot_index]
            return np.r_[75*(knee_point-knee_target), 110*(foot_point-foot_target),
                         200*min(float(foot_world[:,2].min())-0.005,0.0),
                         0.6*(q-baseline),
                         0.15*(q-previous_leg) if previous_leg is not None else np.zeros(6)]

        seeds = [np.clip(baseline, leg_lower+1e-4, leg_upper-1e-4)]
        if previous_leg is not None:
            seeds.insert(0, np.clip(previous_leg, leg_lower+1e-4, leg_upper-1e-4))
        fits = [least_squares(leg_residual, seed, bounds=(leg_lower,leg_upper), max_nfev=180) for seed in seeds]
        fit = min(fits, key=lambda result: np.linalg.norm(result.fun))
        leg_residual(fit.x)
        knee_point = knee_mesh[knee_index] @ data.xmat[knee].reshape(3,3).T + data.xpos[knee]
        foot_point = foot_mesh[foot_index] @ data.xmat[foot].reshape(3,3).T + data.xpos[foot]
        leg_offsets.append(fit.x-baseline)
        previous_leg = fit.x.copy()
        leg_log.append({"time_s":float(t), "knee_contact_error_m":float(np.linalg.norm(knee_point-knee_target)),
                        "foot_contact_error_m":float(np.linalg.norm(foot_point-foot_target))})
    leg_anchors = np.r_[frame(69.65), [frame(t) for t in leg_knots], frame(70.35)]
    leg_values = np.vstack((np.zeros(6),leg_offsets,np.zeros(6)))
    active_leg = np.arange(leg_anchors[0],leg_anchors[-1]+1)
    joint[np.ix_(active_leg,leg_cols)] += PchipInterpolator(leg_anchors,leg_values,axis=0)(active_leg)
    joint[np.ix_(active_leg,leg_cols)] = np.clip(joint[np.ix_(active_leg,leg_cols)],leg_lower,leg_upper)
    # Correct the entire foot sole after interpolation. A vertex-level toe
    # target alone can hide penetration by other vertices as the ankle turns.
    ankle_addr = leg_addr[-2:]
    for k in range(frame(69.68), frame(70.31)):
        if frame(69.8) <= k <= frame(70.2):
            continue
        data.qpos[:3] = prior.body_pos_w[k,0]
        data.qpos[3:7] = prior.body_quat_w[k,0]
        data.qpos[addresses] = joint[k]
        mujoco.mj_forward(model,data)
        original = joint[k,leg_cols[-2:]].copy()
        foot_world = foot_mesh @ data.xmat[foot].reshape(3,3).T + data.xpos[foot]
        toe_index = np.argmin(foot_world[:,0])
        toe_xy = foot_world[toe_index,:2].copy()

        def sole_residual(q):
            data.qpos[ankle_addr] = q
            mujoco.mj_forward(model,data)
            points = foot_mesh @ data.xmat[foot].reshape(3,3).T + data.xpos[foot]
            return np.r_[300*min(float(points[:,2].min())-0.002,0.0),
                         5*(points[toe_index,:2]-toe_xy),0.2*(q-original)]

        fit = least_squares(sole_residual,
                            np.clip(original,leg_lower[-2:]+1e-4,leg_upper[-2:]-1e-4),
                            bounds=(leg_lower[-2:],leg_upper[-2:]),max_nfev=60)
        joint[k,leg_cols[-2:]] = fit.x
    old_candidate = None
    if (OUT / "motion.npz").is_file():
        with np.load(OUT / "motion.npz", allow_pickle=False) as existing:
            old_candidate = existing["joint_pos"][:, arm_cols].astype(np.float64)
    knots = np.r_[np.arange(72.0, 73.81, 0.2), 73.8]
    knots = np.unique(np.round(knots, 4))
    offset_map = {}
    log = []
    solved_poses = {}
    solve_times = sorted(knots, key=lambda t: (abs(t - 73.4), t))
    for t in solve_times:
        k = frame(t)
        baseline = prior.joint_pos[k, arm_cols].astype(np.float64)
        previous = solved_poses[min(solved_poses, key=lambda s: abs(s - t))] if solved_poses else None
        data.qpos[:3] = prior.body_pos_w[k, 0]
        data.qpos[3:7] = prior.body_quat_w[k, 0]
        data.qpos[addresses] = prior.joint_pos[k]
        center = position[k]
        # Two contact regions on the front-left exterior of the basket.
        # They bracket the motion vector, so the forearm can apply the
        # diagonal return force without any finger closure.
        wrist_target = center + np.array([-0.198, 0.098, 0.289 - center[2]])
        elbow_target = center + np.array([-0.068, 0.198, 0.390 - center[2]])

        def residual(q):
            data.qpos[arm_addr] = q
            mujoco.mj_forward(model, data)
            return np.r_[
                100 * (data.xpos[wrist] - wrist_target),
                75 * (data.xpos[elbow] - elbow_target),
                0.4 * (q[:4] - baseline[:4]),
                0.8 * q[4:],
                0.1 * (q - previous) if previous is not None else np.zeros(7),
            ]

        seeds = [np.clip(baseline, lower + 1e-4, upper - 1e-4)]
        if old_candidate is not None:
            seeds.insert(0, np.clip(old_candidate[k], lower + 1e-4, upper - 1e-4))
        if previous is not None:
            seeds.insert(0, np.clip(previous, lower + 1e-4, upper - 1e-4))
        fits = [least_squares(residual, seed, bounds=(lower, upper), max_nfev=150) for seed in seeds]
        solution = min(fits, key=lambda fit: np.linalg.norm(fit.fun))
        residual(solution.x)
        offset_map[t] = solution.x - baseline
        solved_poses[t] = solution.x.copy()
        log.append({"time_s": float(t),
                    "wrist_center_error_m": float(np.linalg.norm(data.xpos[wrist] - wrist_target)),
                    "elbow_center_error_m": float(np.linalg.norm(data.xpos[elbow] - elbow_target)),
                    "wrist_axes_deg": np.rad2deg(solution.x[4:]).tolist()})

    log.sort(key=lambda item: item["time_s"])
    offsets = np.asarray([offset_map[t] for t in knots])
    contact_frames = np.asarray([frame(t) for t in knots])
    anchors = np.r_[frame(71.2), contact_frames, frame(74.4), frame(75.0)]
    values = np.vstack((np.zeros(7), offsets, np.zeros((2, 7))))
    active = np.arange(anchors[0], anchors[-1] + 1)
    joint[np.ix_(active, arm_cols)] += PchipInterpolator(anchors, values, axis=0)(active)
    joint[np.ix_(active, arm_cols)] = np.clip(joint[np.ix_(active, arm_cols)], lower, upper)
    candidate = refresh_fk(replace(prior, joint_pos=joint), MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    candidate.save(OUT / "motion.npz")
    np.savez_compressed(
        OUT / "object_estimate.npz", position_w=position.astype(np.float32),
        quaternion_xyzw=np.repeat([[0.0, 0.0, 0.0, 1.0]], len(position), axis=0).astype(np.float32),
        fps=np.asarray(FPS), status=np.asarray("ground_supported_nonprehensile_estimate"),
    )
    export_human(OUT / "human_motion_world.npz")
    report = {
        "status": "kinematic_candidate_pending_stage35_review",
        "source_range_s_half_open": [69, 76], "fps": FPS,
        "action": ["left_shin_foot_continuous_hook", "left_knee_ground", "right_forearm_return_push"],
        "contact_windows_estimated_s": {"foot": [69.65, 70.55], "right_forearm": [72.0, 73.8]},
        "hand_command": "unchanged_from_body_prior; no grasp or closure schedule",
        "leg_knot_fit": leg_log,
        "initial_basket_xyz_m": position[0].tolist(),
        "outbound_basket_xyz_m": position[frame(70.55)].tolist(),
        "returned_basket_xyz_m": position[frame(73.8)].tolist(),
        "initial_pose_status": "provisional_contact_fit_against_1019_left_shin_not_1012_pose",
        "prior_sha256": sha256(PRIOR), "robot_mjcf_sha256": sha256(MODEL),
        "motion_sha256": sha256(OUT / "motion.npz"),
        "arm_knot_fit": log,
        "physics_verified": False,
    }
    (OUT / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"basket_start": report["initial_basket_xyz_m"],
                      "basket_out": report["outbound_basket_xyz_m"],
                      "basket_return": report["returned_basket_xyz_m"],
                      "wrist_fit_max_m": max(item["wrist_center_error_m"] for item in log),
                      "elbow_fit_max_m": max(item["elbow_center_error_m"] for item in log)}, indent=2))


if __name__ == "__main__":
    main()
