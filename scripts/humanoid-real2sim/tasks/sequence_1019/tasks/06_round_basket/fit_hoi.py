"""Fit right-palm approach/pull for the 69–76 s basket interaction.

Only the seven right-arm joints and commanded Dex3 hand trajectory change.
The basket path is an explicit ground-supported kinematic estimate.
"""

from dataclasses import replace
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


HERE = example_dir(__file__)
ROOT = HERE.parents[1]
PRIOR = HERE / "hoi_v3_nonprehensile/body_prior_69_76.npz"
ORDER = ROOT / "retarget/ground_support_27_end/sidecars/motion_order.json"
MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/"
    "robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)
SOURCE_HUMAN = ROOT / "retarget/ground_support_27_end/sidecars/human_motion_world.npz"
OUT = HERE / "hoi_v2"
FPS = 50
START = 69.0
INITIAL = np.array([0.10, -1.92, 0.001])
HANDLE_LOCAL = np.array([-0.20, 0.0, 0.470])
PALM_LOCAL = np.array([0.10629904, 0.03630376, 0.0])
PALM_NORMAL = np.array([1.0, 0.0, 0.0])
HANDLE_AXIS = np.array([0.0, -1.0, 0.0])
ARM = tuple("right_" + name + "_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
))
FINGERS = tuple("right_hand_" + name + "_joint" for name in (
    "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1",
))
OPEN = np.array([0.234521568, 0, 0, 0, 0, 0, 0])
CLOSED = np.array([0.59, -0.65, -0.90, 1.15, 1.56, 0.77, 0.99])


def frame(seconds):
    return round((seconds - START) * FPS)


def smooth(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3 - 2 * value)


def basket_translation(seconds):
    # The source head/right-hand views show floor-supported pull, not lift.
    lateral = smooth((seconds - 73.45) / 0.50)
    pull = smooth((seconds - 73.52) / 1.03)
    return np.array([-0.05 * lateral, 0.33 * pull, 0.0])


def main():
    baseline = MotionFields.load(PRIOR, ORDER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    arm_cols = np.asarray([baseline.joint_names.index(name) for name in ARM])
    finger_cols = np.asarray([baseline.joint_names.index(name) for name in FINGERS])
    addresses = np.asarray([model.joint(name).qposadr[0] for name in baseline.joint_names])
    arm_addresses = addresses[arm_cols]
    wrist = model.body("right_wrist_yaw_link").id
    low = np.asarray([model.joint(name).range[0] for name in ARM]) + 1e-5
    high = np.asarray([model.joint(name).range[1] for name in ARM]) - 1e-5
    wrist_limit = np.deg2rad(35)
    low[4:] = np.maximum(low[4:], -wrist_limit)
    high[4:] = np.minimum(high[4:], wrist_limit)
    joint = baseline.joint_pos.astype(np.float64).copy()
    solved = []
    records = []
    solved_poses = {}
    # Sparse time-window controls; offsets are interpolated and the entire
    # 50 Hz path is audited after FK, including both splice boundaries.
    knots = [
        (70.7, None),
        (71.4, (-0.23, -1.56, 0.26)),
        (72.0, (-0.23, -1.68, 0.33)),
        (72.5, (-0.19, -1.79, 0.40)),
        (72.9, (-0.15, -1.88, 0.47)),
        (73.25, None), (73.40, (-0.13, -1.92, 0.47)), (73.50, None),
        (73.90, None), (74.30, None), (74.55, None),
        (74.75, (-0.28, -1.43, 0.51)),
        (75.35, None),
    ]
    ordered_knots = sorted(knots[1:-1], key=lambda item: (abs(item[0] - 73.5), item[0]))
    ordered_knots = [knots[0]] + ordered_knots + [knots[-1]]
    for seconds, waypoint in ordered_knots:
        k = frame(seconds)
        prior = baseline.joint_pos[k, arm_cols].astype(np.float64)
        if seconds in (70.7, 75.35):
            solved.append((k, np.zeros(len(arm_cols))))
            continue
        data.qpos[:3] = baseline.body_pos_w[k, 0]
        data.qpos[3:7] = baseline.body_quat_w[k, 0]
        data.qpos[addresses] = baseline.joint_pos[k]
        data.qpos[addresses[finger_cols]] = OPEN
        target = (np.asarray(waypoint) if waypoint is not None else
                  INITIAL + basket_translation(seconds) + HANDLE_LOCAL)
        previous = solved_poses[min(solved_poses, key=lambda t: abs(t - seconds))] if solved_poses else None

        def pose(q):
            data.qpos[arm_addresses] = q
            mujoco.mj_forward(model, data)
            rotation = data.xmat[wrist].reshape(3, 3)
            return data.xpos[wrist] + rotation @ PALM_LOCAL, rotation[:, 1], rotation[:, 2]

        def residual(q):
            eye, normal, axis = pose(q)
            orientation_weight = 3.0 if seconds < 72.5 else 7.0
            return np.r_[
                140 * (eye - target),
                orientation_weight * (normal - PALM_NORMAL),
                (orientation_weight * 0.75) * (axis - HANDLE_AXIS),
                0.25 * (q[:4] - prior[:4]),
                0.6 * q[4:],
                0.5 * (q - previous) if previous is not None else np.zeros(7),
            ]

        seeds = [np.clip(prior, low + 1e-4, high - 1e-4)]
        if previous is not None:
            seeds.insert(0, np.clip(previous, low, high))
        results = [least_squares(residual, seed, bounds=(low, high), max_nfev=180) for seed in seeds]
        result = min(results, key=lambda item: np.linalg.norm(item.fun))
        eye, normal, axis = pose(result.x)
        solved.append((k, result.x - prior))
        solved_poses[seconds] = result.x.copy()
        records.append(dict(
            time_s=seconds, target_m=target.tolist(), palm_m=eye.tolist(),
            palm_error_m=float(np.linalg.norm(eye - target)),
            palm_normal_error_deg=float(np.degrees(np.arccos(np.clip(normal @ PALM_NORMAL, -1, 1)))),
            handle_axis_error_deg=float(np.degrees(np.arccos(np.clip(axis @ HANDLE_AXIS, -1, 1)))),
            wrist_axes_deg=np.rad2deg(result.x[4:]).tolist(),
        ))

    solved.sort(key=lambda item: item[0])
    records.sort(key=lambda item: item["time_s"])
    knot_frames = np.asarray([item[0] for item in solved])
    offsets = np.asarray([item[1] for item in solved])
    active = np.arange(knot_frames[0], knot_frames[-1] + 1)
    joint[np.ix_(active, arm_cols)] += PchipInterpolator(knot_frames, offsets, axis=0)(active)
    joint[np.ix_(active, arm_cols)] = np.clip(joint[np.ix_(active, arm_cols)], low, high)

    for k in range(frame(70.6), frame(75.5) + 1):
        t = START + k / FPS
        opening = smooth((t - 70.6) / 1.25)
        closing = smooth((t - 73.23) / 0.30)
        release = smooth((t - 74.55) / 0.25)
        neutral = baseline.joint_pos[k, finger_cols]
        desired = (1 - closing) * OPEN + closing * CLOSED
        desired = (1 - release) * desired + release * OPEN
        back_to_neutral = smooth((t - 75.15) / 0.35)
        joint[k, finger_cols] = (1 - back_to_neutral) * ((1 - opening) * neutral + opening * desired) + back_to_neutral * neutral

    candidate = refresh_fk(replace(baseline, joint_pos=joint), MODEL)
    OUT.mkdir(exist_ok=True)
    candidate.save(OUT / "motion.npz")
    position = np.repeat(INITIAL[None, :], len(joint), axis=0)
    for k in range(len(position)):
        position[k] += basket_translation(START + k / FPS)
    quat = np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]]), len(joint), axis=0)
    np.savez_compressed(
        OUT / "object_estimate.npz", position_w=position.astype(np.float32),
        quaternion_xyzw=quat.astype(np.float32), fps=np.asarray(FPS),
        status=np.asarray("ground_supported_kinematic_estimate_not_physics"),
    )
    with np.load(SOURCE_HUMAN, allow_pickle=False) as human:
        np.savez_compressed(
            OUT / "human_motion_world.npz",
            positions_w=human["positions_w"][frame(69):frame(76)],
            joint_names=human["joint_names"], edges=human["edges"], fps=human["fps"],
        )
    changed = joint - baseline.joint_pos
    dt = np.diff(joint[:, arm_cols], axis=0) * FPS
    report = dict(
        status="kinematic_candidate_pending_stage35_human_review", fps=FPS,
        source_range_s_half_open=[69, 76],
        prior_sha256=sha256(PRIOR), robot_mjcf_sha256=sha256(MODEL),
        basket_source_blend="round_laundry_basket_v2.blend",
        basket_initial_position_m=INITIAL.tolist(),
        basket_position_revision_from_1012_m=[-0.06, 0.0, 0.0],
        basket_position_revision_status="provisional_from_1019_right_hand_reach",
        basket_handle_contact_local_m=HANDLE_LOCAL.tolist(),
        command_s=dict(open_start=70.6, open_complete=71.85,
                       close_start=73.23, close_complete=73.53,
                       release_start=74.55, release_complete=74.8),
        first_contact_estimate_s=73.5,
        object_motion_estimate_s=[73.45, 74.55],
        object_motion_note="Ground-supported visual estimate; basket x moves -5 cm and y moves +33 cm, no dynamics inferred.",
        unchanged_root_legs_left_arm=True,
        max_arm_delta_rad=float(np.max(np.abs(changed[:, arm_cols]))),
        max_arm_speed_rad_s=float(np.max(np.abs(dt))),
        max_knot_palm_error_m=max(record["palm_error_m"] for record in records),
        max_knot_normal_error_deg=max(record["palm_normal_error_deg"] for record in records),
        knot_metrics=records,
        physics_grasp_verified=False,
        motion_sha256=sha256(OUT / "motion.npz"),
    )
    (OUT / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "max_arm_delta_rad", "max_arm_speed_rad_s", "max_knot_palm_error_m", "max_knot_normal_error_deg"
    )}, indent=2))


if __name__ == "__main__":
    main()
