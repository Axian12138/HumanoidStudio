"""Restore the captured left-foot pitch and fit a supported basket pull at 59–69 s.

This produces a Stage 3.5 kinematic candidate. The video constrains event
timing; the basket path remains a contact-conditioned estimate, not tracking.
"""

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


HERE = example_dir(__file__)
ROOT = HERE.parents[1]
INPUT = HERE / "review_59_76"
OUTPUT = HERE / "hoi_v4_59_76"
ORDER = ROOT / "retarget/ground_support_27_end/sidecars/motion_order.json"
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
             "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
FPS = 50
LEG = tuple("left_" + name + "_joint" for name in (
    "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
))


def frame(seconds: float) -> int:
    return round((seconds - 59.0) * FPS)


def mesh_points(model: mujoco.MjModel, name: str) -> tuple[int, np.ndarray]:
    body = model.body(name).id
    pieces = []
    for geom in range(model.ngeom):
        if (model.geom_bodyid[geom] != body or model.geom_group[geom] == 3
                or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = model.geom_dataid[geom]
        mesh = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pieces.append(Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).apply(mesh) + model.geom_pos[geom])
    return body, np.concatenate(pieces)


def main() -> None:
    source = MotionFields.load(INPUT / "motion.npz", ORDER)
    if len(source.joint_pos) != 850 or source.fps != FPS:
        raise ValueError("expected 59–76 s at 50 Hz")
    with np.load(ROOT / "retarget/body_prior_27_end/sidecars/human.npz", allow_pickle=False) as data:
        begin = round((59.0 - 27.0) * FPS)
        human_rotation = data["rotations_w"][begin:begin + len(source.joint_pos)]
    with np.load(INPUT / "human_motion_world.npz", allow_pickle=False) as data:
        human_names = list(map(str, data["joint_names"]))
    sole = human_names.index("left_sole")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(name).qposadr[0] for name in source.joint_names])
    leg_cols = np.asarray([source.joint_names.index(name) for name in LEG])
    leg_addr = addresses[leg_cols]
    lower = np.asarray([model.joint(name).range[0] for name in LEG]) + 1e-5
    upper = np.asarray([model.joint(name).range[1] for name in LEG]) - 1e-5
    foot, foot_mesh = mesh_points(model, "left_ankle_roll_link")
    knee = model.body("left_knee_link").id

    def pose(k: int, leg: np.ndarray | None = None) -> None:
        data.qpos[:3] = source.body_pos_w[k, 0]
        data.qpos[3:7] = source.body_quat_w[k, 0]
        data.qpos[addresses] = source.joint_pos[k]
        if leg is not None:
            data.qpos[leg_addr] = leg
        mujoco.mj_forward(model, data)

    pose(frame(60.0))
    robot_rest_normal = data.xmat[foot].reshape(3, 3)[:, 2].copy()
    human_rest_rotation = human_rotation[frame(60.0), sole].copy()
    desired_normals = np.einsum(
        "tij,j->ti", human_rotation[:, sole] @ human_rest_rotation.T, robot_rest_normal,
    )
    desired_normals = gaussian_filter1d(desired_normals, sigma=4, axis=0)
    desired_normals /= np.linalg.norm(desired_normals, axis=1, keepdims=True)
    joint = source.joint_pos.astype(np.float64).copy()
    prior = MotionFields.load(HERE / "body_prior.npz", ORDER)
    arm_names = tuple("right_" + name + "_joint" for name in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    ))
    arm_cols = np.asarray([source.joint_names.index(name) for name in arm_names])
    arm_addr = addresses[arm_cols]
    joint[frame(69.0):, arm_cols] = prior.joint_pos[frame(69.0):, arm_cols]
    key_times = np.round(np.arange(60.6, 64.201, 0.1), 4)
    key_frames = np.asarray([frame(float(t)) for t in key_times])
    corrections = []
    fit_log = []
    previous_delta = None
    for t, k in zip(key_times, key_frames, strict=True):
        pose(int(k))
        baseline = source.joint_pos[k, leg_cols].astype(np.float64)
        target_ankle = data.xpos[foot].copy()
        target_knee = data.xpos[knee].copy()
        target_normal = desired_normals[k]

        def residual(q: np.ndarray) -> np.ndarray:
            data.qpos[leg_addr] = q
            mujoco.mj_forward(model, data)
            normal = data.xmat[foot].reshape(3, 3)[:, 2]
            world_foot = foot_mesh @ data.xmat[foot].reshape(3, 3).T + data.xpos[foot]
            floor_gap = min(float(world_foot[:, 2].min()) - 0.001, 0.0)
            return np.r_[
                15 * (normal - target_normal),
                40 * (data.xpos[foot] - target_ankle),
                12 * (data.xpos[knee] - target_knee),
                180 * floor_gap,
                .6 * (q[:4] - baseline[:4]), .2 * (q[4:] - baseline[4:]),
                3 * (q[2] - baseline[2]),
                3 * (q - baseline - previous_delta) if previous_delta is not None else np.zeros(6),
            ]

        seed = np.clip(baseline + (previous_delta if previous_delta is not None else 0), lower + 1e-4, upper - 1e-4)
        result = least_squares(residual, seed, bounds=(lower, upper), max_nfev=110)
        residual(result.x)
        world_foot = foot_mesh @ data.xmat[foot].reshape(3, 3).T + data.xpos[foot]
        normal = data.xmat[foot].reshape(3, 3)[:, 2]
        corrections.append(result.x - baseline)
        previous_delta = result.x - baseline
        fit_log.append({
            "time_s": float(t), "normal_error_deg": float(np.rad2deg(np.arccos(
                np.clip(np.dot(normal, target_normal), -1, 1)))),
            "ankle_displacement_m": float(np.linalg.norm(data.xpos[foot] - target_ankle)),
            "foot_floor_min_m": float(world_foot[:, 2].min()),
        })

    # Taper at both ends; the 59–60 s stance and the later knee support stay intact.
    anchors = np.r_[frame(60.35), key_frames, frame(64.5)]
    values = np.vstack((np.zeros(6), np.asarray(corrections), np.zeros(6)))
    active = np.arange(anchors[0], anchors[-1] + 1)
    joint[np.ix_(active, leg_cols)] += PchipInterpolator(anchors, values, axis=0)(active)
    joint[np.ix_(active, leg_cols)] = np.clip(joint[np.ix_(active, leg_cols)], lower, upper)
    # The source video shows the basket stationary near the table until the
    # sole reaches its outer wall, then moving outward continuously. Fit the
    # centre to both contact surfaces instead of interpolating guessed XY.
    with np.load(INPUT / "object_estimate.npz", allow_pickle=False) as previous_object:
        object_position = previous_object["position_w"].astype(np.float64).copy()
        object_quaternion = previous_object["quaternion_xyzw"].copy()
    shin, shin_mesh = mesh_points(model, "left_knee_link")

    def wall_radius(z: np.ndarray) -> np.ndarray:
        return np.interp(z, [0, .1, .3, .4], [.105, .14, .165, .195])

    basket_knots = np.round(np.arange(61.0, 63.701, 0.1), 4)
    basket_centres = []
    basket_contact_log = []
    previous_center = np.asarray([.13, -2.13])
    previous_ankle = None
    for t in basket_knots:
        k = frame(float(t))
        pose(k, joint[k, leg_cols])
        ankle = data.xpos[foot, :2].copy()
        surfaces = []
        for body, mesh in ((foot, foot_mesh), (shin, shin_mesh)):
            world = mesh @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            world = world[(world[:, 2] > .07) & (world[:, 2] < .35)][::5]
            if not len(world):
                raise ValueError(f"no contact surface at {t:.2f} s")
            surfaces.append(world)

        expected = previous_center if previous_ankle is None else previous_center + .75 * (ankle - previous_ankle)

        def contact_gaps(center: np.ndarray) -> np.ndarray:
            return np.asarray([
                np.min(np.linalg.norm(surface[:, :2] - center, axis=1) - wall_radius(surface[:, 2]))
                for surface in surfaces
            ])

        def centre_residual(center: np.ndarray) -> np.ndarray:
            gaps = contact_gaps(center)
            return np.r_[60 * gaps, 120 * np.minimum(gaps + .002, 0), 2 * (center - expected)]

        lo = np.asarray([-.12, max(-2.3, previous_center[1] - .0005)])
        hi = np.asarray([min(.16, previous_center[0] + .0005), -1.75])
        fitted = least_squares(centre_residual, np.clip(expected, lo + 1e-5, hi - 1e-5),
                               bounds=(lo, hi), max_nfev=90)
        centre = fitted.x
        basket_centres.append(centre)
        basket_contact_log.append({"time_s": float(t), "foot_gap_m": float(contact_gaps(centre)[0]),
                                   "shin_gap_m": float(contact_gaps(centre)[1])})
        previous_center, previous_ankle = centre, ankle
    basket_centres = np.asarray(basket_centres)
    # The foot withdraws at about 64 s. Hold the extracted position until the
    # kneeling forearm return begins; shift the old return path by this offset.
    endpoint = basket_centres[-1]
    times = np.r_[59.0, 60.7, basket_knots, 64.1, 69.0]
    centres = np.vstack((
        basket_centres[0], basket_centres[0], basket_centres, endpoint, endpoint,
    ))
    before = np.arange(frame(69.0))
    object_position[before, :2] = PchipInterpolator(times, centres, axis=0)(59.0 + before / FPS)
    object_position[before, 2] = 0.001
    object_position[frame(69.0):, :2] += endpoint - np.asarray([.25, -1.78])

    # Refit the right arm to the corrected basket path. Preserve the captured
    # root and the left-knee support; no finger joints enter this solve.
    elbow = model.body("right_elbow_link").id
    wrist = model.body("right_wrist_yaw_link").id
    arm_lower = np.asarray([model.joint(name).range[0] for name in arm_names]) + 1e-5
    arm_upper = np.asarray([model.joint(name).range[1] for name in arm_names]) - 1e-5
    arm_lower[4:] = np.maximum(arm_lower[4:], -np.deg2rad(35))
    arm_upper[4:] = np.minimum(arm_upper[4:], np.deg2rad(35))
    arm_times = np.round(np.arange(72.0, 73.801, .2), 4)
    arm_frames = np.asarray([frame(float(t)) for t in arm_times])
    arm_offsets = []
    arm_log = []
    previous_arm_delta = None
    for t, k in zip(arm_times, arm_frames, strict=True):
        pose(int(k))
        baseline = prior.joint_pos[k, arm_cols].astype(np.float64)
        centre = object_position[k]
        wrist_target = np.asarray([centre[0] - .198, centre[1] + .098, .289])
        elbow_target = np.asarray([centre[0] - .068, centre[1] + .198, .390])

        def arm_residual(q: np.ndarray) -> np.ndarray:
            data.qpos[arm_addr] = q
            mujoco.mj_forward(model, data)
            return np.r_[
                100 * (data.xpos[wrist] - wrist_target),
                75 * (data.xpos[elbow] - elbow_target),
                .4 * (q[:4] - baseline[:4]), .8 * q[4:],
                .3 * (q - baseline - previous_arm_delta) if previous_arm_delta is not None else np.zeros(7),
            ]

        seed = np.clip(baseline + (previous_arm_delta if previous_arm_delta is not None else 0),
                       arm_lower + 1e-4, arm_upper - 1e-4)
        solved = least_squares(arm_residual, seed, bounds=(arm_lower, arm_upper), max_nfev=150)
        arm_residual(solved.x)
        previous_arm_delta = solved.x - baseline
        arm_offsets.append(previous_arm_delta)
        arm_log.append({"time_s": float(t),
                        "wrist_error_m": float(np.linalg.norm(data.xpos[wrist] - wrist_target)),
                        "elbow_error_m": float(np.linalg.norm(data.xpos[elbow] - elbow_target))})
    arm_anchors = np.r_[frame(71.2), arm_frames, frame(74.4), frame(75.0)]
    arm_values = np.vstack((np.zeros(7), np.asarray(arm_offsets), np.zeros((2, 7))))
    active_arm = np.arange(arm_anchors[0], arm_anchors[-1] + 1)
    joint[np.ix_(active_arm, arm_cols)] += PchipInterpolator(arm_anchors, arm_values, axis=0)(active_arm)
    joint[np.ix_(active_arm, arm_cols)] = np.clip(joint[np.ix_(active_arm, arm_cols)], arm_lower, arm_upper)

    # The 67 s hand-to-basket event anchors the outbound endpoint. The
    # strawberry is handled by the overlapping task; here we only ensure the
    # robot's left hand can reach over the basket opening at the shared time.
    left_arm_names = tuple(name.replace("right_", "left_", 1) for name in arm_names)
    left_cols = np.asarray([source.joint_names.index(name) for name in left_arm_names])
    left_addr = addresses[left_cols]
    left_lower = np.asarray([model.joint(name).range[0] for name in left_arm_names]) + 1e-5
    left_upper = np.asarray([model.joint(name).range[1] for name in left_arm_names]) - 1e-5
    left_lower[4:] = np.maximum(left_lower[4:], -np.deg2rad(35))
    left_upper[4:] = np.minimum(left_upper[4:], np.deg2rad(35))
    left_index = model.body("left_hand_index_1_link").id
    deposit_times = np.round(np.arange(66.4, 67.101, .1), 4)
    deposit_frames = np.asarray([frame(float(t)) for t in deposit_times])
    deposit_offsets = []
    deposit_log = []
    last_delta = None
    for t, k in zip(deposit_times, deposit_frames, strict=True):
        pose(int(k))
        baseline = source.joint_pos[k, left_cols].astype(np.float64)
        centre = object_position[k]
        target = np.asarray([centre[0], centre[1] + .09, .48])

        def deposit_residual(q: np.ndarray) -> np.ndarray:
            data.qpos[left_addr] = q
            mujoco.mj_forward(model, data)
            return np.r_[
                100 * (data.xpos[left_index] - target),
                .8 * (q - baseline),
                .25 * (q - baseline - last_delta) if last_delta is not None else np.zeros(7),
            ]

        seed = np.clip(baseline + (last_delta if last_delta is not None else 0),
                       left_lower + 1e-4, left_upper - 1e-4)
        fitted = least_squares(deposit_residual, seed, bounds=(left_lower, left_upper), max_nfev=150)
        deposit_residual(fitted.x)
        last_delta = fitted.x - baseline
        deposit_offsets.append(last_delta)
        deposit_log.append({"time_s": float(t), "index_target_error_m": float(np.linalg.norm(
            data.xpos[left_index] - target))})
    deposit_anchors = np.r_[frame(65.8), deposit_frames, frame(67.6), frame(68.0)]
    deposit_values = np.vstack((np.zeros(7), np.asarray(deposit_offsets), np.zeros((2, 7))))
    active_deposit = np.arange(deposit_anchors[0], deposit_anchors[-1] + 1)
    joint[np.ix_(active_deposit, left_cols)] += PchipInterpolator(
        deposit_anchors, deposit_values, axis=0,
    )(active_deposit)
    joint[np.ix_(active_deposit, left_cols)] = np.clip(
        joint[np.ix_(active_deposit, left_cols)], left_lower, left_upper,
    )

    candidate = refresh_fk(replace(source, joint_pos=joint), MODEL)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate.save(OUTPUT / "motion.npz")
    np.savez_compressed(
        OUTPUT / "object_estimate.npz", position_w=object_position.astype(np.float32),
        quaternion_xyzw=object_quaternion.astype(np.float32), fps=np.asarray(FPS),
        status=np.asarray("video_timed_foot_conditioned_ground_supported_candidate"),
    )

    # A dense scan catches the floor crossings missed by knot-only checking.
    floor_min = []
    normal_errors = []
    for k in range(frame(60.6), frame(64.21)):
        pose(k, joint[k, leg_cols])
        world_foot = foot_mesh @ data.xmat[foot].reshape(3, 3).T + data.xpos[foot]
        floor_min.append(float(world_foot[:, 2].min()))
        desired = desired_normals[k]
        actual = data.xmat[foot].reshape(3, 3)[:, 2]
        normal_errors.append(float(np.rad2deg(np.arccos(np.clip(np.dot(actual, desired), -1, 1)))))
    dense_contact = []
    force_bracket = []
    for k in range(frame(61.0), frame(63.7) + 1):
        pose(k, joint[k, leg_cols])
        center = object_position[k, :2]
        contacts = []
        for body, mesh in ((foot, foot_mesh), (shin, shin_mesh)):
            world = mesh @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            world = world[(world[:, 2] > .07) & (world[:, 2] < .35)]
            signed = np.linalg.norm(world[:, :2] - center, axis=1) - wall_radius(world[:, 2])
            closest = int(np.argmin(np.abs(signed)))
            contacts.append((float(np.min(signed)), world[closest, :2]))
        dense_contact.append((59 + k / FPS, contacts[0][0], contacts[1][0]))
        travel = object_position[min(k + 3, len(object_position) - 1), :2] - object_position[max(k - 3, 0), :2]
        if np.linalg.norm(travel) > .0036 and max(abs(contacts[0][0]), abs(contacts[1][0])) < .015:
            cross = [np.linalg.det(np.stack((center - point, travel))) for _, point in contacts]
            force_bracket.append(cross[0] * cross[1] <= 0)
    report = {
        "status": "stage35_candidate_pending_user_review",
        "source_window_s": [59, 76],
        "foot_orientation_source": "human left_sole relative rotation, neutral-calibrated at 60 s",
        "foot_contact_window_estimated_s": [61.0, 63.7],
        "foot_knot_normal_error_deg_max": max(item["normal_error_deg"] for item in fit_log),
        "foot_dense_normal_error_deg_median": float(np.median(normal_errors)),
        "foot_dense_floor_min_m": min(floor_min),
        "max_left_leg_joint_speed_rad_s": float(np.max(np.abs(np.diff(joint[:, leg_cols], axis=0)) * FPS)),
        "basket_path_status": "foot/shin contact-conditioned estimate; source video not metrically tracked",
        "basket_start_xyz_m": object_position[0].tolist(),
        "basket_out_xyz_m": object_position[frame(64.5)].tolist(),
        "basket_return_xyz_m": object_position[frame(73.8)].tolist(),
        "foot_shin_contact_gap_max_abs_m": float(np.max(np.abs([
            [item["foot_gap_m"], item["shin_gap_m"]] for item in basket_contact_log
        ]))),
        "foot_shin_dense_50hz_gap_p95_abs_m": float(np.percentile(np.abs(np.asarray(dense_contact)[:, 1:]), 95)),
        "foot_shin_dense_50hz_max_radial_penetration_m": float(max(0, -np.min(np.asarray(dense_contact)[:, 1:]))),
        "foot_shin_force_direction_bracket_fraction": float(np.mean(force_bracket)) if force_bracket else None,
        "foot_shin_force_direction_bracket_samples": len(force_bracket),
        "arm_wrist_error_max_m": max(item["wrist_error_m"] for item in arm_log),
        "arm_elbow_error_max_m": max(item["elbow_error_m"] for item in arm_log),
        "left_hand_deposit_window_s": [66.4, 67.1],
        "left_hand_index_target_error_max_m": max(item["index_target_error_m"] for item in deposit_log),
        "basket_contact_log": basket_contact_log,
        "dense_contact_log": dense_contact,
        "arm_fit_log": arm_log,
        "deposit_fit_log": deposit_log,
        "fit_log": fit_log,
    }
    (OUTPUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in (
        "fit_log", "basket_contact_log", "dense_contact_log", "arm_fit_log", "deposit_fit_log",
    )}, indent=2))


if __name__ == "__main__":
    main()
