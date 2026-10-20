"""Fit the left Dex3 L-shaped push to the 1019 waste-bin body prior.

Run with ScaleTrack's Python and ``PYTHONPATH=ScaleTrack/retarget3/src``.
This is a kinematic contact candidate; the bin sidecar is an estimate, not a
measured or dynamically simulated trajectory.
"""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from retarget3.ground_stage import mesh_points
from retarget3.motion_io import export_motion
from retarget3.robot import MJCasadiRobot, mujoco
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


ROOT = example_dir(__file__)
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
FPS = 50
KEYS = np.arange(215, 401, 5)
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1")
CONTACT_LINKS = ("left_hand_thumb_2_link", "left_hand_index_1_link")
GUARD_LINKS = (
    "left_wrist_yaw_link", "left_hand_thumb_0_link", "left_hand_thumb_1_link",
    "left_hand_thumb_2_link", "left_hand_index_0_link", "left_hand_index_1_link",
    "left_hand_middle_0_link", "left_hand_middle_1_link",
)


def wall_radius(z):
    """Outer radius from the actual USD mesh rings: 82 mm bottom, 102 mm rim."""
    return 0.082 + 0.020 * np.clip(z, 0, 0.28) / 0.28


def push_directions(bin_position):
    """Smooth travel tangent, extended across the initial/final static frames."""
    directions = np.zeros((len(bin_position), 2))
    for frame in range(len(bin_position)):
        middle = np.clip(frame, 235, 365)
        travel = bin_position[middle + 12, :2] - bin_position[middle - 12, :2]
        directions[frame] = travel / np.linalg.norm(travel)
    return directions


def contact_targets(center, direction):
    # The thumb and index straddle the travel line on the trailing exterior.
    radius = wall_radius(0.18) + 0.002
    lateral = 0.034
    trailing = -direction * np.sqrt(radius * radius - lateral * lateral)
    tangent = np.array([-direction[1], direction[0]])
    xy = center[:2] + trailing + np.array([-lateral * tangent, lateral * tangent])
    return np.column_stack((xy, np.full(2, center[2] + 0.18)))


def main():
    with np.load(ROOT / "retarget/body_prior_scene.npz") as source:
        motion = {key: source[key] for key in source.files}
    with np.load(ROOT / "sidecars/white_waste_bin_estimated.npz") as source:
        bin_position = source["position_w"]
    names = json.loads(ORDER.read_text())["robot_joint_names"]
    robot = MJCasadiRobot()
    model, data = robot.m, robot.d
    q = np.zeros((len(motion["joint_pos"]), model.nq))
    q[:, :3] = motion["body_pos_w"][:, 0]
    q[:, 3:7] = motion["body_quat_w"][:, 0][:, [1, 2, 3, 0]]
    for j, name in enumerate(names):
        q[:, robot.dof_name_to_index[name]] = motion["joint_pos"][:, j]
    if q.shape != (474, 50) or bin_position.shape != (474, 3):
        raise ValueError("expected the reviewed 474-frame body prior and bin sidecar")
    directions = push_directions(bin_position)

    edited = [f"left_{name}_joint" for name in ARM] + [f"left_hand_{name}_joint" for name in HAND]
    ids = np.array([robot.dof_name_to_index[name] for name in edited])
    vertices = {name: mesh_points(model, name) for name in GUARD_LINKS}
    body_ids = {name: model.body(name).id for name in GUARD_LINKS}

    def apply(row):
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]
        data.qpos[7:] = row[7:]
        mujoco.mj_forward(model, data)

    def world_vertices(name, points):
        body_id = body_ids[name]
        return points @ data.xmat[body_id].reshape(3, 3).T + data.xpos[body_id]

    # A mesh vertex on each distal finger is tracked through the temporal solve.
    representative = 275
    apply(q[representative])
    targets = contact_targets(bin_position[representative], directions[representative])
    contact_vertex = {
        name: vertices[name][np.argmin(np.linalg.norm(world_vertices(name, vertices[name]) - target, axis=1))]
        for name, target in zip(CONTACT_LINKS, targets, strict=True)
    }
    last_delta = None
    key_deltas = []
    fit_log = []
    for frame in KEYS:
        center = bin_position[frame]
        target = contact_targets(center, directions[frame])
        base = q[frame].copy()
        original = base[ids].copy()
        lower = robot.q_lower[ids].copy()
        upper = robot.q_upper[ids].copy()
        arm_bound = np.array([0.7, 0.7, 0.7, 1.0, 0.5, 0.5, 0.5])
        lower[:7] = np.maximum(lower[:7], original[:7] - arm_bound)
        upper[:7] = np.minimum(upper[:7], original[:7] + arm_bound)
        upper[9] = min(upper[9], 0.7)  # keep the distal thumb open

        def residual(x):
            row = base.copy()
            row[ids] = x
            apply(row)
            points = np.array([world_vertices(name, point[None])[0] for name, point in contact_vertex.items()])
            terms = list(((points - target) * 50).ravel())
            terms.extend((x[:4] - original[:4]) * 0.25)
            terms.extend((x[4:7] - original[4:7]) * np.array([0.4, 0.6, 0.5]))
            terms.extend((x[7:] - original[7:]) * 0.15)
            if last_delta is not None:
                terms.extend((x - original - last_delta) * 0.45)
            hand_pos = lambda part: data.xpos[model.body(f"left_hand_{part}_link").id]
            thumb_axis = hand_pos("thumb_2") - hand_pos("thumb_0")
            index_axis = hand_pos("index_1") - hand_pos("index_0")
            cosine = np.dot(thumb_axis, index_axis) / (np.linalg.norm(thumb_axis) * np.linalg.norm(index_axis))
            angle = np.arccos(np.clip(cosine, -1, 1))
            terms.append(4 * max(0, np.deg2rad(75) - angle))
            for name in GUARD_LINKS:
                points = world_vertices(name, vertices[name])
                active = (points[:, 2] > 0.015) & (points[:, 2] < 0.276)
                gap = np.linalg.norm(points[:, :2] - center[:2], axis=1) - wall_radius(points[:, 2])
                clearance = 0.012 if name == "left_wrist_yaw_link" else -0.001
                terms.extend(np.where(active, np.minimum(gap - clearance, 0) * 100, 0))
            return np.asarray(terms)

        seed = np.clip(original + (last_delta if last_delta is not None else 0), lower + 1e-6, upper - 1e-6)
        solution = least_squares(
            residual, seed, bounds=(lower, upper), max_nfev=130,
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        last_delta = solution.x - original
        key_deltas.append(last_delta.copy())
        fit_log.append({"frame": int(frame), "success": bool(solution.success), "cost": float(solution.cost)})
    if not all(item["success"] for item in fit_log):
        raise RuntimeError("at least one contact keyframe did not converge; inspect fit_log")

    # Keep the open contact configuration while the original arm withdraws.
    # Blending directly from frame 400 to the body prior sweeps the index
    # through the bin wall around frame 410 despite safe endpoints.
    anchors = np.r_[130, 160, 185, KEYS, 420, 450, 465, 473]
    values = np.vstack((
        np.zeros((3, len(ids))), key_deltas, key_deltas[-1],
        np.zeros((3, len(ids))),
    ))
    correction = PchipInterpolator(anchors, values, axis=0)(np.arange(len(q)))
    correction[:130] = 0
    correction[465:] = 0
    candidate = q.copy()
    candidate[:, ids] += correction
    np.save(ROOT / "retarget/hoi_candidate_q.npy", candidate)
    np.savez_compressed(
        ROOT / "retarget/hoi_fit_keys.npz", frames=KEYS, delta=np.asarray(key_deltas),
        joint_names=np.asarray(edited), contact_vertices=np.asarray(list(contact_vertex.values())),
    )
    export_motion(ROOT / "retarget/hoi_candidate_scene.npz", robot, candidate, FPS)

    audit = {"keyframe_fit": fit_log, "contact_geometry": "tapered USD exterior: 82 mm bottom to 102 mm rim"}
    angles = []
    gaps = {name: [] for name in GUARD_LINKS}
    travel_alignment = []
    travel_between_fingers = []
    for frame, row in enumerate(candidate):
        apply(row)
        center = bin_position[frame]
        points = np.array([world_vertices(name, point[None])[0] for name, point in contact_vertex.items()])
        to_center = center[:2] - points[:, :2]
        to_center /= np.linalg.norm(to_center, axis=1)[:, None]
        push = directions[frame]
        midpoint_ray = center[:2] - points[:, :2].mean(axis=0)
        travel_alignment.append(float(np.dot(midpoint_ray / np.linalg.norm(midpoint_ray), push)))
        cross = to_center[:, 0] * push[1] - to_center[:, 1] * push[0]
        travel_between_fingers.append(bool(cross[0] * cross[1] <= 0))
        thumb_axis = data.xpos[model.body("left_hand_thumb_2_link").id] - data.xpos[model.body("left_hand_thumb_0_link").id]
        index_axis = data.xpos[model.body("left_hand_index_1_link").id] - data.xpos[model.body("left_hand_index_0_link").id]
        cosine = np.dot(thumb_axis, index_axis) / (np.linalg.norm(thumb_axis) * np.linalg.norm(index_axis))
        angles.append(float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))))
        for name in GUARD_LINKS:
            points = world_vertices(name, vertices[name])
            active = (points[:, 2] > 0.02) & (points[:, 2] < 0.275)
            if active.any():
                gap = np.linalg.norm(points[active, :2] - center[:2], axis=1) - wall_radius(points[active, 2])
                gaps[name].append(float(np.min(gap)))
            else:
                gaps[name].append(None)
    audit["contact_window"] = {
        "frames_inclusive": [215, 400],
        "thumb_index_angle_deg_range": [min(angles[215:401]), max(angles[215:401])],
        "mesh_min_gap_m": {
            name: min(value for value in series[215:401] if value is not None)
            for name, series in gaps.items()
        },
        "mesh_max_gap_m": {
            name: max(value for value in series[215:401] if value is not None)
            for name, series in gaps.items()
        },
    }
    audit["moving_push_direction"] = {
        "frames_inclusive": [235, 375],
        "contact_midpoint_direction_cosine_min": min(travel_alignment[235:376]),
        "contact_midpoint_direction_cosine_mean": float(np.mean(travel_alignment[235:376])),
        "motion_ray_between_thumb_and_index_frame_fraction": float(np.mean(travel_between_fingers[235:376])),
        "direction_source": "centered 24-frame displacement of estimated bin trajectory; endpoint tangent extended",
    }
    audit["approach_and_withdrawal"] = {
        "frames_half_open": [130, 465],
        "mesh_min_gap_m": {
            name: min(value for value in series[130:465] if value is not None)
            for name, series in gaps.items()
        },
        "first_thumb_within_2mm_frame": next(
            (frame for frame, value in enumerate(gaps["left_hand_thumb_2_link"]) if value is not None and value <= 0.002),
            None,
        ),
        "first_index_within_2mm_frame": next(
            (frame for frame, value in enumerate(gaps["left_hand_index_1_link"]) if value is not None and value <= 0.002),
            None,
        ),
    }
    audit["max_added_joint_velocity_rad_s"] = float(np.max(np.abs(np.diff(correction, axis=0)) * FPS))
    audit["max_added_joint_acceleration_rad_s2"] = float(np.max(np.abs(np.diff(correction, n=2, axis=0)) * FPS**2))
    edited_motion = candidate[:, ids]
    audit["max_joint_velocity_ratio"] = float(np.max(np.abs(np.diff(edited_motion, axis=0)) * FPS / robot.v_abs[ids - 1]))
    audit["joint_limit_violation_rad"] = float(max(
        np.maximum(robot.q_lower[ids] - edited_motion, 0).max(),
        np.maximum(edited_motion - robot.q_upper[ids], 0).max(),
    ))
    audit["unchanged_root_legs_torso_right_arm"] = bool(np.array_equal(candidate[:, np.setdiff1d(np.arange(50), ids)], q[:, np.setdiff1d(np.arange(50), ids)]))
    (ROOT / "review/hoi_fit_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"frames": len(q), "contact_window": audit["contact_window"], "max_added_joint_velocity_rad_s": audit["max_added_joint_velocity_rad_s"]}, indent=2))


if __name__ == "__main__":
    main()
