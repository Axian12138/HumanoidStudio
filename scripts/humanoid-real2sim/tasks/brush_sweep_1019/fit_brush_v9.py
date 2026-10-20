"""Build an unreviewed right-hand pickup, sweep and return candidate.

The source body prior supplies the locomotion and sweeping arm path. Only the
right arm near the two handle contacts and the commanded Dex3 fingers change.
The brush sidecar is a rigid kinematic estimate, never a contact measurement.
"""

from dataclasses import replace
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256
from scene import BRUSH_POSITION, BRUSH_QUAT_XYZW


ROOT = example_dir(__file__)
PRIOR = ROOT / "retarget/body_prior_8_28/motion.npz"
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
FIT = Path("/home/galbot/workspace/ScaleTrack/data/1012test/1012-1/stage4-brush/review/handle_fit.json")
OUT = ROOT / "retarget/brush_sweep_kinematic_v9"
START_S = 8.0
FPS = 50
ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")


def frame(t):
    return round((t - START_S) * FPS)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def main():
    baseline = MotionFields.load(PRIOR, ORDER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    fit = json.loads(FIT.read_text())
    center = np.asarray(fit["center_in_wrist_m"])
    brush_rot = Rotation.from_quat(BRUSH_QUAT_XYZW)
    grip = np.asarray(BRUSH_POSITION) + brush_rot.apply(np.asarray(fit["object_grip_local_m"]))
    axis = brush_rot.apply([1.0, 0.0, 0.0])
    joint = baseline.joint_pos.astype(float).copy()
    arm_cols = np.asarray([baseline.joint_names.index("right_" + part + "_joint") for part in ARM_PARTS])
    finger_cols = np.asarray([baseline.joint_names.index("right_hand_" + part + "_joint") for part in FINGERS])
    addresses = np.asarray([model.joint(name).qposadr[0] for name in baseline.joint_names])
    arm_addresses = addresses[arm_cols]
    lower = np.asarray([model.joint(baseline.joint_names[j]).range[0] for j in arm_cols]) + 1e-6
    upper = np.asarray([model.joint(baseline.joint_names[j]).range[1] for j in arm_cols]) - 1e-6
    wrist = model.body("right_wrist_yaw_link").id
    palm_normal_target = np.array([0.0, 0.0, -1.0])
    # The Dex3 fingers bend toward wrist-local +Y; that is the palm side.
    # Their grasp point must approach the handle along that face, not the back.
    wrist_limit = np.deg2rad(35.0)
    lower[4:] = np.maximum(lower[4:], -wrist_limit)
    upper[4:] = np.minimum(upper[4:], wrist_limit)
    open_pose = np.asarray(fit["closure_angles"][0])
    hand_meshes = []
    for bid in range(model.nbody):
        if not model.body(bid).name.startswith("right_hand_"):
            continue
        meshes = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != bid or model.geom_group[geom] == 3 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[geom]
            points = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            meshes.append(Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).apply(points) + model.geom_pos[geom])
        if meshes:
            vertices = np.unique(np.concatenate(meshes), axis=0)
            hand_meshes.append((bid, vertices[ConvexHull(vertices).vertices]))
    windows = [
        [(11.1, None), (11.45, 0.15), (11.8, 0.08), (12.15, 0.015),
         (12.4, 0.0), (12.6, 0.0), (12.8, 0.0), (13.0, 0.02),
         (13.4, 0.06), (14.3, None)],
        [(24.0, None), (24.5, 0.14), (24.9, 0.08), (25.2, 0.025),
         (25.45, 0.0), (25.65, 0.0), (25.85, 0.0), (26.2, 0.06),
         (26.8, None)],
    ]
    residuals = []
    pick_wrist_rotation = None
    for window_number, keys in enumerate(windows):
        solved = []
        last = None
        for seconds, clearance in keys:
            k = frame(seconds)
            prior = baseline.joint_pos[k, arm_cols]
            if clearance is None:
                solved.append((k, np.zeros(len(arm_cols))))
                last = None
                continue
            data.qpos[:3] = baseline.body_pos_w[k, 0]
            data.qpos[3:7] = baseline.body_quat_w[k, 0]
            data.qpos[addresses] = baseline.joint_pos[k]
            data.qpos[addresses[finger_cols]] = open_pose
            target = grip + np.array([0.0, 0.0, clearance])

            def pose(q):
                data.qpos[arm_addresses] = q
                mujoco.mj_forward(model, data)
                rotation = data.xmat[wrist].reshape(3, 3)
                return data.xpos[wrist] + rotation @ center, rotation[:, 2], rotation[:, 1]

            def residual(q):
                eye, direction, palm_normal = pose(q)
                rotation = Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))
                orientation_error = (
                    (pick_wrist_rotation.inv() * rotation).as_rotvec()
                    if pick_wrist_rotation is not None else np.zeros(3)
                )
                return np.r_[60.0 * (eye - target), 4.0 * (direction - axis),
                             7.0 * (palm_normal - palm_normal_target),
                             0.8 * orientation_error,
                             0.20 * (q[:4] - prior[:4]),
                             0.8 * (q[4:] - np.clip(prior[4:], -wrist_limit, wrist_limit))]

            seeds = [np.clip(prior, lower, upper)]
            if last is not None:
                seeds.insert(0, np.clip(last, lower, upper))
            results = [least_squares(residual, seed, bounds=(lower, upper), max_nfev=140) for seed in seeds]
            result = min(results, key=lambda item: np.linalg.norm(item.fun))
            eye, direction, palm_normal = pose(result.x)
            solved.append((k, result.x - prior))
            residuals.append({"source_time_s": seconds, "fist_eye_error_m": float(np.linalg.norm(eye - target)),
                              "handle_axis_error_deg": float(np.degrees(np.arccos(np.clip(direction @ axis, -1, 1)))),
                              "palm_to_handle_error_deg": float(np.degrees(np.arccos(np.clip(palm_normal @ palm_normal_target, -1, 1)))),
                              "wrist_axes_rad": result.x[4:].tolist()})
            last = result.x
        times = np.asarray([item[0] for item in solved])
        offsets = np.asarray([item[1] for item in solved])
        indices = np.arange(times[0], times[-1] + 1)
        taper = smooth((indices - times[0]) / 20.0) * smooth((times[-1] - indices) / 20.0)
        joint[np.ix_(indices, arm_cols)] += PchipInterpolator(times, offsets, axis=0)(indices) * taper[:, None]
        if window_number == 0:
            k = frame(12.8)
            data.qpos[:3] = baseline.body_pos_w[k, 0]
            data.qpos[3:7] = baseline.body_quat_w[k, 0]
            data.qpos[addresses] = joint[k]
            mujoco.mj_forward(model, data)
            pick_wrist_rotation = Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))
            pick_wrist_position = data.xpos[wrist].copy()
            local_brush_position = pick_wrist_rotation.inv().apply(np.asarray(BRUSH_POSITION) - pick_wrist_position)
            local_brush_rotation = pick_wrist_rotation.inv() * brush_rot

            bristles_dir = ROOT.parent / "brush_pick_1012/assets/green_hand_brush/collision_hulls"
            bristle_vertices = np.concatenate([
                trimesh.load(path, force="mesh").vertices
                for path in sorted(bristles_dir.glob("Hull_bristles*.obj"))
            ])
            sweep_keys = [
                (15.2, None), (15.7, 0.09), (15.9, 0.05),
                (16.0, 0.035), (16.1, 0.025), (16.2, 0.020),
                (16.45, 0.020), (16.7, 0.018), (17.0, 0.015),
                (17.3, 0.015), (17.6, 0.015), (17.9, 0.015),
                (18.2, 0.020), (18.7, 0.045), (19.2, 0.045),
                (19.6, 0.020), (20.0, 0.015), (20.4, 0.015),
                (20.7, 0.020), (21.2, 0.07), (21.8, None),
            ]
            sweep_solved = []
            last_sweep = None
            for seconds, clearance in sweep_keys:
                k = frame(seconds)
                prior = baseline.joint_pos[k, arm_cols]
                if clearance is None:
                    sweep_solved.append((k, np.zeros(len(arm_cols))))
                    last_sweep = None
                    continue
                data.qpos[:3] = baseline.body_pos_w[k, 0]
                data.qpos[3:7] = baseline.body_quat_w[k, 0]
                data.qpos[addresses] = baseline.joint_pos[k]
                mujoco.mj_forward(model, data)
                prior_wrist_pos = data.xpos[wrist].copy()
                prior_wrist_rot = Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))

                def sweep_residual(q):
                    data.qpos[arm_addresses] = q
                    mujoco.mj_forward(model, data)
                    wrist_pos = data.xpos[wrist]
                    wrist_rot = Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))
                    brush_pos = wrist_pos + wrist_rot.apply(local_brush_position)
                    brush_rotation = wrist_rot * local_brush_rotation
                    world = brush_rotation.apply(bristle_vertices) + brush_pos
                    min_z = np.min(world[:, 2])
                    bristle_center_x = np.mean(world[:, 0])
                    deepest_hand = 0.0
                    for bid, local in hand_meshes:
                        points = local @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
                        inside = ((points[:, 0] >= 0.76) & (points[:, 0] <= 1.36)
                                  & (points[:, 1] >= -0.71) & (points[:, 1] <= 0.69))
                        if inside.any():
                            deepest_hand = max(deepest_hand, float(0.420 - np.min(points[inside, 2])))
                    return np.r_[70.0 * (min_z - (0.401 + clearance)),
                                 30.0 * (bristle_center_x - 0.85),
                                 100.0 * max(deepest_hand, 0.0),
                                 5.0 * (wrist_pos[1] - prior_wrist_pos[1]),
                                 0.4 * (prior_wrist_rot.inv() * wrist_rot).as_rotvec(),
                                 0.13 * (q - prior)]

                seeds = [np.clip(prior, lower, upper)]
                if last_sweep is not None:
                    seeds.insert(0, np.clip(last_sweep, lower, upper))
                results = [least_squares(sweep_residual, seed, bounds=(lower, upper), max_nfev=120) for seed in seeds]
                result = min(results, key=lambda item: np.linalg.norm(item.fun))
                sweep_solved.append((k, result.x - prior))
                last_sweep = result.x
            sweep_frames = np.asarray([item[0] for item in sweep_solved])
            sweep_offsets = np.asarray([item[1] for item in sweep_solved])
            sweep_indices = np.arange(sweep_frames[0], sweep_frames[-1] + 1)
            sweep_taper = smooth((sweep_indices - sweep_frames[0]) / 20.0) * smooth((sweep_frames[-1] - sweep_indices) / 20.0)
            joint[np.ix_(sweep_indices, arm_cols)] += PchipInterpolator(sweep_frames, sweep_offsets, axis=0)(sweep_indices) * sweep_taper[:, None]

    closed_pose = np.asarray(fit["angles"]).copy()
    # Refit closure on the exact offsethand model used in the RSS editor.
    # This is mesh contact geometry only; force and retention need playback.
    contact_frame = frame(12.8)
    data.qpos[:3] = baseline.body_pos_w[contact_frame, 0]
    data.qpos[3:7] = baseline.body_quat_w[contact_frame, 0]
    data.qpos[addresses] = baseline.joint_pos[contact_frame]
    data.qpos[arm_addresses] = joint[contact_frame, arm_cols]
    grip_mesh = trimesh.load(ROOT.parent / "brush_pick_1012/assets/green_hand_brush/collision_hulls/Hull_grip.obj", force="mesh")
    grip_mesh.apply_transform(trimesh.transformations.quaternion_matrix(
        np.r_[BRUSH_QUAT_XYZW[3], BRUSH_QUAT_XYZW[:3]]))
    grip_mesh.apply_translation(BRUSH_POSITION)
    closure_gaps = {}
    for parts, distal in (((0, 1, 2), "thumb_2"), ((3, 4), "middle_1"), ((5, 6), "index_1")):
        finger_addresses = addresses[finger_cols[list(parts)]]
        finger_body = model.body("right_hand_" + distal + "_link").id
        vertices = next(vertices for bid, vertices in hand_meshes if bid == finger_body)
        finger_bounds = [tuple(model.joint(baseline.joint_names[finger_cols[p]]).range) for p in parts]
        seed = closed_pose[list(parts)]

        def signed_gap(angles):
            data.qpos[finger_addresses] = angles
            mujoco.mj_forward(model, data)
            world = vertices @ data.xmat[finger_body].reshape(3, 3).T + data.xpos[finger_body]
            return float(trimesh.proximity.signed_distance(grip_mesh, world).max())

        def finger_objective(angles):
            gap = signed_gap(angles)
            return 10000.0 * (gap - 0.0005) ** 2 + 0.1 * np.sum((angles - seed) ** 2)

        fitted = minimize(finger_objective, seed, bounds=finger_bounds,
                          method="SLSQP", options={"maxiter": 100, "ftol": 1e-10})
        closed_pose[list(parts)] = fitted.x
        closure_gaps[distal] = signed_gap(fitted.x)
    for k in range(frame(10.5), frame(27.0) + 1):
        t = START_S + k / FPS
        opening = smooth((t - 10.5) / 1.1)
        closing = smooth((t - 11.9) / 0.9)
        release = smooth((t - 25.45) / 0.80)
        neutral = baseline.joint_pos[k, finger_cols]
        desired = open_pose * (1 - closing) + closed_pose * closing
        desired = desired * (1 - release) + open_pose * release
        return_neutral = smooth((t - 26.5) / 0.5)
        joint[k, finger_cols] = ((1 - opening) * neutral + opening * desired) * (1 - return_neutral) + return_neutral * neutral

    candidate = refresh_fk(replace(baseline, joint_pos=joint), MODEL)
    wrist_name = candidate.body_names.index("right_wrist_yaw_link")
    wrist_pos = candidate.body_pos_w[:, wrist_name]
    wrist_rot = Rotation.from_quat(candidate.body_quat_w[:, wrist_name][:, [1, 2, 3, 0]])
    eye = wrist_pos + wrist_rot.apply(np.broadcast_to(center, wrist_pos.shape))
    hold = frame(12.8)
    release = frame(25.65)
    local_position = wrist_rot[hold].inv().apply(np.asarray(BRUSH_POSITION) - wrist_pos[hold])
    local_rotation = wrist_rot[hold].inv() * brush_rot
    position = np.broadcast_to(BRUSH_POSITION, wrist_pos.shape).copy()
    orientation = np.broadcast_to(BRUSH_QUAT_XYZW, (len(position), 4)).copy()
    held = np.arange(hold + 1, release + 1)
    position[held] = wrist_pos[held] + wrist_rot[held].apply(np.broadcast_to(local_position, (len(held), 3)))
    orientation[held] = (wrist_rot[held] * local_rotation).as_quat()
    position[release + 1:] = position[release]
    orientation[release + 1:] = orientation[release]

    OUT.mkdir(parents=True, exist_ok=True)
    candidate.save(OUT / "motion.npz")
    np.savez_compressed(OUT / "object_estimate.npz", position_w=position.astype("f4"),
                        quaternion_xyzw=orientation.astype("f4"), first_grasp_frame=np.array(hold),
                        release_frame=np.array(release), status=np.array("kinematic_estimate_not_physics"))
    max_speed = float(np.max(abs(candidate.joint_vel[:, arm_cols])))
    report = {
        "status": "kinematic_candidate_unreviewed",
        "fps": FPS,
        "source_window_s_half_open": [START_S, START_S + len(joint) / FPS],
        "base_motion_sha256": sha256(PRIOR),
        "robot_mjcf_sha256": sha256(MODEL),
        "grip_fit_source_sha256": sha256(FIT),
        "edited_arm": "right only, pickup and return windows",
        "root_legs_torso_left_arm_unchanged": True,
        "commanded_hand_events_s": {"open_start": 10.5, "open_complete": 11.6,
                                    "close_start": 11.9, "close_complete": 12.8,
                                    "release_start": 25.45, "release_complete": 26.25},
        "estimated_grasp_s": 12.8,
        "estimated_release_s": 25.65,
        "offsethand_closed_finger_angles_rad": closed_pose.tolist(),
        "closed_finger_signed_grip_gaps_m": closure_gaps,
        "fist_eye_errors": residuals,
        "grip_world_m": grip.tolist(),
        "estimated_return_position_error_m": float(np.linalg.norm(position[release] - BRUSH_POSITION)),
        "estimated_return_orientation_error_deg": float((Rotation.from_quat(orientation[release]).inv() * brush_rot).magnitude() * 180 / np.pi),
        "max_edited_arm_joint_speed_rad_s": max_speed,
        "motion_sha256": sha256(OUT / "motion.npz"),
        "physics_grasp_verified": False,
        "physical_table_contact_verified": False,
    }
    (OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "estimated_return_position_error_m", "estimated_return_orientation_error_deg", "max_edited_arm_joint_speed_rad_s")}, indent=2))


if __name__ == "__main__":
    main()
