"""Kinematic body-prior-preserving 6–12 s brush-handle retarget."""
from __future__ import annotations

import json
from dataclasses import replace
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
MODEL = Path('/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/robot.xml')
ORDER = Path('/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json')
BASE = ROOT / 'retarget/kinematic_motion.npz'
FIT = Path('/home/galbot/workspace/ScaleTrack/data/1012test/1012-1/stage4-brush/review/handle_fit.json')
OUT = ROOT / 'retarget/handle_kinematic_v1'
ARM_PARTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
FINGERS = ('thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1')
# (frame, vertical clearance over the fixed handle grip). Negative means use original path.
KEYS = ((450, None), (460, None), (480, .23), (495, .15), (510, .06),
        (522, 0.), *((frame, 0.) for frame in range(524, 561, 2)), (565, .025), (575, .08),
        (590, None), (615, None), (625, None))


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t * t * (3 - 2 * t)


def fit_closed_fingers(model, data, motion, frame, q, seed, names):
    """Fit a fixed closure seed to the real grip hull at the reviewed grasp pose."""
    data.qpos[:3] = motion.body_pos_w[frame, 0]
    data.qpos[3:7] = motion.body_quat_w[frame, 0]
    data.qpos[[model.joint(n).qposadr[0] for n in motion.joint_names]] = q[frame]
    grip_mesh = trimesh.load(ASSET_GRIP, force='mesh')
    grip_mesh.apply_transform(trimesh.transformations.quaternion_matrix(
        np.r_[BRUSH_QUAT_XYZW[3], BRUSH_QUAT_XYZW[:3]]))
    grip_mesh.apply_translation(BRUSH_POSITION)
    closed = seed.copy()
    gaps = {}
    for parts, distal in (((0, 1, 2), 'thumb_2'), ((3, 4), 'middle_1'), ((5, 6), 'index_1')):
        addresses = [model.joint('right_hand_' + FINGERS[i] + '_joint').qposadr[0] for i in parts]
        bounds = [tuple(model.joint('right_hand_' + FINGERS[i] + '_joint').range) for i in parts]
        body = model.body('right_hand_' + distal + '_link').id
        vertices = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != body or model.geom_group[geom] == 3:
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = model.geom_dataid[geom]
            points = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            vertices.append(Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).apply(points) + model.geom_pos[geom])
        vertices = np.unique(np.concatenate(vertices), axis=0)
        vertices = vertices[ConvexHull(vertices).vertices]

        def gap(angles):
            data.qpos[addresses] = angles
            mujoco.mj_forward(model, data)
            world = vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            return float(trimesh.proximity.signed_distance(grip_mesh, world).max())

        source = seed[list(parts)]

        def objective(angles):
            distance = gap(angles)
            return (10000. * (distance + .0004) ** 2 + .1 * np.sum((angles - source) ** 2)
                    + 100000. * max(distance - .0005, 0.) ** 2)

        result = minimize(objective, source, bounds=bounds, method='SLSQP',
                          options={'maxiter': 80, 'ftol': 1e-10})
        final_gap = gap(result.x)
        if not result.success or not -.002 <= final_gap <= .0005:
            raise RuntimeError(f'finger fit {distal} failed: {result.message}, gap={final_gap}')
        closed[list(parts)] = result.x
        gaps[distal] = final_gap
    return closed, gaps


ASSET_GRIP = ROOT / 'assets/green_hand_brush/collision_hulls/Hull_grip.obj'


def main():
    baseline = MotionFields.load(BASE, ORDER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    arm_cols = np.array([baseline.joint_names.index('right_' + p + '_joint') for p in ARM_PARTS])
    all_addr = np.array([model.joint(n).qposadr[0] for n in baseline.joint_names])
    arm_addr = all_addr[arm_cols]
    lo = np.array([model.joint(baseline.joint_names[i]).range[0] for i in arm_cols])
    hi = np.array([model.joint(baseline.joint_names[i]).range[1] for i in arm_cols])
    wrist_id = model.body('right_wrist_yaw_link').id
    fit = json.loads(FIT.read_text())
    center = np.asarray(fit['center_in_wrist_m'])
    brush_rot = Rotation.from_quat(BRUSH_QUAT_XYZW)
    grip_local = np.asarray(fit['object_grip_local_m']) + np.array([-.015, 0., 0.])
    grip = np.asarray(BRUSH_POSITION) + brush_rot.apply(grip_local)
    axis = brush_rot.apply([1., 0., 0.])
    original_eye = baseline.body_pos_w[:, baseline.body_names.index('right_wrist_yaw_link')] + Rotation.from_quat(
        baseline.body_quat_w[:, baseline.body_names.index('right_wrist_yaw_link')][:, [1, 2, 3, 0]]
    ).apply(np.broadcast_to(center, (len(baseline.joint_pos), 3)))
    joint = baseline.joint_pos.astype(float).copy()
    solved = []
    last = None
    for frame, clearance in KEYS:
        if clearance is None:
            solved.append((frame, baseline.joint_pos[frame, arm_cols].copy()))
            last = None
            continue
        target = grip + [0., 0., clearance]
        data.qpos[:3] = baseline.body_pos_w[frame, 0]
        data.qpos[3:7] = baseline.body_quat_w[frame, 0]
        data.qpos[all_addr] = baseline.joint_pos[frame]
        prior = last if last is not None and frame >= 510 else baseline.joint_pos[frame, arm_cols]

        def forward(q):
            data.qpos[arm_addr] = q
            mujoco.mj_forward(model, data)
            rot = data.xmat[wrist_id].reshape(3, 3)
            return data.xpos[wrist_id] + rot @ center, rot[:, 2]

        def residual(q):
            eye, direction = forward(q)
            # The cylinder needs to pass through the fist eye. Wrist angles
            # beyond 65 degrees are penalized, while shoulder/elbow can adapt.
            return np.r_[50. * (eye - target), 3. * (direction - axis),
                         .16 * (q - prior), 8. * np.maximum(np.abs(q[4:]) - np.deg2rad(65.), 0.)]

        starts = [prior]
        if last is not None:
            starts.insert(0, last)
        candidates = []
        for seed in starts:
            result = least_squares(residual, np.clip(seed, lo + 1e-6, hi - 1e-6),
                                   bounds=(lo + 1e-6, hi - 1e-6), max_nfev=120)
            candidates.append((np.linalg.norm(result.fun), result.x))
        q = min(candidates, key=lambda item: item[0])[1]
        solved.append((frame, q))
        last = q
    key_frames = np.array([f for f, _ in solved])
    delta_keys = np.stack([q - baseline.joint_pos[f, arm_cols] for f, q in solved])
    frames = np.arange(key_frames[0], key_frames[-1] + 1)
    joint[np.ix_(frames, arm_cols)] += PchipInterpolator(key_frames, delta_keys, axis=0)(frames)

    finger_cols = np.array([baseline.joint_names.index('right_hand_' + p + '_joint') for p in FINGERS])
    open_pose = baseline.joint_pos[500, finger_cols]
    closed, finger_gaps = fit_closed_fingers(model, data, baseline, 560, joint,
                                             np.asarray(fit['angles']), FINGERS)
    for frame in range(500, 661):
        closure = smooth((frame - 518.) / 42.)
        return_to_prior = smooth((frame - 600.) / 60.)
        grasp = open_pose * (1. - closure) + closed * closure
        joint[frame, finger_cols] = (1. - return_to_prior) * grasp + return_to_prior * baseline.joint_pos[frame, finger_cols]
    motion = refresh_fk(replace(baseline, joint_pos=joint), MODEL)
    wrist = motion.body_names.index('right_wrist_yaw_link')
    eye = motion.body_pos_w[:, wrist] + Rotation.from_quat(
        motion.body_quat_w[:, wrist][:, [1, 2, 3, 0]]
    ).apply(np.broadcast_to(center, (len(joint), 3)))
    axis_world = Rotation.from_quat(motion.body_quat_w[:, wrist][:, [1, 2, 3, 0]]).apply(
        np.broadcast_to([0., 0., 1.], (len(joint), 3)))
    speed = np.max(np.abs(motion.joint_vel[300:625]), axis=0)
    edge = np.max(np.abs(joint[300:625] - baseline.joint_pos[300:625]), axis=0)
    contact = np.arange(522, 546)
    report = {
        'status': 'kinematic_candidate_unreviewed', 'fps': 50,
        'source_window_frames': [300, 600], 'edited_frames': [450, 660],
        'base_motion': str(BASE), 'base_sha256': sha256(BASE),
        'model_sha256': sha256(MODEL), 'handle_fit_sha256': sha256(FIT),
        'brush_position_m': list(BRUSH_POSITION), 'brush_quat_xyzw': list(BRUSH_QUAT_XYZW),
        'grip_world_m': grip.tolist(), 'grip_local_m': grip_local.tolist(),
        'handle_axis_world': axis.tolist(),
        'fixed_hand_commands_s': {'open_start': 8.9, 'open_complete': 9.8,
                                  'close_start': 10.36, 'close_complete': 11.2},
        'contact_prior_frames': [522, 560], 'lift_start_frame': 560,
        'contact_eye_error_m': {'mean': float(np.mean(np.linalg.norm(eye[contact] - grip, axis=1))),
                                'max': float(np.max(np.linalg.norm(eye[contact] - grip, axis=1)))},
        'contact_axis_error_deg': {'mean': float(np.mean(np.rad2deg(np.arccos(np.clip(axis_world[contact] @ axis, -1, 1))))),
                                   'max': float(np.max(np.rad2deg(np.arccos(np.clip(axis_world[contact] @ axis, -1, 1)))))},
        'grip_error_before_m': float(np.linalg.norm(original_eye[525] - grip)),
        'grip_error_after_m': float(np.linalg.norm(eye[525] - grip)),
        'closed_finger_surface_gaps_m': finger_gaps,
        'closed_finger_angles_rad': closed.tolist(),
        'max_joint_speed_rad_s': float(speed.max()),
        'max_arm_delta_rad': float(edge[arm_cols].max()),
        'root_legs_torso_left_arm_unchanged': True,
        'body_prior': 'stage8 inherited global/stance reference; no new full-body solve',
        'physics_grasp_verified': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    motion.save(OUT / 'motion.npz')
    np.savez_compressed(OUT / 'grasp_geometry.npz', frame=np.arange(300, 625),
                        fist_eye_w=eye[300:625], handle_grip_w=np.broadcast_to(grip, (325, 3)),
                        hand_axis_w=axis_world[300:625])
    # Estimated held-object path, kept separate from the motion and from RSS physics.
    # The brush remains at its gravity-tested pose until closure completes.
    wrist_rotation = Rotation.from_quat(motion.body_quat_w[:, wrist][:, [1, 2, 3, 0]])
    wrist_position = motion.body_pos_w[:, wrist]
    hold = 560
    brush_position = np.broadcast_to(BRUSH_POSITION, (len(joint), 3)).copy()
    brush_quaternion = np.broadcast_to(BRUSH_QUAT_XYZW, (len(joint), 4)).copy()
    object_local = wrist_rotation[hold].inv().apply(np.asarray(BRUSH_POSITION) - wrist_position[hold])
    object_rotation_local = wrist_rotation[hold].inv() * brush_rot
    brush_position[hold + 1:] = wrist_position[hold + 1:] + wrist_rotation[hold + 1:].apply(
        np.broadcast_to(object_local, (len(joint) - hold - 1, 3))
    )
    brush_quaternion[hold + 1:] = (wrist_rotation[hold + 1:] * object_rotation_local).as_quat()
    np.savez_compressed(OUT / 'object_estimate.npz', position_w=brush_position,
                        quaternion_xyzw=brush_quaternion, first_grasp_frame=np.array(hold),
                        status=np.array('kinematic_estimate_not_physics'))
    report['estimated_brush_displacement_11p2_to_12s_m'] = (
        brush_position[600] - brush_position[hold]
    ).tolist()
    report['estimated_brush_lift_11p2_to_12s_m'] = float(brush_position[600, 2] - brush_position[hold, 2])
    report['motion_sha256'] = sha256(OUT / 'motion.npz')
    (OUT / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
