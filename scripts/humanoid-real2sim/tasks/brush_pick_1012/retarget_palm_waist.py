"""Palm-first brush retarget with a small waist adjustment."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
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
OUT = ROOT / 'retarget/palm_waist_v3'
PREVIOUS = ROOT / 'retarget/palm_approach_v2/motion.npz'
WAIST_PARTS = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')
ARM_PARTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
FINGERS = ('thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1')
# Eye targets stay outside the basket until the final lateral entry. The
# last approach is from negative Y toward the exposed handle, not through it
# from above. None joins the unchanged full-body prior.
KEYS = ((450, None), (460, None), (470, (0., -.18, .18)),
        (480, (0., -.18, .12)), (495, (0., -.18, .04)),
        (510, (0., -.12, 0.)), (518, (0., -.045, 0.)),
        (522, (0., 0., 0.)), *((frame, (0., 0., 0.)) for frame in range(524, 561, 2)),
        (565, (-.003, 0., .04)), (575, (-.005, 0., .09)),
        (590, None), (615, None), (625, None))


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t * t * (3 - 2 * t)


def fit_closed_fingers(model, data, motion, frame, q, seed, names, *, target_gap=-.0004):
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
            return (10000. * (distance - target_gap) ** 2 + .1 * np.sum((angles - source) ** 2)
                    + 100000. * max(distance - target_gap - .0009, 0.) ** 2)

        result = minimize(objective, source, bounds=bounds, method='SLSQP',
                          options={'maxiter': 80, 'ftol': 1e-10})
        final_gap = gap(result.x)
        if not result.success or not target_gap - .0016 <= final_gap <= target_gap + .0009:
            raise RuntimeError(f'finger fit {distal} failed: {result.message}, gap={final_gap}')
        closed[list(parts)] = result.x
        gaps[distal] = final_gap
    return closed, gaps


ASSET_GRIP = ROOT / 'assets/green_hand_brush/collision_hulls/Hull_grip.obj'


def main():
    baseline = MotionFields.load(BASE, ORDER)
    previous = MotionFields.load(PREVIOUS, ORDER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    arm_cols = np.array([baseline.joint_names.index('right_' + p + '_joint') for p in ARM_PARTS])
    waist_cols = np.array([baseline.joint_names.index(p) for p in WAIST_PARTS])
    edit_cols = np.r_[waist_cols, arm_cols]
    all_addr = np.array([model.joint(n).qposadr[0] for n in baseline.joint_names])
    edit_addr = all_addr[edit_cols]
    lo = np.array([model.joint(baseline.joint_names[i]).range[0] for i in edit_cols])
    hi = np.array([model.joint(baseline.joint_names[i]).range[1] for i in edit_cols])
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
    palm_target = np.array([0., 1., .2])
    palm_target /= np.linalg.norm(palm_target)
    for frame, offset in KEYS:
        if offset is None:
            solved.append((frame, baseline.joint_pos[frame, edit_cols].copy()))
            last = None
            continue
        if frame <= 480:
            q = previous.joint_pos[frame, edit_cols].copy()
            solved.append((frame, q))
            last = q
            continue
        target = grip + np.asarray(offset)
        data.qpos[:3] = baseline.body_pos_w[frame, 0]
        data.qpos[3:7] = baseline.body_quat_w[frame, 0]
        data.qpos[all_addr] = baseline.joint_pos[frame]
        prior = baseline.joint_pos[frame, edit_cols]
        previous_q = previous.joint_pos[frame, edit_cols]
        lower = lo.copy()
        upper = hi.copy()
        lower[:3] = np.maximum(lower[:3], prior[:3] - .2)
        upper[:3] = np.minimum(upper[:3], prior[:3] + .2)
        lower[7:] = np.maximum(lower[7:], -.9)
        upper[7:] = np.minimum(upper[7:], .9)

        def forward(q):
            data.qpos[edit_addr] = q
            mujoco.mj_forward(model, data)
            rot = data.xmat[wrist_id].reshape(3, 3)
            return data.xpos[wrist_id] + rot @ center, rot[:, 2], rot[:, 1]

        def residual(q):
            eye, direction, palm = forward(q)
            # The hand must face the exposed handle while entering from the
            # basket's near side. Wrist cost applies below the old 65° limit.
            return np.r_[150. * (eye - target), 10. * (direction - axis),
                         4. * (palm - palm_target),
                         4. * (q[:3] - prior[:3]),
                         np.array([.5, .5, .5, 2.]) * (q[3:7] - prior[3:7]),
                         np.array([2.5, 4., 2.5]) * q[7:],
                         .5 * (q - last) if last is not None else np.zeros(10)]

        starts = [prior, previous_q]
        if last is not None:
            starts.insert(0, last)
        candidates = []
        for seed in starts:
            result = least_squares(residual, np.clip(seed, lower + 1e-6, upper - 1e-6),
                                   bounds=(lower + 1e-6, upper - 1e-6), max_nfev=160)
            candidates.append((np.linalg.norm(result.fun), result.x))
        q = min(candidates, key=lambda item: item[0])[1]
        solved.append((frame, q))
        last = q
    key_frames = np.array([f for f, _ in solved])
    delta_keys = np.stack([q - baseline.joint_pos[f, edit_cols] for f, q in solved])
    frames = np.arange(key_frames[0], key_frames[-1] + 1)
    joint[np.ix_(frames, edit_cols)] += PchipInterpolator(key_frames, delta_keys, axis=0)(frames)
    # Smooth keyframe branch transitions while preserving both splice poses.
    window = np.arange(450, 591)
    delta = joint[np.ix_(window, edit_cols)] - baseline.joint_pos[np.ix_(window, edit_cols)]
    delta = gaussian_filter1d(delta, 1.6, axis=0, mode='nearest')
    taper = smooth((window - 450) / 10.) * smooth((590 - window) / 10.)
    joint[np.ix_(window, edit_cols)] = baseline.joint_pos[np.ix_(window, edit_cols)] + delta * taper[:, None]

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
    palm_world = Rotation.from_quat(motion.body_quat_w[:, wrist][:, [1, 2, 3, 0]]).apply(
        np.broadcast_to([0., 1., 0.], (len(joint), 3)))
    approach_velocity = eye[511:523] - eye[509:521]
    approach_velocity /= np.maximum(np.linalg.norm(approach_velocity, axis=1, keepdims=True), 1e-8)
    palm_alignment = np.sum(approach_velocity * palm_world[510:522], axis=1)
    wrist_cols = arm_cols[4:]
    wrist_norm = np.linalg.norm(joint[470:561, wrist_cols], axis=1)
    old = MotionFields.load(ROOT / 'retarget/handle_kinematic_v1/motion.npz', ORDER)
    old_wrist_norm = np.linalg.norm(old.joint_pos[470:561, wrist_cols], axis=1)
    speed = np.max(np.abs(motion.joint_vel[300:625]), axis=0)
    edge = np.max(np.abs(joint[300:625] - baseline.joint_pos[300:625]), axis=0)
    contact = np.arange(522, 546)
    report = {
        'status': 'kinematic_candidate_unreviewed', 'fps': 50,
        'source_window_frames': [300, 600], 'edited_frames': [450, 660],
        'approach': 'palm-facing lateral entry from negative world Y',
        'palm_target_world': palm_target.tolist(),
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
        'palm_approach_alignment_510_521': {'mean': float(palm_alignment.mean()),
                                            'min': float(palm_alignment.min())},
        'wrist_3axis_norm_470_560_rad': {'old_max': float(old_wrist_norm.max()),
                                         'new_max': float(wrist_norm.max()),
                                         'old_mean': float(old_wrist_norm.mean()),
                                         'new_mean': float(wrist_norm.mean())},
        'grip_error_before_m': float(np.linalg.norm(original_eye[525] - grip)),
        'grip_error_after_m': float(np.linalg.norm(eye[525] - grip)),
        'closed_finger_surface_gaps_m': finger_gaps,
        'closed_finger_angles_rad': closed.tolist(),
        'max_joint_speed_rad_s': float(speed.max()),
        'max_arm_delta_rad': float(edge[arm_cols].max()),
        'root_and_legs_unchanged': True,
        'max_waist_delta_rad': float(np.abs(joint[450:591, waist_cols] - baseline.joint_pos[450:591, waist_cols]).max()),
        'body_prior': 'stage8 inherited root/stance; waist adjusts up to 0.2 rad in HOI window',
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
