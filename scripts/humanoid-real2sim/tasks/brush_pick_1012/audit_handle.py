"""Audit the kinematic brush handle candidate against source limits and basket hulls."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields
from scene import ASSETS, BASKET_POSITION, BASKET_QUAT_XYZW
from retarget_handle import BASE, MODEL, ORDER, OUT


def mesh_points(model, body):
    points = []
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name != body or model.geom_group[g] == 3:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        v = Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]]).apply(v) + model.geom_pos[g]
        points.append(v)
    if not points:
        return np.empty((0, 3))
    v = np.unique(np.concatenate(points), axis=0)
    return v[ConvexHull(v).vertices]


def main():
    motion = MotionFields.load(OUT / 'motion.npz', ORDER)
    base = MotionFields.load(BASE, ORDER)
    tree = ET.parse('/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/body.urdf')
    limits = {j.get('name'): float(j.find('limit').get('velocity', '20'))
              for j in tree.findall('joint') if j.find('limit') is not None}
    speeds = np.array([limits.get(name, 3.) for name in motion.joint_names])
    ratio = np.abs(np.diff(motion.joint_pos.astype(float), axis=0)) * motion.fps / speeds
    maximum = np.unravel_index(np.argmax(ratio[300:661]), ratio[300:661].shape)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    qaddr = np.array([model.joint(name).qposadr[0] for name in motion.joint_names])
    hand_names = [model.body(b).name for b in range(model.nbody)
                  if model.body(b).name.startswith('right_hand_') or model.body(b).name.startswith('right_wrist_')]
    hand = [(model.body(name).id, mesh_points(model, name)) for name in hand_names]
    hand = [(bid, v) for bid, v in hand if len(v)]
    basket_rotation = Rotation.from_quat(BASKET_QUAT_XYZW)
    basket = []
    for path in sorted((ASSETS / 'white_storage_box/collision_hulls').glob('*.obj')):
        mesh = trimesh.load(path, force='mesh', process=False)
        v = basket_rotation.apply(mesh.vertices) + BASKET_POSITION
        hull = ConvexHull(v)
        normal = hull.equations[:, :3]
        offset = hull.equations[:, 3]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        basket.append((path.name, np.min(v, axis=0), np.max(v, axis=0), normal, offset))
    hits = []
    for frame in range(450, 601):
        data.qpos[:3] = motion.body_pos_w[frame, 0]
        data.qpos[3:7] = motion.body_quat_w[frame, 0]
        data.qpos[qaddr] = motion.joint_pos[frame]
        mujoco.mj_forward(model, data)
        for bid, local in hand:
            points = local @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
            for name, lower, upper, normal, offset in basket:
                near = points[np.all((points >= lower - .01) & (points <= upper + .01), axis=1)]
                if len(near) == 0:
                    continue
                depth = np.max(near @ normal.T + offset, axis=1)
                inside = depth < -1e-5
                if np.any(inside):
                    hits.append({'frame': frame, 'hand_body': model.body(bid).name,
                                 'basket_hull': name, 'vertex_count': int(np.sum(inside)),
                                 'max_vertex_depth_m': float(-np.min(depth[inside]))})
    object_track = np.load(OUT / 'object_estimate.npz')
    brush_hulls = [trimesh.load(path, force='mesh', process=False).vertices
                   for path in sorted((ASSETS / 'green_hand_brush/collision_hulls').glob('*.obj'))]
    object_hits = []
    for frame in range(560, 601):
        rotation = Rotation.from_quat(object_track['quaternion_xyzw'][frame])
        position = object_track['position_w'][frame]
        for vertices in brush_hulls:
            points = rotation.apply(vertices) + position
            for name, lower, upper, normal, offset in basket:
                near = points[np.all((points >= lower - .01) & (points <= upper + .01), axis=1)]
                if len(near) and np.any(np.max(near @ normal.T + offset, axis=1) < -.001):
                    object_hits.append({'frame': frame, 'basket_hull': name})
    diff = np.abs(motion.joint_pos - base.joint_pos)
    changed = [name for name, delta in zip(motion.joint_names, np.max(diff, axis=0)) if delta > 1e-6]
    changed_columns = [motion.joint_names.index(name) for name in changed]
    acceleration = np.diff(motion.joint_pos.astype(float), n=2, axis=0) * motion.fps ** 2
    old_acceleration = np.diff(base.joint_pos.astype(float), n=2, axis=0) * motion.fps ** 2
    arm_columns = [column for column in changed_columns if 'hand_' not in motion.joint_names[column]]
    report = {
        'status': 'kinematic_audit',
        'changed_joints': changed,
        'root_leg_torso_left_arm_identical': bool(np.array_equal(motion.body_pos_w[:, 0], base.body_pos_w[:, 0])
                                                   and all(n.startswith('right_') for n in changed)),
        'outside_edit_window_identical': bool(np.array_equal(motion.joint_pos[:450], base.joint_pos[:450])
                                              and np.array_equal(motion.joint_pos[661:], base.joint_pos[661:])),
        'maximum_speed_limit_ratio_6_to_13p2s': float(np.max(ratio[300:661])),
        'maximum_speed_joint': motion.joint_names[maximum[1]],
        'maximum_speed_frame': int(maximum[0] + 300),
        'changed_joint_maximum_speed_limit_ratio': float(np.max(ratio[300:661, changed_columns])),
        'changed_joint_maximum_acceleration_rad_s2': float(np.max(np.abs(acceleration[448:660, changed_columns]))),
        'prior_same_joint_maximum_acceleration_rad_s2': float(np.max(np.abs(old_acceleration[448:660, changed_columns]))),
        'arm_position_equal_to_prior_at_seams': bool(
            np.array_equal(motion.joint_pos[450, arm_columns], base.joint_pos[450, arm_columns])
            and np.array_equal(motion.joint_pos[615, arm_columns], base.joint_pos[615, arm_columns])),
        'finger_position_equal_to_prior_at_final_seam': bool(
            np.array_equal(motion.joint_pos[660, changed_columns], base.joint_pos[660, changed_columns])),
        'hand_basket_vertex_inside_count': len(hits),
        'hand_basket_vertex_inside_first_20': hits[:20],
        'estimated_brush_basket_vertices_deeper_than_1mm': len(object_hits),
        'collision_audit_limit': 'Convex hand-mesh vertices versus basket hull halfspaces; edge-only intersections not covered.',
        'physics_grasp_verified': False,
    }
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
