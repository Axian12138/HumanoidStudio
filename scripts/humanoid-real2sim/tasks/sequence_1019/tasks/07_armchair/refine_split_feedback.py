"""Revise chair hand contact and fit a separate tabletop brace candidate.

Outputs are new, task-local run artifacts. The two actions overlap by 1.0 s
for later stitching but are reviewed and trained independently.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from robosimstudio.humanoid_real2sim.paths import run_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

from fit_hoi import HERE, MODEL, ORDER, frame, smoothstep


CHAIR_BEGIN, CHAIR_END = frame(75), frame(84.5)
TABLE_BEGIN, TABLE_END = frame(83.5), frame(91)
TABLE_Z = 0.754
TABLE_Y_EDGE = -2.11905405


def pose(model, data, motion, k, addresses):
    data.qpos[:3] = motion.body_pos_w[k, 0]
    data.qpos[3:7] = motion.body_quat_w[k, 0]
    data.qpos[addresses] = motion.joint_pos[k]
    mujoco.mj_forward(model, data)


def arm_columns(model, motion, side, *, fingers=True):
    names = [f"{side}_{part}_joint" for part in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )]
    if fingers:
        names += [n for n in motion.joint_names if n.startswith(f"{side}_hand_")]
    cols = np.asarray([motion.joint_names.index(n) for n in names])
    low = np.asarray([model.joint(n).range[0] + 1e-5 for n in names])
    high = np.asarray([model.joint(n).range[1] - 1e-5 for n in names])
    return names, cols, low, high


def revise_chair_hand(mode, base, chair_position, chair_rotation, human=None):
    strip_contact = mode.endswith("_strip")
    backface_contact = mode.endswith("_backface") or mode.endswith("_backface_flip")
    flip_thumb_face = mode.endswith("_backface_flip")
    surface_contact = mode.endswith("_surface") or strip_contact or backface_contact
    edge_contact = surface_contact or mode.endswith("_edge")
    hand_mode = mode.split("_")[0]
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(n).qposadr[0] for n in base.joint_names])
    joints = base.joint_pos.astype(np.float64).copy()
    reports = {}
    for side, sign in (("left", 1), ("right", -1)):
        names, cols, low, high = arm_columns(model, base, side)
        addr = addresses[cols]
        if hand_mode == "thumb":
            root_index = names.index(f"{side}_hand_thumb_0_joint")
            low[root_index], high[root_index] = -0.001, 0.001
            if surface_contact:
                # The broad chair back rewards a straight thumb laid across
                # its surface, unlike a small cylindrical grasp.
                thumb1 = names.index(f"{side}_hand_thumb_1_joint")
                thumb2 = names.index(f"{side}_hand_thumb_2_joint")
                if flip_thumb_face:
                    low[thumb1], high[thumb1] = -.001, .001
                    low[thumb2], high[thumb2] = -.001, .001
                else:
                    low[thumb1], high[thumb1] = -.08, .08
                    low[thumb2], high[thumb2] = ((1e-5, .08) if side == "left"
                                                 else (-.08, -1e-5))
        else:
            # The index and middle fingers stay nearly straight against the
            # backrest. Joint zero is straight in this Dex3 MJCF.
            for digit in ("index", "middle"):
                for joint in ("0", "1"):
                    i = names.index(f"{side}_hand_{digit}_{joint}_joint")
                    if sign == 1:
                        low[i], high[i] = -0.12, -1e-5
                    else:
                        low[i], high[i] = 1e-5, 0.12
        tip_ids = [model.body(f"{side}_hand_{digit}_finger_tip").id
                   for digit in ("thumb", "index", "middle")]
        proximal = model.geom(f"{side}_hand_thumb_1_collision").id
        distal = model.geom(f"{side}_hand_thumb_2_collision").id
        palm = model.geom(f"{side}_hand_palm_collision").id
        keys = np.arange(frame(79.2), frame(83.4) + 1, 5 if backface_contact else 10)
        if surface_contact:
            approach_keys = np.arange(frame(78.0), frame(79.1) + 1, 5)
            solve_order = np.r_[keys, approach_keys[::-1]]
        else:
            solve_order = keys
        fitted_offsets, errors = {}, []
        previous = None
        contact_palm = None
        contact_normal = None
        for k in solve_order:
            pose(model, data, base, k, addresses)
            q0 = base.joint_pos[k, cols].astype(np.float64)
            chair_r = chair_rotation[k]
            chair_p = chair_position[k]
            is_approach = surface_contact and k < frame(79.2)
            if is_approach and k == approach_keys[-1]:
                previous = fitted_offsets[frame(79.2)][0]

            def objective(q):
                data.qpos[addr] = q
                mujoco.mj_forward(model, data)
                local = lambda point: chair_r.T @ (point - chair_p)
                tips = np.asarray([local(data.xpos[tip]) for tip in tip_ids])
                continuous = 1.1 * (q - previous) if previous is not None else np.zeros(len(q))
                normal = chair_r.T @ (data.geom_xmat[palm].reshape(3, 3)[:, 0]
                                      * (-1 if side == "left" else 1))
                if is_approach:
                    wrist_key = 13 if side == "left" else 14
                    human_local = local(human[k, wrist_key])
                    x = sign * np.clip(abs(human_local[0]), .16, .40)
                    minimum_y = np.interp(abs(x), [.16, .28, .40], [.37, .34, .12])
                    outside_target = np.array([x, max(human_local[1] + .03, minimum_y),
                                               max(human_local[2] - .02, .86)])
                    blend = float(smoothstep((75 + k / 50 - 78.0) / 1.15))
                    target = (1 - blend) * outside_target + blend * contact_palm
                    target_normal = (1 - blend) * np.array([0, -1, 0]) + blend * contact_normal
                    points = np.vstack((tips, local(data.geom_xpos[proximal]),
                                        local(data.geom_xpos[distal])))
                    # Before contact the palm and fingers pass outside the
                    # side or above the top rail, never through the backrest.
                    near_side = smoothstep((.34 - np.abs(points[:, 0])) / .05)
                    below_top = smoothstep((.82 - points[:, 2]) / .05)
                    avoid = (1 - blend) * 45 * near_side * below_top * np.maximum(0, .31 - points[:, 1])
                    return np.r_[
                        40 * (local(data.geom_xpos[palm]) - target),
                        8 * (normal - target_normal), avoid,
                        .35 * (q[:7] - q0[:7]), .1 * (q[7:] - q0[7:]), continuous,
                    ]
                if hand_mode == "thumb":
                    if backface_contact:
                        # The thumb mesh is long along geom-local Z and has
                        # its broad side normal along geom-local X. The two
                        # signed faces require opposite whole-hand rolls;
                        # keep the thumb's downward long axis fixed.
                        if side == "left":
                            progress = np.clip((k - frame(79.2)) / (frame(83.4) - frame(79.2)), 0, 1)
                            x_contact = .178 - .003 * progress
                            y_contact = .257 - .006 * progress
                            thumb_tip = [x_contact, y_contact, .670]
                            thumb_proximal = [.155, .216, .750]
                            thumb_distal = [x_contact, y_contact, .700]
                        else:
                            thumb_tip = [-.169, .241, .670]
                            thumb_proximal = [-.160, .225, .750]
                            thumb_distal = [-.169, .241, .700]
                            late_clearance = .004 * np.clip((75 + k / 50 - 81.5) / 1.2, 0, 1)
                            for target in (thumb_tip, thumb_distal):
                                target[0] -= .53 * late_clearance
                                target[1] += .85 * late_clearance
                        # Mesh curvature makes the last two contact knots
                        # slightly deeper than their center targets. Keep a
                        # smooth few-millimeter normal clearance there.
                        bump = (.003 if flip_thumb_face else .006) * np.exp(
                            -.5 * ((75 + k / 50 - 83.3) / .11) ** 2)
                        for target in (thumb_tip, thumb_distal):
                            target[0] += sign * .53 * bump
                            target[1] += .85 * bump
                            if flip_thumb_face:
                                # The opposite thumb face changes the mesh's
                                # exposed surface by a few millimeters.
                                target[0] -= sign * .53 * .003
                                target[1] -= .85 * .003
                    elif strip_contact:
                        # The thumb's long axis crosses the top edge in the
                        # *tangent plane*. Keep the proximal and distal links
                        # on the sloping outer strip. A tip-only target makes
                        # an unstable point contact and was rejected in review.
                        progress = np.clip((k - frame(79.2)) / (frame(83.4) - frame(79.2)), 0, 1)
                        proximal_lift = (.004 + .010 * progress if side == "left"
                                         else .002 + .002 * progress)
                        thumb_tip = [sign * .182, .138, .716]
                        thumb_proximal = [sign * .182, .209, .757 + proximal_lift]
                        thumb_distal = [sign * .182, .169, .732]
                    elif surface_contact:
                        desired_tips = np.asarray([
                            [sign * .18, .225, .68],
                            [sign * .172, .220, .705],
                            [sign * .190, .200, .683],
                        ])
                        thumb_proximal = [sign * .18, .285, .74]
                        thumb_distal = [sign * .18, .255, .715]
                    elif edge_contact:
                        # The top rail runs along chair-local X. Bring the
                        # thumb across it in the YZ plane, using the wrist
                        # rather than rotating thumb root joint zero.
                        desired_tips = np.asarray([
                            [sign * .18, .155, .69],
                            [sign * .172, .155, .705],
                            [sign * .190, .135, .683],
                        ])
                        thumb_proximal = [sign * .18, .21, .735]
                        thumb_distal = [sign * .18, .18, .71]
                    else:
                        desired_tips = np.asarray([
                            [sign * .145, .235, .750],
                            [sign * .172, .155, .705],
                            [sign * .190, .135, .683],
                        ])
                        thumb_proximal = [sign * .18, .21, .735]
                        thumb_distal = [sign * .16, .225, .75]
                    if backface_contact:
                        thumb_r = data.geom_xmat[distal].reshape(3, 3)
                        thumb_a_normal = chair_r.T @ (
                            thumb_r[:, 0] * (1 if flip_thumb_face else -1))
                        thumb_down_axis = chair_r.T @ (thumb_r[:, 2] * sign)
                        return np.r_[
                            100 * (tips[0] - thumb_tip),
                            200 * (local(data.geom_xpos[proximal]) - thumb_proximal),
                            200 * (local(data.geom_xpos[distal]) - thumb_distal),
                            (20 if flip_thumb_face else 8) * (
                                thumb_a_normal - [-sign * .53, -.85, 0]),
                            (20 if flip_thumb_face else 8) * (
                                thumb_down_axis - [0, 0, -1]),
                            .25 * (q[:7] - q0[:7]), .1 * (q[7:] - q0[7:]), continuous,
                        ]
                    if strip_contact:
                        return np.r_[
                            55 * (tips[0] - thumb_tip),
                            90 * (local(data.geom_xpos[proximal]) - thumb_proximal),
                            90 * (local(data.geom_xpos[distal]) - thumb_distal),
                            .25 * (q[:7] - q0[:7]), .1 * (q[7:] - q0[7:]), continuous,
                        ]
                    return np.r_[
                        30 * (tips - desired_tips).ravel(),
                        60 * (local(data.geom_xpos[proximal]) - thumb_proximal),
                        (100 if surface_contact else 40 if edge_contact else 35) * (
                            local(data.geom_xpos[distal]) - thumb_distal),
                        2.5 * (normal - [0, -1, 0]) if surface_contact else np.zeros(3),
                        .25 * (q[:7] - q0[:7]), .1 * (q[7:] - q0[7:]), continuous,
                    ]
                if edge_contact:
                    finger_proximal = np.asarray([
                        local(data.geom_xpos[model.geom(f"{side}_hand_{digit}_0_collision").id])
                        for digit in ("index", "middle")
                    ])
                    finger_goal = np.asarray([
                        [sign * .20, .242 if surface_contact else .17, .69],
                        [sign * .14, .242 if surface_contact else .17, .67],
                    ])
                    finger_alignment = np.r_[40 * (tips[1:] - finger_goal).ravel(),
                                              45 * (tips[1:, 0] - finger_proximal[:, 0])]
                else:
                    finger_alignment = 10 * np.maximum(0, tips[:, 1] - .24)
                return np.r_[
                    35 * (local(data.geom_xpos[palm]) - [sign * .15, .315 if surface_contact else .255, .72]),
                    (8 if edge_contact else 11) * (normal - [0, -1, 0]),
                    finger_alignment,
                    .4 * (q[:7] - q0[:7]), .1 * (q[7:] - q0[7:]), continuous,
                ]

            seed = np.clip(q0 if previous is None else previous, low + 1e-8, high - 1e-8)
            result = least_squares(objective, seed, bounds=(low, high), max_nfev=140)
            previous = result.x.copy()
            fitted_offsets[k] = (result.x.copy(), result.x - q0)
            if surface_contact and k == frame(79.2):
                data.qpos[addr] = result.x
                mujoco.mj_forward(model, data)
                contact_palm = chair_r.T @ (data.geom_xpos[palm] - chair_p)
                contact_normal = chair_r.T @ (
                    data.geom_xmat[palm].reshape(3, 3)[:, 0] * (-1 if side == "left" else 1))
            errors.append(float(np.linalg.norm(result.fun)))
        keys = np.sort(np.asarray(list(fitted_offsets)))
        knots = np.r_[frame(77.45 if surface_contact else 78.3), keys, frame(84.45)]
        offsets = np.vstack((np.zeros(len(cols)),
                             [fitted_offsets[k][1] for k in keys], np.zeros(len(cols))))
        correction = (PchipInterpolator(knots, offsets, axis=0) if backface_contact
                      else CubicSpline(knots, offsets, axis=0, bc_type="clamped"))
        active = slice(frame(77.45 if surface_contact else 78.3), frame(84.45) + 1)
        joints[active, cols] += correction(np.arange(active.start, active.stop))
        for i, name in enumerate(names):
            physical = model.joint(name).range
            joints[active, cols[i]] = np.clip(joints[active, cols[i]],
                                               physical[0] + 1e-5, physical[1] - 1e-5)
        if hand_mode == "thumb":
            # Interpolation of offsets otherwise leaves a small root swivel
            # between knots. Pin it across the entire sustained contact.
            joints[frame(79.2):frame(83.4) + 1, cols[root_index]] = 0.0
            if surface_contact:
                sustained = slice(frame(79.2), frame(83.4) + 1)
                for i in (thumb1, thumb2):
                    joints[sustained, cols[i]] = np.clip(joints[sustained, cols[i]],
                                                          low[i], high[i])
        reports[side] = {"mean_knot_residual": float(np.mean(errors)),
                         "max_knot_residual": float(max(errors))}
    if surface_contact:
        correction = gaussian_filter1d(joints - base.joint_pos, sigma=4, axis=0)
        joints = base.joint_pos + correction
    result = refresh_fk(replace(base, joint_pos=joints.astype(np.float32)), MODEL)
    return result, reports


def right_hand_vertices_in_wrist(model, data):
    wrist = model.body("right_wrist_yaw_link").id
    wrist_p = data.xpos[wrist]
    wrist_r = data.xmat[wrist].reshape(3, 3)
    pieces = []
    for g in range(model.ngeom):
        if not model.geom(g).name.startswith("right_hand_") or model.geom_group[g] != 3:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        pieces.append((world - wrist_p) @ wrist_r)
    return np.concatenate(pieces)


def capsule_bottom(model, data, geom):
    size = model.geom_size[geom]
    z_axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
    return float(data.geom_xpos[geom, 2] - size[0] - size[1] * abs(z_axis[2]))


def revise_table_support(base, rigid):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(n).qposadr[0] for n in base.joint_names])
    joints = base.joint_pos.astype(np.float64).copy()
    waist = base.joint_names.index("waist_pitch_joint")
    right_names, right_cols, right_low, right_high = arm_columns(model, base, "right", fingers=False)
    right_cols = np.r_[waist, right_cols]
    right_names = ["waist_pitch_joint", *right_names]
    right_low = np.r_[model.joint("waist_pitch_joint").range[0] + 1e-5, right_low]
    right_high = np.r_[model.joint("waist_pitch_joint").range[1] - 1e-5, right_high]
    right_addr = addresses[right_cols]
    wrist = model.body("right_wrist_yaw_link").id
    elbow = model.body("right_elbow_link").id
    palm = model.geom("right_hand_palm_collision").id
    forearm = model.geom("right_elbow_yaw_collision").id
    wrist_geom = model.geom("right_wrist_collision").id
    anchor = frame(86.0)
    source_rotation = Rotation.from_quat(rigid["right_rigid_quat_xyzw"])
    source_position = rigid["right_rigid_pos_w"]
    wrist_anchor = np.array([-.27, -2.28, .817])
    elbow_anchor = np.array([-.20, -2.11, .803])
    # First find an anatomically feasible contact orientation at the anchor.
    pose(model, data, base, anchor, addresses)
    q0 = base.joint_pos[anchor, right_cols].astype(np.float64)

    def anchor_objective(q):
        data.qpos[right_addr] = q
        mujoco.mj_forward(model, data)
        normal = data.geom_xmat[palm].reshape(3, 3)[:, 0]
        return np.r_[35 * (data.xpos[wrist] - wrist_anchor),
                     35 * (data.xpos[elbow] - elbow_anchor),
                     5 * (normal - [0, 0, -1]),
                     .5 * (q[0] - q0[0]), .2 * (q[1:] - q0[1:])]

    fitted = least_squares(anchor_objective, np.clip(q0, right_low, right_high),
                           bounds=(right_low, right_high), max_nfev=200)
    data.qpos[right_addr] = fitted.x
    mujoco.mj_forward(model, data)
    anchor_wrist_rotation = Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))
    anchor_wrist_position = data.xpos[wrist].copy()
    anchor_elbow_position = data.xpos[elbow].copy()
    source_anchor_rotation = source_rotation[anchor]
    source_anchor_position = source_position[anchor]
    # The hand-back marker mount is unknown. Preserve its measured *relative*
    # motion after choosing an anatomically feasible wrist at 86 s.
    keys = np.unique(np.r_[
        np.arange(frame(83.4), frame(85.1) + 1, 2),
        np.arange(frame(85.1), frame(88.8) + 1, 5),
    ])
    corrections = []
    previous = None
    for k in keys:
        pose(model, data, base, k, addresses)
        # Finger closure changes during the approach; use this frame's exact
        # hand shape rather than assuming the 86 s fist geometry throughout.
        key_hand_vertices = right_hand_vertices_in_wrist(model, data)
        key_hand_vertices = key_hand_vertices[ConvexHull(key_hand_vertices).vertices]
        q0 = base.joint_pos[k, right_cols].astype(np.float64)
        relative_rotation = source_rotation[k] * source_anchor_rotation.inv()
        desired_rotation = relative_rotation * anchor_wrist_rotation
        translation = source_position[k] - source_anchor_position
        desired_wrist = anchor_wrist_position + np.array([
            translation[0], translation[1], np.clip(translation[2], -.015, .015)
        ])
        desired_elbow = anchor_elbow_position + np.array([0, translation[1] * .35, 0])
        contact = smoothstep((k - frame(84.1)) / frame(.9 + 75)) * (
            1 - smoothstep((k - frame(88.0)) / frame(.8 + 75)))
        contact = float(np.clip(contact, 0, 1))

        def objective(q):
            data.qpos[right_addr] = q
            mujoco.mj_forward(model, data)
            normal = data.geom_xmat[palm].reshape(3, 3)[:, 0]
            wrist_r = Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))
            orientation_error = (desired_rotation * wrist_r.inv()).as_rotvec()
            wrist_matrix = data.xmat[wrist].reshape(3, 3)
            hand_world = key_hand_vertices @ wrist_matrix.T + data.xpos[wrist]
            hand_bottom = float(hand_world[:, 2].min())
            inside = ((hand_world[:, 0] >= -1.03) & (hand_world[:, 0] <= .57)
                      & (hand_world[:, 1] >= -2.97) & (hand_world[:, 1] <= TABLE_Y_EDGE))
            table_bottom = float(hand_world[inside, 2].min()) if np.any(inside) else 10.0
            forearm_bottom = capsule_bottom(model, data, forearm)
            wrist_bottom = capsule_bottom(model, data, wrist_geom)
            return np.r_[
                (15 + 18 * contact) * (data.xpos[wrist] - desired_wrist),
                (12 + 20 * contact) * (data.xpos[elbow] - desired_elbow),
                (2 + 3 * contact) * orientation_error,
                5 * contact * (normal - [0, 0, -1]),
                65 * contact * (hand_bottom - (TABLE_Z + .006)),
                200 * contact * max(0, TABLE_Z + .002 - hand_bottom),
                500 * max(0, TABLE_Z + .005 - table_bottom),
                20 * contact * (wrist_bottom - (TABLE_Z + .014)),
                20 * contact * (forearm_bottom - (TABLE_Z + .006)),
                .5 * (q[0] - q0[0]), .25 * (q[1:] - q0[1:]),
                1.0 * (q - previous) if previous is not None else np.zeros(len(q)),
            ]

        seed = np.clip(q0 if previous is None else previous, right_low + 1e-8, right_high - 1e-8)
        fitted = least_squares(objective, seed, bounds=(right_low, right_high), max_nfev=100)
        previous = fitted.x.copy()
        corrections.append(fitted.x - q0)
    knots = np.r_[frame(83.0), keys, frame(89.5)]
    corrections = np.vstack((np.zeros(len(right_cols)), corrections, np.zeros(len(right_cols))))
    spline = CubicSpline(knots, corrections, axis=0, bc_type="clamped")
    active = slice(frame(83.0), frame(89.5) + 1)
    joints[active, right_cols] += spline(np.arange(active.start, active.stop))
    joints[active, right_cols] = np.clip(joints[active, right_cols], right_low, right_high)

    # The original left hand travels through the tabletop near 87 s. Fit a
    # raised, reachable path following the rigid marker's XY trend. This does
    # not claim that the human marker is a calibrated robot wrist target.
    left_names, left_cols, left_low, left_high = arm_columns(model, base, "left", fingers=False)
    left_addr = addresses[left_cols]
    left_wrist = model.body("left_wrist_yaw_link").id
    left_palm = model.geom("left_hand_palm_collision").id
    left_keys = np.arange(frame(85.0), frame(88.3) + 1, 10)
    left_offsets = []
    previous = None
    for k in left_keys:
        data.qpos[:3] = base.body_pos_w[k, 0]
        data.qpos[3:7] = base.body_quat_w[k, 0]
        data.qpos[addresses] = base.joint_pos[k]
        data.qpos[addresses[waist]] = joints[k, waist]
        mujoco.mj_forward(model, data)
        q0 = base.joint_pos[k, left_cols].astype(np.float64)
        human = rigid["left_rigid_pos_w"][k]
        target = np.array([human[0] - .06, max(human[1] + .12, -2.28),
                           max(human[2] - .07, .88)])

        def objective(q):
            data.qpos[left_addr] = q
            mujoco.mj_forward(model, data)
            palm_bottom = data.geom_xpos[left_palm, 2] - model.geom_size[left_palm, 0]
            inside_table = smoothstep((TABLE_Y_EDGE - data.geom_xpos[left_palm, 1]) / .08)
            return np.r_[
                20 * (data.xpos[left_wrist] - target),
                35 * inside_table * max(0, TABLE_Z + .018 - palm_bottom),
                .45 * (q - q0),
                1.1 * (q - previous) if previous is not None else np.zeros(len(q)),
            ]

        seed = np.clip(q0 if previous is None else previous, left_low + 1e-8, left_high - 1e-8)
        fitted = least_squares(objective, seed, bounds=(left_low, left_high), max_nfev=100)
        previous = fitted.x.copy()
        left_offsets.append(fitted.x - q0)
    knots = np.r_[frame(84.5), left_keys, frame(89.0)]
    corrections = np.vstack((np.zeros(len(left_cols)), left_offsets, np.zeros(len(left_cols))))
    spline = CubicSpline(knots, corrections, axis=0, bc_type="clamped")
    active = slice(frame(84.5), frame(89.0) + 1)
    joints[active, left_cols] += spline(np.arange(active.start, active.stop))
    joints[active, left_cols] = np.clip(joints[active, left_cols], left_low, left_high)
    return refresh_fk(replace(base, joint_pos=joints.astype(np.float32)), MODEL)


def slice_npz(source: Path, target: Path, begin: int, end: int):
    with np.load(source, allow_pickle=False) as data:
        values = {}
        for key in data.files:
            value = np.asarray(data[key])
            if value.ndim and value.shape[0] == 800:
                value = value[begin:end]
            if key == "source_range_s":
                value = np.asarray([75 + begin / 50, 75 + end / 50])
            values[key] = value
    np.savez_compressed(target, **values)


def save_clip(motion, source, out, begin, end, *, include_chair=True):
    out.mkdir(parents=True, exist_ok=False)
    sliced = replace(motion, **{field: getattr(motion, field)[begin:end]
                                for field in ("joint_pos", "joint_vel", "body_pos_w",
                                              "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")})
    sliced = refresh_fk(sliced, MODEL)
    sliced.save(out / "motion.npz")
    slice_npz(source / "human_motion_world.npz", out / "human_motion_world.npz", begin, end)
    slice_npz(HERE / "human.npz", out / "human.npz", begin, end)
    slice_npz(HERE / "source_sync.npz", out / "source_sync.npz", begin, end)
    if include_chair:
        slice_npz(source / "object_estimate.npz", out / "object_estimate.npz", begin, end)
    return sliced


def audit_chair(mode, motion, out, chair_position, chair_rotation):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mesh = trimesh.load_mesh(HERE / "grey_armchair_visual.obj")
    addresses = np.asarray([model.joint(n).qposadr[0] for n in motion.joint_names])
    scores = {side: {part: [] for part in ("thumb_tip", "thumb_proximal", "thumb_distal", "index_tip", "middle_tip")}
              for side in ("left", "right")}
    for k in range(frame(79.2), frame(83.4) + 1, 5):
        pose(model, data, motion, k, addresses)
        for side in scores:
            for part in scores[side]:
                digit, location = part.split("_", 1)
                if location == "tip":
                    point = data.xpos[model.body(f"{side}_hand_{digit}_finger_tip").id]
                else:
                    which = "1" if location == "proximal" else "2"
                    point = data.geom_xpos[model.geom(f"{side}_hand_thumb_{which}_collision").id]
                local = chair_rotation[k].T @ (point - chair_position[k])
                _, distance, _ = trimesh.proximity.closest_point(mesh, local[None])
                scores[side][part].append(float(distance[0]))
    report = {side: {part: {"mean_m": float(np.mean(values)), "max_m": float(max(values))}
                     for part, values in parts.items()} for side, parts in scores.items()}
    if mode.startswith("thumb"):
        report["thumb_root_max_abs_rad"] = max(float(np.max(np.abs(
            motion.joint_pos[frame(79.2):frame(83.4) + 1,
                             motion.joint_names.index(f"{side}_hand_thumb_0_joint")])))
            for side in ("left", "right"))
    (out / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_table(motion, out):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(n).qposadr[0] for n in motion.joint_names])
    palm = model.geom("right_hand_palm_collision").id
    left_palm = model.geom("left_hand_palm_collision").id
    forearm = model.geom("right_elbow_yaw_collision").id
    wrist = model.geom("right_wrist_collision").id
    stats = {key: [] for key in ("right_palm_y", "right_palm_bottom_z", "right_palm_down_cos",
                                 "right_forearm_bottom_z", "right_wrist_bottom_z", "left_palm_bottom_z")}
    for k in range(frame(85.3) - TABLE_BEGIN, frame(87.8) - TABLE_BEGIN + 1, 5):
        pose(model, data, motion, k, addresses)
        stats["right_palm_y"].append(float(data.geom_xpos[palm, 1]))
        stats["right_palm_bottom_z"].append(float(data.geom_xpos[palm, 2] - model.geom_size[palm, 0]))
        stats["right_palm_down_cos"].append(float(-data.geom_xmat[palm].reshape(3, 3)[2, 0]))
        stats["right_forearm_bottom_z"].append(capsule_bottom(model, data, forearm))
        stats["right_wrist_bottom_z"].append(capsule_bottom(model, data, wrist))
        stats["left_palm_bottom_z"].append(float(data.geom_xpos[left_palm, 2] - model.geom_size[left_palm, 0]))
    report = {key: {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}
              for key, values in stats.items()}
    exact = {}
    for side in ("left", "right"):
        geometries = []
        for geom in range(model.ngeom):
            if not model.geom(geom).name.startswith(f"{side}_hand_"):
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = model.geom_dataid[geom]
            vertices = model.mesh_vert[
                model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
            ]
            geometries.append((geom, vertices))
        minimum, worst, penetrated = float("inf"), None, 0
        for k in range(0, frame(90) - TABLE_BEGIN):
            pose(model, data, motion, k, addresses)
            for geom, vertices in geometries:
                world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                inside = ((world[:, 0] >= -1.03) & (world[:, 0] <= .57)
                          & (world[:, 1] >= -2.97) & (world[:, 1] <= TABLE_Y_EDGE))
                if not inside.any():
                    continue
                height = float(world[inside, 2].min())
                if height < minimum:
                    minimum, worst = height, 83.5 + k / 50
                penetrated += int(height < TABLE_Z)
        exact[side] = {"minimum_inside_tabletop_z_m": minimum,
                       "worst_recording_time_s": worst,
                       "penetrating_geom_frame_count": penetrated}
    report["exact_hand_mesh_tabletop_sweep"] = exact
    report["maximum_joint_speed_rad_s"] = float(np.max(np.abs(
        np.gradient(motion.joint_pos, 1 / motion.fps, axis=0)
    )))
    limits = np.asarray([model.joint(name).range for name in motion.joint_names])
    report["joint_limit_max_violation_rad"] = float(max(
        np.max(limits[:, 0] - motion.joint_pos),
        np.max(motion.joint_pos - limits[:, 1]), 0,
    ))
    report["tabletop_z_m"] = TABLE_Z
    report["table_near_edge_y_m"] = TABLE_Y_EDGE
    (out / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--edge-candidates-only", action="store_true")
    parser.add_argument("--surface-candidates-only", action="store_true")
    parser.add_argument("--thumb-strip-only", action="store_true")
    parser.add_argument("--thumb-backface-only", action="store_true")
    parser.add_argument("--thumb-backface-flip-only", action="store_true")
    args = parser.parse_args()
    if sum((args.edge_candidates_only, args.surface_candidates_only,
            args.thumb_strip_only, args.thumb_backface_only,
            args.thumb_backface_flip_only)) > 1:
        parser.error("choose only one candidate group")
    out = args.out if args.out else run_dir("sequence_1019/armchair_split_feedback")
    out.mkdir(parents=True, exist_ok=True)
    with np.load(HERE / "hoi_v3_thumb_contact/object_estimate.npz", allow_pickle=False) as chair:
        chair_position = chair["position_w"]
        chair_quat = chair["quaternion_xyzw"]
        chair_rotation = Rotation.from_quat(chair_quat).as_matrix()
    if (args.surface_candidates_only or args.thumb_strip_only or args.thumb_backface_only
            or args.thumb_backface_flip_only):
        # Give both hands clearance during approach without changing the
        # post-push chair pose shared with the accepted table-brace clip.
        chair_position = chair_position.copy()
        for k in range(len(chair_position)):
            time = 75 + k / 50
            chair_position[k, 1] -= .06 * smoothstep((time - 77.45) / .75) * (
                1 - smoothstep((time - 80.0) / 3.4))
        with np.load(HERE / "human.npz", allow_pickle=False) as source_human:
            human = source_human["keypoints_w"]
    else:
        human = None
    with np.load(HERE / "hand_camera_reference.npz", allow_pickle=False) as rigid:
        rigid = {key: rigid[key] for key in rigid.files}
    result = {"output": str(out), "source": str(HERE), "clips": {}}
    modes = (("thumb_backface_flip",) if args.thumb_backface_flip_only else
             ("thumb_backface",) if args.thumb_backface_only else
             ("thumb_strip",) if args.thumb_strip_only else
             ("thumb_surface", "palm_surface") if args.surface_candidates_only else
             ("thumb_edge", "palm_edge") if args.edge_candidates_only else ("thumb", "palm"))
    for mode in modes:
        source = HERE / f"hoi_v3_{'thumb_contact' if mode.startswith('thumb') else 'palm_push'}"
        base = MotionFields.load(source / "motion.npz", ORDER)
        refined, fitting = revise_chair_hand(mode, base, chair_position, chair_rotation, human)
        target = out / f"chair_{mode}"
        sliced = save_clip(refined, source, target, CHAIR_BEGIN, CHAIR_END)
        if (args.surface_candidates_only or args.thumb_strip_only or args.thumb_backface_only
                or args.thumb_backface_flip_only):
            np.savez_compressed(
                target / "object_estimate.npz",
                position_w=chair_position[CHAIR_BEGIN:CHAIR_END],
                quaternion_xyzw=chair_quat[CHAIR_BEGIN:CHAIR_END],
                fps=np.asarray(50),
                source_range_s=np.asarray([75.0, 84.5]),
                trajectory_source=np.asarray("human_wrist_fit_approach_shift_6cm_kinematic"),
            )
        audit = audit_chair(mode, sliced, target, chair_position, chair_rotation)
        result["clips"][f"chair_{mode}"] = {"source_range_s": [75, 84.5],
                                            "fitting": fitting, "audit": audit}
    if not (args.edge_candidates_only or args.surface_candidates_only
            or args.thumb_strip_only or args.thumb_backface_only
            or args.thumb_backface_flip_only):
        source = HERE / "hoi_v3_palm_push"
        base = MotionFields.load(source / "motion.npz", ORDER)
        refined = revise_table_support(base, rigid)
        target = out / "table_support"
        sliced = save_clip(refined, source, target, TABLE_BEGIN, TABLE_END)
        report = audit_table(sliced, target)
        result["clips"]["table_support"] = {"source_range_s": [83.5, 91], "audit": report}
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
