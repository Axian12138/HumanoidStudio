"""DexAgent-inspired contact-point grasp proposal for the G1 Dex3 brush task.

This deliberately optimizes a virtual wrist SE(3) and seven finger joints only.
It does not claim that the whole-body tracker can attain the proposed wrist.
"""
from __future__ import annotations

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256
from scene import BRUSH_POSITION, BRUSH_QUAT_XYZW

ROOT = example_dir(__file__)
MODEL = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/robot.xml")
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
SOURCE = ROOT / "retarget/handle_kinematic_v1/motion.npz"
OUT = ROOT / "retarget/dexagent_hand_only_v1"
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
TIP_BODIES = ("right_hand_thumb_finger_tip", "right_hand_index_finger_tip", "right_hand_middle_finger_tip")
FRAME = 560


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def main():
    motion = MotionFields.load(SOURCE, ORDER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    qaddr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    f_names = ["right_hand_" + n + "_joint" for n in FINGERS]
    f_cols = np.array([motion.joint_names.index(n) for n in f_names])
    f_addr = np.array([model.joint(n).qposadr[0] for n in f_names])
    lower = np.array([model.joint(n).range[0] for n in f_names])
    upper = np.array([model.joint(n).range[1] for n in f_names])
    data.qpos[:3] = motion.body_pos_w[FRAME, 0]
    data.qpos[3:7] = motion.body_quat_w[FRAME, 0]
    data.qpos[qaddr] = motion.joint_pos[FRAME]
    mujoco.mj_forward(model, data)
    wrist = model.body("right_wrist_yaw_link").id
    tips = [model.body(n).id for n in TIP_BODIES]
    wrist_p = data.xpos[wrist].copy()
    wrist_R = data.xmat[wrist].reshape(3, 3).copy()
    brush_R = Rotation.from_quat(BRUSH_QUAT_XYZW)
    brush_p = np.asarray(BRUSH_POSITION)
    # Two opposing sides of the actual thin grip. The source motion supplies
    # the contact frame and index/middle ordering; the target surfaces supply
    # the DexAgent-style contact constraints.
    surface_local = np.array([
        [-0.080, 0.003, 0.0518],  # thumb, under the handle
        [-0.050, 0.003, 0.0693],  # index, above the handle
        [-0.110, 0.003, 0.0693],  # middle, above the handle
    ])
    normals_local = np.array([[0, 0, -1], [0, 0, 1], [0, 0, 1]], float)
    # A fingertip body is a kinematic landmark, not its collision surface.
    # Keep 4 mm outside each surface; collision distance is audited separately.
    targets = brush_p + brush_R.apply(surface_local + 0.004 * normals_local)
    prior_f = motion.joint_pos[FRAME, f_cols].astype(float)
    collision = []
    for name in ("thumb_2", "index_1", "middle_1"):
        gid = model.geom("right_hand_" + name + "_collision").id
        mid = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        # MuJoCo's compiled mesh vertices are in the geom frame.
        collision.append([gid, vertices.copy(), vertices[::max(1, len(vertices) // 80)].copy()])
    grip_mesh = trimesh.load(ROOT / "assets/green_hand_brush/collision_hulls/Hull_grip.obj", force="mesh")
    grip_mesh.apply_transform(np.block([[brush_R.as_matrix(), brush_p[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))

    def tip_positions(q):
        data.qpos[f_addr] = q
        mujoco.mj_forward(model, data)
        return np.stack([data.xpos[b] for b in tips])

    def residual(x):
        delta_p, delta_r, finger = x[:3], x[3:6], x[6:]
        rot = Rotation.from_rotvec(delta_r)
        proposed = wrist_p + delta_p + rot.apply(tip_positions(finger) - wrist_p)
        contact_local = brush_R.inv().apply(proposed - brush_p)
        contact_error = contact_local - (surface_local + 0.004 * normals_local)
        mesh_pen = []
        for gid, _, vertices in collision:
            world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            moved = wrist_p + delta_p + rot.apply(world - wrist_p)
            signed = trimesh.proximity.signed_distance(grip_mesh, moved)
            mesh_pen.append(max(float(np.max(signed)) - 0.001, 0.0) / 0.001)
        return np.r_[
            (contact_error / np.array([0.020, 0.005, 0.005])).ravel(),
            mesh_pen,
            delta_p / 0.08,
            delta_r / 0.6,
            (finger - prior_f) / 1.0,
        ]

    bounds_l = np.r_[[-0.15] * 3, [-1.0] * 3, lower + 1e-5]
    bounds_u = np.r_[[0.15] * 3, [1.0] * 3, upper - 1e-5]
    seeds = [np.r_[np.zeros(6), np.clip(prior_f, lower + 1e-5, upper - 1e-5)]]
    # A second seed uses the earlier fitted closed hand; it is only a prior.
    fit = json.loads((ROOT / "retarget/handle_kinematic_v1/review.json").read_text())
    seeds.append(np.r_[np.zeros(6), np.clip(fit["closed_finger_angles_rad"], lower + 1e-5, upper - 1e-5)])
    solutions = [least_squares(residual, s, bounds=(bounds_l, bounds_u), max_nfev=150)
                 for s in seeds]
    best = min(solutions, key=lambda s: np.linalg.norm(s.fun))
    # Add the actual deepest collision-mesh vertices to the constraint set.
    # Uniform subsampling can miss a narrow fingertip ridge.
    for _ in range(3):
        tip_positions(best.x[6:])
        rot = Rotation.from_rotvec(best.x[3:6])
        for item in collision:
            gid, full, selected = item
            world = full @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            moved = wrist_p + best.x[:3] + rot.apply(world - wrist_p)
            signed = trimesh.proximity.signed_distance(grip_mesh, moved)
            deep = full[np.argsort(signed)[-120:]]
            item[2] = np.unique(np.concatenate([selected, deep]), axis=0)
        best = least_squares(residual, best.x, bounds=(bounds_l, bounds_u), max_nfev=100)
    x = best.x
    achieved = wrist_p + x[:3] + Rotation.from_rotvec(x[3:6]).apply(tip_positions(x[6:]) - wrist_p)
    wrist_target_p = wrist_p + x[:3]
    wrist_target_R = Rotation.from_rotvec(x[3:6]).as_matrix() @ wrist_R
    tip_gap_mm = {}
    for name in ("thumb_2", "index_1", "middle_1"):
        body = model.body("right_hand_" + name + "_link").id
        vertices = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != body or model.geom_group[gid] != 3 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[gid]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            vertices.append(v)
        if vertices:
            v = np.concatenate(vertices)
            gid = model.geom("right_hand_" + name + "_collision").id
            world = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            moved = wrist_target_p + (world - wrist_p) @ wrist_R @ wrist_target_R.T
            signed = trimesh.proximity.signed_distance(grip_mesh, moved)
            tip_gap_mm[name] = {"closest_surface_mm": float(1000 * np.min(np.abs(signed))),
                                "maximum_penetration_mm": float(1000 * max(np.max(signed), 0.0))}
    # Roll the hand-only proposal into source time. The source's body and arm
    # trajectories are not modified; consumers must solve their own IK.
    frames = np.arange(450, 661)
    weight = smooth((frames - 480) / 42) * (1 - smooth((frames - 600) / 60))
    finger_open = motion.joint_pos[500, f_cols].astype(float)
    closing = smooth((frames - 518) / 42)
    opening = smooth((frames - 450) / 50)
    preclose = motion.joint_pos[np.ix_(frames, f_cols)] * (1 - opening[:, None]) + finger_open[None] * opening[:, None]
    finger_q = preclose * (1 - closing[:, None]) + x[6:][None] * closing[:, None]
    wrist_p_series = motion.body_pos_w[frames, motion.body_names.index("right_wrist_yaw_link")].astype(float) + weight[:, None] * x[:3]
    wrist_quat_series = np.empty((len(frames), 4))
    for i, f in enumerate(frames):
        prior_q = motion.body_quat_w[f, motion.body_names.index("right_wrist_yaw_link"), [1, 2, 3, 0]]
        wrist_quat_series[i] = (Rotation.from_rotvec(weight[i] * x[3:6]) * Rotation.from_quat(prior_q)).as_quat()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "hand_targets.npz", frame=frames, wrist_position_w=wrist_p_series,
                        wrist_quaternion_xyzw=wrist_quat_series, finger_joint_names=np.array(f_names),
                        finger_joint_q=finger_q, contact_frame=np.array(FRAME))
    report = {
        "status": "hand_only_kinematic_proposal_unreviewed",
        "method": "Manual brush-surface contacts and DexAgent-inspired multi-seed wrist/finger IK on G1 Dex3 FK; no DexAgent tool or model executed",
        "dexagent_source": "https://github.com/dexagent1/DexAgent/tree/main/tool_library/skills",
        "source_motion": str(SOURCE), "source_motion_sha256": sha256(SOURCE),
        "model_sha256": sha256(MODEL), "contact_frame": FRAME, "fps": motion.fps,
        "changed_variables": "virtual right wrist SE(3) and seven right Dex3 finger joints only",
        "wrist_prior_position_w": wrist_p.tolist(),
        "wrist_target_position_w": wrist_target_p.tolist(),
        "wrist_target_quaternion_xyzw": Rotation.from_matrix(wrist_target_R).as_quat().tolist(),
        "wrist_translation_m": x[:3].tolist(), "wrist_rotation_rad": x[3:6].tolist(),
        "finger_joint_names": f_names, "finger_joint_q": x[6:].tolist(),
        "contact_surface_local_m": surface_local.tolist(),
        "fingertip_landmark_target_w": targets.tolist(),
        "fingertip_landmark_achieved_w": achieved.tolist(),
        "fingertip_landmark_error_mm": (1000 * np.linalg.norm(achieved - targets, axis=1)).tolist(),
        "tip_collision_mesh_to_grip": tip_gap_mm,
        "optimizer_cost": float(best.cost), "optimizer_success": bool(best.success),
        "hand_targets_sha256": sha256(OUT / "hand_targets.npz"),
        "arm_ik_verified": False, "physics_grasp_verified": False,
        "limitation": "Fingertip body landmarks are not contact pads; requires mesh penetration audit, arm IK, tracker playback and RSS free-object lift before training use.",
    }
    (OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "wrist_translation_m", "fingertip_landmark_error_mm", "optimizer_success")}, indent=2))


if __name__ == "__main__":
    main()
