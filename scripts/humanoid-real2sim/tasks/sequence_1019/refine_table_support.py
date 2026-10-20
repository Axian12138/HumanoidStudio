"""Fit the bracing right palm to the dining tabletop during 84.5–88.8 s."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from scipy.spatial.transform import Rotation as R

from retarget3.motion_io import export_motion
from retarget3.robot import MJCasadiRobot
from retarget3.robot import ca
from retarget3.robot import mujoco


ROOT = example_dir(__file__)
OUT = ROOT / "retarget/ground_support_27_end"
FPS = 50
SURFACE_Z = .754
BODY = "right_wrist_yaw_link"


def right_hand_collision_vertices(robot, q):
    """Express fixed-fist collision vertices in the wrist link frame."""
    model, data = robot.m, robot.d
    data.qpos[:3] = q[:3]
    data.qpos[3:7] = q[[6, 3, 4, 5]]
    data.qpos[7:] = q[7:]
    mujoco.mj_forward(model, data)
    wrist = model.body(BODY).id
    wrist_pos = data.xpos[wrist]
    wrist_rot = data.xmat[wrist].reshape(3, 3)
    pieces = []
    for g in range(model.ngeom):
        name = model.geom(g).name
        if not name.startswith("right_hand_") or model.geom_group[g] != 3:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        mesh = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        world = mesh @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        pieces.append((world-wrist_pos) @ wrist_rot)
    all_vertices = np.concatenate(pieces)
    # Sample support extrema in many directions. The exact collision meshes are
    # evaluated once more in the output audit, so approximation is explicit.
    k = np.arange(512)
    z = 1 - 2*(k+.5)/len(k)
    angle = k * np.pi * (3-np.sqrt(5))
    directions = np.column_stack([np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z])
    chosen = np.unique(np.argmin(all_vertices @ directions.T, axis=0))
    return all_vertices[chosen]


def main():
    robot = MJCasadiRobot()
    prior = np.load(OUT / "intermediate/support_refined_q.npy")
    names = [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    ids = np.asarray([robot.dof_name_to_index[name] for name in names])
    begin = round((84.0 - 27.0) * FPS)
    clear_end = round((90.0 - 27.0) * FPS)
    end = round((90.5 - 27.0) * FPS)
    core_begin = round((84.5 - 27.0) * FPS)
    core_end = round((88.8 - 27.0) * FPS)
    vertices = right_hand_collision_vertices(robot, prior[core_begin])
    pos, quat = robot.forward(prior[core_begin:core_end], [BODY])
    rot = R.from_quat(quat[:, 0][:, [1, 2, 3, 0]]).as_matrix()
    center = pos[:, 0] + np.einsum("tij,j->ti", rot, [.084, 0, 0])
    anchor = np.median(center[:, :2], axis=0)
    x = ca.SX.sym("right_arm", len(ids))
    row = ca.SX.sym("body_row", robot.nq)
    seed = ca.SX(row)
    seed[ids.tolist()] = x
    world_p = robot.world_pos(BODY, seed)
    world_r = robot.world_rotmat(BODY, seed)
    palm = world_p + world_r @ ca.DM([.084, 0, 0])
    visible_z = world_p[2] + ca.mtimes(world_r[2, :], ca.DM(vertices.T))
    bottom = -1e-4 * ca.logsumexp(-visible_z.T / 1e-4)
    ref = ca.SX.sym("joint_ref", len(ids))
    previous = ca.SX.sym("previous", len(ids))
    target_xy = ca.SX.sym("table_xy", 2)
    gate = ca.SX.sym("gate")
    # The same measured surface drives both placement and penetration bounds.
    objective = 4*ca.sumsqr(x-ref) + 8*ca.sumsqr(x-previous)
    objective += gate*(800*ca.sumsqr(palm[:2]-target_xy)
        + 30000*(bottom-(SURFACE_Z+.0015))**2
        + 250*(world_r[2, 2]-1)**2)
    solver = ca.nlpsol("table_support", "ipopt", {
        "x": x, "p": ca.vertcat(row, ref, previous, target_xy, gate), "f": objective,
        "g": bottom,
    }, {"print_time": False, "ipopt": {"print_level": 0, "max_iter": 500, "tol": 1e-6}})
    output = prior.copy()
    previous_x = prior[begin, ids].copy()
    statuses = []
    for k in range(begin, end):
        in_weight = np.clip((k-begin)/(core_begin-begin), 0, 1)
        out_weight = np.clip((round((89.3-27.0)*FPS)-k)/(round(.5*FPS)), 0, 1)
        amount = min(in_weight, out_weight)
        amount = amount*amount*(3-2*amount)
        bound_low = SURFACE_Z+.0007 if core_begin <= k < core_end else (SURFACE_Z+.0003 if core_end <= k < clear_end else -np.inf)
        bound_high = SURFACE_Z+.003 if core_begin <= k < core_end else np.inf
        params = np.r_[prior[k], prior[k, ids], previous_x, anchor, amount]
        sol = solver(x0=previous_x, p=params, lbx=robot.q_lower[ids], ubx=robot.q_upper[ids],
                     lbg=[bound_low], ubg=[bound_high])
        if not solver.stats()["success"]:
            raise RuntimeError(f"Table support fit failed at absolute frame {k+27*FPS}: {solver.stats()['return_status']}")
        previous_x = np.asarray(sol["x"]).ravel()
        output[k, ids] = previous_x
        statuses.append({"frame": k, "gate": float(amount), "bottom_z": float(sol["g"]),
                         "status": solver.stats()["return_status"]})
    # Smoothly rejoin the original arm at both ends of the support window.
    for k in range(begin, end):
        feather = min(1.0, (k-begin)/25, (end-1-k)/(end-clear_end-1))
        feather = feather*feather*(3-2*feather)
        output[k, ids] = prior[k, ids] + feather*(output[k, ids]-prior[k, ids])
    # Exact seam and speed check before promoting the support-adjusted prior.
    ratio = np.abs(np.diff(output[:, 7:], axis=0))*FPS/robot.v_abs[6:]
    report = {
        "support_s": [84.5, 88.8], "clearance_until_s": 90.0, "tabletop_z_m": SURFACE_Z,
        "anchor_xy_m": anchor.tolist(), "max_joint_velocity_ratio": float(ratio.max()),
        "max_joint_step_at_window_edges_rad": float(max(
            np.max(abs(output[begin, ids]-prior[begin, ids])),
            np.max(abs(output[end-1, ids]-prior[end-1, ids])))),
        "minimum_hand_collision_z_m": float(min(x["bottom_z"] for x in statuses)),
        "hand_collision_z_m_during_core": [
            float(min(x["bottom_z"] for x in statuses if core_begin <= x["frame"] < core_end)),
            float(max(x["bottom_z"] for x in statuses if core_begin <= x["frame"] < core_end))],
    }
    (OUT / "review/table_support.json").write_text(json.dumps({"summary": report, "frames": statuses}, indent=2) + "\n")
    if ratio.max() > 1.001:
        raise RuntimeError(f"Table support exceeds joint speed limit: {ratio.max():.3f}")
    np.save(OUT / "intermediate/body_prior_supported_q.npy", output)
    export_motion(OUT / "supported_motion.npz", robot, output, FPS)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
