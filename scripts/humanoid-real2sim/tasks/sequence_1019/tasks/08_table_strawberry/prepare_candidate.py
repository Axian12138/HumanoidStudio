"""Reproducible table strawberry candidate; reports failures without approval."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

REPO = Path(__file__).resolve().parents[6]
SNAP = REPO / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot"
ST = Path("/home/galbot/workspace/ScaleTrack")
MODEL = ST / "source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
ASSETS = ST / "data/1012test/hoi_stage2/scene/assets"
ORDER = SNAP / "retarget/ground_support_27_end/sidecars/motion_order.json"


def main():
    import mujoco
    import trimesh
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--version", default="v02")
    parser.add_argument("--grasp-design", type=Path)
    parser.add_argument("--geometry-config", type=Path)
    parser.add_argument("--release-time", type=float, default=96.7)
    args = parser.parse_args()
    release_time = args.release_time
    release_k = round((release_time - 90) * 50)
    release_shift = release_time - 96.7
    design = json.loads(args.grasp_design.read_text()) if args.grasp_design else None
    out = args.run / "retarget/candidates/08_table_strawberry" / args.version
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / "prepare_candidate_source.py")
    if args.grasp_design:
        shutil.copy2(args.grasp_design, out / "grasp_design_input.json")
    geometry = json.loads(args.geometry_config.read_text()) if args.geometry_config else {}
    if args.geometry_config:
        shutil.copy2(args.geometry_config, out / "geometry_config.json")
    room_transform = np.array(geometry.get("T_room_correction", np.eye(4)))
    prior = MotionFields.load(SNAP / "tasks/08_table_strawberry/body_prior.npz", ORDER)
    n = len(prior.joint_pos)
    times = 90 + np.arange(n) / 50
    q = prior.joint_pos.astype(float).copy()
    roots = prior.body_pos_w.copy()
    rootquat = prior.body_quat_w.copy()
    # Preserve already accepted overlapping 07 and 09 robot states exactly.
    for package, start, mask in [
        ("brace-dining-table-kinematic-v1", 83.5, times < 91),
        ("lined-rect-basket-kinematic-v1", 98, times >= 98),
    ]:
        z = np.load(ST / "data/1019test" / package / "motion.npz")
        indices = np.rint((times[mask] - start) * 50).astype(int)
        q[mask] = z["joint_pos"][indices]
        roots[mask, 0] = z["body_pos_w"][indices, 0]
        rootquat[mask, 0] = z["body_quat_w"][indices, 0]
    protected = q.copy()

    # Offset blending fixes joins without altering either accepted overlap.
    def blend(u):
        u = np.clip(u, 0, 1)
        return u * u * (3 - 2 * u)

    for k in range(50, 80):
        q[k] += (protected[49] - prior.joint_pos[49]) * (1 - blend((k - 49) / 31))
    for k in range(365, 400):
        q[k] += (protected[400] - prior.joint_pos[400]) * blend((k - 365) / 35)
    protected[50:400] = q[50:400]
    basket = np.load(ST / "data/1019test/lined-rect-basket-kinematic-v1/lined_rect_basket_trajectory.npz")
    bi = np.maximum(np.rint((times - 98) * 50).astype(int), 0)
    bp = basket["position_w"][bi]
    bq = basket["quaternion_wxyz"][bi][:, [1, 2, 3, 0]]
    br = Rotation.from_quat(bq).as_matrix()
    berry_mesh = trimesh.load_mesh(ASSETS / "object_strawberry_existing_0_0.obj")
    # The inherited 1012 strawberry y=-2.071 lies outside the measured table.
    # This is a support-consistent 1019 candidate, still pending camera fitting.
    initial = np.array([-0.822655201, -2.20, 0.754 - berry_mesh.bounds[0, 2]])
    initial = np.array(geometry.get("berry_position_world_m", initial))
    initial_quat = np.array(geometry.get("berry_quaternion_xyzw", [0, 0, 0, 1]))
    final_local = np.array([0.0, 0.0, 0.008 - berry_mesh.bounds[0, 2]])
    release = bp[release_k] + br[release_k] @ final_local
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.array([model.joint(name).qposadr[0] for name in prior.joint_names])
    arm = np.array(
        [
            prior.joint_names.index(f"right_{part}_joint")
            for part in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ]
    )
    hand = np.array([i for i, name in enumerate(prior.joint_names) if name.startswith("right_hand")])
    palm = model.geom("right_hand_palm_collision").id
    spec = importlib.util.spec_from_file_location(
        "basket_geometry", Path(__file__).parents[1] / "09_lined_rect_basket/fit_hoi.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    samples = [
        (g, geo.collision_points(model, g))
        for g in range(model.ngeom)
        if (model.geom(g).name or "").startswith("right_")
        and (model.geom(g).name or "").endswith("_collision")
        and any(s in model.geom(g).name for s in ("hand_", "wrist", "elbow_yaw"))
    ]
    lo = np.array([model.joint(prior.joint_names[i]).range[0] for i in arm]) + 1e-5
    hi = np.array([model.joint(prior.joint_names[i]).range[1] for i in arm]) - 1e-5
    # Original world hand transit is retained between measured pickup/place anchors.
    data.qpos[:3], data.qpos[3:7] = roots[400, 0], rootquat[400, 0]
    data.qpos[addresses] = q[400]
    mujoco.mj_forward(model, data)
    design_offset = np.array(design["object_in_palm_position"] if design else [0.055, 0, 0])
    successor_center = data.geom_xpos[palm] + data.geom_xmat[palm].reshape(3, 3) @ design_offset
    successor_face = data.geom_xmat[palm].reshape(3, 3)[:, 0].copy()
    way_t = [
        91.4,
        92.2,
        92.4,
        92.6,
        93.2,
        94,
        95,
        95.8,
        96.35 + release_shift * 0.5,
        release_time,
        97.1 + release_shift * 0.7,
        97.5 + release_shift * 0.3,
        97.8,
        98.0,
    ]
    outside = br[0] @ np.array([-1.0, 0.0, 0.0])
    way_p = [
        initial + np.array([0, 0.08, 0.12]),
        initial,
        initial,
        initial + np.array([0, 0, 0.015]),
        initial + np.array([-0.08, 0.05, 0.17]),
        [-1.14, -1.65, 0.94],
        [-1.4, -2.02, 0.97],
        bp[0] + [-0.12, 0, 0.25],
        bp[0] + [0, 0, 0.22],
        release + np.array([0, 0, 0.055]),
        successor_center + np.array([0, 0, 0.25]) + 0.12 * outside,
        successor_center + np.array([0, 0, 0.16]) + 0.10 * outside,
        successor_center + np.array([0, 0, 0.05]) + 0.06 * outside,
        successor_center,
    ]
    target = PchipInterpolator(way_t, way_p, axis=0)(np.clip(times, way_t[0], way_t[-1]))
    offset = np.array(design["object_in_palm_position"] if design else [0.055, 0, 0])
    desired_rotation = np.array(design["palm_rotation_world"]) if design else None
    errors = []
    for k in range(50, 400):
        t = times[k]
        data.qpos[:3], data.qpos[3:7] = roots[k, 0], rootquat[k, 0]
        # Command timing is fixed, not a freely optimized grasp parameter.
        close = np.clip((t - 91.9) / 0.5, 0, 1) * (1 - np.clip((t - (release_time - 0.2)) / 0.5, 0, 1))
        open_q = np.array([0.3, 0.08, 0.08, -0.2, 0.08, 0.08, -0.3])
        # Canonical closure designed once against the actual three-finger FK.
        # The command is then frozen across the carry, not refitted per frame.
        close_q = np.array([-0.042184, 1.014660, 0.886488, 0.724302, 0.996495, 0.902858, -0.649225])
        if design:
            close_q = np.array(design["joint_pos"][-7:])
        q[k, hand] = open_q * (1 - close) + close_q * close
        data.qpos[addresses] = q[k]
        successor_blend = blend((t - 97.3) / 0.7)
        baseline = prior.joint_pos[k, arm] * (1 - successor_blend) + protected[400, arm] * successor_blend
        seed = np.clip(q[k - 1, arm], lo, hi)

        def residual(x, k=k, successor_blend=successor_blend, baseline=baseline, seed=seed):
            data.qpos[addresses[arm]] = x
            mujoco.mj_forward(model, data)
            rot = data.geom_xmat[palm].reshape(3, 3)
            center = data.geom_xpos[palm] + rot @ offset
            clearance = []
            for g, points in samples:
                world = points @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                # Table footprint comes from the inherited dining-table mesh.
                room_local = (world - room_transform[:3, 3]) @ room_transform[:3, :3]
                inside = (
                    (room_local[:, 0] > -1.03)
                    & (room_local[:, 0] < 0.57)
                    & (room_local[:, 1] < -2.11905405)
                    & (room_local[:, 1] > -2.97)
                )
                clearance.append(np.maximum(0.756 - room_local[:, 2], 0) * inside)
                clearance.append(np.maximum(0.002 - geo.basket_sdf((world - bp[k]) @ br[k]), 0))
            face = desired_rotation[:, 0] if design else np.array([0, 0, -1])
            face_target = face * (1 - successor_blend) + successor_face * successor_blend
            axis_residual = (
                0.5 * (rot[:, 2] - desired_rotation[:, 2]) * (1 - successor_blend) if design else np.zeros(3)
            )
            return np.r_[
                18 * (center - target[k]),
                0.7 * (rot[:, 0] - face_target),
                axis_residual,
                0.09 * (x - baseline),
                0.12 * (x - seed),
                25 * np.concatenate(clearance),
            ]

        fit = least_squares(
            residual, seed, bounds=(np.maximum(lo, seed - 0.10), np.minimum(hi, seed + 0.10)), max_nfev=65
        )
        weight = blend((t - 91.0) / 0.8)
        q[k, arm] = protected[k, arm] * (1 - weight) + fit.x * weight
        hand_blend = blend((t - 97.0) / 1.0)
        hand_target = q[k, hand] * (1 - hand_blend) + protected[400, hand] * hand_blend
        q[k, hand] = protected[k, hand] * (1 - weight) + hand_target * weight
        errors.append(float(np.linalg.norm(residual(fit.x)[:3]) / 18))
    # Rejoin the immutable successor pose without extrapolating its incoming
    # velocity backward through the basket wall.
    for k in range(380, 400):
        weight = blend((k - 380) / 20)
        join_target = protected[400]
        q[k, arm] = q[k, arm] * (1 - weight) + join_target[arm] * weight
    # Project the short transition onto a collision-free neighborhood. This
    # changes only the candidate right arm; the successor remains untouched.
    dense_samples = []
    for g, points in samples:
        dense_points = points
        mesh = model.geom_dataid[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            first = model.mesh_vertadr[mesh]
            dense_points = model.mesh_vert[first : first + model.mesh_vertnum[mesh] : 10].copy()
        dense_samples.append((g, dense_points))
    for k in range(380, 400):
        data.qpos[:3], data.qpos[3:7] = roots[k, 0], rootquat[k, 0]
        data.qpos[addresses] = q[k]
        original = q[k, arm].copy()

        def transition_residual(x, k=k, original=original):
            data.qpos[addresses[arm]] = x
            mujoco.mj_forward(model, data)
            penalties = []
            for g, points in dense_samples:
                world = points @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                penalties.append(np.maximum(0.003 - geo.basket_sdf((world - bp[k]) @ br[k]), 0))
            return np.r_[0.1 * (x - original), 0.03 * (x - q[k - 1, arm]), 50 * np.concatenate(penalties)]

        fit = least_squares(
            transition_residual,
            original,
            bounds=(np.maximum(lo, original - 0.3), np.minimum(hi, original + 0.3)),
            max_nfev=80,
        )
        q[k, arm] = fit.x
    unrelated = np.setdiff1d(np.arange(len(prior.joint_names)), np.r_[arm, hand])
    if not np.array_equal(q[:, unrelated], protected[:, unrelated]):
        raise AssertionError("HOI modified non-interacting joints")
    motion = refresh_fk(replace(prior, joint_pos=q, body_pos_w=roots, body_quat_w=rootquat), MODEL)
    motion.save(out / "motion.npz")
    prior.save(out / "body_prior.npz")
    objp = np.tile(initial, (n, 1))
    objq = np.tile(initial_quat, (n, 1))
    contact_k = round((92.4 - 90) * 50)
    # Derive one immutable local object rotation at first held frame.
    palm_rotations = []
    palm_positions = []
    clearance_min = []
    for k in range(n):
        data.qpos[:3], data.qpos[3:7] = roots[k, 0], rootquat[k, 0]
        data.qpos[addresses] = q[k]
        mujoco.mj_forward(model, data)
        r = data.geom_xmat[palm].reshape(3, 3).copy()
        palm_rotations.append(r)
        palm_positions.append(data.geom_xpos[palm].copy())
        allpts = np.concatenate(
            [points @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] for g, points in samples]
        )
        clearance_min.append(float(geo.basket_sdf((allpts - bp[k]) @ br[k]).min()))
    palm_rotations = np.array(palm_rotations)
    palm_positions = np.array(palm_positions)
    held = (times >= 92.4) & (times < release_time)
    local_rotation = palm_rotations[contact_k].T @ Rotation.from_quat(initial_quat).as_matrix()
    grasp_offset = offset if design else palm_rotations[contact_k].T @ (initial - palm_positions[contact_k])
    objp[held] = palm_positions[held] + np.einsum("nij,j->ni", palm_rotations[held], grasp_offset)
    objq[held] = Rotation.from_matrix(palm_rotations[held] @ local_rotation).as_quat()
    released = times >= release_time
    release_pos = palm_positions[release_k] + palm_rotations[release_k] @ grasp_offset
    release_rot = palm_rotations[release_k] @ local_rotation
    bottom = float((np.asarray(berry_mesh.vertices) @ release_rot.T)[:, 2].min())
    final_local = br[release_k].T @ (release_pos - bp[release_k])
    final_local[2] = 0.008 - bottom
    release_local_rotation = br[release_k].T @ release_rot
    objp[released] = bp[released] + np.einsum("nij,j->ni", br[released], final_local)
    # Ballistic drop estimate, explicitly not a physics-verified resting result.
    height = max(0.0, release_pos[2] - (bp[release_k, 2] + final_local[2]))
    objp[released, 2] += np.maximum(height - 0.5 * 9.81 * (times[released] - release_time) ** 2, 0)
    objq[released] = Rotation.from_matrix(br[released] @ release_local_rotation).as_quat()
    for name, pos, quat in [("table_strawberry", objp, objq), ("lined_rect_basket", bp, bq)]:
        np.savez_compressed(out / f"{name}.npz", position_w=pos, quaternion_xyzw=quat, fps=50, source_time_s=times)
    # Full successor state continues inside the same 09 container pose.
    bfull = basket["position_w"]
    rq = Rotation.from_quat(basket["quaternion_wxyz"][:, [1, 2, 3, 0]])
    np.savez_compressed(
        out / "table_strawberry_successor_09.npz",
        position_w=bfull + rq.apply(final_local),
        quaternion_xyzw=Rotation.from_matrix(rq.as_matrix() @ release_local_rotation).as_quat(),
        fps=50,
        source_time_s=basket["recording_time_s"],
    )
    for name in ("source_sync.npz", "human.npz"):
        shutil.copy2(SNAP / "tasks/08_table_strawberry" / name, out / name)
    shutil.copy2(ORDER, out / "motion_order.json")
    (out / "tracks.json").write_text(
        json.dumps(
            {"objects": [{"name": name, "path": f"{name}.npz"} for name in ("table_strawberry", "lined_rect_basket")]},
            indent=2,
        )
    )
    report = {
        "release_time_s": release_time,
        "stage_1": "source timing must be reviewed against camera timestamps",
        "stage_2": "pending_live_same_camera_and_free_rest",
        "stage_3_1": "pending_agent_visual_review",
        "stage_3_2": "candidate_not_passed",
        "physics_interaction_verified": False,
        "training_data_ready": False,
        "max_target_residual_m": max(errors),
        "max_joint_speed_rad_s": float(np.abs(motion.joint_vel).max()),
        "held_target_residual_max_m": max(
            error for k, error in enumerate(errors, start=50) if 92.4 <= times[k] <= release_time
        ),
        "fixed_grasp_local_position_m": grasp_offset.tolist(),
        "release_drop_estimate_m": height,
        "joint_limit_violation_count": sum(
            int(((q[:, i] < model.joint(name).range[0] - 1e-5) | (q[:, i] > model.joint(name).range[1] + 1e-5)).sum())
            for i, name in enumerate(prior.joint_names)
        ),
        "basket_sample_min_distance_m": min(clearance_min),
        "basket_penetration_frames": [int(i) for i, x in enumerate(clearance_min) if x < -0.002],
        "protected_overlap_joint_max_delta": float(
            np.abs(q[(times < 91) | (times >= 98)] - protected[(times < 91) | (times >= 98)]).max()
        ),
        "object_transition_jump_m": {
            "pickup": float(np.linalg.norm(objp[120] - objp[119])),
            "release": float(np.linalg.norm(objp[release_k] - objp[release_k - 1])),
        },
        "unverified": [
            "continuous whole-body collision sweep",
            "fixed grasp closure envelope",
            "release support orientation",
            "camera calibration",
            "RSS live scene and free rest",
            "Stage 3.5 visual review",
        ],
        "inputs": {
            str(p): sha256(p)
            for p in [
                MODEL,
                ORDER,
                SNAP / "tasks/08_table_strawberry/body_prior.npz",
                ST / "data/1019test/lined-rect-basket-kinematic-v1/lined_rect_basket_trajectory.npz",
            ]
        },
    }
    (out / "review.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
