"""Fit local arms/digits to the two-object candidate; export with explicit review gates."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--right-checkpoint", type=Path)
    p.add_argument("--only-side", choices=["left", "right"])
    p.add_argument("--left-push-checkpoint", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    snapshot = a.run.parents[1] / "curated_snapshot"
    order = snapshot / "retarget/ground_support_27_end/sidecars/motion_order.json"
    prior = MotionFields.load(a.run / "retarget/body_prior/10_flower_bouquet/body_prior.npz", order)
    modelpath = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(modelpath))
    d = mujoco.MjData(m)
    address = np.array([m.joint(x).qposadr[0] for x in prior.joint_names])
    joints = prior.joint_pos.copy()
    n = len(joints)
    t = 108 + np.arange(n) / 50
    flower = np.load(a.tracks / "flower_bouquet.npz")
    basket = np.load(a.tracks / "small_wicker_basket.npz")
    R = Rotation.from_quat(flower["quaternion_xyzw"])
    geometry = json.loads((a.run / "scene/10_flower_bouquet/basket_registered_candidate.json").read_text())
    dx, _dy, dz = geometry["dimensions_world_aligned_m"]
    import trimesh
    from scipy.spatial import ConvexHull

    folder = a.assets or a.run / "scene/10_flower_bouquet/registered_assets"
    profile_path = folder / "wrapper_profile.json"
    profile = (
        json.loads(profile_path.read_text())
        if profile_path.exists()
        else {
            "lower_radius_m": 0.035,
            "upper_radius_m": 0.035,
            "height_m": 0.360078,
        }
    )

    def wrapper_radius(z):
        return profile["lower_radius_m"] + (profile["upper_radius_m"] - profile["lower_radius_m"]) * np.clip(
            z / profile["height_m"], 0, 1
        )

    grip_radius = wrapper_radius(0.25)
    collision_objects = {}
    collision_bounds = {}
    for name in ["flower_bouquet", "small_wicker_basket", "cabinet_support", "cabinet_side"]:
        paths = (
            list(folder.glob(name + "_hull*.obj"))
            if name in ["flower_bouquet", "small_wicker_basket"]
            else [folder / (name + ".obj")]
        )
        meshes = [trimesh.load(path, force="mesh", process=False).vertices for path in paths]
        collision_bounds[name] = [(v.min(axis=0), v.max(axis=0)) for v in meshes]
        collision_objects[name] = [
            np.unique(
                np.round(ConvexHull(trimesh.load(path, force="mesh", process=False).vertices).equations, 8), axis=0
            )
            for path in paths
        ]
    collision_samples = {}
    for g in range(m.ngeom):
        label = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not label.endswith("_collision"):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            voxel = 0.006 if "palm" in label else 0.004
            _, sample_indices = np.unique(np.round(v / voxel), axis=0, return_index=True)
            collision_samples[g] = (label, v[sample_indices].copy())
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius, half = m.geom_size[g, :2]
            v = np.array(
                [
                    [radius * np.cos(theta), radius * np.sin(theta), z]
                    for z in np.linspace(-half, half, 7)
                    for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False)
                ]
            )
            collision_samples[g] = (label, v)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            half = m.geom_size[g]
            v = np.array(
                [
                    [x, y, z]
                    for x in [-half[0], 0.0, half[0]]
                    for y in [-half[1], 0.0, half[1]]
                    for z in [-half[2], 0.0, half[2]]
                ]
            )
            collision_samples[g] = (label, v)
    reports = {}
    for side, intervals in [
        ("right", [(111.5, 113.2, 123.2, 125.2)]),
        ("left", [(116.8, 118.7, 120.4, 121.6), (124.1, 125.0, 127.8, 128.5)]),
    ]:
        if a.only_side and side != a.only_side:
            continue
        names = [
            f"{side}_{part}_joint"
            for part in [
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            ]
        ] + [x for x in prior.joint_names if x.startswith(side + "_hand_")]
        cols = np.array([prior.joint_names.index(x) for x in names])
        addr = address[cols]
        if side == "right" and a.right_checkpoint:
            saved = np.load(a.right_checkpoint)
            if saved["joint_names"].tolist() != names:
                raise ValueError("right checkpoint joint order mismatch")
            saved_keys = saved["keys"]
            span = np.arange(saved_keys[0], saved_keys[-1] + 1)
            joints[np.ix_(span, cols)] = PchipInterpolator(saved_keys, saved["solutions"], axis=0)(span)
            reports[side] = [{"reused_checkpoint": str(a.right_checkpoint)}]
            continue
        lo = np.array([m.joint(x).range[0] - 1e-7 for x in names])
        hi = np.array([m.joint(x).range[1] + 1e-7 for x in names])
        palm = m.geom(side + "_hand_palm_collision").id
        tips = [m.body(side + "_hand_" + x + "_finger_tip").id for x in ["thumb", "index", "middle"]]
        previous = None
        rows = []
        for begin, contact, release, finish in intervals:
            if side == "left" and begin > 124 and a.left_push_checkpoint:
                saved = np.load(a.left_push_checkpoint)
                if saved["joint_names"].tolist() != names:
                    raise ValueError("left checkpoint joint order mismatch")
                joints[np.ix_(saved["keys"], cols)] = saved["solutions"]
                continue
            previous = None
            keys = np.unique(
                np.r_[
                    np.arange(round((begin - 108) * 50), round((finish - 108) * 50) + 1, 1), round((finish - 108) * 50)
                ]
            ).astype(int)
            solutions = []
            endpoint = prior.joint_pos[keys[-1], cols].astype(float)
            planning_speed = 2.8  # Solve every 50 Hz sample; retain margin below the 3 rad/s fallback.
            for k in keys:
                d.qpos[:3] = prior.body_pos_w[k, 0]
                d.qpos[3:7] = prior.body_quat_w[k, 0]
                d.qpos[address] = prior.joint_pos[k]
                mujoco.mj_forward(m, d)
                base = prior.joint_pos[k, cols].astype(float)
                p0 = d.geom_xpos[palm].copy()
                tips0 = d.xpos[tips].copy()
                matrix0 = d.geom_xmat[palm].reshape(3, 3).copy()
                ramp = min(
                    np.clip((t[k] - begin) / (contact - begin), 0, 1),
                    np.clip((finish - t[k]) / (finish - release), 0, 1),
                )
                w = ramp * ramp * (3 - 2 * ramp)
                basket_frame = Rotation.identity()
                normal = np.zeros(3)
                axis = np.array([0.0, 0.0, 1.0])
                tangent = np.array([0.0, 1.0, 0.0])
                if side == "right":
                    center = flower["position_w"][k] + R[k].apply([0, 0, 0.25])
                    axis = R[k].as_matrix()[:, 2]
                    normal = R[k].apply([0.0, 1.0, 0])
                    tangent = np.cross(axis, normal)
                    pt = center + 0.075 * normal - 0.040 * tangent
                    tt = np.array(
                        [
                            center - 0.024 * normal + 0.014 * tangent,
                            center - 0.023 * normal - 0.018 * tangent + 0.02 * axis,
                            center - 0.023 * normal - 0.018 * tangent - 0.02 * axis,
                        ]
                    )
                    face = -normal
                    along = tangent
                else:
                    center = basket["position_w"][k]
                    pull = contact < 120
                    if pull:
                        pt = center + np.array([dx / 2 - 0.025, 0, dz + 0.065])
                        tt = np.array(
                            [
                                center + np.array([dx / 2 + 0.020, -0.03, dz - 0.02]),
                                center + np.array([dx / 2 - 0.025, 0.01, dz - 0.02]),
                                center + np.array([dx / 2 - 0.025, 0.04, dz - 0.02]),
                            ]
                        )
                        face = np.array([0, 0, -1])
                        along = np.array([1, 0, 0])
                    else:
                        pt = center + np.array([dx / 2 + 0.055, 0, dz * 0.62])
                        tt = np.array(
                            [
                                center + np.array([dx / 2 + 0.022, -0.035, dz * 0.55]),
                                center + np.array([dx / 2 + 0.022, 0.01, dz * 0.35]),
                                center + np.array([dx / 2 + 0.022, 0.04, dz * 0.35]),
                            ]
                        )
                        face = np.array([-1, 0, 0])
                        along = np.array([0, 0, -1])
                    basket_yaw = Rotation.from_quat(basket["quaternion_xyzw"][k]).as_euler("xyz")[2] - np.pi / 2
                    basket_frame = Rotation.from_euler("z", basket_yaw)
                    pt = center + basket_frame.apply(pt - center)
                    tt = center + basket_frame.apply(tt - center)
                    face = basket_frame.apply(face)
                    along = basket_frame.apply(along)
                target = (1 - w) * p0 + w * pt
                targettips = (1 - w) * tips0 + w * tt
                if side == "right":
                    hover = np.array([0.12 if t[k] > release else 0.0, 0.0, 0.16 if t[k] > release else 0.10]) * np.sin(
                        np.pi * ramp
                    )
                    target += hover
                    targettips += hover
                if side == "left":
                    hover = np.array([-0.04 if contact < 120 else 0.04, 0.0, 0.14]) * np.sin(np.pi * ramp)
                    target += hover
                    targettips += hover
                object_frames = {
                    name: (Rotation.from_quat(track["quaternion_xyzw"][k]).as_matrix(), track["position_w"][k])
                    for name, track in [("flower_bouquet", flower), ("small_wicker_basket", basket)]
                }
                active_samples = {
                    g: v
                    for g, (label, v) in collision_samples.items()
                    if label.startswith(side + "_hand_")
                    or label in [side + "_wrist_collision", side + "_elbow_yaw_collision"]
                }

                face0 = (1 if side == "right" else -1) * matrix0[:, 0]
                ft = (1 - w) * face0 + w * face
                at = (1 - w) * matrix0[:, 2] + w * along

                def residual(
                    q,
                    addr=addr,
                    palm=palm,
                    side=side,
                    target=target,
                    tips=tips,
                    targettips=targettips,
                    ft=ft,
                    at=at,
                    base=base,
                    previous=previous,
                    center=center,
                    axis=axis,
                    tangent=tangent,
                    normal=normal,
                    w=w,
                    object_frames=object_frames,
                    active_samples=active_samples,
                    contact=contact,
                    basket_frame=basket_frame,
                    ramp=ramp,
                ):
                    d.qpos[addr] = q
                    mujoco.mj_forward(m, d)
                    mat = d.geom_xmat[palm].reshape(3, 3)
                    f = (1 if side == "right" else -1) * mat[:, 0]
                    tip_error = 35 * (d.xpos[tips] - targettips).ravel()
                    if side == "left" and contact < 120:
                        tip_error[6:9] *= 1 - w
                    if side == "right":
                        delta = d.xpos[tips] - center
                        axial = delta @ axis
                        radial = np.linalg.norm(delta - axial[:, None] * axis, axis=1)
                        across = delta @ tangent
                        surface = np.r_[
                            10 * (radial - grip_radius - 0.008),
                            40 * np.maximum(abs(axial) - 0.065, 0),
                            25 * (across - np.array([0.025, -0.025, -0.025])),
                        ]
                        tip_error = (1 - w) * tip_error + w * surface
                    clearance = []
                    pad_contact = []
                    for g, v in active_samples.items():
                        world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                        label = collision_samples[g][0]
                        if side == "left":
                            basket_local = basket_frame.inv().apply(world - center)
                            is_pad = any(
                                part in label
                                for part in (
                                    ["thumb_2", "index_1"] if contact < 120 else ["thumb_2", "index_1", "middle_1"]
                                )
                            )
                            if is_pad:
                                face_x = dx / 2 if contact > 120 or "thumb_2" in label else dx / 2 - 0.012
                                valid_height = (basket_local[:, 2] > 0.01) & (basket_local[:, 2] < dz)
                                face_gap = np.min(
                                    np.where(
                                        valid_height,
                                        np.abs(basket_local[:, 0] - face_x),
                                        1 + np.abs(basket_local[:, 2] - dz),
                                    )
                                )
                                if contact < 120:
                                    lateral = -0.03 if "thumb_2" in label else (0.01 if "index_1" in label else 0.04)
                                    desired = np.array([face_x, lateral, dz - 0.015])
                                    desired += np.array([.03 if "thumb_2" in label else -.04,0,.10])*np.sin(np.pi*ramp)
                                    nearest = int(np.argmin(np.linalg.norm(basket_local - desired, axis=1)))
                                    pad_contact.extend(300 * w * (basket_local[nearest] - desired))
                                else:
                                    pad_contact.append(150 * w * face_gap)
                        if side == "left" and contact > 120:
                            basket_local = basket_frame.inv().apply(world - center)
                            clearance.extend(np.maximum(dx / 2 + 0.001 - basket_local[:, 0], 0.0))
                        if side == "right" and any(part in label for part in ["thumb_2", "index_1", "middle_1"]):
                            intended_side = 1.0 if "thumb_2" in label else -1.0
                            axial_offset = 0.0 if "thumb_2" in label else (0.03 if "index_1" in label else -0.03)
                            surface_normal = intended_side * 0.85 * tangent + np.sqrt(1 - 0.85**2) * normal
                            contact_target = (
                                center + axial_offset * axis + wrapper_radius(0.25 + axial_offset) * surface_normal
                            )
                            nearest = int(np.argmin(np.linalg.norm(world - contact_target, axis=1)))
                            pad_contact.extend(300 * w * (world[nearest] - contact_target))
                        for name, planes in collision_objects.items():
                            rot, pos = object_frames.get(name, (np.eye(3), np.zeros(3)))
                            local = (world - pos) @ rot
                            hull_distances = []
                            for e, (low, high) in zip(planes, collision_bounds[name], strict=True):
                                near = np.all((local >= low - 0.005) & (local <= high + 0.005), axis=1)
                                distance = np.full(len(local), 0.005)
                                distance[near] = np.max(local[near] @ e[:, :3].T + e[:, 3], axis=1)
                                hull_distances.append(distance)
                            distances = np.min(hull_distances, axis=0)
                            clearance.extend(np.maximum(0.001 - distances, 0.0))
                    return np.r_[
                        2000 * np.asarray(clearance),
                        pad_contact,
                        (35 if side == "right" else 80) * (d.geom_xpos[palm] - target),
                        tip_error,
                        10 * (f - ft),
                        10 * (mat[:, 2] - at),
                        (0.5 + 100 * (1 - w) ** 4) * (q[:7] - base[:7]),
                        (0.08 + 100 * (1 - w) ** 4) * (q[7:] - base[7:]),
                        0.12 * (q - previous) if previous is not None else np.zeros(len(q)),
                    ]

                delta_t = 0.0 if previous is None else (k - keys[len(solutions) - 1]) / 50
                remaining = (keys[-1] - k) / 50
                lower = np.maximum(lo, endpoint - planning_speed * remaining)
                upper = np.minimum(hi, endpoint + planning_speed * remaining)
                if previous is not None:
                    lower = np.maximum(lower, previous - planning_speed * delta_t)
                    upper = np.minimum(upper, previous + planning_speed * delta_t)
                if k in [keys[0], keys[-1]]:
                    lower, upper = base - 1e-9, base + 1e-9
                collapsed = lower >= upper
                if np.any(collapsed):
                    if np.max((lower - upper)[collapsed]) > 1e-6:
                        raise ValueError(
                            f"{side} infeasible rate/endpoint bounds at source{t[k]}: {(lower - upper).tolist()}"
                        )
                    midpoint = (lower[collapsed] + upper[collapsed]) / 2
                    lower[collapsed] = midpoint - 1e-8
                    upper[collapsed] = midpoint + 1e-8
                seed = np.clip(base if previous is None else previous, lower, upper)
                sol = least_squares(residual, seed, bounds=(lower, upper), max_nfev=65)
                previous = sol.x
                solutions.append(base.copy() if k in [keys[0], keys[-1]] else sol.x.copy())
                if len(solutions) % 50 == 0:
                    (a.out / "progress.json").write_text(
                        json.dumps({"side": side, "source_s": float(t[k]), "interval": [begin, finish]})
                    )
                residual(sol.x)
                rows.append(
                    {
                        "source_s": float(t[k]),
                        "contact_weight": float(w),
                        "palm_target_error_m": float(np.linalg.norm(d.geom_xpos[palm] - target)),
                        "tips_target_error_max_m": float(np.max(np.linalg.norm(d.xpos[tips] - targettips, axis=1))),
                        "right_stem_radial_gap_m": (
                            (
                                np.linalg.norm(
                                    (d.xpos[tips] - center) - ((d.xpos[tips] - center) @ axis)[:, None] * axis, axis=1
                                )
                                - grip_radius
                                - 0.008
                            ).tolist()
                            if side == "right"
                            else None
                        ),
                        "right_tip_tangent_positions_m": (
                            ((d.xpos[tips] - center) @ tangent).tolist() if side == "right" else None
                        ),
                    }
                )
            span = np.arange(keys[0], keys[-1] + 1)
            joints[np.ix_(span, cols)] = PchipInterpolator(keys, np.array(solutions), axis=0)(span)
            np.savez(
                a.out / f"checkpoint_{side}_{begin:g}.npz",
                keys=keys,
                joint_names=np.array(names),
                solutions=np.array(solutions),
            )
        reports[side] = rows
    free_finger_changes = {}
    for side, intervals in [
        ("right", [(111.5, 125.2)]),
        ("left", [(116.8, 121.6), (124.1, 128.5)]),
    ]:
        columns = np.array([i for i, name in enumerate(prior.joint_names) if name.startswith(side + "_hand_")])
        free = np.arange(n) >= 50
        for begin, finish in intervals:
            free &= ~((t >= begin - 1e-8) & (t <= finish + 1e-8))
        boundaries = np.diff(np.r_[False, free, False].astype(int))
        before = joints[:, columns].copy()
        for start, end_exclusive in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1), strict=True):
            for k in range(start, end_exclusive):
                lower = joints[k - 1, columns].astype(float) - 3 / 50
                upper = joints[k - 1, columns].astype(float) + 3 / 50
                if end_exclusive < n:
                    lower = np.maximum(lower, joints[end_exclusive, columns] - 3 * (end_exclusive - k) / 50)
                    upper = np.minimum(upper, joints[end_exclusive, columns] + 3 * (end_exclusive - k) / 50)
                if np.any(lower > upper + 1e-6):
                    raise ValueError(f"{side} free finger interval cannot reach fixed contact boundary at frame{k}")
                joints[k, columns] = np.clip(joints[k, columns], lower, upper)
        free_finger_changes[side] = int(np.count_nonzero(abs(joints[:, columns] - before) > 1e-7))
    # Prefix q/root remains exactly the accepted 09 overlap. Derivatives are refreshed centrally.
    joints[:50] = prior.joint_pos[:50]
    motion = refresh_fk(replace(prior, joint_pos=joints), modelpath)
    motion.save(a.out / "motion.npz")
    (a.out / "review.json").write_text(
        json.dumps(
            {
                "status": "unreviewed IK candidate; full collision audit required before acceptance",
                "body_prior_visual_gate": "independent review completed; see review/stage3/10_body_prior_independent_review.json",
                "tip_metric_note": "right tip Cartesian targets are orientation guides; actual mesh pad radial gap and radial direction constrain the frustum contact",
                "source_range_s": [108, float(t[-1])],
                "collision_assets": str(folder),
                "wrapper_profile": profile,
                "velocity_contract": "absolute 50Hz states, 2.8 rad/s adjacent bounds with endpoint reachability; independently audited against existing joint contract",
                "free_finger_rate_projection_changes": free_finger_changes,
                "root_and_legs_unchanged": True,
                "overlap_q_max_diff": float(np.max(abs(joints[:50] - prior.joint_pos[:50]))),
                "residuals": reports,
                "unverified": [
                    "all hand collision geometry",
                    "cabinet and object penetration",
                    "continuous hand contact",
                    "source-video directional comparison",
                    "final support rest",
                ],
            },
            indent=2,
        )
    )
    print(a.out)


if __name__ == "__main__":
    main()
