"""Clear the free middle finger and the thumb's pre-push fold using actual hand geometry."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    for name in ["run", "motion", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--window", type=float, nargs=2, default=[116.0, 128.5])
    p.add_argument("--arm-delta", type=float, default=0.2)
    p.add_argument("--midpoints", action="store_true")
    p.add_argument("--target-motion", type=Path)
    p.add_argument("--wall-contact", action="store_true")
    p.add_argument("--push-any-pad", action="store_true")
    p.add_argument("--self-margin", type=float, default=0.005)
    p.add_argument("--retreat-reference", type=Path)
    p.add_argument("--retreat-contact-until", type=float, default=120.4)
    p.add_argument("--retreat-match-from", type=float, default=120.8)
    p.add_argument("--surface-samples", action="store_true")
    p.add_argument("--retreat-orientation-weight", type=float, default=0)
    p.add_argument("--native-shoulder", action="store_true")
    p.add_argument("--boundary-ramp-s", type=float, default=0)
    p.add_argument("--boundary-tangent", action="store_true")
    p.add_argument("--joint-refinement-only", action="store_true")
    p.add_argument("--acceleration-bound", type=float, default=60.0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    seed = MotionFields.load(a.motion, order)
    target_motion = MotionFields.load(a.target_motion, order) if a.target_motion else seed
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(robot))
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in seed.joint_names])
    names = [
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_thumb_1_joint",
    ]
    names += [
        n
        for n in seed.joint_names
        if n.startswith("left_") and any(v in n for v in ["shoulder_", "elbow_", "wrist_", "hand_"]) and n not in names
    ]
    step = np.array([2.95 if "hand_" in n else 6.0 for n in names]) / 50
    cols = [seed.joint_names.index(n) for n in names]
    aa = addr[cols]
    q = seed.joint_pos.copy()
    first, last = [round((t - 108) * 50) for t in a.window]
    retreat_targets = {}
    retreat_rotations = {}
    if a.retreat_reference:
        reference = MotionFields.load(a.retreat_reference, order)
        palm_id = model.geom("left_hand_palm_collision").id
        anchor_frame = round((a.retreat_contact_until - 108) * 50)
        anchor_points = []
        anchor_rotations = []
        for motion in [seed, reference]:
            data.qpos[:3] = motion.body_pos_w[anchor_frame, 0]
            data.qpos[3:7] = motion.body_quat_w[anchor_frame, 0]
            data.qpos[addr] = motion.joint_pos[anchor_frame]
            mujoco.mj_forward(model, data)
            anchor_points.append(data.geom_xpos[palm_id].copy())
            anchor_rotations.append(data.geom_xmat[palm_id].reshape(3, 3).copy())
        offset = anchor_points[0] - anchor_points[1]
        rotation_offset = Rotation.from_matrix(anchor_rotations[0] @ anchor_rotations[1].T).as_rotvec()
        for k in range(first, last + 1):
            time = 108 + k / 50
            if time < a.retreat_contact_until:
                continue
            data.qpos[:3] = reference.body_pos_w[k, 0]
            data.qpos[3:7] = reference.body_quat_w[k, 0]
            data.qpos[addr] = reference.joint_pos[k]
            mujoco.mj_forward(model, data)
            weight = np.clip((time - a.retreat_contact_until) / (a.retreat_match_from - a.retreat_contact_until), 0, 1)
            weight = weight * weight * (3 - 2 * weight)
            retreat_targets[k] = data.geom_xpos[palm_id].copy() + (1 - weight) * offset
            retreat_rotations[k] = Rotation.from_rotvec((1 - weight) * rotation_offset).as_matrix() @ data.geom_xmat[
                palm_id
            ].reshape(3, 3)
    previous = q[first, cols].astype(float)
    endpoint = q[last, cols].astype(float)
    lo = np.array([model.joint(n).range[0] - 1e-7 for n in names])
    hi = np.array([model.joint(n).range[1] + 1e-7 for n in names])
    hand = {
        n: model.geom("left_hand_" + n + "_collision").id
        for n in ["palm", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    }
    pairs = [
        (hand[mid], g)
        for mid in ["middle_0", "middle_1"]
        for name, g in hand.items()
        if not name.startswith("middle") and name != "palm"
    ] + [(hand["thumb_2"], hand["palm"])]
    visuals = {}
    for name, proxy in hand.items():
        ids = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == model.geom_bodyid[proxy]
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and not model.geom_contype[g]
            and not model.geom_conaffinity[g]
        ]
        if len(ids) == 1:
            visuals[name] = ids[0]
    pairs.extend((visuals[mid], visuals[thumb]) for mid in ["middle_0", "middle_1"] for thumb in ["thumb_1", "thumb_2"])
    digits = ["thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
    for i, n1 in enumerate(digits):
        for n2 in digits[i + 1 :]:
            if n1.split("_")[0] != n2.split("_")[0]:
                pairs.extend([(hand[n1], hand[n2]), (visuals[n1], visuals[n2])])
    pairs = list(dict.fromkeys(tuple(sorted(pair)) for pair in pairs))
    pairs.extend(
        (model.geom("left_shoulder_yaw_collision").id, model.geom(n).id)
        for n in ["torso_collision1", "torso_collision2", "torso_collision3"]
    )
    if a.native_shoulder:
        native_ids = []
        for body_name in ["left_shoulder_yaw_link", "torso_link"]:
            ids = [
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == model.body(body_name).id
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and not model.geom_contype[g]
                and not model.geom_conaffinity[g]
                and model.mesh(model.geom_dataid[g]).name == body_name
            ]
            if len(ids) != 1:
                raise ValueError((body_name, ids))
            native_ids.append(ids[0])
        pairs.append(tuple(native_ids))
    points = {}
    for name in ["middle_0", "middle_1", "thumb_1", "thumb_2", "index_0", "index_1", "palm"]:
        g = hand[name]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            half = model.geom_size[g]
            points[g] = np.array(
                [
                    [x, y, z]
                    for x in np.linspace(-half[0], half[0], 5)
                    for y in np.linspace(-half[1], half[1], 5)
                    for z in np.linspace(-half[2], half[2], 5)
                ]
            )
            continue
        mi = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mi] : model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]
        if a.surface_samples and name in ["index_0", "thumb_1"]:
            faces = model.mesh_face[model.mesh_faceadr[mi] : model.mesh_faceadr[mi] + model.mesh_facenum[mi]]
            v, _ = trimesh.remesh.subdivide_to_size(v, faces, max_edge=0.003, max_iter=8)
        _, ix = np.unique(np.round(v / 0.001), axis=0, return_index=True)
        points[g] = v[ix]
    for name in ["middle_0", "middle_1", "thumb_1", "thumb_2", "index_0", "index_1", "palm"]:
        if name not in visuals:
            continue
        g = visuals[name]
        mi = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mi] : model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]
        if a.surface_samples and name in ["index_0", "thumb_1"]:
            faces = model.mesh_face[model.mesh_faceadr[mi] : model.mesh_faceadr[mi] + model.mesh_facenum[mi]]
            v, _ = trimesh.remesh.subdivide_to_size(v, faces, max_edge=0.003, max_iter=8)
        _, ix = np.unique(np.round(v / 0.001), axis=0, return_index=True)
        points[g] = v[ix]
    assets = a.run / "scene/10_flower_bouquet/registered_assets_paper_wrapper_v1"
    basket_dims = json.loads((a.run / "scene/10_flower_bouquet/basket_registered_candidate.json").read_text())[
        "dimensions_world_aligned_m"
    ]
    objects = {}
    for name in ["flower_bouquet", "small_wicker_basket"]:
        track = np.load(a.run / f"retarget/hoi/10_flower_bouquet/tracks_v8/{name}.npz")
        hulls = []
        for path in sorted(assets.glob(name + "_hull*.obj")):
            v = trimesh.load(path, force="mesh", process=False).vertices
            hulls.append((np.unique(np.round(ConvexHull(v).equations, 10), axis=0), v.min(axis=0), v.max(axis=0)))
        objects[name] = (track, hulls)

    def object_residual(object_frames):
        values = []
        for g, v in points.items():
            world = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            for name, (_, hulls) in objects.items():
                rot, pos = object_frames[name]
                local = (world - pos) @ rot
                dist = []
                for e, low, high in hulls:
                    near = np.all((local >= low - 0.005) & (local <= high + 0.005), axis=1)
                    signed = np.full(len(local), 0.005)
                    signed[near] = np.max(local[near] @ e[:, :3].T + e[:, 3], axis=1)
                    dist.append(signed)
                penetration = np.maximum(0.0002 - np.min(dist, axis=0), 0)
                worst = np.sort(penetration)[-16:]
                values.extend(30000 * np.pad(worst, (16 - len(worst), 0)))
        return values

    records = []
    joint_residuals = []
    future_low = np.tile(lo, (len(q), 1))
    future_high = np.tile(hi, (len(q), 1))
    for k in range(first, last + 1):
        time = 108 + k / 50
        envelope = max(0, min(1, (time - 116) / 0.7, (128.5 - time) / 0.7))
        bound_center = seed.joint_pos[k, cols].astype(float).copy()
        if a.boundary_ramp_s:
            ramp = np.clip(min(time - a.window[0], a.window[1] - time) / a.boundary_ramp_s, 0, 1)
            blend = ramp**3 * (10 - 15 * ramp + 6 * ramp**2)
            envelope *= blend
            if a.boundary_tangent:
                if k - first <= last - k:
                    tangent = seed.joint_pos[first, cols] + (k - first) * (
                        seed.joint_pos[first, cols] - seed.joint_pos[first - 1, cols]
                    )
                else:
                    tangent = seed.joint_pos[last, cols] + (k - last) * (
                        seed.joint_pos[last + 1, cols] - seed.joint_pos[last, cols]
                    )
                bound_center = blend * bound_center + (1 - blend) * tangent
        for j in range(2, len(names)):
            future_low[k, j] = max(lo[j], bound_center[j] - (0.35 if "hand_" in names[j] else a.arm_delta) * envelope)
            future_high[k, j] = min(hi[j], bound_center[j] + (0.35 if "hand_" in names[j] else a.arm_delta) * envelope)
    future_low[last] = endpoint
    future_high[last] = endpoint
    for k in range(last - 1, first - 1, -1):
        future_low[k] = np.maximum(future_low[k], future_low[k + 1] - step)
        future_high[k] = np.minimum(future_high[k], future_high[k + 1] + step)
    for k in range(first, last + 1):
        time = 108 + k / 50
        data.qpos[:3] = seed.body_pos_w[k, 0]
        data.qpos[3:7] = seed.body_quat_w[k, 0]
        data.qpos[addr] = seed.joint_pos[k]
        mujoco.mj_forward(model, data)
        baseline = q[k, cols].astype(float)
        open_weight = max(0, min(1, time - 116, 128.5 - time))
        finger_reference = baseline.copy()
        finger_reference[:2] *= 1 - open_weight
        push_weight = max(0, min(1, (time - 123.8), (128.5 - time) / 0.7))
        finger_reference[:2] = (1 - push_weight) * finger_reference[:2] + push_weight * np.array([-1.15, -1.0])
        frames = {
            name: (Rotation.from_quat(tr["quaternion_xyzw"][k]).as_matrix(), tr["position_w"][k])
            for name, (tr, _) in objects.items()
        }
        mid_frames = {}
        if a.midpoints and k > first:
            for name, (tr, _) in objects.items():
                qa, qb = tr["quaternion_xyzw"][k - 1], tr["quaternion_xyzw"][k]
                qm = qa + (qb if qa @ qb >= 0 else -qb)
                mid_frames[name] = (
                    Rotation.from_quat(qm / np.linalg.norm(qm)).as_matrix(),
                    (tr["position_w"][k - 1] + tr["position_w"][k]) / 2,
                )
        anchors = []
        wall_anchors = []
        push_frame = None
        if a.push_any_pad and 125 <= time <= 127.8:
            rot, pos = frames["small_wicker_basket"]
            push_frame = (rot @ Rotation.from_euler("z", -np.pi / 2).as_matrix(), pos)
        if 116 <= time <= 120.4 or 123.8 <= time <= 127.8:
            data.qpos[addr] = target_motion.joint_pos[k]
            mujoco.mj_forward(model, data)
            for part in ["thumb_2", "index_1"]:
                if push_frame is not None:
                    continue
                g = visuals[part]
                vertices = points[g]
                world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                rot, pos = frames["small_wicker_basket"]
                local = (world - pos) @ rot
                distance = np.min(
                    [
                        np.abs(np.max(local @ e[:, :3].T + e[:, 3], axis=1))
                        for e, _, _ in objects["small_wicker_basket"][1]
                    ],
                    axis=0,
                )
                ix = np.argmin(distance)
                if a.wall_contact and 125 <= time <= 127.8:
                    rect_rot = rot @ Rotation.from_euler("z", -np.pi / 2).as_matrix()
                    rect = (world - pos) @ rect_rot
                    plane = basket_dims[0] / 2
                    distance = (
                        (rect[:, 0] - plane) ** 2
                        + np.maximum(abs(rect[:, 1]) - basket_dims[1] / 2, 0) ** 2
                        + np.maximum(rect[:, 2] - basket_dims[2], 0) ** 2
                        + np.maximum(-rect[:, 2], 0) ** 2
                    )
                    ix = np.argmin(distance)
                    wall_anchors.append((g, vertices[ix], rect_rot, pos, plane, rect[ix]))
                else:
                    anchors.append((g, vertices[ix], world[ix]))
            data.qpos[addr] = seed.joint_pos[k]
            mujoco.mj_forward(model, data)

        def residual(
            x,
            frames=frames,
            baseline=baseline,
            finger_reference=finger_reference,
            anchors=anchors,
            wall_anchors=wall_anchors,
            k=k,
            previous=previous,
            mid_frames=mid_frames,
            push_frame=push_frame,
        ):
            data.qpos[aa] = x
            mujoco.mj_forward(model, data)
            gaps = np.array([mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None) for g1, g2 in pairs])
            values = list(10000 * np.maximum(a.self_margin - gaps, 0))
            values.extend(object_residual(frames))
            values.extend(5 * (x - finger_reference))
            if k in retreat_targets:
                values.extend(1500 * (data.geom_xpos[palm_id] - retreat_targets[k]))
                values.extend(
                    a.retreat_orientation_weight
                    * (data.geom_xmat[palm_id].reshape(3, 3) - retreat_rotations[k]).ravel()
                )
            if push_frame is not None:
                rect_rot, pos = push_frame
                best = []
                for part in ["thumb_2", "index_1", "middle_1"]:
                    g = visuals[part]
                    world = points[g] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                    rect = (world - pos) @ rect_rot
                    distance = np.sqrt(
                        (rect[:, 0] - basket_dims[0] / 2) ** 2
                        + np.maximum(abs(rect[:, 1]) - basket_dims[1] / 2, 0) ** 2
                        + np.maximum(rect[:, 2] - basket_dims[2], 0) ** 2
                        + np.maximum(-rect[:, 2], 0) ** 2
                    )
                    best.append(float(distance.min()))
                values.append(20000 * max(0, min(best) - 0.001))
            for g, vertex, target in anchors:
                delta = vertex @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - target
                values.append(20000 * max(0, np.linalg.norm(delta) - 0.001))
            for g, vertex, rect_rot, pos, plane, target in wall_anchors:
                current = (vertex @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - pos) @ rect_rot
                values.append(200000 * max(0, abs(current[0] - plane - 0.0005) - 0.0005))
                values.extend(
                    200000
                    * np.maximum([abs(current[1]) - basket_dims[1] / 2, -current[2], current[2] - basket_dims[2]], 0)
                )
                values.extend(1000 * (current[1:] - target[1:]))
            if a.midpoints and k > first:
                saved = data.qpos.copy()
                data.qpos[:3] = (seed.body_pos_w[k - 1, 0] + seed.body_pos_w[k, 0]) / 2
                quat = seed.body_quat_w[k - 1, 0] + seed.body_quat_w[k, 0]
                data.qpos[3:7] = quat / np.linalg.norm(quat)
                data.qpos[addr] = (q[k - 1] + seed.joint_pos[k]) / 2
                data.qpos[aa] = (previous + x) / 2
                mujoco.mj_forward(model, data)
                values.extend(object_residual(mid_frames))
                gaps = np.array([mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None) for g1, g2 in pairs])
                values.extend(10000 * np.maximum(0.002 - gaps, 0))
                data.qpos[:] = saved
                mujoco.mj_forward(model, data)
            return np.asarray(values)

        lower = np.maximum(future_low[k], previous - step)
        upper = np.minimum(future_high[k], previous + step)
        joint_residuals.append((k, residual))
        if k in [first, last] or a.joint_refinement_only:
            result = baseline
        else:
            eq = lower >= upper
            if np.any(eq):
                if np.max((lower - upper)[eq]) > 1e-6:
                    raise ValueError((time, "infeasible finger bounds", lower - upper))
                mid = (lower[eq] + upper[eq]) / 2
                lower[eq] = mid - 1e-8
                upper[eq] = mid + 1e-8
            free = upper - lower > 1e-6
            fixed = (lower + upper) / 2

            def reduced(y, fixed=fixed, free=free, residual=residual):
                x = fixed.copy()
                x[free] = y
                return residual(x)

            result = fixed.copy()
            if np.any(free):
                result[free] = least_squares(
                    reduced,
                    np.clip(previous[free], lower[free], upper[free]),
                    bounds=(lower[free], upper[free]),
                    max_nfev=40,
                    ftol=1e-5,
                    xtol=1e-5,
                    gtol=1e-5,
                ).x
        q[k, cols] = result
        previous = result.copy()
        residual(result)
        records.append(
            {
                "source_s": time,
                "min_native_gap_m": min(
                    float(mujoco.mj_geomDistance(model, data, g1, g2, 0.03, None)) for g1, g2 in pairs
                ),
                "joint_delta_rad": (result - baseline).tolist(),
                "retreat_palm_error_m": None
                if k not in retreat_targets
                else float(np.linalg.norm(data.geom_xpos[palm_id] - retreat_targets[k])),
                "retreat_palm_rotation_error_deg": None
                if k not in retreat_targets
                else float(
                    np.rad2deg(
                        Rotation.from_matrix(data.geom_xmat[palm_id].reshape(3, 3) @ retreat_rotations[k].T).magnitude()
                    )
                ),
            }
        )
        if k % 25 == 0:
            (a.out / "progress.json").write_text(json.dumps(records[-1]))
            np.save(a.out / "joint_pos_checkpoint.npy", q)
    joint_summary = None
    if a.joint_refinement_only:
        # Optimize the full window together. The free middle finger is fixed to
        # the seed; all seven arm and five thumb/index joints remain coupled.
        variable_cols = np.array(cols[2:])
        variable_count = len(variable_cols)
        interior_count = last - first - 1
        initial = q[first + 1 : last, variable_cols].astype(float).copy()

        def assemble(values):
            result = q[first - 1 : last + 2, cols].astype(float).copy()
            result[2:-2, 2:] = values.reshape(interior_count, variable_count)
            return result

        blocks = []
        for k, frame_residual in joint_residuals:
            data.qpos[:3] = seed.body_pos_w[k, 0]
            data.qpos[3:7] = seed.body_quat_w[k, 0]
            data.qpos[addr] = q[k]
            blocks.append(len(frame_residual(q[k, cols], previous=q[k - 1, cols])))
        extended = assemble(initial.ravel())
        d1_shape, d2_shape = np.diff(extended, axis=0).shape, np.diff(extended, n=2, axis=0).shape
        sparsity = lil_matrix((sum(blocks) + np.prod(d1_shape) + 2 * np.prod(d2_shape), initial.size), dtype=int)

        def mark(row_start, count, frames):
            for frame in frames:
                if first < frame < last:
                    offset = (frame - first - 1) * variable_count
                    sparsity[row_start : row_start + count, offset : offset + variable_count] = 1

        row = 0
        for (k, _), count in zip(joint_residuals, blocks, strict=True):
            mark(row, count, [k - 1, k])
            row += count
        for index in range(d1_shape[0]):
            mark(row, len(cols), [first - 1 + index, first + index])
            row += len(cols)
        for _ in range(2):
            for index in range(d2_shape[0]):
                mark(row, len(cols), [first - 1 + index, first + index, first + 1 + index])
                row += len(cols)
        evaluations = 0

        def combined(values):
            nonlocal evaluations
            sequence = assemble(values)
            output = []
            for k, frame_residual in joint_residuals:
                data.qpos[:3] = seed.body_pos_w[k, 0]
                data.qpos[3:7] = seed.body_quat_w[k, 0]
                data.qpos[addr] = q[k]
                output.extend(frame_residual(sequence[k - first + 1], previous=sequence[k - first]))
            velocity_step = np.diff(sequence, axis=0)
            acceleration_step = np.diff(sequence, n=2, axis=0)
            output.extend((100000 * np.maximum(abs(velocity_step) - step, 0)).ravel())
            output.extend((10000 * np.maximum(abs(acceleration_step) - a.acceleration_bound / 50**2, 0)).ravel())
            output.extend((100 * acceleration_step).ravel())
            evaluations += 1
            if evaluations % 25 == 0:
                (a.out / "joint_progress.json").write_text(
                    json.dumps(
                        {
                            "evaluations": evaluations,
                            "residual_norm": float(np.linalg.norm(output)),
                            "max_acceleration_rad_s2": float(abs(acceleration_step).max() * 50**2),
                        }
                    )
                )
            return np.asarray(output)

        lower = future_low[first + 1 : last, 2:].ravel()
        upper = future_high[first + 1 : last, 2:].ravel()
        solution = least_squares(
            combined,
            np.clip(initial.ravel(), lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper),
            jac_sparsity=sparsity.tocsr(),
            max_nfev=35,
            ftol=1e-5,
            xtol=1e-6,
            gtol=1e-5,
            verbose=2,
        )
        q[first + 1 : last, variable_cols] = solution.x.reshape(interior_count, variable_count)
        joint_summary = {
            "success": bool(solution.success),
            "message": solution.message,
            "cost": float(solution.cost),
            "acceleration_planning_bound_rad_s2": a.acceleration_bound,
            "evaluations": evaluations,
        }
    refresh_fk(replace(seed, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "report.json").write_text(
        json.dumps(
            {
                "status": "unreviewed candidate; middle contact not assumed",
                "finger_names": names,
                "retreat_reference": str(a.retreat_reference) if a.retreat_reference else None,
                "surface_samples": a.surface_samples,
                "retreat_orientation_weight": a.retreat_orientation_weight,
                "native_shoulder_convex_constraint": a.native_shoulder,
                "boundary_ramp_s": a.boundary_ramp_s,
                "boundary_tangent": a.boundary_tangent,
                "joint_refinement": joint_summary,
                "records_scope": "Pre-joint-refinement seed diagnostics"
                if a.joint_refinement_only
                else "Exported sequential candidate",
                "retreat_target": "Bodyprior palm plus initial grasp offset smoothly removed after contact; no high retreat waypoint",
                "records": records,
            },
            indent=2,
        )
    )
    print(a.out)


if __name__ == "__main__":
    main()
