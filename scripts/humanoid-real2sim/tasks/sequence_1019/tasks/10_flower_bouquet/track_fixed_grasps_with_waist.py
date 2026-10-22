"""Track verified palm/object relative grasp shapes without per-frame finger branch changes."""

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--attachments", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pose-only", action="store_true")
    p.add_argument("--source-begin", type=float, default=111.5)
    p.add_argument("--release-seed", type=Path)
    p.add_argument("--self-clearance", action="store_true")
    p.add_argument("--prefix-motion", type=Path)
    p.add_argument("--open-right-release", action="store_true")
    p.add_argument("--waist-weight", type=float, default=3.0)
    p.add_argument("--seed-weight", type=float, default=0.0)
    p.add_argument("--arm-delta-bound", type=float)
    p.add_argument("--release-fingers", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "configuration.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}, indent=2)
    )
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    seed = MotionFields.load(a.seed, order)
    release_seed = MotionFields.load(a.release_seed, order) if a.release_seed else seed
    attachments = json.loads(a.attachments.read_text())["attachments"]
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(robot))
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in seed.joint_names])
    q = seed.joint_pos.copy()
    if a.open_right_release:
        fingers = [i for i, n in enumerate(seed.joint_names) if n.startswith("right_hand_")]
        start = round((123.2 - 108) * 50)
        closed = q[start, fingers].copy()
        for k in range(start + 1, len(q)):
            u = np.clip((108 + k / 50 - 123.2) / 0.8, 0, 1)
            desired = (1 - u * u * (3 - 2 * u)) * closed
            q[k, fingers] = np.clip(desired, q[k - 1, fingers] - 2.95 / 50, q[k - 1, fingers] + 2.95 / 50)
    if a.release_fingers:
        path = np.load(a.release_fingers)
        fingers = [seed.joint_names.index(str(name)) for name in path["joint_names"]]
        for row, time in zip(path["joint_pos"], path["source_time_s"], strict=True):
            q[round((float(time) - 108) * 50), fingers] = row
        q[round((float(path["source_time_s"][-1]) - 108) * 50) + 1 :, fingers] = path["joint_pos"][-1]
    urdf = ET.parse("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/body.urdf")
    speed = {
        j.get("name"): float(j.find("limit").get("velocity", "20"))
        for j in urdf.findall("joint")
        if j.find("limit") is not None
    }
    assets = a.run / "scene/10_flower_bouquet/registered_assets_paper_wrapper_v1"
    tracks = a.run / "retarget/hoi/10_flower_bouquet/tracks_v8"
    objects = {}
    for name in ["flower_bouquet", "small_wicker_basket", "cabinet_support", "cabinet_side", "room_back_wall"]:
        paths = (
            sorted(assets.glob(name + "_hull*.obj"))
            if name in ["flower_bouquet", "small_wicker_basket"]
            else [assets / (name + ".obj")]
        )
        hulls = []
        for path in paths:
            vertices = trimesh.load(path, force="mesh", process=False).vertices
            hulls.append(
                (
                    np.unique(np.round(ConvexHull(vertices).equations, 10), axis=0),
                    vertices.min(axis=0),
                    vertices.max(axis=0),
                )
            )
        track = np.load(tracks / (name + ".npz")) if name in ["flower_bouquet", "small_wicker_basket"] else None
        objects[name] = (track, hulls)

    def set_pose(motion, k, joints):
        data.qpos[:3] = motion.body_pos_w[k, 0]
        data.qpos[3:7] = motion.body_quat_w[k, 0]
        data.qpos[addr] = joints
        mujoco.mj_forward(model, data)

    samples = {}
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if not name.endswith("_collision") or not any(
            name.startswith(s)
            for s in ["left_hand_", "right_hand_", "left_wrist", "right_wrist", "left_elbow_yaw", "right_elbow_yaw"]
        ):
            continue
        kind = model.geom_type[g]
        size = model.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mi = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[mi] : model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]
            _, indices = np.unique(np.round(v / 0.003), axis=0, return_index=True)
            v = v[indices]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            v = np.array(
                [
                    [x, y, z]
                    for x in [-size[0], 0, size[0]]
                    for y in [-size[1], 0, size[1]]
                    for z in [-size[2], 0, size[2]]
                ]
            )
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            v = trimesh.creation.capsule(height=2 * size[1], radius=size[0], count=[8, 12]).vertices
        else:
            raise ValueError((name, kind))
        samples[g] = (name, v)
    if a.pose_only:
        samples = {}
    names = [
        n
        for n in seed.joint_names
        if n.startswith("waist_")
        or any(n.startswith(side + "_" + part) for side in ["left", "right"] for part in ["shoulder", "elbow", "wrist"])
    ]
    cols = np.array([seed.joint_names.index(n) for n in names])
    armaddr = addr[cols]
    waist = np.array([i for i, n in enumerate(names) if n.startswith("waist_")])
    low = np.array([model.joint(n).range[0] + 1e-7 for n in names])
    high = np.array([model.joint(n).range[1] - 1e-7 for n in names])
    vmax = np.array([min(6.0, speed.get(n, 3.0) * 0.95) for n in names])
    first, last = round((a.source_begin - 108) * 50), 860
    if a.prefix_motion:
        prefix = MotionFields.load(a.prefix_motion, order)
        q[: first + 1] = prefix.joint_pos[: first + 1]
    endpoint = seed.joint_pos[last, cols].astype(float)
    previous = q[first, cols].astype(float)
    rows = []
    release_frames = {}
    for side, release in [("left", 120.4), ("right", 123.2)]:
        k = round((release - 108) * 50)
        set_pose(release_seed, k, release_seed.joint_pos[k])
        palm = model.geom(side + "_hand_palm_collision").id
        release_frames[side] = (data.geom_xpos[palm].copy(), data.geom_xmat[palm].reshape(3, 3).copy())
    self_pairs = []
    if a.self_clearance:
        for side in ["left", "right"]:
            self_pairs.append((model.geom(side + "_shoulder_yaw_collision").id, model.geom("torso_collision1").id))
            for g in range(model.ngeom):
                name = model.geom(g).name or ""
                if name.startswith(side + "_hand_") and name.endswith("_collision"):
                    for hip in ["left_hip_collision", "right_hip_collision"]:
                        self_pairs.append((g, model.geom(hip).id))
    for k in range(first, last + 1):
        time = 108 + k / 50
        set_pose(seed, k, seed.joint_pos[k])
        targets = {}
        for side, obj, begin, contact, release, finish in [
            ("right", "flower_bouquet", 111.5, 113.2, 123.2, 125.2),
            ("left", "small_wicker_basket", 116.8, 118.7, 120.4, 121.6),
        ]:
            palm = model.geom(side + "_hand_palm_collision").id
            ramp = min(
                np.clip((time - begin) / (contact - begin), 0, 1), np.clip((finish - time) / (finish - release), 0, 1)
            )
            weight = ramp * ramp * (3 - 2 * ramp)
            track = objects[obj][0]
            ro = Rotation.from_quat(track["quaternion_xyzw"][k]).as_matrix()
            attached = track["position_w"][k] + ro @ attachments[side]["palm_in_object_position"]
            attachedR = ro @ attachments[side]["palm_in_object_rotation"]
            pbase = data.geom_xpos[palm].copy()
            rbase = data.geom_xmat[palm].reshape(3, 3).copy()
            target = (1 - weight) * pbase + weight * attached
            targetR = (
                Rotation.from_rotvec(weight * Rotation.from_matrix(attachedR @ rbase.T).as_rotvec()).as_matrix() @ rbase
            )
            if a.release_seed and release < time < finish:
                kr = round((release - 108) * 50)
                rr = Rotation.from_quat(track["quaternion_xyzw"][kr]).as_matrix()
                start = track["position_w"][kr] + rr @ attachments[side]["palm_in_object_position"]
                startR = rr @ attachments[side]["palm_in_object_rotation"]
                set_pose(release_seed, k, release_seed.joint_pos[k])
                base = data.geom_xpos[palm].copy()
                baseR = data.geom_xmat[palm].reshape(3, 3).copy()
                u = (time - release) / (finish - release)
                decay = 1 - u * u * (3 - 2 * u)
                offset = np.array([0.10, 0, 0.12]) if side == "left" else np.array([0.12, 0, 0.14])
                target = base + decay * (start - release_frames[side][0]) + np.sin(np.pi * u) * offset
                targetR = (
                    Rotation.from_rotvec(
                        decay * Rotation.from_matrix(startR @ release_frames[side][1].T).as_rotvec()
                    ).as_matrix()
                    @ baseR
                )
                if side == "right" and a.open_right_release:
                    hold_s = 0.6 if a.release_fingers else 0.2
                    u = np.clip((time - release - hold_s) / (finish - release - hold_s), 0, 1)
                    blend = u * u * (3 - 2 * u)
                    radial = np.array(attachments[side]["palm_in_object_position"])
                    radial[2] = 0
                    outward = rr @ radial / np.linalg.norm(radial)
                    target = (
                        (1 - blend) * start
                        + blend * base
                        + np.sin(np.pi * u) * (0.1 * outward + np.array([0, 0, 0.08]))
                    )
                    targetR = (
                        Rotation.from_rotvec(blend * Rotation.from_matrix(baseR @ startR.T).as_rotvec()).as_matrix()
                        @ startR
                    )
                    if time <= release + hold_s:
                        weight = 1.0
                set_pose(seed, k, seed.joint_pos[k])
            if side == "left" and 125.0 <= time <= 127.8:
                weight = 1.0
            targets[side] = (palm, target, targetR, float(weight))
        data.qpos[addr] = q[k]
        mujoco.mj_forward(model, data)
        frames = {
            n: (Rotation.from_quat(tr["quaternion_xyzw"][k]).as_matrix(), tr["position_w"][k])
            if tr is not None
            else (np.eye(3), np.zeros(3))
            for n, (tr, _) in objects.items()
        }
        baseline = seed.joint_pos[k, cols].astype(float)

        def residual(x, targets=targets, frames=frames, baseline=baseline, previous=previous, time=time):
            data.qpos[armaddr] = x
            mujoco.mj_forward(model, data)
            values = []
            for palm, target, targetR, weight in targets.values():
                scale = 50 if weight == 1 else 1
                values.extend(800 * scale * (data.geom_xpos[palm] - target))
                values.extend((80 * scale * (data.geom_xmat[palm].reshape(3, 3) - targetR)).ravel())
            for g, (_label, vertices) in samples.items():
                world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                for n, (_, hulls) in objects.items():
                    rot, pos = frames[n]
                    local = (world - pos) @ rot
                    dist = []
                    for e, lo, hi in hulls:
                        near = np.all((local >= lo - 0.005) & (local <= hi + 0.005), axis=1)
                        signed = np.full(len(local), 0.005)
                        signed[near] = np.max(local[near] @ e[:, :3].T + e[:, 3], axis=1)
                        dist.append(signed)
                    values.extend(2000 * np.maximum(0.0003 - np.min(dist, axis=0), 0))
            for g1, g2 in self_pairs:
                distance = mujoco.mj_geomDistance(model, data, g1, g2, 0.02, None)
                values.append(2000 * max(0.002 - distance, 0))
            values.extend(a.waist_weight * (x[waist] - baseline[waist]))
            values.extend(a.seed_weight * (x - baseline))
            values.extend(0.05 * (x - previous))
            return np.asarray(values)

        lower = np.maximum(low, endpoint - vmax * (last - k) / 50)
        upper = np.minimum(high, endpoint + vmax * (last - k) / 50)
        lower = np.maximum(lower, previous - vmax / 50)
        upper = np.minimum(upper, previous + vmax / 50)
        if a.arm_delta_bound is not None:
            arms = np.array([i for i, name in enumerate(names) if not name.startswith("waist_")])
            lower[arms] = np.maximum(lower[arms], baseline[arms] - a.arm_delta_bound)
            upper[arms] = np.minimum(upper[arms], baseline[arms] + a.arm_delta_bound)
        lower[waist] = np.maximum(lower[waist], baseline[waist] - 0.15)
        upper[waist] = np.minimum(upper[waist], baseline[waist] + 0.15)
        if k in [first, last]:
            fixed = q[first, cols].astype(float) if k == first else baseline
            lower = fixed - 1e-9
            upper = fixed + 1e-9
        equal = lower >= upper
        if np.any(equal):
            if np.max((lower - upper)[equal]) > 1e-6:
                raise ValueError("infeasible bounds")
            midpoint = (lower[equal] + upper[equal]) / 2
            lower[equal] = midpoint - 1e-8
            upper[equal] = midpoint + 1e-8
        result = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper), max_nfev=50)
        q[k, cols] = result.x
        previous = result.x.copy()
        residual(result.x)
        row = {"source_s": time, "waist_delta_rad": (result.x[waist] - baseline[waist]).tolist()}
        for side, (palm, target, targetR, weight) in targets.items():
            row[side] = {
                "weight": weight,
                "palm_error_m": float(np.linalg.norm(data.geom_xpos[palm] - target)),
                "rotation_error_deg": float(
                    np.rad2deg(Rotation.from_matrix(data.geom_xmat[palm].reshape(3, 3) @ targetR.T).magnitude())
                ),
            }
        rows.append(row)
        if k % 25 == 0:
            (a.out / "progress.json").write_text(json.dumps(row))
            np.save(a.out / "joint_pos_checkpoint.npy", q)
    q[:50] = seed.joint_pos[:50]
    refresh_fk(replace(seed, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "solve_report.json").write_text(
        json.dumps({"status": "unreviewed candidate", "waist_delta_bound_rad": 0.15, "records": rows}, indent=2)
    )
    print(a.out)


if __name__ == "__main__":
    main()
