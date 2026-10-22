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
    p.add_argument("--left-anchor-motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    seed = MotionFields.load(a.seed, order)
    left_anchor = MotionFields.load(a.left_anchor_motion, order)
    robot = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(robot))
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in seed.joint_names])
    q = seed.joint_pos.copy()
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
    report = {}
    for side, obj, anchor_time, begin, contact, release, finish in [
        ("right", "flower_bouquet", 116.0, 111.5, 113.2, 123.2, 125.2),
        ("left", "small_wicker_basket", 119.0, 116.8, 118.7, 120.4, 121.6),
    ]:
        source = seed if side == "right" else left_anchor
        k0 = round((anchor_time - 108) * 50)
        set_pose(source, k0, source.joint_pos[k0])
        palm = model.geom(side + "_hand_palm_collision").id
        track = objects[obj][0]
        ro = Rotation.from_quat(track["quaternion_xyzw"][k0]).as_matrix()
        po = track["position_w"][k0]
        relative_position = ro.T @ (data.geom_xpos[palm] - po)
        relative_rotation = ro.T @ data.geom_xmat[palm].reshape(3, 3)
        fingers = np.array([i for i, n in enumerate(seed.joint_names) if n.startswith(side + "_hand_")])
        anchor_fingers = source.joint_pos[k0, fingers].copy()
        names = [
            side + "_" + s + "_joint"
            for s in [
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            ]
        ]
        cols = np.array([seed.joint_names.index(n) for n in names])
        armaddr = addr[cols]
        low = np.array([model.joint(n).range[0] - 1e-7 for n in names])
        high = np.array([model.joint(n).range[1] + 1e-7 for n in names])
        vmax = np.array([min(6.0, speed.get(n, 3.0) * 0.95) for n in names])
        first, last = round((begin - 108) * 50), round((finish - 108) * 50)
        endpoint = seed.joint_pos[last, cols].astype(float)
        previous = seed.joint_pos[first, cols].astype(float)
        rows = []
        for k in range(first, last + 1):
            time = 108 + k / 50
            ramp = min(
                np.clip((time - begin) / (contact - begin), 0, 1), np.clip((finish - time) / (finish - release), 0, 1)
            )
            weight = ramp * ramp * (3 - 2 * ramp)
            set_pose(seed, k, seed.joint_pos[k])
            pbase = data.geom_xpos[palm].copy()
            rbase = data.geom_xmat[palm].reshape(3, 3).copy()
            ro = Rotation.from_quat(track["quaternion_xyzw"][k]).as_matrix()
            po = track["position_w"][k]
            target = (1 - weight) * pbase + weight * (po + ro @ relative_position)
            rattached = ro @ relative_rotation
            rdelta = Rotation.from_matrix(rattached @ rbase.T).as_rotvec()
            target_rotation = Rotation.from_rotvec(weight * rdelta).as_matrix() @ rbase
            q[k, fingers] = (1 - weight) * seed.joint_pos[k, fingers] + weight * anchor_fingers
            if k > first:
                q[k, fingers] = np.clip(q[k, fingers], q[k - 1, fingers] - 2.95 / 50, q[k - 1, fingers] + 2.95 / 50)
            set_pose(seed, k, q[k])
            frames = {
                n: (Rotation.from_quat(tr["quaternion_xyzw"][k]).as_matrix(), tr["position_w"][k])
                if tr is not None
                else (np.eye(3), np.zeros(3))
                for n, (tr, _) in objects.items()
            }

            def residual(
                x,
                target=target,
                target_rotation=target_rotation,
                frames=frames,
                side=side,
                palm=palm,
                armaddr=armaddr,
                previous=previous,
            ):
                data.qpos[armaddr] = x
                mujoco.mj_forward(model, data)
                collision = []
                for g, (label, vertices) in samples.items():
                    if not label.startswith(side + "_"):
                        continue
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
                        collision.extend(np.maximum(0.0003 - np.min(dist, axis=0), 0))
                return np.r_[
                    400 * (data.geom_xpos[palm] - target),
                    40 * (data.geom_xmat[palm].reshape(3, 3) - target_rotation).ravel(),
                    2000 * np.asarray(collision),
                    0.05 * (x - previous),
                ]

            lower = np.maximum(low, endpoint - vmax * (last - k) / 50)
            upper = np.minimum(high, endpoint + vmax * (last - k) / 50)
            lower = np.maximum(lower, previous - vmax / 50)
            upper = np.minimum(upper, previous + vmax / 50)
            if k in [first, last]:
                lower = seed.joint_pos[k, cols].astype(float) - 1e-9
                upper = lower + 2e-9
            equal = lower >= upper
            if np.any(equal):
                if np.max((lower - upper)[equal]) > 1e-6:
                    raise ValueError("infeasible endpoint speed bounds")
                midpoint = (lower[equal] + upper[equal]) / 2
                lower[equal] = midpoint - 1e-8
                upper[equal] = midpoint + 1e-8
            result = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper), max_nfev=60)
            q[k, cols] = result.x
            previous = result.x.copy()
            residual(result.x)
            rows.append(
                {
                    "source_s": time,
                    "weight": float(weight),
                    "palm_position_error_m": float(np.linalg.norm(data.geom_xpos[palm] - target)),
                    "palm_rotation_error_deg": float(
                        np.rad2deg(
                            Rotation.from_matrix(data.geom_xmat[palm].reshape(3, 3) @ target_rotation.T).magnitude()
                        )
                    ),
                }
            )
            if k % 50 == 0:
                (a.out / "progress.json").write_text(json.dumps({"side": side, "source_s": time}))
        report[side] = {
            "anchor_source_s": anchor_time,
            "anchor_motion": str(a.seed if side == "right" else a.left_anchor_motion),
            "palm_in_object_position": relative_position.tolist(),
            "palm_in_object_rotation": relative_rotation.tolist(),
            "finger_names": [seed.joint_names[i] for i in fingers],
            "finger_q": anchor_fingers.tolist(),
            "records": rows,
        }
    q[:50] = seed.joint_pos[:50]
    refresh_fk(replace(seed, joint_pos=q), robot).save(a.out / "motion.npz")
    (a.out / "grasp_attachments.json").write_text(
        json.dumps(
            {"status": "unreviewed candidate; full-window contacts/collision/velocity required", "attachments": report},
            indent=2,
        )
        + "\n"
    )
    print(a.out)


if __name__ == "__main__":
    main()
