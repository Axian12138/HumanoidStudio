"""Check held hat collision meshes against static room surfaces in the requested window."""

import argparse
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import sha256


def main():
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("--room-mesh", type=Path)
    p.add_argument("--room-transform", type=Path)
    p.add_argument("--output-name", default="mesh_crossing_audit.json")
    p.add_argument("--motion-file", default="motion.npz")
    p.add_argument("--order-file", default="motion_order.json")
    p.add_argument("--source-start-s", type=float, default=75.0)
    p.add_argument("--window-s", type=float, nargs=2, default=[83.4, 84.5])
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--static-room-only", action="store_true", help="Exclude the fixed post-push chair outside its valid window."
    )
    a = p.parse_args()
    order = json.loads((a.candidate / a.order_file).read_text())["robot_joint_names"]
    with np.load(a.candidate / a.motion_file) as source:
        z = {key: source[key] for key in ("joint_pos", "body_pos_w", "body_quat_w")}
    first, last = [int(round((t - a.source_start_s) * 100)) for t in a.window_s]
    if first < 0 or last < first or last > 2 * (len(z["joint_pos"]) - 1):
        raise ValueError("Audit window must be inside the 50 Hz motion.")
    if any(abs(t - (a.source_start_s + tick / 100)) > 1e-7 for t, tick in zip(a.window_s, (first, last), strict=True)):
        raise ValueError("Audit window must align with the 100 Hz sampling grid.")
    model = mujoco.MjModel.from_xml_path(str(a.candidate.parent.parent / "scene/robot_props_cylinder_hat.xml"))
    hat = np.load(a.candidate / "scene_tracks/black_ribbon_hat_free.npz")
    free = model.joint("black_ribbon_hat_free").qposadr[0]
    data = mujoco.MjData(model)
    adr = [int(model.joint(n).qposadr[0]) for n in order]
    snap = Path("outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot")
    room_path = a.room_mesh or snap / "assets/ikea_1012_room.obj"
    room = trimesh.load_mesh(room_path, force="mesh")
    room = trimesh.Trimesh(room.vertices, room.faces[room.triangles[:, :, 2].max(axis=1) > 0.01], process=False)
    if a.room_transform:
        transform = np.array(json.loads(a.room_transform.read_text())["T_room_correction"])
        room.vertices = room.vertices @ transform[:3, :3].T + transform[:3, 3]
    chair = trimesh.load_mesh(snap / "tasks/07_armchair/grey_armchair_visual.obj", force="mesh")
    ct = np.load(
        "/home/galbot/workspace/ScaleTrack/data/1019test/brace-dining-table-kinematic-v1/metadata/object_trajectories/grey_armchair.npz"
    )
    chair.vertices = (
        chair.vertices @ Rotation.from_quat(ct["quaternion_wxyz"][-1][[1, 2, 3, 0]]).as_matrix().T
        + ct["position_w"][-1]
    )
    surfaces = {"room": room, "post_push_chair": chair}
    if a.static_room_only:
        surfaces.pop("post_push_chair")
    engines = {n: RayMeshIntersector(m) for n, m in surfaces.items()}
    trees = {n: m.triangles_tree for n, m in surfaces.items()}
    colliders = []
    for g in range(model.ngeom):
        if not model.geom_contype[g] or model.geom_bodyid[g] != model.body("black_ribbon_hat").id:
            continue
        kind, size = model.geom_type[g], model.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            i = model.geom_dataid[g]
            vertices = model.mesh_vert[model.mesh_vertadr[i] : model.mesh_vertadr[i] + model.mesh_vertnum[i]].copy()
            faces = model.mesh_face[model.mesh_faceadr[i] : model.mesh_faceadr[i] + model.mesh_facenum[i]].copy()
            mesh = trimesh.Trimesh(vertices, faces, process=False)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(height=2 * size[1], radius=size[0], count=[12, 16])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * size)
        else:
            raise ValueError((g, kind))
        colliders.append((g, mesh, RayMeshIntersector(mesh)))

    def crosses(engine, start, end):
        vector = end - start
        lengths = np.linalg.norm(vector, axis=1)
        mask = lengths > 1e-7
        start, vector, lengths = start[mask], vector[mask], lengths[mask]
        hits, rays, _ = engine.intersects_location(start, vector / lengths[:, None], multiple_hits=False)
        if not len(rays):
            return 0
        distances = np.linalg.norm(hits - start[rays], axis=1)
        return int(np.sum((distances > 1e-5) & (distances < lengths[rays] - 1e-5)))

    rows, floor = [], []
    for step, tick in enumerate(range(first, last + 1)):
        frame = tick // 2
        nxt, weight = min(frame + 1, len(z["joint_pos"]) - 1), (tick % 2) / 2
        data.qpos[adr] = (1 - weight) * z["joint_pos"][frame] + weight * z["joint_pos"][nxt]
        data.qpos[:3] = (1 - weight) * z["body_pos_w"][frame, 0] + weight * z["body_pos_w"][nxt, 0]
        quat = (1 - weight) * z["body_quat_w"][frame, 0] + weight * z["body_quat_w"][nxt, 0]
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[free : free + 3] = (1 - weight) * hat["position_w"][frame] + weight * hat["position_w"][nxt]
        from scipy.spatial.transform import Slerp

        rot = Slerp([0, 1], Rotation.from_quat(hat["quaternion_xyzw"][[frame, nxt]]))([weight]).as_quat()[0]
        data.qpos[free + 3 : free + 7] = rot[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        for g, mesh, collider_engine in colliders:
            rot, pos = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
            world = mesh.vertices @ rot.T + pos
            floor.append(float(world[:, 2].min()))
            for name, surface in surfaces.items():
                ids = list(trees[name].intersection(np.r_[world.min(axis=0), world.max(axis=0)]))
                if not ids:
                    continue
                edges = mesh.edges_unique
                count = crosses(engines[name], world[edges[:, 0]], world[edges[:, 1]])
                triangles = surface.triangles[ids]
                left = (triangles[:, [0, 1, 2]].reshape(-1, 3) - pos) @ rot
                right = (triangles[:, [1, 2, 0]].reshape(-1, 3) - pos) @ rot
                reverse = crosses(collider_engine, left, right)
                if count or reverse:
                    rows.append(
                        {
                            "source_time_s": a.source_start_s + tick / 100,
                            "geom": model.geom(g).name,
                            "geom_center_world": pos.tolist(),
                            "geom_bounds_world": [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
                            "surface": name,
                            "robot_edges": count,
                            "furniture_edges": reverse,
                        }
                    )
        if step % 20 == 0:
            print(step, len(rows), flush=True)
    report = {
        "scope": "100Hz bidirectional mesh boundary crossing only; containment, self collision and physical stability not proved",
        "source_window_s": a.window_s,
        "frames_100hz": last - first + 1,
        "colliders": len(colliders),
        "minimum_sampled_world_z_m": min(floor),
        "room_transform": str(a.room_transform) if a.room_transform else None,
        "room_mesh": str(room_path),
        "room_mesh_sha256": sha256(room_path),
        "motion_sha256": sha256(a.candidate / a.motion_file),
        "order_sha256": sha256(a.candidate / a.order_file),
        "chair_state": (
            "excluded; moving-chair interaction is outside this audit"
            if a.static_room_only
            else "accepted world-frame post-push pose; not transformed with static room"
        ),
        "crossings": rows,
        "accepted": False,
    }
    (a.output or a.candidate / a.output_name).write_text(json.dumps(report, indent=2) + "\n")
    print("crossing records", len(rows))


if __name__ == "__main__":
    main()
