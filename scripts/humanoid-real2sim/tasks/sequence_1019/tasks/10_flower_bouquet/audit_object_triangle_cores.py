"""Test hand triangles against inset convex object cores at 100 Hz."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import trimesh
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    from scipy.spatial.transform import Rotation
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--inset-m", type=float, default=0.002)
    p.add_argument("--all-robot", action="store_true")
    p.add_argument("--hand-visuals-only", action="store_true")
    p.add_argument("--source-range", type=float, nargs=2)
    a = p.parse_args()
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    motion = MotionFields.load(a.motion, order)
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    addr = [m.joint(x).qposadr[0] for x in motion.joint_names]
    objects = {}
    for name in ["flower_bouquet", "small_wicker_basket"]:
        track = np.load(a.tracks / (name + ".npz"))
        cores = []
        for path in sorted(a.assets.glob(name + "_hull*.obj")):
            v = trimesh.load(path, force="mesh", process=False).vertices
            equations = ConvexHull(v).equations.copy()
            equations[:, 3] += a.inset_m
            interior = v.mean(axis=0)
            if np.max(equations[:, :3] @ interior + equations[:, 3]) >= 0:
                raise ValueError("inset removes declared convex interior: " + str(path))
            vertices = HalfspaceIntersection(equations, interior).intersections
            surface = trimesh.convex.convex_hull(vertices)
            cores.append((path.name, surface, RayMeshIntersector(surface), equations))
        objects[name] = (track, cores)
    robot = []
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        if a.hand_visuals_only:
            mesh_name = m.mesh(m.geom_dataid[g]).name or "" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else ""
            if (
                not mesh_name.startswith(("left_hand_", "right_hand_"))
                or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                or m.geom_contype[g]
                or m.geom_conaffinity[g]
            ):
                continue
        elif a.all_robot:
            if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_bodyid[g] == 0:
                continue
        else:
            if not name.endswith("_collision") or not any(
                name.startswith(s)
                for s in ["left_hand_", "right_hand_", "left_wrist", "right_wrist", "left_elbow_yaw", "right_elbow_yaw"]
            ):
                continue
        kind = m.geom_type[g]
        size = m.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mi] : m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].copy()
            f = m.mesh_face[m.mesh_faceadr[mi] : m.mesh_faceadr[mi] + m.mesh_facenum[mi]].copy()
            mesh = trimesh.Trimesh(v, f, process=False)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * size)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(height=2 * size[1], radius=size[0], count=[12, 16])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=24)
        else:
            raise ValueError((name, kind))
        robot.append((g, mesh, RayMeshIntersector(mesh)))

    def crosses(engine, start, end):
        vector = end - start
        length = np.linalg.norm(vector, axis=1)
        mask = length > 1e-7
        start, vector, length = start[mask], vector[mask], length[mask]
        hit, ray, _ = engine.intersects_location(start, vector / length[:, None], multiple_hits=False)
        if not len(ray):
            return 0
        distance = np.linalg.norm(hit - start[ray], axis=1)
        return int(np.sum((distance > 1e-5) & (distance < length[ray] - 1e-5)))

    rows = []
    sampled = 0
    for tick in range(2 * len(motion.joint_pos) - 1):
        if a.source_range and not a.source_range[0] <= 108 + tick / 100 <= a.source_range[1]:
            continue
        sampled += 1
        k = tick // 2
        nxt = min(k + 1, len(motion.joint_pos) - 1)
        w = (tick % 2) / 2
        d.qpos[addr] = (1 - w) * motion.joint_pos[k] + w * motion.joint_pos[nxt]
        d.qpos[:3] = (1 - w) * motion.body_pos_w[k, 0] + w * motion.body_pos_w[nxt, 0]
        q = (1 - w) * motion.body_quat_w[k, 0] + w * motion.body_quat_w[nxt, 0]
        d.qpos[3:7] = q / np.linalg.norm(q)
        mujoco.mj_forward(m, d)
        for name, (track, cores) in objects.items():
            pos = (1 - w) * track["position_w"][k] + w * track["position_w"][nxt]
            q0 = track["quaternion_xyzw"][k]
            q1 = track["quaternion_xyzw"][nxt]
            if q0 @ q1 < 0:
                q1 = -q1
            q = (1 - w) * q0 + w * q1
            rot = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()
            for g, mesh, engine in robot:
                gr = d.geom_xmat[g].reshape(3, 3)
                gp = d.geom_xpos[g]
                world = mesh.vertices @ gr.T + gp
                local = (world - pos) @ rot
                for part, core, core_engine, equations in cores:
                    if np.any(local.max(axis=0) < core.bounds[0]) or np.any(local.min(axis=0) > core.bounds[1]):
                        continue
                    edges = mesh.edges_unique
                    direct = crosses(core_engine, local[edges[:, 0]], local[edges[:, 1]])
                    vertices = (core.vertices @ rot.T + pos - gp) @ gr
                    edges = core.edges_unique
                    reverse = crosses(engine, vertices[edges[:, 0]], vertices[edges[:, 1]])
                    near = np.all((local >= core.bounds[0]) & (local <= core.bounds[1]), axis=1)
                    inside = int(np.sum(np.max(local[near] @ equations[:, :3].T + equations[:, 3], axis=1) < 0))
                    if direct or reverse or inside:
                        rows.append(
                            {
                                "source_s": 108 + tick / 100,
                                "geom": m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}:visual:{g}",
                                "mesh": m.mesh(m.geom_dataid[g]).name
                                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                                else None,
                                "object": name,
                                "core": part,
                                "robot_edge_crossings": direct,
                                "object_edge_crossings": reverse,
                                "robot_vertices_inside_core": inside,
                            }
                        )
        if tick % 200 == 0:
            print(tick, len(rows), flush=True)
    a.out.write_text(
        json.dumps(
            {
                "status": "diagnostic; no automatic acceptance",
                "method": "100Hz bidirectional triangle-edge intersections and robot vertices inside uniformly inset convex object cores",
                "inset_m": a.inset_m,
                "all_robot_colliders": a.all_robot,
                "hand_visuals_only": a.hand_visuals_only,
                "source_range_s": a.source_range,
                "frames_100hz": sampled,
                "motion_sha256": sha256(a.motion),
                "scope": (
                    "actual native hand visual meshes" if a.hand_visuals_only else "selected robot collision geoms"
                )
                + " versus task collision proxies; object fully contained within robot mesh, continuous between100Hz samples, and robot self collision not proved",
                "events": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print("events", len(rows))


if __name__ == "__main__":
    main()
