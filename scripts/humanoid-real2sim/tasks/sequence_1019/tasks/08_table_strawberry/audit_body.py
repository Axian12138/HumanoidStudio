"""Whole-body collider sweep and source-contact foot audit, independently of IK."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields


def main():
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--motion-file", default="motion.npz")
    ap.add_argument("--output-name", default="body_audit")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--room-world-mesh", type=Path)
    args = ap.parse_args()
    p = args.candidate
    out = p / args.output_name
    out.mkdir(exist_ok=False)
    thresholds = {
        "ground_penetration_m": 0.005,
        "furniture_contact_numerical_tolerance_m": 0.002,
        "support_slide_interval_m": 0.02,
        "support_point_speed_m_s": 0.10,
        "support_height_m": 0.015,
        "acceleration": "No model acceleration limits identified; report numerical peaks and seam jumps, do not invent hardware PASS.",
        "rationale": "Foot criteria are diagnostic source-contact consistency tolerances, not measured physical stability limits. Registered before this audit.",
    }
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    motion = MotionFields.load(p / args.motion_file, p / "motion_order.json")
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    data = mujoco.MjData(model)
    adr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    snap = Path("outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot")
    room_path = args.room_world_mesh or snap / "assets/ikea_1012_room.obj"
    room = trimesh.load_mesh(room_path, force="mesh")
    geometry = json.loads((p / "geometry_config.json").read_text()) if (p / "geometry_config.json").exists() else {}
    if geometry and not args.room_world_mesh:
        room.apply_transform(np.array(geometry["T_room_correction"]))
    import hashlib

    (out / "room_provenance.json").write_text(
        json.dumps(
            {
                "path": str(room_path.resolve()),
                "sha256": hashlib.sha256(room_path.read_bytes()).hexdigest(),
                "already_world": bool(args.room_world_mesh),
                "chair": "accepted07 original world, independently included",
            },
            indent=2,
        )
    )
    # Ground is audited analytically; remove only near-horizontal floor faces.
    keep = room.triangles[:, :, 2].max(axis=1) > 0.01
    room = trimesh.Trimesh(room.vertices, room.faces[keep], process=False)
    chair = trimesh.load_mesh(snap / "tasks/07_armchair/grey_armchair_visual.obj", force="mesh")
    ct = np.load(
        "/home/galbot/workspace/ScaleTrack/data/1019test/brace-dining-table-kinematic-v1/metadata/object_trajectories/grey_armchair.npz"
    )
    cq = ct["quaternion_wxyz"][-1]
    chair.vertices = chair.vertices @ Rotation.from_quat(cq[[1, 2, 3, 0]]).as_matrix().T + ct["position_w"][-1]
    surfaces = {"static_room": room, "inherited07_chair_final": chair}
    engines = {n: RayMeshIntersector(m) for n, m in surfaces.items()}
    trees = {n: m.triangles_tree for n, m in surfaces.items()}
    spec = importlib.util.spec_from_file_location(
        "basket_geo", Path(__file__).parents[1] / "09_lined_rect_basket/fit_hoi.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    colliders = []
    collider_engines = {}
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]) or model.geom_bodyid[g] == 0:
            continue
        typ = model.geom_type[g]
        size = model.geom_size[g]
        if typ == mujoco.mjtGeom.mjGEOM_MESH:
            i = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[i] : model.mesh_vertadr[i] + model.mesh_vertnum[i]].copy()
            f = model.mesh_face[model.mesh_faceadr[i] : model.mesh_faceadr[i] + model.mesh_facenum[i]].copy()
            mesh = trimesh.Trimesh(v, f, process=False)
        elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(height=2 * size[1], radius=size[0], count=[12, 16])
        elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
        elif typ == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * size)
        else:
            raise ValueError((g, typ))
        colliders.append((g, mesh.vertices, mesh.edges_unique))
        if args.bidirectional:
            collider_engines[g] = RayMeshIntersector(mesh)
    basket = np.load(p / "lined_rect_basket.npz")
    br = Rotation.from_quat(basket["quaternion_xyzw"]).as_matrix()
    times = 90 + np.arange(899) / 100
    ground = np.empty((899, len(colliders)))
    basket_clear = ground.copy()
    feet = np.empty((899, 2, 7, 3))
    feet_bottom = np.empty((899, 2, 7))
    intersections = []
    for step, t in enumerate(times):
        k = step // 2
        j = min(k + 1, 449)
        w = (step % 2) / 2
        data.qpos[adr] = (1 - w) * motion.joint_pos[k] + w * motion.joint_pos[j]
        data.qpos[:3] = (1 - w) * motion.body_pos_w[k, 0] + w * motion.body_pos_w[j, 0]
        q = (1 - w) * motion.body_quat_w[k, 0] + w * motion.body_quat_w[j, 0]
        data.qpos[3:7] = q / np.linalg.norm(q)
        mujoco.mj_forward(model, data)
        for gi, (g, vertices, edges) in enumerate(colliders):
            rot = data.geom_xmat[g].reshape(3, 3)
            world = vertices @ rot.T + data.geom_xpos[g]
            ground[step, gi] = world[:, 2].min()
            basket_clear[step, gi] = geo.basket_sdf((world - basket["position_w"][k]) @ br[k]).min()
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ground[step, gi] = data.geom_xpos[g, 2] - abs(rot[2, 2]) * model.geom_size[g, 1] - model.geom_size[g, 0]
            name = model.geom(g).name
            if "_foot" in name:
                side = int(name.startswith("right"))
                idx = int(name.split("foot")[1].split("_")[0]) - 1
                feet[step, side, idx] = data.geom_xpos[g]
                feet_bottom[step, side, idx] = ground[step, gi]
            bounds = np.r_[world.min(axis=0), world.max(axis=0)]
            for label, tree in trees.items():
                nearby_faces = list(tree.intersection(bounds))
                if not nearby_faces:
                    continue
                if args.bidirectional:
                    triangles = surfaces[label].triangles[nearby_faces]
                    starts = triangles[:, [0, 1, 2]].reshape(-1, 3)
                    ends = triangles[:, [1, 2, 0]].reshape(-1, 3)
                    local_starts = (starts - data.geom_xpos[g]) @ rot
                    vectors = (ends - starts) @ rot
                    sizes = np.linalg.norm(vectors, axis=1)
                    valid_edges = sizes > 1e-7
                    local_starts, vectors, sizes = local_starts[valid_edges], vectors[valid_edges], sizes[valid_edges]
                    hits, rays, _ = collider_engines[g].intersects_location(
                        local_starts, vectors / sizes[:, None], multiple_hits=False
                    )
                    if len(rays):
                        distances = np.linalg.norm(hits - local_starts[rays], axis=1)
                        inside_edges = (distances > 1e-5) & (distances < sizes[rays] - 1e-5)
                        if inside_edges.any():
                            intersections.append(
                                {
                                    "time_s": float(t),
                                    "geom": name,
                                    "surface": label,
                                    "direction": "furniture_edges_against_robot",
                                    "crossing_edges": int(inside_edges.sum()),
                                    "first_hit_world": (hits[inside_edges][0] @ rot.T + data.geom_xpos[g]).tolist(),
                                }
                            )
                origin = world[edges[:, 0]]
                vector = world[edges[:, 1]] - origin
                lengths = np.linalg.norm(vector, axis=1)
                valid = lengths > 1e-7
                origin = origin[valid]
                vector = vector[valid]
                lengths = lengths[valid]
                hits, rays, _ = engines[label].intersects_location(
                    origin, vector / lengths[:, None], multiple_hits=False
                )
                if not len(rays):
                    continue
                dist = np.linalg.norm(hits - origin[rays], axis=1)
                interior = (dist > 1e-5) & (dist < lengths[rays] - 1e-5)
                if interior.any():
                    # Endpoint depth along an intersecting edge is descriptive,
                    # not a signed Euclidean penetration estimate for open meshes.
                    margins = np.minimum(dist[interior], lengths[rays[interior]] - dist[interior])
                    intersections.append(
                        {
                            "time_s": float(t),
                            "geom": name,
                            "surface": label,
                            "crossing_edges": int(interior.sum()),
                            "max_edge_interior_margin_m": float(margins.max()),
                            "first_hit_world": hits[interior][0].tolist(),
                        }
                    )
        if step % 100 == 0:
            print(f"{step}/899 intersections={len(intersections)}", flush=True)
    contact = np.load(p / "human.npz")["foot_contact"]
    foot_report = {}
    for side, name in enumerate(["left", "right"]):
        # Capsule center displacement is a conservative stable-anchor proxy;
        # exact lowest capsule points move as orientation changes during rocking.
        points = feet[::2, side].mean(axis=1)
        speed = np.linalg.norm(np.gradient(points[:, :2], 0.02, axis=0), axis=1)
        mask = contact[:, side].astype(bool)
        changes = np.diff(np.r_[False, mask, False].astype(int))
        intervals = []
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
            delta = np.linalg.norm(points[start:end, :2] - points[start, :2], axis=1)
            intervals.append(
                {
                    "start_s": 90 + start / 50,
                    "end_s": 90 + end / 50,
                    "max_xy_anchor_drift_m": float(delta.max()),
                    "max_xy_speed_m_s": float(speed[start:end].max()),
                    "min_sole_z_m": float(feet_bottom[2 * start : min(2 * end, 899), side].min()),
                    "max_lowest_sole_z_m": float(feet_bottom[2 * start : min(2 * end, 899), side].min(axis=1).max()),
                }
            )
        foot_report[name] = {"source_contact_intervals": intervals, "max_contact_speed_m_s": float(speed[mask].max())}
    vel = np.gradient(motion.joint_pos, 0.02, axis=0)
    acc = np.gradient(vel, 0.02, axis=0)
    seams = {}
    for k in [50, 120, 335, 380, 400]:
        seams[str(90 + k / 50)] = {
            "max_position_step_rad": float(abs(motion.joint_pos[k] - motion.joint_pos[k - 1]).max()),
            "max_acceleration_rad_s2": float(abs(acc[max(0, k - 2) : k + 3]).max()),
            "joint_at_acceleration_peak": motion.joint_names[
                int(np.unravel_index(abs(acc[max(0, k - 2) : k + 3]).argmax(), acc[max(0, k - 2) : k + 3].shape)[1])
            ],
        }
    names = [model.geom(g).name for g, _, _ in colliders]
    report = {
        "scope": "100Hz all enabled robot collision geoms; exact capsule floor minima; triangulated primitive/mesh edge intersections with static room and inherited07 chair. Open surfaces mean intersection evidence, not volumetric penetration proof. Basket vertex SDF uses inherited09 geometry.",
        "collider_count": len(colliders),
        "bidirectional_edge_intersection": args.bidirectional,
        "mesh_vertices_per_frame": sum(len(v) for _, v, _ in colliders),
        "foot_support": foot_report,
        "ground_minimum_by_geom_m": dict(zip(names, ground.min(axis=0).tolist(), strict=True)),
        "basket_minimum_new08_by_geom_m": dict(zip(names, basket_clear[100:800].min(axis=0).tolist(), strict=True)),
        "surface_intersections": intersections,
        "seams": seams,
        "joint_acceleration_max_rad_s2": dict(zip(motion.joint_names, abs(acc).max(axis=0).tolist(), strict=True)),
        "accepted": False,
        "geometry_config": geometry,
    }
    np.savez_compressed(
        out / "sweep.npz",
        source_time_s=times,
        ground_minimum=ground,
        basket_minimum=basket_clear,
        foot_capsule_centers=feet,
        foot_capsule_bottom=feet_bottom,
        geom_names=np.array(names),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("finished", out, flush=True)


if __name__ == "__main__":
    main()
