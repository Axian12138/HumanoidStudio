"""Check bidirectional actual visual surfaces of the hand and moving drawer."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation, Slerp
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("--window", nargs=2, type=float, default=[49.8, 54])
    p.add_argument("--active-window", nargs=2, type=float, default=[51.2, 53.38])
    a = p.parse_args()
    m, z = mujoco.MjModel.from_xml_path(str(a.model)), np.load(a.candidate / "motion.npz")
    d = mujoco.MjData(m)
    order = json.loads((a.candidate / "motion_order.json").read_text())
    addr = [m.joint(n).qposadr[0] for n in order["robot_joint_names"]]
    times = np.arange(round(a.window[0] * 100), round(a.window[1] * 100) + 1) / 100
    source = 27 + np.arange(len(z["joint_pos"])) / 50
    root = Slerp(source, Rotation.from_quat(z["body_quat_w"][:, 0, [1, 2, 3, 0]]))(times).as_quat()
    slide = np.load(a.candidate / "drawer_slide.npz")["position"]
    drawer_gid = next(
        g
        for g in range(m.ngeom)
        if m.geom_dataid[g] == m.mesh("drawer_visual").id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    )

    def mesh_for(g):
        mi = int(m.geom_dataid[g])
        va, vn, fa, fn = map(int, [m.mesh_vertadr[mi], m.mesh_vertnum[mi], m.mesh_faceadr[mi], m.mesh_facenum[mi]])
        return trimesh.Trimesh(m.mesh_vert[va : va + vn], m.mesh_face[fa : fa + fn], process=True)

    def prepare(mesh):
        mesh.fix_normals()
        if not mesh.is_watertight:
            raise ValueError("Native containment requires a watertight part")
        points = []
        for tri in mesh.triangles:
            n = max(1, int(np.ceil(np.linalg.norm(tri - np.roll(tri, 1, axis=0), axis=1).max() / 0.002)))
            for u in range(n + 1):
                v = np.arange(n + 1 - u) / n
                points.append(tri[0] + (tri[1] - tri[0]) * (u / n) + v[:, None] * (tri[2] - tri[0]))
        samples = np.unique(np.round(np.vstack(points), 7), axis=0)
        return mesh, RayMeshIntersector(mesh), samples

    drawer = [prepare(part) for part in mesh_for(drawer_gid).split(only_watertight=False)]
    hands = []
    for g in range(m.ngeom):
        if (
            m.body(int(m.geom_bodyid[g])).name.startswith("right_hand")
            and not m.geom_contype[g]
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            hands.append((g, prepare(mesh_for(g))))

    def inside_depth(points, prepared):
        mesh, ray, _ = prepared
        use = np.all((points >= mesh.bounds[0] - 1e-7) & (points <= mesh.bounds[1] + 1e-7), axis=1)
        points = points[use]
        if not len(points):
            return 0.0
        interior = points[ray.contains_points(points)]
        if not len(interior):
            return 0.0
        return float(trimesh.proximity.closest_point(mesh, interior)[1].max())

    records = []
    for k, t in enumerate(times):
        d.qpos[:3] = [np.interp(t, source, z["body_pos_w"][:, 0, j]) for j in range(3)]
        d.qpos[3:7] = root[k, [3, 0, 1, 2]]
        d.qpos[addr] = [np.interp(t, source, z["joint_pos"][:, j]) for j in range(len(addr))]
        d.qpos[m.joint("drawer_slide").qposadr[0]] = np.interp(t, source, slide)
        mujoco.mj_forward(m, d)
        dr, dp = d.geom_xmat[drawer_gid].reshape(3, 3), d.geom_xpos[drawer_gid]
        for g, hand in hands:
            hr, hp = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            hpoints = ((hand[2] @ hr.T + hp) - dp) @ dr
            forward, reverse, nearest = 0.0, 0.0, 0.3
            for part in drawer:
                forward = max(forward, inside_depth(hpoints, part))
                dpoints = ((part[2] @ dr.T + dp) - hp) @ hr
                reverse = max(reverse, inside_depth(dpoints, hand))
                near = np.all((hpoints >= part[0].bounds[0] - 0.02) & (hpoints <= part[0].bounds[1] + 0.02), axis=1)
                if near.any():
                    nearest = min(nearest, float(trimesh.proximity.closest_point(part[0], hpoints[near])[1].min()))
            records.append(
                {
                    "source_s": float(t),
                    "body": m.body(int(m.geom_bodyid[g])).name,
                    "hand_surface_in_drawer_m": forward,
                    "drawer_surface_in_hand_m": reverse,
                    "nearest_sampled_surface_distance_m": nearest,
                }
            )
        if k % 50 == 0:
            print({"sample": k, "total": len(times)}, flush=True)
    report = {
        "status": "bidirectional_native_surface_diagnostic",
        "sampling": "100Hz linear q/position with root SLERP; actual triangle surfaces, grid edge<=2mm, deduplicated at1e-7m.",
        "scope": "Six watertight drawer visual components and seven watertight right-hand visual meshes; surface penetration distances, not force/contact feasibility.",
        "model": str(a.model),
        "active_window_s": a.active_window,
        "max_hand_surface_in_drawer_m": max(x["hand_surface_in_drawer_m"] for x in records),
        "max_drawer_surface_in_hand_m": max(x["drawer_surface_in_hand_m"] for x in records),
        "active_max_nearest_sampled_surface_distance_m": max(
            min(x["nearest_sampled_surface_distance_m"] for x in records if x["source_s"] == float(t))
            for t in times
            if a.active_window[0] <= t <= a.active_window[1]
        ),
        "records": records,
    }
    (a.candidate / "native_drawer_hand_100hz.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "records"})


if __name__ == "__main__":
    main()
