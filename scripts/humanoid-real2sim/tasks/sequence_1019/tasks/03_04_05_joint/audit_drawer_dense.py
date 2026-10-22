"""Compare every robot/drawer collision pair at interpolated 100 Hz poses."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--visual-surface", action="store_true")
    p.add_argument("--model", type=Path)
    p.add_argument("--window", type=float, nargs=2, default=[51, 54])
    p.add_argument("--output-name", default="drawer_100hz.json")
    p.add_argument("--active-window", nargs=2, type=float, default=[51.2, 53.38])
    a = p.parse_args()
    model_path = a.model or a.run / "scene/03_04_05/robot_props_drawer.xml"
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    z = np.load(a.candidate / "motion.npz")
    order = json.loads((a.candidate / "motion_order.json").read_text())
    addr = [m.joint(n).qposadr[0] for n in order["robot_joint_names"]]
    rb = {m.body(n).id for n in order["robot_body_names"]}
    db = m.body("drawer").id
    drawer_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == db and m.geom_contype[g]]
    hand_geoms = [
        g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith("right_hand") and m.geom_contype[g]
    ]
    surfaces = []
    if a.visual_surface:
        for g in range(m.ngeom):
            if (
                not m.body(int(m.geom_bodyid[g])).name.startswith("right_hand")
                or m.geom_contype[g]
                or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
            ):
                continue
            mi = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[mi]), int(m.mesh_vertnum[mi])
            fa, fn = int(m.mesh_faceadr[mi]), int(m.mesh_facenum[mi])
            vertices = m.mesh_vert[va : va + vn]
            points = []
            for face in m.mesh_face[fa : fa + fn]:
                tri = vertices[face]
                n = max(1, int(np.ceil(np.linalg.norm(tri - np.roll(tri, 1, axis=0), axis=1).max() / 0.002)))
                for u in range(n + 1):
                    v = np.arange(n + 1 - u) / n
                    points.append(tri[0] + (tri[1] - tri[0]) * (u / n) + v[:, None] * (tri[2] - tri[0]))
            surfaces.append((g, np.vstack(points)))
    da = m.joint("drawer_slide").qposadr[0]
    slide = np.load(a.candidate / "drawer_slide.npz")["position"]
    ts = 27 + np.arange(len(slide)) / 50
    times = np.arange(round(a.window[0] * 100), round(a.window[1] * 100) + 1) / 100
    root = Slerp(ts, Rotation.from_quat(z["body_quat_w"][:, 0, [1, 2, 3, 0]]))(np.clip(times, ts[0], ts[-1])).as_quat()
    q = np.column_stack([np.interp(times, ts, z["joint_pos"][:, j]) for j in range(len(addr))])
    xyz = np.column_stack([np.interp(times, ts, z["body_pos_w"][:, 0, j]) for j in range(3)])
    slider = np.interp(times, ts, slide)
    records, bad = [], []
    visual_records = []
    for k, t in enumerate(times):
        d.qpos[:3] = xyz[k]
        d.qpos[3:7] = root[k, [3, 0, 1, 2]]
        d.qpos[addr] = q[k]
        d.qpos[da] = slider[k]
        mujoco.mj_forward(m, d)
        pairs = []
        for c in d.contact:
            b1, b2 = map(int, m.geom_bodyid[[c.geom1, c.geom2]])
            if db not in (b1, b2):
                continue
            other = b2 if b1 == db else b1
            if other not in rb:
                continue
            pair = {
                "body": m.body(other).name,
                "depth_m": max(0.0, float(-c.dist)),
                "geoms": [m.geom(c.geom1).name, m.geom(c.geom2).name],
            }
            pairs.append(pair)
            if c.dist < -0.005:
                bad.append({"source_s": float(t), **pair})
        closest = min(mujoco.mj_geomDistance(m, d, h, g, 0.3, None) for h in hand_geoms for g in drawer_geoms)
        records.append({"source_s": float(t), "pairs": pairs, "minimum_hand_drawer_distance_m": float(closest)})
        if a.visual_surface:
            for g, points in surfaces:
                world = points @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                deepest = 0.0
                for box in drawer_geoms:
                    if m.geom_type[box] != mujoco.mjtGeom.mjGEOM_BOX:
                        raise ValueError("Surface-depth method expects authored drawer boxes")
                    local = (world - d.geom_xpos[box]) @ d.geom_xmat[box].reshape(3, 3)
                    deepest = max(deepest, float(np.min(m.geom_size[box] - abs(local), axis=1).max()))
                visual_records.append(
                    {
                        "source_s": float(t),
                        "body": m.body(int(m.geom_bodyid[g])).name,
                        "sampled_surface_depth_m": deepest,
                    }
                )
    report = {
        "status": "failed_5mm_depth" if bad else "passed_5mm_depth",
        "scope": "All authored robot/drawer collision pairs; does not validate actual visual meshes or contact force.",
        "model": str(model_path),
        "samples": len(times),
        "window_s": a.window,
        "active_window_s": a.active_window,
        "violations": bad,
        "active_max_nearest_distance_m": max(
            x["minimum_hand_drawer_distance_m"]
            for x in records
            if a.active_window[0] <= x["source_s"] <= a.active_window[1]
        ),
        "records": records,
    }
    (a.candidate / a.output_name).write_text(json.dumps(report, indent=2) + "\n")
    if a.visual_surface:
        visual = {
            "status": "diagnostic_one_direction_actual_surface_to_authored_boxes",
            "maximum_sampled_depth_m": max(x["sampled_surface_depth_m"] for x in visual_records),
            "surface_covering_distance_bound_m": 0.002,
            "scope": "Every actual right-hand visual triangle sampled with maximum lattice edge 2mm. Not reverse drawer-surface containment in hand; not actual curved drawer mesh.",
            "records": visual_records,
        }
        (a.candidate / "drawer_visual_surface_100hz.json").write_text(json.dumps(visual, indent=2) + "\n")
    print({"status": report["status"], "violations": len(bad)})


if __name__ == "__main__":
    main()
