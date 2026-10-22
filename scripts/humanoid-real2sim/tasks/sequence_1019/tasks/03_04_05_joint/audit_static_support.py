"""Compute gravity COM/support margins from actual collision foot geometry and mass."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial import ConvexHull

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    a = p.parse_args()
    b = a.run / "retarget/candidates" / a.candidate
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    data = mujoco.MjData(model)
    z = np.load(b / "motion.npz")
    order = json.loads((b / "motion_order.json").read_text())
    addr = np.array([model.joint(n).qposadr[0] for n in order["robot_joint_names"]])
    foot_geoms = {}
    for side in ["left", "right"]:
        body = model.body(side + "_ankle_roll_link").id
        gs = []
        for g in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[g]
                start = model.mesh_vertadr[m]
                vertices = model.mesh_vert[start : start + model.mesh_vertnum[m]].copy()
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                vertices = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]) * model.geom_size[g]
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius, half_length = model.geom_size[g, :2]
                vertices = np.array(
                    [
                        [
                            radius * np.sin(phi) * np.cos(theta),
                            radius * np.sin(phi) * np.sin(theta),
                            radius * np.cos(phi) + end * half_length,
                        ]
                        for end in [-1, 1]
                        for phi in np.linspace(0, np.pi, 17)
                        for theta in np.linspace(0, 2 * np.pi, 32, endpoint=False)
                    ]
                )
            else:
                raise ValueError("Unsupported foot geom")
            gs.append((g, vertices))
        if not gs:
            raise ValueError(side + " no foot collision")
        foot_geoms[side] = gs
    records = []
    for k in range(875, 1126):
        data.qpos[:3] = z["body_pos_w"][k, 0]
        data.qpos[3:7] = z["body_quat_w"][k, 0]
        data.qpos[addr] = z["joint_pos"][k]
        mujoco.mj_forward(model, data)
        com = np.average(data.xipos[1:], axis=0, weights=model.body_mass[1:])
        support = []
        floor = {}
        for side, geoms in foot_geoms.items():
            verts = np.concatenate([v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] for g, v in geoms])
            floor[side] = float(verts[:, 2].min())
            if floor[side] < 0.01:
                support.extend(verts[verts[:, 2] < verts[:, 2].min() + 0.015, :2])
        hull = ConvexHull(support)
        edges = hull.equations
        dist = (edges[:, :2] @ com[:2] + edges[:, 2]) / np.linalg.norm(edges[:, :2], axis=1)
        records.append(
            {
                "source_s": 27 + k / 50,
                "com_world_m": com.tolist(),
                "static_support_margin_m": float(-dist.max()),
                "foot_min_z_m": floor,
            }
        )
    summary = {
        "status": "static_geometry_only_not_dynamic_balance",
        "minimum_com_margin_m": min(x["static_support_margin_m"] for x in records),
        "release_window_47_48_margin_m": min(
            x["static_support_margin_m"] for x in records if 47 <= x["source_s"] <= 48
        ),
        "support_foot_height_threshold_m": 0.01,
        "minimum_foot_z_m": {s: min(x["foot_min_z_m"][s] for x in records) for s in foot_geoms},
        "maximum_foot_z_m": {s: max(x["foot_min_z_m"][s] for x in records) for s in foot_geoms},
        "limitations": [
            "Foot collision bottom15mm is projected as support polygon; actual load-bearing contact and ZMP are not measured.",
            "This cannot certify balance during acceleration.",
        ],
        "frames": records,
    }
    (b / "static_support.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({k: v for k, v in summary.items() if k != "frames"})


if __name__ == "__main__":
    main()
