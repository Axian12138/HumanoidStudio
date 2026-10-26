"""Resolve conservative bound overlaps using original component triangle surfaces."""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()
    import json
    from pathlib import Path

    import mujoco
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    r = Path(args.run) if args.run else Path(Path("/tmp/wipes_run").read_text())
    p = r / "retarget" / args.version
    z = np.load(p / "motion.npz")
    names = json.loads((p / "motion_order.json").read_text())["robot_joint_names"]
    cfg = json.loads((r / "scene/config.json").read_text())
    R = Rotation.from_quat(cfg["quaternion_xyzw"]).as_matrix()
    pos = np.array(cfg["position_w"])
    sweep = np.load(r / "review" / f"sweep_{args.version}.npz")["values"]
    ticks = np.unique(
        np.r_[np.rint((sweep[sweep[:, 1:-1].min(1) < 0, 0] - 83.5) * 100), np.arange(300, 371, 10)]
    ).astype(int)
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    adr = [m.joint(n).qposadr[0] for n in names]
    samples = []
    for g in range(m.ngeom):
        n = m.geom(g).name or ""
        if n.startswith("left_hand") and n.endswith("_collision") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            samples.append((g, m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()))
    parts = [trimesh.load(x["mesh"], force="mesh") for x in cfg["parts"]]
    rows = []
    for tick in ticks:
        k = tick // 2
        j = min(k + 1, len(z["joint_pos"]) - 1)
        u = tick % 2 * 0.5
        d.qpos[adr] = (1 - u) * z["joint_pos"][k] + u * z["joint_pos"][j]
        d.qpos[:3] = (1 - u) * z["body_pos_w"][k, 0] + u * z["body_pos_w"][j, 0]
        qt = (1 - u) * z["body_quat_w"][k, 0] + u * z["body_quat_w"][j, 0]
        d.qpos[3:7] = qt / np.linalg.norm(qt)
        mujoco.mj_forward(m, d)
        v = (np.concatenate([a @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, a in samples]) - pos) @ R
        values = []
        for mesh in parts:
            bounds = mesh.bounds
            near = np.all((v >= bounds[0] - 0.005) & (v <= bounds[1] + 0.005), axis=1)
            if near.any():
                values.append(float(-trimesh.proximity.signed_distance(mesh, v[near]).max()))
            else:
                values.append(None)
        rows.append({"source_s": 83.5 + tick / 100, "signed_clearance_m": values})
    vals = [x for row in rows for x in row["signed_clearance_m"] if x is not None]
    report = {
        "scope": "original triangle signed distances at every100Hz sample with AABB overlap plus contact samples; vertex sampling is not physics",
        "minimum_surface_clearance_m": min(vals),
        "rows": rows,
    }
    (r / "review" / f"exact_contact_{args.version}.json").write_text(json.dumps(report, indent=2))
    print("minimum", min(vals))


if __name__ == "__main__":
    main()
