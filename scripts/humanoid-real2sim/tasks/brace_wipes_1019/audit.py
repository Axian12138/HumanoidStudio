"""Independent full-hand sweep against conservative component bounds and tabletop."""


def main():
    import argparse
    import json
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--window-s", nargs=2, type=float)
    a = ap.parse_args()
    p = a.run / "retarget" / a.version
    z = np.load(p / "motion.npz")
    b = np.load(a.run / "retarget/baseline/motion.npz")
    names = json.loads((p / "motion_order.json").read_text())["robot_joint_names"]
    cfg = json.loads((a.run / "scene/config.json").read_text())
    R = Rotation.from_quat(cfg["quaternion_xyzw"]).as_matrix()
    pos = np.array(cfg["position_w"])
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    adr = [m.joint(n).qposadr[0] for n in names]
    samples = []
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        if name.startswith("left_") and name.endswith("_collision") and any(x in name for x in ["hand_", "wrist"]):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()
                samples.append((g, v))
    rows = []
    ticks = range(2 * len(z["joint_pos"]) - 1)
    if a.window_s:
        ticks = [tick for tick in ticks if a.window_s[0] <= 83.5 + tick / 100 <= a.window_s[1]]
    for tick in ticks:
        k = tick // 2
        j = min(k + 1, len(z["joint_pos"]) - 1)
        u = tick % 2 * 0.5
        d.qpos[adr] = (1 - u) * z["joint_pos"][k] + u * z["joint_pos"][j]
        d.qpos[:3] = (1 - u) * z["body_pos_w"][k, 0] + u * z["body_pos_w"][j, 0]
        qt = (1 - u) * z["body_quat_w"][k, 0] + u * z["body_quat_w"][j, 0]
        d.qpos[3:7] = qt / np.linalg.norm(qt)
        mujoco.mj_forward(m, d)
        v = np.concatenate([x @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, x in samples])
        local = (v - pos) @ R
        mins = []
        for part in cfg["parts"]:
            bounds = np.array(part["bounds"])
            center = bounds.mean(0)
            half = (bounds[1] - bounds[0]) / 2
            delta = np.abs(local - center) - half
            sdf = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(1), 0)
            mins.append(float(sdf.min()))
        inside = (v[:, 0] > -1.03) & (v[:, 0] < 0.57) & (v[:, 1] < -2.11905405) & (v[:, 1] > -2.97)
        rows.append([83.5 + tick / 100, *mins, float(v[inside, 2].min() - 0.754) if inside.any() else 1.0])
    rows = np.array(rows)
    np.savez_compressed(a.run / "review" / ("sweep_" + a.version + ".npz"), values=rows)
    untouched = [
        i
        for i, n in enumerate(names)
        if not (n.startswith("left_") and any(s in n for s in ["shoulder", "elbow", "wrist"]))
    ]
    assert np.array_equal(z["joint_pos"][:, untouched], b["joint_pos"][:, untouched])
    mask = (rows[:, 0] >= 86.5) & (rows[:, 0] <= 87.2)
    report = {
        "scope": "all left hand/wrist mesh vertices at100Hz; component local AABBs conservative, not force validation",
        "window_s": a.window_s,
        "untouched_joints_exact": True,
        "root_exact": bool(np.array_equal(z["body_pos_w"][:, 0], b["body_pos_w"][:, 0])),
        "minimum_component_m": {v["name"]: float(rows[:, i + 1].min()) for i, v in enumerate(cfg["parts"])},
        "press_packet_signed_distance_m": [float(rows[mask, 1].min()), float(rows[mask, 1].max())],
        "table_minimum_m": float(rows[:, -1].min()),
        "penetration_over_1mm_times": rows[(rows[:, 1:-1].min(1) < -0.001), 0].tolist(),
    }
    (a.run / "review" / ("contact_" + a.version + ".json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
