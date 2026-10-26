"""Prepare the requested wipes asset and a contact-preserving left-fist candidate."""


def main():
    import json
    import shutil
    import sys
    from dataclasses import replace
    from pathlib import Path

    import mujoco
    import numpy as np
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.paths import run_dir
    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget3/src")
    import retarget3.runtime  # noqa: F401
    from pxr import Usd, UsdGeom

    r = run_dir("sequence_1019/brace_left_fist_wipes")
    Path("/tmp/wipes_run").write_text(str(r))
    scene = r / "scene"
    scene.mkdir()
    out = r / "retarget/v1"
    out.mkdir(parents=True)
    (r / "review").mkdir()
    (r / "logs").mkdir()
    archive = Path("/home/galbot/workspace/ScaleTrack/data/1019test/brace-dining-table-kinematic-v1")
    z = np.load(archive / "motion.npz")
    fields = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "fps"]
    np.savez_compressed(out / "motion.npz", **{k: z[k] for k in fields})
    for n in ["motion_order.json", "human.npz", "human_motion_world.npz", "source_sync.npz", "chair_track.npz"]:
        shutil.copy2(archive / "metadata" / n, out / n)
    base = r / "retarget/baseline"
    shutil.copytree(out, base)
    asset = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/objects/1012/wet_wipes_v3.usd")
    stage = Usd.Stage.Open(str(asset))
    cache = UsdGeom.XformCache()
    parts = []
    partinfo = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or "/CollisionSet/" in str(prim.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        v = np.array([cache.GetLocalToWorldTransform(prim).Transform(x) for x in mesh.GetPointsAttr().Get()])
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        faces = []
        j = 0
        for count in counts:
            f = indices[j : j + count]
            faces.extend([[f[0], f[k], f[k + 1]] for k in range(1, count - 1)])
            j += count
        t = trimesh.Trimesh(v, np.asarray(faces), process=False)
        name = prim.GetName()
        t.export(scene / (name + ".obj"))
        parts.append(t)
        partinfo.append({"name": name, "mesh": str(scene / (name + ".obj")), "bounds": t.bounds.tolist()})
    trimesh.util.concatenate(parts).export(scene / "wet_wipes_visual.obj")
    position = np.array([0.285, -2.38, 0.754])
    rot = Rotation.from_euler("z", -90, degrees=True)
    config = {
        "asset_blend": str(asset.with_suffix(".blend")),
        "asset_usd": str(asset),
        "parts": partinfo,
        "position_w": position.tolist(),
        "quaternion_xyzw": rot.as_quat().tolist(),
        "scale": [1, 1, 1],
        "press_source_s": [86.5, 87.2],
        "table_height_m": 0.754,
        "physics": "kinematic_only_rigid_packet_no_deformation_or_force_simulation",
    }
    (scene / "config.json").write_text(json.dumps(config, indent=2))
    b = MotionFields.load(base / "motion.npz", base / "motion_order.json")
    q = b.joint_pos.astype(float).copy()
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(model))
    d = mujoco.MjData(m)
    adr = np.array([m.joint(n).qposadr[0] for n in b.joint_names])
    cols = np.array(
        [
            b.joint_names.index("left_" + p + "_joint")
            for p in [
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            ]
        ]
    )
    palm = m.geom("left_hand_palm_collision").id
    samples = []
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        if (
            name.startswith("left_hand")
            and name.endswith("_collision")
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()
            samples.append((g, v))
    lo = np.array([m.joint(b.joint_names[i]).range[0] for i in cols]) + 1e-6
    hi = np.array([m.joint(b.joint_names[i]).range[1] for i in cols]) - 1e-6

    def smooth(u):
        u = np.clip(u, 0, 1)
        return u * u * u * (10 + u * (-15 + 6 * u))

    report = []
    # Use the existing fist and downward approach. Only the final landing is adjusted.
    for k in range(100, 225):
        t = 83.5 + k / 50
        w = smooth((t - 85.5) / 0.9) * (1 - smooth((t - 87.2) / 0.8))
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = q[k]
        mujoco.mj_forward(m, d)
        p = d.geom_xpos[palm].copy()
        R = d.geom_xmat[palm].reshape(3, 3).copy()
        verts = np.concatenate([v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, v in samples])
        tip = verts[verts[:, 2].argmin()]
        offset = R.T @ (tip - p)
        # Maintain a stationary fist contact patch over the packet front, away from its open lid.
        target = tip * (1 - w) + np.array([0.245, -2.38, 0.8063]) * w
        original = q[k, cols].copy()

        def residual(x, offset=offset, target=target, R=R, original=original, k=k):
            d.qpos[adr[cols]] = x
            mujoco.mj_forward(m, d)
            RR = d.geom_xmat[palm].reshape(3, 3)
            return np.r_[
                200 * (d.geom_xpos[palm] + RR @ offset - target),
                3 * Rotation.from_matrix(RR @ R.T).as_rotvec(),
                0.1 * (x - original),
                0.1 * (x - q[k - 1, cols]),
            ]

        fit = least_squares(residual, np.clip(q[k - 1, cols], lo, hi), bounds=(lo, hi), max_nfev=40)
        q[k, cols] = fit.x
        report.append([t, float(np.linalg.norm(residual(fit.x)[:3]) / 200)])
    refresh_fk(replace(b, joint_pos=q), model).save(out / "motion.npz")
    np.savez_compressed(
        out / "wet_wipes.npz",
        position_w=np.tile(position, (len(q), 1)),
        quaternion_xyzw=np.tile(rot.as_quat(), (len(q), 1)),
        fps=50.0,
    )
    (out / "tracks.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"name": "grey_armchair", "path": "chair_track.npz"},
                    {"name": "wet_wipes", "path": "wet_wipes.npz"},
                ]
            },
            indent=2,
        )
    )
    (out / "fit.json").write_text(
        json.dumps(
            {"point_errors": report, "scope": "left arm only; existing closed fist and right support preserved"},
            indent=2,
        )
    )
    # Track manifests use absolute recording time, including the inherited chair.
    for filename in ["chair_track.npz", "wet_wipes.npz"]:
        track = dict(np.load(out / filename))
        track["source_time_s"] = 83.5 + np.arange(len(q)) / 50
        np.savez_compressed(out / filename, **track)
    lines = ["mtllib wet_wipes_visual.mtl"]
    materials = []
    first_vertex = 1
    for i, mesh in enumerate(parts):
        color = (0.25, 0.62, 0.82) if i < 3 else ((0.06, 0.07, 0.08) if i == 4 else (0.9, 0.9, 0.92))
        material = f"part{i}"
        lines.extend(["o " + material, "usemtl " + material])
        lines.extend("v " + " ".join(map(str, v)) for v in mesh.vertices)
        lines.extend("f " + " ".join(str(int(x) + first_vertex) for x in face) for face in mesh.faces)
        first_vertex += len(mesh.vertices)
        materials.extend(["newmtl " + material, "Kd " + " ".join(map(str, color)), "d 1"])
    (scene / "wet_wipes_visual.obj").write_text("\n".join(lines) + "\n")
    (scene / "wet_wipes_visual.mtl").write_text("\n".join(materials) + "\n")
    print(r)


if __name__ == "__main__":
    main()
