"""Reach forward through negative shoulder pitch, with roll near the initial posture."""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--version", default="v3")
    args = parser.parse_args()
    import json
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    r = Path(args.run) if args.run else Path(Path("/tmp/wipes_extended_run").read_text())
    src = r / "retarget/baseline"
    out = r / "retarget" / args.version
    shutil.copytree(r / "retarget/previous", out)
    cfg = json.loads((r / "scene/config.json").read_text())
    cfg["position_w"] = [0.10, -2.48, 0.754]
    cfg["quaternion_xyzw"] = [0, 0, 1, 0]
    cfg["placement_note"] = (
        "10cm farther forward (-Y),18.5cm toward body midline; rotated90deg with open lid beyond the fist."
    )
    (r / "scene/config.json").write_text(json.dumps(cfg, indent=2))
    b = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = b.joint_pos.astype(float).copy()
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(model))
    d = mujoco.MjData(m)
    adr = np.array([m.joint(n).qposadr[0] for n in b.joint_names])
    ids = np.array(
        [
            b.joint_names.index("left_" + s + "_joint")
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
    )
    palm = m.geom("left_hand_palm_collision").id
    samples = []
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        if name.startswith("left_") and name.endswith("_collision") and any(x in name for x in ["hand_", "wrist"]):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()
                samples.append((g, v))
    k = 175
    d.qpos[:3] = b.body_pos_w[k, 0]
    d.qpos[3:7] = b.body_quat_w[k, 0]
    d.qpos[adr] = q[k]
    mujoco.mj_forward(m, d)
    R = d.geom_xmat[palm].reshape(3, 3).copy()
    p = d.geom_xpos[palm].copy()
    v = np.concatenate([x @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, x in samples])
    anchor = R.T @ (v[v[:, 2].argmin()] - p)
    roll0 = q[0, ids[1]]
    sparse = [(g, v[::80]) for g, v in samples]
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "contact_geometry", "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/09_lined_rect_basket/fit_hoi.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        if (
            name.startswith("left_")
            and name.endswith("_collision")
            and any(x in name for x in ["shoulder", "elbow", "wrist"])
            and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            sparse.append((g, geo.collision_points(m, g)))
    objR = Rotation.from_quat(cfg["quaternion_xyzw"]).as_matrix()
    objp = np.array(cfg["position_w"])
    bounds = [np.array(x["bounds"]) for x in cfg["parts"]]
    lo = np.array([m.joint(b.joint_names[i]).range[0] for i in ids]) + 1e-6
    hi = np.array([m.joint(b.joint_names[i]).range[1] for i in ids]) - 1e-6

    def smooth(u):
        u = np.clip(u, 0, 1)
        return u * u * u * (10 + u * (-15 + 6 * u))

    errors = []
    for k in range(75, 220):
        t = 83.5 + k / 50
        w = smooth((t - 85.0) / 1.35) * (1 - smooth((t - 87.2) / 0.7))
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = b.joint_pos[k]
        mujoco.mj_forward(m, d)
        R = d.geom_xmat[palm].reshape(3, 3).copy()
        old = d.geom_xpos[palm] + R @ anchor
        target = old * (1 - w) + np.array([0.10, -2.44, 0.8064]) * w
        u = np.clip((t - 87.2) / 0.7, 0, 1)
        target[2] += 0.05 * np.sin(np.pi * u) ** 2
        base = b.joint_pos[k, ids].copy()
        prior = base * (1 - w) + np.array([-2.0, roll0, 0.72, -0.3, 0.40, 0, 0]) * w

        def fun(x, target=target, w=w, R=R, prior=prior, k=k):
            d.qpos[adr[ids]] = x
            mujoco.mj_forward(m, d)
            RR = d.geom_xmat[palm].reshape(3, 3)
            verts = np.concatenate([v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, v in sparse])
            local = (verts - objp) @ objR
            clear = []
            for bound in bounds:
                delta = np.abs(local - bound.mean(0)) - (bound[1] - bound[0]) / 2
                sdf = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(1), 0)
                clear.append(np.maximum(0.002 - sdf, 0))
            inside = (verts[:, 0] > -1.03) & (verts[:, 0] < 0.57) & (verts[:, 1] < -2.11905405) & (verts[:, 1] > -2.97)
            return np.r_[
                250 * (d.geom_xpos[palm] + RR @ anchor - target),
                4 * w * (x[1] - roll0),
                0.7 * w * (x[3] + 0.3),
                0.15 * w * x[4],
                2.0 * w * x[5:],
                0.15 * w * (x[2] - 0.72),
                0.25 * Rotation.from_matrix(RR @ R.T).as_rotvec(),
                0.12 * (x - prior),
                0.12 * (x - q[k - 1, ids]),
                200 * np.concatenate(clear),
                400 * np.maximum(0.761 - verts[:, 2], 0) * inside,
            ]

        seed = np.clip(q[k - 1, ids], lo, hi)
        if k == 140:
            seed = np.clip(prior, lo, hi)
        fit = least_squares(fun, seed, bounds=(lo, hi), max_nfev=55)
        q[k, ids] = fit.x
        errors.append([t, float(np.linalg.norm(fun(fit.x)[:3]) / 250)])
    # Preserve both existing seams exactly. The final correction vanishes smoothly.
    for k in range(210, 220):
        w = smooth((k - 210) / 10)
        q[k, ids] = q[k, ids] * (1 - w) + b.joint_pos[k, ids] * w
    refresh_fk(replace(b, joint_pos=q), model).save(out / "motion.npz")
    np.savez_compressed(
        out / "wet_wipes.npz",
        position_w=np.tile(objp, (len(q), 1)),
        quaternion_xyzw=np.tile(cfg["quaternion_xyzw"], (len(q), 1)),
        fps=50.0,
        source_time_s=83.5 + np.arange(len(q)) / 50,
    )
    (out / "fit.json").write_text(
        json.dumps(
            {
                "point_errors": errors,
                "release_arm_order": [b.joint_names[i] for i in ids],
                "press_arm_degrees": np.degrees(q[175, ids]).tolist(),
                "initial_roll_degrees": float(np.degrees(roll0)),
                "scope": "left arm only; original fist/right support/root/legs unchanged",
            },
            indent=2,
        )
    )
    print((out / "fit.json").read_text()[-650:])


if __name__ == "__main__":
    main()
