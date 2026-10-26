"""Jointly fit a smooth redundant arm path with basket clearance."""


def main():

    import importlib.util
    import json
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    r = Path(Path("/tmp/berry_release_run").read_text())
    src = r / "retarget/v7"
    out = r / "retarget/v8"
    shutil.copytree(src, out)
    old = Path(
        "outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800/retarget/candidates/08_table_strawberry/v25_room_v4"
    )
    b = MotionFields.load(old / "motion.npz", old / "motion_order.json")
    a = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(model))
    d = mujoco.MjData(m)
    adr = np.array([m.joint(n).qposadr[0] for n in b.joint_names])
    ids = np.array(
        [
            b.joint_names.index("right_" + s + "_joint")
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
    palm = m.geom("right_hand_palm_collision").id
    spec = importlib.util.spec_from_file_location(
        "geo", "scripts/humanoid-real2sim/tasks/sequence_1019/tasks/09_lined_rect_basket/fit_hoi.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    samples = []
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        if (
            name.startswith("right_")
            and name.endswith("_collision")
            and any(x in name for x in ["hand_", "wrist", "elbow_yaw"])
        ):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                pts = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid] : 150].copy()
            else:
                pts = geo.collision_points(m, g)
            samples.append((g, pts))
    berry = dict(np.load(old / "table_strawberry.npz"))
    basket = np.load(old / "lined_rect_basket.npz")
    br = Rotation.from_quat(basket["quaternion_xyzw"]).as_matrix()
    start = 245
    stop = 405
    n = stop - start
    centers = []
    rots = []
    offset = None
    for k in range(start, stop):
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = b.joint_pos[k]
        mujoco.mj_forward(m, d)
        rot = d.geom_xmat[palm].reshape(3, 3).copy()
        if offset is None:
            offset = rot.T @ (berry["position_w"][k] - d.geom_xpos[palm])
        centers.append(d.geom_xpos[palm] + rot @ offset)
        rots.append(rot)
    base = b.joint_pos[start:stop, ids]
    times = 90 + np.arange(start, stop) / 50
    w = np.minimum(np.clip((times - 95) / 1.2, 0, 1), np.clip((98 - times) / 0.8, 0, 1))
    w = w * w * (3 - 2 * w)
    target = base[:, 5:] * (1 - 0.75 * w[:, None])
    et = base[:, 3] * (1 - w) + 1.7 * w
    nc = sum(len(pts) for _, pts in samples)
    nr = 3 + 2 + 1 + 1 + 7 + nc + 7
    # Each frame depends on its own seven joints; temporal regularizers couple neighbors.
    sparsity = lil_matrix((n * nr + (n - 1) * 7 + (n - 2) * 7, n * 7), dtype=int)
    for i in range(n):
        sparsity[i * nr : (i + 1) * nr, i * 7 : (i + 1) * 7] = 1
    for i in range(n - 1):
        sparsity[n * nr + i * 7 : n * nr + (i + 1) * 7, i * 7 : (i + 2) * 7] = 1
    for i in range(n - 2):
        sparsity[n * nr + (n - 1) * 7 + i * 7 : n * nr + (n - 1) * 7 + (i + 1) * 7, i * 7 : (i + 3) * 7] = 1
    fixed = np.zeros(n)
    fixed[:5] = 100
    fixed[-5:] = 100
    lo = np.array([m.joint(b.joint_names[i]).range[0] for i in ids]) + 1e-5
    hi = np.array([m.joint(b.joint_names[i]).range[1] for i in ids]) - 1e-5

    def fun(flat):
        x = flat.reshape(n, 7)
        res = []
        for i, k in enumerate(range(start, stop)):
            d.qpos[:3] = b.body_pos_w[k, 0]
            d.qpos[3:7] = b.body_quat_w[k, 0]
            d.qpos[adr] = b.joint_pos[k]
            d.qpos[adr[ids]] = x[i]
            mujoco.mj_forward(m, d)
            rot = d.geom_xmat[palm].reshape(3, 3)
            pts = np.concatenate([v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, v in samples])
            clear = geo.basket_sdf((pts - basket["position_w"][k]) @ br[k])
            res.extend(
                np.r_[
                    300 * (d.geom_xpos[palm] + rot @ offset - centers[i]),
                    1.2 * (x[i, 5:] - target[i]),
                    0.8 * (x[i, 3] - et[i]),
                    0.25 * (x[i, 4] - base[i, 4]),
                    0.12 * (x[i] - base[i]),
                    60 * np.maximum(0.004 - clear, 0),
                    fixed[i] * (x[i] - base[i]),
                ]
            )
        return np.r_[res, 0.8 * np.diff(x, axis=0).ravel(), 5 * np.diff(x, n=2, axis=0).ravel()]

    x0 = np.clip(a.joint_pos[start:stop, ids], lo + 1e-6, hi - 1e-6)
    fit = least_squares(
        fun,
        x0.ravel(),
        bounds=(np.tile(lo, n), np.tile(hi, n)),
        jac_sparsity=sparsity.tocsr(),
        max_nfev=45,
        ftol=1e-5,
        verbose=1,
    )
    q = b.joint_pos.astype(float).copy()
    q[start:stop, ids] = fit.x.reshape(n, 7)
    q[:250] = b.joint_pos[:250]
    q[400:] = b.joint_pos[400:]
    result = refresh_fk(replace(b, joint_pos=q), model)
    result.save(out / "motion.npz")
    (out / "trajectory_fit.json").write_text(
        json.dumps(
            {
                "cost": fit.cost,
                "optimality": fit.optimality,
                "nfev": fit.nfev,
                "release_arm_degrees": np.degrees(q[350, ids]).tolist(),
                "max_arm_speed": float(abs(result.joint_vel[250:400, ids]).max()),
            },
            indent=2,
        )
    )
    print((out / "trajectory_fit.json").read_text())


if __name__ == "__main__":
    main()
