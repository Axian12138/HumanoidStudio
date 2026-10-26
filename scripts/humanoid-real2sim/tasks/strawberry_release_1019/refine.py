"""Redistribute release arm posture while retaining the original palm transform."""


def main():

    import json
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    SRC = Path(
        "outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800/retarget/candidates/08_table_strawberry/v25_room_v4"
    )
    MODEL = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    r = Path(Path("/tmp/berry_release_run").read_text())
    out = r / "retarget/v6"
    out.mkdir(parents=True)
    for p in SRC.iterdir():
        if p.is_file() and (
            p.suffix == ".npz" or p.name in ["motion_order.json", "geometry_config.json", "tracks.json"]
        ):
            shutil.copy2(p, out / p.name)
    b = MotionFields.load(SRC / "motion.npz", SRC / "motion_order.json")
    q = b.joint_pos.astype(float).copy()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
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
    lo = np.array([m.joint(b.joint_names[i]).range[0] for i in ids]) + 1e-6
    hi = np.array([m.joint(b.joint_names[i]).range[1] for i in ids]) - 1e-6
    errors = []
    berry = dict(np.load(SRC / "table_strawberry.npz"))
    oldrot = Rotation.from_quat(berry["quaternion_xyzw"]).as_matrix()
    delta_rot = []
    for k in range(250, 400):
        t = 90 + k / 50
        w = min(max((t - 95) / 1.2, 0), max((98 - t) / 0.8, 0), 1)
        w = w * w * (3 - 2 * w)
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = q[k]
        mujoco.mj_forward(m, d)
        pos = d.geom_xpos[palm].copy()
        rot = d.geom_xmat[palm].reshape(3, 3).copy()
        base = q[k, ids].copy()
        target = base[4:].copy()
        target[1:] *= 1 - 0.75 * w
        offset = rot.T @ (berry["position_w"][k] - pos)
        if k == 250:
            held_offset = offset.copy()
        offset = held_offset

        def fun(x, offset=offset, pos=pos, rot=rot, target=target, base=base, w=w, k=k):
            d.qpos[adr[ids]] = x
            mujoco.mj_forward(m, d)
            return np.r_[
                200 * (d.geom_xpos[palm] + d.geom_xmat[palm].reshape(3, 3) @ offset - (pos + rot @ offset)),
                0.25 * Rotation.from_matrix(d.geom_xmat[palm].reshape(3, 3) @ rot.T).as_rotvec(),
                2 * (x[5:] - target[1:]),
                0.2 * (x[4] - base[4]),
                0.15 * (x[:4] - base[:4]),
                1.5 * (x[3] - (base[3] * (1 - w) + 1.7 * w)),
                0.5 * (x - q[k - 1, ids]),
            ]

        lower = np.maximum(lo, base - np.array([1.0, 1.0, 0.8, 0.9, 0.5, 2.0, 2.0]))
        upper = np.minimum(hi, base + np.array([1.0, 1.0, 0.8, 0.9, 0.5, 2.0, 2.0]))
        fit = least_squares(
            fun,
            np.clip(q[k - 1, ids], lower, upper),
            bounds=(lower, upper),
            max_nfev=80,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        q[k, ids] = fit.x
        errors.append([t, float(np.linalg.norm(fun(fit.x)[:3]) / 200), float(np.linalg.norm(fun(fit.x)[3:6]) / 0.08)])
        delta_rot.append(d.geom_xmat[palm].reshape(3, 3) @ rot.T)
        if t <= 97:
            berry["quaternion_xyzw"][k] = Rotation.from_matrix(delta_rot[-1] @ oldrot[k]).as_quat()
    release_delta = Rotation.from_matrix(delta_rot[100]).as_rotvec()
    for k in range(351, 376):
        u = (k - 350) / 25
        w = 1 - u * u * (3 - 2 * u)
        berry["quaternion_xyzw"][k] = Rotation.from_matrix(
            Rotation.from_rotvec(release_delta * w).as_matrix() @ oldrot[k]
        ).as_quat()
    np.savez_compressed(out / "table_strawberry.npz", **berry)
    result = refresh_fk(replace(b, joint_pos=q), MODEL)
    result.save(out / "motion.npz")
    report = {
        "source": str(SRC.resolve()),
        "eef_errors": errors,
        "release_before_deg": np.degrees(b.joint_pos[350, ids]).tolist(),
        "release_after_deg": np.degrees(q[350, ids]).tolist(),
        "arm_order": [b.joint_names[i] for i in ids],
        "max_speed": float(abs(result.joint_vel[:, ids]).max()),
        "max_acceleration": float(abs(np.gradient(result.joint_vel[:, ids], 0.02, axis=0)).max()),
    }
    (out / "refinement.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "eef_errors"})


if __name__ == "__main__":
    main()
