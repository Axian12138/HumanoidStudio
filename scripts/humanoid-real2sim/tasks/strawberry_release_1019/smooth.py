"""Smooth the local correction and recompute held strawberry FK consistently."""


def main():

    import json
    import shutil
    from dataclasses import replace
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

    r = Path(Path("/tmp/berry_release_run").read_text())
    src = r / "retarget/v6"
    out = r / "retarget/v7"
    shutil.copytree(src, out)
    old = Path(
        "outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800/retarget/candidates/08_table_strawberry/v25_room_v4"
    )
    b = MotionFields.load(old / "motion.npz", old / "motion_order.json")
    a = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = b.joint_pos.astype(float).copy()
    ids = [
        i
        for i, n in enumerate(b.joint_names)
        if n.startswith("right_") and any(x in n for x in ["shoulder", "elbow", "wrist"])
    ]
    delta = gaussian_filter1d(a.joint_pos[:, ids] - b.joint_pos[:, ids], 5, axis=0)
    for k in range(250, 400):
        u = min((k - 250) / 10, (400 - k) / 10, 1)
        w = u * u * (3 - 2 * u)
        q[k, ids] += delta[k] * w
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(model))
    d = mujoco.MjData(m)
    adr = np.array([m.joint(n).qposadr[0] for n in b.joint_names])
    palm = m.geom("right_hand_palm_collision").id
    berry = dict(np.load(old / "table_strawberry.npz"))
    origp = berry["position_w"].copy()
    origr = Rotation.from_quat(berry["quaternion_xyzw"]).as_matrix()
    position_errors = []
    for k in range(250, 351):
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = b.joint_pos[k]
        mujoco.mj_forward(m, d)
        p = d.geom_xpos[palm].copy()
        rot = d.geom_xmat[palm].reshape(3, 3).copy()
        offset = rot.T @ (origp[k] - p)
        d.qpos[adr] = q[k]
        mujoco.mj_forward(m, d)
        nr = d.geom_xmat[palm].reshape(3, 3).copy()
        berry["position_w"][k] = d.geom_xpos[palm] + nr @ offset
        berry["quaternion_xyzw"][k] = Rotation.from_matrix(nr @ rot.T @ origr[k]).as_quat()
        position_errors.append(float(np.linalg.norm(berry["position_w"][k] - origp[k])))
    rd = Rotation.from_matrix(nr @ rot.T).as_rotvec()
    pd = berry["position_w"][350] - origp[350]
    for k in range(351, 376):
        u = (k - 350) / 25
        w = 1 - u * u * (3 - 2 * u)
        berry["position_w"][k] = origp[k] + pd * w
        berry["quaternion_xyzw"][k] = Rotation.from_matrix(
            Rotation.from_rotvec(rd * w).as_matrix() @ origr[k]
        ).as_quat()
    np.savez_compressed(out / "table_strawberry.npz", **berry)
    result = refresh_fk(replace(b, joint_pos=q), model)
    result.save(out / "motion.npz")
    report = {
        "held_center_max_change_m": max(position_errors),
        "release_center_change_m": float(np.linalg.norm(pd)),
        "release_arm_degrees": np.degrees(q[350, ids]).tolist(),
        "max_right_arm_speed": float(abs(result.joint_vel[250:400, ids]).max()),
        "max_right_arm_acceleration": float(abs(np.gradient(result.joint_vel[250:400, ids], 0.02, axis=0)).max()),
        "thumb_root_max": {n: float(abs(q[:, i]).max()) for i, n in enumerate(b.joint_names) if "thumb_0" in n},
        "physics": "not_run; postrelease orientation blend is kinematic, not a new settling simulation",
    }
    (out / "smoothing.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
