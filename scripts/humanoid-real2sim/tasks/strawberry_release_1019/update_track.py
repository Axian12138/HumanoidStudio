"""Update the held object transform and check the modified release geometry."""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v8")
    parser.add_argument("--run")
    args = parser.parse_args()

    import json
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

    r = Path(args.run) if args.run else Path(Path("/tmp/berry_release_run").read_text())
    out = r / "retarget" / args.version
    old = Path(
        "outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800/retarget/candidates/08_table_strawberry/v25_room_v4"
    )
    b = MotionFields.load(old / "motion.npz", old / "motion_order.json")
    a = MotionFields.load(out / "motion.npz", out / "motion_order.json")
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
    errors = []
    for k in range(250, 351):
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = b.joint_pos[k]
        mujoco.mj_forward(m, d)
        p = d.geom_xpos[palm].copy()
        rot = d.geom_xmat[palm].reshape(3, 3).copy()
        offset = rot.T @ (origp[k] - p)
        d.qpos[adr] = a.joint_pos[k]
        mujoco.mj_forward(m, d)
        nr = d.geom_xmat[palm].reshape(3, 3).copy()
        berry["position_w"][k] = d.geom_xpos[palm] + nr @ offset
        berry["quaternion_xyzw"][k] = Rotation.from_matrix(nr @ rot.T @ origr[k]).as_quat()
        errors.append(float(np.linalg.norm(berry["position_w"][k] - origp[k])))
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
    report = {
        "held_center_max_change_m": max(errors),
        "release_center_change_m": float(np.linalg.norm(pd)),
        "physics": "not_run; postrelease orientation blend is kinematic, no claim to retain original settling validation",
    }
    (out / "track_update.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
