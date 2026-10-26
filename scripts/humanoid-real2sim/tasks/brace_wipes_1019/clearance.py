"""Refine fist support on the packet/frame and lift before withdrawing past the lid."""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--source", default="v1")
    parser.add_argument("--version", default="v2")
    parser.add_argument("--lift", type=float, default=0.035)
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

    r = Path(args.run) if args.run else Path(Path("/tmp/wipes_run").read_text())
    src = r / "retarget" / args.source
    out = r / "retarget" / args.version
    shutil.copytree(src, out)
    b = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = b.joint_pos.astype(float).copy()
    cfg = json.loads((r / "scene/config.json").read_text())
    objR = Rotation.from_quat(cfg["quaternion_xyzw"]).as_matrix()
    objp = np.array(cfg["position_w"])
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
            samples.append((g, m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()))
    lo = np.array([m.joint(b.joint_names[i]).range[0] for i in cols]) + 1e-6
    hi = np.array([m.joint(b.joint_names[i]).range[1] for i in cols]) - 1e-6
    changes = []
    for k in range(130, 225):
        t = 83.5 + k / 50
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = q[k]
        mujoco.mj_forward(m, d)
        p = d.geom_xpos[palm].copy()
        R = d.geom_xmat[palm].reshape(3, 3).copy()
        v = np.concatenate([x @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, x in samples])
        local = (v - objp) @ objR
        needed = 0.0
        for part in cfg["parts"][:5]:
            bounds = np.array(part["bounds"])
            inside = np.all((local[:, :2] >= bounds[0, :2]) & (local[:, :2] <= bounds[1, :2]), axis=1)
            if inside.any():
                needed = max(needed, float(bounds[1, 2] + 0.0004 - local[inside, 2].min()))
        # A compact smooth upward detour after the press protects the open lid.
        u = np.clip((t - 87.12) / 0.8, 0, 1)
        lift = args.lift * np.sin(np.pi * u) ** 2
        dz = max(needed, lift)
        target = p + np.array([0, 0, dz])
        original = q[k, cols].copy()

        def residual(x, target=target, R=R, original=original):
            d.qpos[adr[cols]] = x
            mujoco.mj_forward(m, d)
            rot = d.geom_xmat[palm].reshape(3, 3)
            return np.r_[
                200 * (d.geom_xpos[palm] - target),
                5 * Rotation.from_matrix(rot @ R.T).as_rotvec(),
                0.2 * (x - original),
            ]

        fit = least_squares(residual, np.clip(original, lo, hi), bounds=(lo, hi), max_nfev=30)
        q[k, cols] = fit.x
        changes.append([t, dz])
    refresh_fk(replace(b, joint_pos=q), model).save(out / "motion.npz")
    (out / "clearance.json").write_text(
        json.dumps(
            {
                "height_changes": changes,
                "scope": "support on actual packet/frame envelope and retract vertically before passing lid",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
