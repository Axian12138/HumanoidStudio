"""Join the accepted chair motion and raise the tabletop entry before landing."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

ST = Path("/home/galbot/workspace/ScaleTrack")
MODEL = ST / "source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "fps")


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (10 + x * (-15 + 6 * x))


def main():
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation, Slerp

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="v1")
    a = p.parse_args()
    r = a.run.resolve()
    src = ST / "data/1019test/brace-dining-table-kinematic-v1"
    chair = ST / "data/1019test/push-armchair-thumb-kinematic-v1"
    base = r / "retarget/baseline"
    base.mkdir(parents=True, exist_ok=True)
    out = r / "retarget" / a.version
    out.mkdir(exist_ok=False)
    for folder in ["review", "logs", "scene"]:
        (r / folder).mkdir(exist_ok=True)
    z = np.load(src / "motion.npz")
    np.savez_compressed(base / "motion.npz", **{k: z[k] for k in FIELDS})
    shutil.copy2(src / "metadata/motion_order.json", base / "motion_order.json")
    b = MotionFields.load(base / "motion.npz", base / "motion_order.json")
    ch = np.load(chair / "motion.npz")
    n = len(b.joint_pos)
    times = 83.5 + np.arange(n) / 50
    q = b.joint_pos.astype(float).copy()
    end = 80
    startq = ch["joint_pos"][425].astype(float)
    v0 = (ch["joint_pos"][426] - ch["joint_pos"][424]) * 25
    # Cubic endpoint interpolation for torso and the non-active arm; roots and legs
    # are already identical at the seam and remain untouched.
    upper = [
        i
        for i, name in enumerate(b.joint_names)
        if name.startswith(("waist", "left_", "right_")) and not any(s in name for s in ["hip", "knee", "ankle"])
    ]
    duration = end / 50
    u = np.arange(end + 1) / end
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    vend = (q[end + 1] - q[end - 1]) * 25
    for j in upper:
        q[: end + 1, j] = h00 * startq[j] + h10 * duration * v0[j] + h01 * q[end, j] + h11 * duration * vend[j]
    # Explicit hand-state schedule, not free optimization variables.
    fingers = [i for i, name in enumerate(b.joint_names) if "_hand_" in name]
    f = smooth((times[: end + 1] - 83.55) / 1.1)
    q[: end + 1, fingers] = startq[fingers] + f[:, None] * (b.joint_pos[end, fingers] - startq[fingers])
    thumbs = [b.joint_names.index(s + "_hand_thumb_0_joint") for s in ["left", "right"]]
    q[:, thumbs] = 0
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    adr = np.array([m.joint(name).qposadr[0] for name in b.joint_names])
    cols = [
        b.joint_names.index("right_" + s + "_joint")
        for s in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    w = m.body("right_wrist_yaw_link").id

    def pose(k, x):
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = q[k]
        d.qpos[adr[cols]] = x
        mujoco.mj_kinematics(m, d)

    pose(0, startq[cols])
    p0 = d.xpos[w].copy()
    rot0 = Rotation.from_matrix(d.xmat[w].reshape(3, 3))
    pose(end, b.joint_pos[end, cols])
    p1 = d.xpos[w].copy()
    rot1 = Rotation.from_matrix(d.xmat[w].reshape(3, 3))
    waytimes = np.array([83.5, 83.56, 83.92, 84.30, 84.62, 85.1])
    wp = np.array(
        [
            p0,
            ch["body_pos_w"][428, b.body_names.index("right_wrist_yaw_link")],
            [-0.62, -1.86, 1.01],
            [-0.43, -2.08, 1.08],
            [-0.28, -2.25, 1.02],
            p1,
        ]
    )
    vel0 = (
        ch["body_pos_w"][426, b.body_names.index("right_wrist_yaw_link")]
        - ch["body_pos_w"][424, b.body_names.index("right_wrist_yaw_link")]
    ) * 25
    vel1 = (
        b.body_pos_w[end + 1, b.body_names.index("right_wrist_yaw_link")]
        - b.body_pos_w[end - 1, b.body_names.index("right_wrist_yaw_link")]
    ) * 25
    targets = CubicSpline(waytimes, wp, bc_type=((1, vel0), (1, vel1)))(times[: end + 1])
    ori = Slerp([0, 1], Rotation.from_quat([rot0.as_quat(), rot1.as_quat()]))(smooth((times[: end + 1] - 83.56) / 0.9))
    geoms = []
    for g in range(m.ngeom):
        name = m.body(m.geom_bodyid[g]).name
        if (
            (
                name.startswith("right_hand")
                or name
                in ["right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link", "right_elbow_link"]
            )
            and not m.geom_contype[g]
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            mi = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mi] : m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
            _, keep = np.unique(np.round(v / 0.004), axis=0, return_index=True)
            geoms.append((g, v[keep]))
    lo = np.array([m.joint(b.joint_names[j]).range[0] + 1e-6 for j in cols])
    hi = np.array([m.joint(b.joint_names[j]).range[1] - 1e-6 for j in cols])
    records = []
    previous = startq[cols].copy()
    for k in range(end + 1):
        reference = q[k, cols].copy()
        target = targets[k]
        desired = ori[k]
        margin = 0.15 * float(smooth((times[k] - 83.8) / 0.3)) * (1 - float(smooth((times[k] - 84.62) / 0.48))) + 0.004

        def residual(x, k=k, margin=margin, target=target, desired=desired, reference=reference, previous=previous):
            pose(k, x)
            pen = []
            for g, v in geoms:
                world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                mask = (world[:, 0] >= -1.05) & (world[:, 0] <= 0.59) & (world[:, 1] >= -2.99) & (world[:, 1] <= -2.099)
                bottom = world[mask, 2].min() if mask.any() else 10.0
                pen.append(max(0.754 + margin - bottom, 0))
            return np.r_[
                (d.xpos[w] - target) * 100,
                (Rotation.from_matrix(d.xmat[w].reshape(3, 3)) * desired.inv()).as_rotvec() * 2.5,
                (x - reference) * 0.25,
                (x - previous) * 0.5,
                np.array(pen) * 350,
            ]

        fit = least_squares(residual, np.clip(previous, lo, hi), bounds=(lo, hi), max_nfev=120)
        if k == 0:
            fit.x = startq[cols]
        if k == end:
            fit.x = b.joint_pos[end, cols]
        q[k, cols] = fit.x
        previous = fit.x.copy()
        pose(k, fit.x)
        records.append(
            {
                "source_s": float(times[k]),
                "target_wrist": target.tolist(),
                "wrist": d.xpos[w].tolist(),
                "position_error_m": float(np.linalg.norm(d.xpos[w] - target)),
            }
        )
    result = refresh_fk(replace(b, joint_pos=q), MODEL)
    result.save(out / "motion.npz")
    for name in ["human.npz", "human_motion_world.npz", "source_sync.npz", "motion_order.json"]:
        shutil.copy2(src / "metadata" / name, out / name)
    track = np.load(src / "metadata/object_trajectories/grey_armchair.npz")
    np.savez_compressed(
        out / "chair_track.npz",
        position_w=track["position_w"],
        quaternion_xyzw=track["quaternion_wxyz"][:, [1, 2, 3, 0]],
        fps=50,
    )
    (out / "review.json").write_text(
        json.dumps(
            {
                "status": "candidate_pending_checks",
                "chair_source": str(chair),
                "brace_source": str(src),
                "source_window_s": [83.5, 90.98],
                "refined_window_s": [83.5, 85.1],
                "table_z_m": 0.754,
                "table_near_edge_y_m": -2.11905405,
                "planned_clearance_m": 0.15,
                "records": records,
            },
            indent=2,
        )
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()
