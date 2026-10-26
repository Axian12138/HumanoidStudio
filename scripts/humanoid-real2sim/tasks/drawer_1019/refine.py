"""Jointly smooth the right-arm window with explicit hand surface clearance."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    import mujoco
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--max-nfev", type=int, default=25)
    a = p.parse_args()
    r = a.run.resolve()
    src = r / "retarget/baseline"
    out = r / "retarget" / a.version
    out.mkdir(exist_ok=False)
    cfg = json.loads((r / "scene/config.json").read_text())
    m = mujoco.MjModel.from_xml_path(cfg["robot_and_early_props"])
    d = mujoco.MjData(m)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = motion.joint_pos.astype(float).copy()
    n = len(q)
    t = 49.5 + np.arange(n) / 50
    cols = [
        motion.joint_names.index("right_" + name + "_joint")
        for name in [
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
    ]
    addr = np.array([m.joint(name).qposadr[0] for name in motion.joint_names])
    wrist = m.body("right_wrist_yaw_link").id
    q[:, motion.joint_names.index("right_hand_thumb_0_joint")] = 0
    slide = np.load(src / "scene_tracks/drawer_slide.npz")["position"]
    da = m.joint("drawer_slide").qposadr[0]
    boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("drawer").id and m.geom_contype[g]]
    assert all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in boxes)
    hands = []
    for g in range(m.ngeom):
        name = m.body(int(m.geom_bodyid[g])).name
        if (
            not (name.startswith("right_hand") or name == "right_wrist_yaw_link")
            or m.geom_contype[g]
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mi = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mi] : m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
        _, keep = np.unique(np.round(v / 0.004), axis=0, return_index=True)
        hands.append((g, v[keep]))
    print("surface_groups", len(hands), "samples", sum(len(v) for _, v in hands), flush=True)
    target_p = []
    target_r = []
    boxp = []
    boxr = []

    def set_frame(k, x):
        d.qpos[:3] = motion.body_pos_w[k, 0]
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = q[k]
        d.qpos[addr[cols]] = x
        d.qpos[da] = slide[k]
        mujoco.mj_kinematics(m, d)

    for k in range(n):
        set_frame(k, q[k, cols])
        target_p.append(d.xpos[wrist].copy())
        target_r.append(d.xmat[wrist].reshape(3, 3).copy())
        boxp.append(d.geom_xpos[boxes].copy())
        boxr.append(d.geom_xmat[boxes].reshape(-1, 3, 3).copy())
    target_p = np.array(target_p)
    target_r = np.array(target_r)
    boxp = np.array(boxp)
    boxr = np.array(boxr)
    ref = gaussian_filter1d(q[:, cols], sigma=2, axis=0, mode="nearest")
    fade = np.clip((t - 49.8) / 0.6, 0, 1) * np.clip((54.9 - t) / 0.6, 0, 1)
    fade = fade**2 * (3 - 2 * fade)
    ref = q[:, cols] + fade[:, None] * (ref - q[:, cols])
    # Contact policy is fixed; the optimizer only changes the seven arm joints.
    contact = np.minimum(np.clip((t - 50.95) / 0.15, 0, 1), np.clip((54.22 - t) / 0.12, 0, 1))
    bsize = m.geom_size[boxes]
    ng = len(hands)
    nf = 7 + 3 + 3 + ng + 1

    def residual(flat):
        x = flat.reshape(n, 7)
        parts = []
        for k in range(n):
            set_frame(k, x[k])
            dist = []
            for g, v in hands:
                world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                local = np.einsum("bvi,bij->bvj", world[None] - boxp[k, :, None, :], boxr[k])
                delta = np.abs(local) - bsize[:, None, :]
                sdf = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(axis=-1), 0)
                dist.append(sdf.min())
            dist = np.array(dist)
            prior_weight = 0.8 if 50.1 < t[k] < 54.6 else 30.0
            parts.extend(
                [
                    (x[k] - ref[k]) * prior_weight,
                    (d.xpos[wrist] - target_p[k]) * 30,
                    Rotation.from_matrix(d.xmat[wrist].reshape(3, 3) @ target_r[k].T).as_rotvec() * 1.2,
                    np.minimum(dist - 0.0003, 0) * 800,
                    np.array([max(dist.min() - 0.001, 0) * 250 * contact[k]]),
                ]
            )
        parts.extend([np.diff(x, axis=0).ravel() * 2, np.diff(x, n=2, axis=0).ravel() * 45])
        return np.concatenate(parts)

    shape = n * nf + (n - 1) * 7 + (n - 2) * 7
    sparsity = lil_matrix((shape, n * 7), dtype=int)
    for k in range(n):
        sparsity[k * nf : (k + 1) * nf, k * 7 : (k + 1) * 7] = 1
    base = n * nf
    for k in range(n - 1):
        for j in range(7):
            sparsity[base + k * 7 + j, [k * 7 + j, (k + 1) * 7 + j]] = 1
    base += (n - 1) * 7
    for k in range(n - 2):
        for j in range(7):
            sparsity[base + k * 7 + j, [k * 7 + j, (k + 1) * 7 + j, (k + 2) * 7 + j]] = 1
    limits = np.array([m.joint(motion.joint_names[j]).range for j in cols])
    lo = np.tile(limits[:, 0] + 1e-6, n)
    hi = np.tile(limits[:, 1] - 1e-6, n)
    fit = least_squares(
        residual,
        np.clip(ref.ravel(), lo, hi),
        bounds=(lo, hi),
        jac_sparsity=sparsity.tocsr(),
        max_nfev=a.max_nfev,
        verbose=2,
        ftol=1e-5,
    )
    q[:, cols] = fit.x.reshape(n, 7)
    # Exact outer states protect splicing; the whole window is checked after this assignment.
    q[[0, -1], :] = motion.joint_pos[[0, -1], :]
    q[:, motion.joint_names.index("right_hand_thumb_0_joint")] = 0
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(motion, joint_pos=q), model)
    result.save(out / "motion.npz")
    for name in ["human_motion_world.npz", "source_sync.npz", "motion_order.json"]:
        shutil.copy2(src / name, out / name)
    shutil.copytree(src / "scene_tracks", out / "scene_tracks")
    report = {
        "status": "candidate_pending_independent_checks",
        "cost": fit.cost,
        "evaluations": fit.nfev,
        "modified_joints": [motion.joint_names[i] for i in cols] + ["right_hand_thumb_0_joint"],
        "right_arm_peak_acceleration_before_rad_s2": float(
            np.abs(np.diff(motion.joint_pos[:, cols], n=2, axis=0) * 2500).max()
        ),
        "right_arm_peak_acceleration_after_rad_s2": float(np.abs(np.diff(q[:, cols], n=2, axis=0) * 2500).max()),
        "contact_window_s": [51.1, 54.1],
        "source_geometry": "unchanged registered drawer and room",
        "dynamics": "not_run",
        "right_thumb_0_exact_zero": True,
    }
    (out / "review.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
