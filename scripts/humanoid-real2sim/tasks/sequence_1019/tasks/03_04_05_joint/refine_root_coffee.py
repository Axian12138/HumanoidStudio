"""Local root translation with both feet and wrists preserved in world coordinates.

Diagnostic candidate only: requires independent full room, self, speed and source review.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", default="03_04_05_joint_v50")
    p.add_argument("--version", default="root_v1")
    a = p.parse_args()
    src = a.run / "retarget/candidates" / a.candidate
    out = a.run / "repairs/04_right_stance_coffee" / a.version
    out.mkdir(parents=True, exist_ok=False)
    motion = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    mesh = trimesh.load_mesh(a.run / "scene/03_04_05/coffee_top_raw.obj")
    hull = ConvexHull(mesh.vertices)
    assert mesh.is_watertight and abs(abs(mesh.volume) - hull.volume) < 1e-5
    eq = np.unique(np.round(hull.equations, 10), axis=0)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    addr = np.array([m.joint(n).qposadr[0] for n in motion.joint_names])
    legnames = [
        s + "_" + n + "_joint"
        for s in ["left", "right"]
        for n in ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
    ]
    legcols = np.array([motion.joint_names.index(n) for n in legnames])
    armnames = [
        [
            s + "_" + n + "_joint"
            for n in [
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            ]
        ]
        for s in ["left", "right"]
    ]
    armcols = [np.array([motion.joint_names.index(n) for n in names]) for names in armnames]
    feet = [m.body(s + "_ankle_roll_link").id for s in ["left", "right"]]
    wrists = [m.body(s + "_wrist_yaw_link").id for s in ["left", "right"]]
    geoms = []
    for g in range(m.ngeom):
        name = m.body(m.geom_bodyid[g]).name
        if (
            m.geom_contype[g]
            and any(name.startswith(s + "_") for s in ["left", "right"])
            and any(n in name for n in ["hip", "knee", "ankle"])
        ):
            assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            local = np.zeros((31, 3))
            local[:, 2] = np.linspace(-m.geom_size[g, 1], m.geom_size[g, 1], 31)
            geoms.append((g, local, m.geom_size[g, 0]))
    time = 27 + np.arange(len(motion.joint_pos)) / 50
    rows = np.flatnonzero((time >= 42.3) & (time <= 43.5))
    allq = motion.joint_pos.copy()
    positions = motion.body_pos_w.copy()
    limits = np.array([m.joint(n).range for n in legnames])
    previous = np.zeros(15)
    reports = []
    for k in rows[1:-1]:
        prior = motion.joint_pos[k].astype(float)
        root = motion.body_pos_w[k, 0].astype(float)
        d.qpos[:3] = root
        d.qpos[3:7] = motion.body_quat_w[k, 0]
        d.qpos[addr] = prior
        mujoco.mj_forward(m, d)
        targets = [(d.xpos[b].copy(), Rotation.from_matrix(d.xmat[b].reshape(3, 3))) for b in feet + wrists]
        fade = min((time[k] - 42.3) / 0.2, (43.5 - time[k]) / 0.2, 1)
        fade = max(fade, 0)
        fade = fade * fade * (3 - 2 * fade)
        lower = np.r_[-np.ones(3) * 0.06 * fade, limits[:, 0] - prior[legcols]]
        upper = np.r_[np.ones(3) * 0.06 * fade, limits[:, 1] - prior[legcols]]

        def residual(x, root=root, prior=prior, targets=targets, previous=previous):
            d.qpos[:3] = root + x[:3]
            d.qpos[addr[legcols]] = prior[legcols] + x[3:]
            mujoco.mj_forward(m, d)
            terms = []
            for b, (pos, rot) in zip(feet, targets[:2], strict=True):
                terms.extend((d.xpos[b] - pos) * 20000)
                terms.extend((Rotation.from_matrix(d.xmat[b].reshape(3, 3)) * rot.inv()).as_rotvec() * 5000)
            for g, local, radius in geoms:
                world = local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                separation = np.max(world @ eq[:, :3].T + eq[:, 3], axis=1)
                terms.extend(np.maximum(radius + 0.002 - separation, 0) * 5000)
                terms.append(max(0, radius - world[:, 2].min()) * 1000)
            terms.extend(x[:3] * 30)
            terms.extend(x[3:] * 0.5)
            terms.extend((x - previous) * 2)
            return np.array(terms)

        fit = least_squares(
            residual,
            np.clip(previous, lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper),
            max_nfev=100,
            ftol=1e-8,
            xtol=1e-8,
        )
        previous = fit.x.copy()
        residual(fit.x)
        foot_errors = [
            [
                float(np.linalg.norm(d.xpos[b] - pos)),
                float((Rotation.from_matrix(d.xmat[b].reshape(3, 3)) * rot.inv()).magnitude()),
            ]
            for b, (pos, rot) in zip(feet, targets[:2], strict=True)
        ]
        allq[k, legcols] = prior[legcols] + fit.x[3:]
        positions[k, 0] = root + fit.x[:3]
        arm_errors = []
        for cols, b, (pos, rot) in zip(armcols, wrists, targets[2:], strict=True):
            bounds = np.array([m.joint(motion.joint_names[c]).range for c in cols])

            def arm_residual(q, cols=cols, b=b, pos=pos, rot=rot, prior=prior):
                d.qpos[addr[cols]] = q
                mujoco.mj_forward(m, d)
                return np.r_[
                    (d.xpos[b] - pos) * 5000,
                    (Rotation.from_matrix(d.xmat[b].reshape(3, 3)) * rot.inv()).as_rotvec() * 1000,
                    (q - prior[cols]) * 0.05,
                ]

            armfit = least_squares(
                arm_residual,
                np.clip(prior[cols], bounds[:, 0] + 1e-8, bounds[:, 1] - 1e-8),
                bounds=(bounds[:, 0], bounds[:, 1]),
                max_nfev=80,
            )
            allq[k, cols] = armfit.x
            arm_residual(armfit.x)
            arm_errors.append(
                [
                    float(np.linalg.norm(d.xpos[b] - pos)),
                    float((Rotation.from_matrix(d.xmat[b].reshape(3, 3)) * rot.inv()).magnitude()),
                ]
            )
        margins = []
        for g, local, radius in geoms:
            world = local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            margins.extend(np.maximum(radius + 0.002 - np.max(world @ eq[:, :3].T + eq[:, 3], axis=1), 0))
        row = {
            "source_s": float(time[k]),
            "root_delta_m": fit.x[:3].tolist(),
            "feet_errors_m_rad": foot_errors,
            "wrist_errors_m_rad": arm_errors,
            "coffee_margin_violation_m": float(max(margins)),
        }
        reports.append(row)
        if k % 10 == 0:
            print(row, flush=True)
    result = refresh_fk(replace(motion, joint_pos=allq, body_pos_w=positions), MODEL)
    result.save(out / "motion.npz")
    np.savez_compressed(out / "overlay.npz", rows=rows, joint_pos=allq[rows], root_position_w=positions[rows, 0])
    (out / "review.json").write_text(
        json.dumps(
            {
                "status": "diagnostic_pending_full_geometry_source_speed_and_support_checks",
                "base": str(src),
                "root_rotation_unchanged": True,
                "object_tracks_unchanged": True,
                "window_s": [42.3, 43.5],
                "max_root_delta_m": float(np.linalg.norm(positions[:, 0] - motion.body_pos_w[:, 0], axis=1).max()),
                "max_joint_delta_rad": float(abs(allq - motion.joint_pos).max()),
                "samples": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
