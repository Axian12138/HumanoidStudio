"""Refine the unfinished right-arm seams while retaining the selected grasp trajectory.

This clip-specific continuation preserves root, legs, waist, left arm, fingers,
object tracks and window boundary poses. It does not certify physical support.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

ROBOT = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
    "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    import mujoco
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion", "order", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--position-tolerance", type=float, default=0.0002)
    parser.add_argument("--rotation-tolerance", type=float, default=0.002)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--arm-speed", type=float, default=5.9)
    parser.add_argument("--source-retreat", type=Path)
    parser.add_argument("--release-only", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=120)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    seed = MotionFields.load(args.motion, args.order)
    warm = MotionFields.load(args.warm_start, args.order) if args.warm_start else seed
    model, data = mujoco.MjModel.from_xml_path(str(ROBOT)), None
    data = mujoco.MjData(model)
    address = np.array([model.joint(n).qposadr[0] for n in seed.joint_names])
    cols = np.array(
        [
            i
            for i, n in enumerate(seed.joint_names)
            if n.startswith("right_") and any(s in n for s in ("shoulder_", "elbow_", "wrist_"))
        ]
    )
    palm = model.geom("right_hand_palm_collision").id
    pairs = [
        (model.geom("right_shoulder_yaw_collision").id, model.geom(f"torso_collision{i}").id) for i in (1, 2, 3)
    ] + [
        (model.geom("right_elbow_yaw_collision").id, model.geom("torso_collision2").id),
        (model.geom("right_elbow_yaw_collision").id, model.geom("right_wrist_collision").id),
    ]
    npose = 6 + len(pairs)
    q = seed.joint_pos.copy()
    reports = []
    windows = [(123.2, 125.6)] if args.release_only else [(120.0, 121.1), (123.2, 125.6)]
    for begin, end in windows:
        first, last = [round((t - 108) * seed.fps) for t in (begin, end)]
        indices = np.arange(first - 1, last + 2)
        baseline = seed.joint_pos[np.ix_(indices, cols)].astype(float)
        size, joints = len(indices) - 4, len(cols)
        targets_p, targets_r, roots, target_gaps = [], [], [], []
        for k in indices[2:-2]:
            data.qpos[:3] = seed.body_pos_w[k, 0]
            data.qpos[3:7] = seed.body_quat_w[k, 0]
            data.qpos[address] = seed.joint_pos[k]
            mujoco.mj_forward(model, data)
            target_gaps.append([mujoco.mj_geomDistance(model, data, g, h, 0.03, None) for g, h in pairs])
            roots.append(data.qpos.copy())
            targets_p.append(data.geom_xpos[palm].copy())
            targets_r.append(data.geom_xmat[palm].reshape(3, 3).copy())
        targets_p, targets_r = np.array(targets_p), np.array(targets_r)
        if args.source_retreat and begin == 123.2:
            from scipy.spatial.transform import Slerp

            with np.load(args.source_retreat) as source:
                hand = source["right_rigid_pos_w"]
                hand_rotation = Rotation.from_quat(source["right_rigid_quat_xyzw"]).as_matrix()
                valid = source["right_valid"]
            if not valid[first : last + 1].all():
                raise ValueError("Source wrist tracking is invalid during retreat")
            data.qpos[:3] = seed.body_pos_w[first, 0]
            data.qpos[3:7] = seed.body_quat_w[first, 0]
            data.qpos[address] = seed.joint_pos[first]
            mujoco.mj_forward(model, data)
            start_p = data.geom_xpos[palm].copy()
            start_r = data.geom_xmat[palm].reshape(3, 3).copy()
            times = 108 + indices[2:-2] / seed.fps
            # Use independently measured wrist rotation relative to release;
            # do not inherit the late orientation transition from the old IK.
            for j, k in enumerate(indices[2:-2]):
                u = np.clip((times[j] - 124.4) / (125.6 - 124.4), 0, 1)
                blend = u**3 * (10 - 15 * u + 6 * u * u)
                source_p = start_p + hand[k] - hand[first]
                targets_p[j] = (1 - blend) * source_p + blend * targets_p[j]
            for j, k in enumerate(indices[2:-2]):
                u = np.clip((times[j] - 124.4) / 1.2, 0, 1)
                blend = u**3 * (10 - 15 * u + 6 * u * u)
                source_r = hand_rotation[k] @ hand_rotation[first].T @ start_r
                targets_r[j] = Slerp([0, 1], Rotation.from_matrix([source_r, targets_r[j]]))([blend]).as_matrix()[0]
        held = 108 + indices[2:-2] / seed.fps <= 123.2
        pscale = np.where(held, args.position_tolerance, 0.004)
        rscale = np.where(held, args.rotation_tolerance, 0.025)

        def residual(
            x,
            baseline=baseline,
            size=size,
            joints=joints,
            roots=roots,
            targets_p=targets_p,
            targets_r=targets_r,
            pscale=pscale,
            rscale=rscale,
            target_gaps=target_gaps,
        ):
            full = baseline.copy()
            full[2:-2] = x.reshape(size, joints)
            pose = np.zeros((size, npose))
            for j in range(size):
                data.qpos[:] = roots[j]
                data.qpos[address[cols]] = full[j + 2]
                mujoco.mj_forward(model, data)
                pose[j, :3] = (data.geom_xpos[palm] - targets_p[j]) / pscale[j]
                pose[j, 3:6] = (
                    Rotation.from_matrix(data.geom_xmat[palm].reshape(3, 3) @ targets_r[j].T).as_rotvec() / rscale[j]
                )
                gaps = np.array([mujoco.mj_geomDistance(model, data, g, h, 0.03, None) for g, h in pairs])
                desired_gaps = np.minimum(np.array(target_gaps[j]), 0.003)
                pose[j, 6:] = np.maximum(desired_gaps - gaps, 0) / 0.00002
            velocity = np.diff(full, axis=0) * seed.fps
            accel = np.diff(full, n=2, axis=0) * seed.fps**2
            return np.r_[
                pose.ravel(),
                ((full[2:-2] - baseline[2:-2]) / 0.25).ravel(),
                (accel / 35.0).ravel(),
                (np.maximum(abs(velocity) - args.arm_speed, 0) / 0.02).ravel(),
            ]

        total = size * npose + size * joints + (size + 2) * joints + (size + 3) * joints
        sparsity = lil_matrix((total, size * joints), dtype=int)
        for j in range(size):
            sparsity[j * npose : (j + 1) * npose, j * joints : (j + 1) * joints] = 1
            for c in range(joints):
                col = j * joints + c
                sparsity[size * npose + col, col] = 1
                offset = size * (npose + joints)
                for d in (j, j + 1, j + 2):
                    sparsity[offset + d * joints + c, col] = 1
                offset += (size + 2) * joints
                for d in (j + 1, j + 2):
                    sparsity[offset + d * joints + c, col] = 1
        limits = np.array([model.joint(seed.joint_names[c]).range for c in cols])
        x0 = warm.joint_pos[np.ix_(indices[2:-2], cols)].astype(float).ravel()
        result = least_squares(
            residual,
            x0,
            jac_sparsity=sparsity.tocsr(),
            bounds=(np.tile(limits[:, 0], size), np.tile(limits[:, 1], size)),
            max_nfev=args.max_nfev,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-6,
        )
        q[np.ix_(indices[2:-2], cols)] = result.x.reshape(size, joints)
        final_residual = residual(result.x)[: size * npose].reshape(size, npose)
        after = baseline.copy()
        after[2:-2] = result.x.reshape(size, joints)
        row = {
            "window_s": [begin, end],
            "success": bool(result.success),
            "nfev": result.nfev,
            "cost": result.cost,
            "max_palm_delta_m": float(np.max(np.linalg.norm(final_residual[:, :3] * pscale[:, None], axis=1))),
            "max_palm_delta_rad": float(np.max(np.linalg.norm(final_residual[:, 3:6] * rscale[:, None], axis=1))),
            "max_joint_delta_rad": float(abs(after - baseline).max()),
            "peak_acceleration_before": float(abs(np.diff(baseline, n=2, axis=0) * seed.fps**2).max()),
            "peak_acceleration_after": float(abs(np.diff(after, n=2, axis=0) * seed.fps**2).max()),
            "peak_speed_after": float(abs(np.diff(after, axis=0) * seed.fps).max()),
        }
        reports.append(row)
        print(json.dumps(row), flush=True)
    refresh_fk(replace(seed, joint_pos=q), ROBOT).save(args.out / "motion.npz")
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "status": "candidate_pending_geometry_and_visual_review",
                "seed_sha256": sha256(args.motion),
                "windows": reports,
                "preserved": "All other joints, root poses, fingers, object tracks and outside-window poses.",
                "physics": "not_run",
                "source_retreat": str(args.source_retreat) if args.source_retreat else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
