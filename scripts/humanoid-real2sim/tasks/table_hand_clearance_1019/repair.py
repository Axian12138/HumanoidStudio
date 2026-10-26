"""Local arm-only tabletop clearance repair; preserve all other NPZ members."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256
from robosimstudio.humanoid_real2sim.track.launcher import DEFAULT_ROBOT


def main():
    import mujoco
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--freeze-first", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fields = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "fps"]
    source = a.source or a.package / "motion.npz"
    with np.load(source) as z:
        values = {k: z[k] for k in fields}
    order = json.loads((a.package / "metadata/motion_order.json").read_text())
    b = MotionFields(
        joint_names=tuple(order["robot_joint_names"]), body_names=tuple(order["robot_body_names"]), **values
    )
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_ROBOT))
    d = mujoco.MjData(m)
    adr = np.array([m.joint(n).qposadr[0] for n in b.joint_names])
    q = b.joint_pos.astype(float).copy()
    # Authored MORBYLANGA tabletop envelope in current room USD, world metres.
    low = np.array([-0.99, -2.800686, 0.713])
    high = np.array([0.610005, -1.949053, 0.753003])
    center, half = (low + high) / 2, (high - low) / 2
    samples = {}
    dense = {}
    for side in ["left", "right"]:
        samples[side], dense[side] = [], []
        for g in range(m.ngeom):
            name = m.geom(g).name
            if not (name.startswith(side + "_hand") and name.endswith("_collision")):
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
                hull = ConvexHull(v)
                small = v[hull.vertices]
                # Triangle centroids and edge midpoints catch faces crossing the slab.
                triangles = v[hull.simplices]
                small = np.concatenate([small, triangles.mean(1), (triangles[:, 0] + triangles[:, 1]) / 2])
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                grid = np.array(np.meshgrid(*[[-1.0, 0.0, 1.0]] * 3)).reshape(3, -1).T
                v = small = grid * m.geom_size[g]
            else:
                continue
            samples[side].append((g, small))
            dense[side].append((g, v))

    def pose(k, joints):
        d.qpos[:3] = b.body_pos_w[k, 0]
        d.qpos[3:7] = b.body_quat_w[k, 0]
        d.qpos[adr] = joints
        mujoco.mj_forward(m, d)

    def points(side, full=False):
        return np.concatenate(
            [v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, v in (dense if full else samples)[side]]
        )

    def sdf(v):
        delta = abs(v - center) - half
        return np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(1), 0)

    report = {
        "source_sha256": sha256(source),
        "tabletop_bounds": [low.tolist(), high.tolist()],
        "sides": {},
    }
    for side in ["left", "right"]:
        ids = np.array(
            [
                i
                for i, n in enumerate(b.joint_names)
                if n.startswith(side + "_") and any(s in n for s in ["shoulder_", "elbow_", "wrist_"])
            ]
        )
        lo = np.array([m.joint(b.joint_names[i]).range[0] for i in ids]) + 1e-6
        hi = np.array([m.joint(b.joint_names[i]).range[1] for i in ids]) - 1e-6
        distances = []
        for k in range(len(q)):
            pose(k, b.joint_pos[k])
            distances.append(sdf(points(side, True)).min())
        active = maximum_filter1d(np.array(distances) < 0.003, size=21)
        if a.freeze_first:
            active[0] = False
        indices = np.flatnonzero(active)
        print(a.package.name, side, "penetration", min(distances), "active", len(indices), flush=True)
        for iteration in range(3):
            prior = q[:, ids].copy()
            if iteration:
                correction = gaussian_filter1d(q[:, ids] - b.joint_pos[:, ids], 1.5, axis=0)
                prior = b.joint_pos[:, ids] + correction
            for k in indices:
                pose(k, b.joint_pos[k])
                wrist = m.body(side + "_wrist_yaw_link").id
                baseR = d.xmat[wrist].reshape(3, 3).copy()

                def residual(x, k=k, ids=ids, side=side, prior=prior, wrist=wrist, baseR=baseR):
                    trial = q[k].copy()
                    trial[ids] = x
                    pose(k, trial)
                    clear = np.maximum(0.008 - sdf(points(side)), 0)
                    return np.r_[
                        1000 * clear,
                        0.5 * (x - b.joint_pos[k, ids]),
                        2 * (x - prior[k]),
                        0.3 * Rotation.from_matrix(d.xmat[wrist].reshape(3, 3) @ baseR.T).as_rotvec(),
                    ]

                fit = least_squares(residual, np.clip(prior[k], lo, hi), bounds=(lo, hi), max_nfev=35, ftol=1e-7)
                q[k, ids] = fit.x
        report["sides"][side] = {
            "original_min_signed_distance_m": float(min(distances)),
            "changed_frames": indices.tolist(),
        }
    result = refresh_fk(MotionFields(**{**b.__dict__, "joint_pos": q}), DEFAULT_ROBOT)
    # Preserve all unaffected rows, joints, and body fields exactly, including their velocities.
    changed = np.any(abs(q - b.joint_pos) > 1e-7, axis=1)
    affected = maximum_filter1d(changed, size=5)
    replacements = {}
    armcols = [i for i, n in enumerate(b.joint_names) if any(s in n for s in ["shoulder_", "elbow_", "wrist_"])]
    armbodies = [
        i for i, n in enumerate(b.body_names) if any(s in n for s in ["shoulder_", "elbow_", "wrist_", "hand_"])
    ]
    for key in fields:
        value = values[key].copy()
        if key in ["joint_pos", "joint_vel"]:
            value[np.ix_(affected, armcols)] = getattr(result, key)[np.ix_(affected, armcols)]
        elif key.startswith("body_"):
            value[np.ix_(affected, armbodies)] = getattr(result, key)[np.ix_(affected, armbodies)]
        replacements[key] = value
    np.savez_compressed(a.out / "core.npz", **replacements)
    checks = []
    for tick in range(2 * len(q) - 1):
        k = tick // 2
        j = min(k + 1, len(q) - 1)
        u = (tick % 2) / 2
        pose(k, (1 - u) * q[k] + u * q[j])
        d.qpos[:3] = (1 - u) * b.body_pos_w[k, 0] + u * b.body_pos_w[j, 0]
        qt = (1 - u) * b.body_quat_w[k, 0] + u * b.body_quat_w[j, 0]
        d.qpos[3:7] = qt / np.linalg.norm(qt)
        mujoco.mj_forward(m, d)
        checks.append([float(sdf(points(side, True)).min()) for side in ["left", "right"]])
    check = np.array(checks)
    report.update(
        minimum_distance_100hz_m=check.min(0).tolist(),
        changed_frames=np.flatnonzero(changed).tolist(),
        max_joint_change_rad=float(abs(q - b.joint_pos).max()),
        max_joint_velocity_rad_s=float(abs(replacements["joint_vel"][:, armcols]).max()),
        visual_fields_stale=True,
    )
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    if check.min() < -0.0005:
        raise RuntimeError(f"clearance failed: {check.min()}")
    if a.core_only:
        print("COMPLETE CORE", a.out, flush=True)
        return
    # Stream other members so multi-GB visual arrays remain byte-identical without a RAM spike.
    with (
        zipfile.ZipFile(a.package / "motion.npz") as src,
        zipfile.ZipFile(a.out / "motion.npz", "w", compression=zipfile.ZIP_DEFLATED) as dst,
    ):
        for name in src.namelist():
            key = name.removesuffix(".npy")
            if key in replacements:
                buf = io.BytesIO()
                np.save(buf, replacements[key], allow_pickle=False)
                dst.writestr(name, buf.getvalue())
            else:
                with src.open(name) as r, dst.open(name, "w", force_zip64=True) as w:
                    import shutil

                    shutil.copyfileobj(r, w, 1024 * 1024)
    print("COMPLETE", a.out, report["minimum_distance_100hz_m"], flush=True)


if __name__ == "__main__":
    main()
