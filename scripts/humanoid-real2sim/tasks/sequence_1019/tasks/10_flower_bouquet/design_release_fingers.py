"""Generate a bounded finger opening path using actual pad vertices and flower hulls."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

    p = argparse.ArgumentParser()
    for name in ["run", "motion", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    motion = MotionFields.load(a.motion, order)
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    cols = [i for i, n in enumerate(motion.joint_names) if n.startswith("right_hand_")]
    faddr = addr[cols]
    k = 760
    data.qpos[:3] = motion.body_pos_w[k, 0]
    data.qpos[3:7] = motion.body_quat_w[k, 0]
    data.qpos[addr] = motion.joint_pos[k]
    mujoco.mj_forward(model, data)
    tr = np.load(a.run / "retarget/hoi/10_flower_bouquet/tracks_v8/flower_bouquet.npz")
    rot = Rotation.from_quat(tr["quaternion_xyzw"][k]).as_matrix()
    pos = tr["position_w"][k]
    assets = a.run / "scene/10_flower_bouquet/registered_assets_paper_wrapper_v1"
    hulls = [
        np.unique(np.round(ConvexHull(trimesh.load(path, force="mesh", process=False).vertices).equations, 10), axis=0)
        for path in sorted(assets.glob("flower_bouquet_hull*.obj"))
    ]
    points = {}
    targets = {}
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if not name.startswith("right_hand_") or not name.endswith("_collision"):
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mi = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mi] : model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]
        _, ix = np.unique(np.round(v / 0.003), axis=0, return_index=True)
        points[g] = v[ix]
        if any(part in name for part in ["thumb_2", "index_1", "middle_1"]):
            local = (v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - pos) @ rot
            signed = np.max(local @ hulls[0][:, :3].T + hulls[0][:, 3], axis=1)
            ix = np.argmin(abs(signed))
            normal = local[ix].copy()
            normal[2] = 0
            normal /= np.linalg.norm(normal)
            targets[g] = (v[ix].copy(), local[ix].copy(), normal)
    lo = np.array([model.joint(motion.joint_names[i]).range[0] + 1e-7 for i in cols])
    hi = np.array([model.joint(motion.joint_names[i]).range[1] - 1e-7 for i in cols])
    previous = motion.joint_pos[k, cols].astype(float)
    sequence = [previous.copy()]
    records = []
    for step in range(1, 31):
        u = step / 30
        blend = u * u * (3 - 2 * u)

        def residual(x, blend=blend, previous=previous):
            data.qpos[faddr] = x
            mujoco.mj_forward(model, data)
            values = []
            for g, (v, start, normal) in targets.items():
                local = (v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - pos) @ rot
                values.extend(600 * (local - start - 0.025 * blend * normal))
            for g, v in points.items():
                local = (v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - pos) @ rot
                dist = np.min([np.max(local @ e[:, :3].T + e[:, 3], axis=1) for e in hulls], axis=0)
                values.extend(2000 * np.maximum(0.0002 - dist, 0))
            values.extend(0.01 * x)
            values.extend(0.03 * (x - previous))
            return np.asarray(values)

        lower = np.maximum(lo, previous - 2.95 / 50)
        upper = np.minimum(hi, previous + 2.95 / 50)
        res = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper), max_nfev=60)
        previous = res.x.copy()
        sequence.append(previous.copy())
        records.append({"source_s": 123.2 + step / 50, "cost": float(np.linalg.norm(res.fun))})
    np.savez(
        a.out / "finger_path.npz",
        joint_names=np.array([motion.joint_names[i] for i in cols]),
        joint_pos=np.array(sequence),
        source_time_s=123.2 + np.arange(31) / 50,
        fps=50,
    )
    (a.out / "report.json").write_text(
        json.dumps(
            {
                "status": "candidate only; full geometry and transition audit required",
                "fixed_palm_source_s": 123.2,
                "pad_radial_target_m": 0.025,
                "records": records,
                "start": sequence[0].tolist(),
                "end": sequence[-1].tolist(),
            },
            indent=2,
        )
    )
    print(a.out)


if __name__ == "__main__":
    main()
