"""Design one fixed three-digit closure against the real strawberry geometry."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields


def main():
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    folder = args.candidate
    prior = MotionFields.load(folder / "motion.npz", folder / "motion_order.json")
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    data = mujoco.MjData(model)
    addresses = np.array([model.joint(n).qposadr[0] for n in prior.joint_names])
    arm = [
        prior.joint_names.index(f"right_{p}_joint")
        for p in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
    hand = [i for i, n in enumerate(prior.joint_names) if n.startswith("right_hand")]
    cols = np.array(arm + hand)
    data.qpos[:3], data.qpos[3:7] = prior.body_pos_w[120, 0], prior.body_quat_w[120, 0]
    data.qpos[addresses] = prior.joint_pos[120]
    berry_track = np.load(folder / "table_strawberry.npz")
    berry = berry_track["position_w"][0]
    berry_rotation = Rotation.from_quat(berry_track["quaternion_xyzw"][0]).as_matrix()
    geometry = (
        json.loads((folder / "geometry_config.json").read_text()) if (folder / "geometry_config.json").exists() else {}
    )
    room_transform = np.array(geometry.get("T_room_correction", np.eye(4)))
    mesh = trimesh.load_mesh(
        "/home/galbot/workspace/ScaleTrack/data/1012test/hoi_stage2/scene/assets/object_strawberry_existing_0_0.obj"
    )
    hull = ConvexHull(mesh.vertices)
    planes = hull.equations
    spec = importlib.util.spec_from_file_location(
        "basket_geometry", Path(__file__).parents[1] / "09_lined_rect_basket/fit_hoi.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    samples = []
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if name.startswith("right_hand") and name.endswith("_collision"):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = model.geom_dataid[g]
                start = model.mesh_vertadr[mesh_id]
                points = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id] : 30].copy()
            else:
                points = geo.collision_points(model, g)
            samples.append((g, points))
    palm = model.geom("right_hand_palm_collision").id
    seed = prior.joint_pos[120, cols].astype(float)
    lower = np.array([model.joint(prior.joint_names[i]).range[0] for i in cols]) + 1e-5
    upper = np.array([model.joint(prior.joint_names[i]).range[1] for i in cols]) - 1e-5
    best = None
    for attempt in range(2):

        def residual(q, seed=seed):
            data.qpos[addresses[cols]] = q
            mujoco.mj_forward(model, data)
            penalty = []
            near = []
            for g, points in samples:
                world = points @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                distance = ((world - berry) @ berry_rotation @ planes[:, :3].T + planes[:, 3]).max(1)
                penalty.extend(50 * np.minimum(distance + 0.0005, 0))
                room_local = (world - room_transform[:3, 3]) @ room_transform[:3, :3]
                inside = (
                    (room_local[:, 0] > -1.03)
                    & (room_local[:, 0] < 0.57)
                    & (room_local[:, 1] < -2.11905405)
                    & (room_local[:, 1] > -2.97)
                )
                penalty.extend(35 * np.minimum(room_local[:, 2] - 0.756, 0) * inside)
                if any(part in model.geom(g).name for part in ("thumb_2", "index_1", "middle_1")):
                    near.append(12 * distance.min())
            relative = (berry - data.geom_xpos[palm]) @ data.geom_xmat[palm].reshape(3, 3)
            return np.r_[
                penalty, near, 0.03 * (q[:7] - seed[:7]), 0.002 * (q[7:] - seed[7:]), 2 * (relative - [0.065, 0, 0])
            ]

        fit = least_squares(residual, seed, bounds=(lower, upper), max_nfev=350, diff_step=1e-4)
        if best is None or fit.cost < best.cost:
            best = fit
        seed = fit.x.copy()
        seed[-7] = 0.5 if attempt == 0 else seed[-7]
    residual(best.x)
    report = {
        "joint_names": [prior.joint_names[i] for i in cols],
        "joint_pos": best.x.tolist(),
        "cost": best.cost,
        "object_center_w": berry.tolist(),
        "palm_rotation_world": data.geom_xmat[palm].reshape(3, 3).tolist(),
        "object_in_palm_position": ((berry - data.geom_xpos[palm]) @ data.geom_xmat[palm].reshape(3, 3)).tolist(),
        "acceptance": "single-frame design only; dense geometry and continuous approach still required",
        "method": "fixed object and table, three distal-finger surface contacts, sampled nonpenetration, weak arm/hand prior",
    }
    (folder / "grasp_design.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
