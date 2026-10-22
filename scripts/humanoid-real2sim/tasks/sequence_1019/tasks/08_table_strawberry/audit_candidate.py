"""Independent mesh-vertex sweep of the table-strawberry candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields


def main():
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    folder = args.candidate
    geometry = (
        json.loads((folder / "geometry_config.json").read_text()) if (folder / "geometry_config.json").exists() else {}
    )
    room_transform = np.array(geometry.get("T_room_correction", np.eye(4)))
    model_path = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    motion = MotionFields.load(folder / "motion.npz", folder / "motion_order.json")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = np.array([model.joint(name).qposadr[0] for name in motion.joint_names])
    spec = importlib.util.spec_from_file_location(
        "basket_geometry", Path(__file__).parents[1] / "09_lined_rect_basket/fit_hoi.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    samples = []
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if not name.startswith("right_") or not name.endswith("_collision"):
            continue
        if not any(part in name for part in ("hand_", "wrist", "elbow_yaw")):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            first = model.mesh_vertadr[mesh]
            points = model.mesh_vert[first : first + model.mesh_vertnum[mesh]].copy()
        else:
            points = geo.collision_points(model, g)
        samples.append((g, points))
    basket = np.load(folder / "lined_rect_basket.npz")
    berry = np.load(folder / "table_strawberry.npz")
    basket_rot = Rotation.from_quat(basket["quaternion_xyzw"]).as_matrix()
    times = 90 + np.arange(2 * len(motion.joint_pos) - 1) / 100
    minimum = []
    table_minimum = []
    for step in range(len(times)):
        k = step // 2
        following = min(k + 1, len(motion.joint_pos) - 1)
        mix = (step % 2) / 2
        data.qpos[addresses] = motion.joint_pos[k] * (1 - mix) + motion.joint_pos[following] * mix
        data.qpos[:3] = motion.body_pos_w[k, 0] * (1 - mix) + motion.body_pos_w[following, 0] * mix
        rootq = motion.body_quat_w[k, 0] * (1 - mix) + motion.body_quat_w[following, 0] * mix
        data.qpos[3:7] = rootq / np.linalg.norm(rootq)
        mujoco.mj_forward(model, data)
        world = np.concatenate(
            [points @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] for g, points in samples]
        )
        local = (world - basket["position_w"][k]) @ basket_rot[k]
        minimum.append(float(geo.basket_sdf(local).min()))
        room_local = (world - room_transform[:3, 3]) @ room_transform[:3, :3]
        inside = (
            (room_local[:, 0] > -1.03)
            & (room_local[:, 0] < 0.57)
            & (room_local[:, 1] < -2.11905405)
            & (room_local[:, 1] > -2.97)
        )
        table_minimum.append(float((room_local[inside, 2] - 0.754).min()) if inside.any() else 1.0)
    minimum = np.array(minimum)
    table_minimum = np.array(table_minimum)
    np.savez_compressed(
        folder / "dense_hand_sweep.npz",
        source_time_s=times,
        basket_signed_distance_m=minimum,
        table_clearance_m=table_minimum,
    )
    mesh = trimesh.load_mesh(
        "/home/galbot/workspace/ScaleTrack/data/1012test/hoi_stage2/scene/assets/object_strawberry_existing_0_0.obj"
    )
    k = 150
    data.qpos[addresses] = motion.joint_pos[k]
    data.qpos[:3], data.qpos[3:7] = motion.body_pos_w[k, 0], motion.body_quat_w[k, 0]
    mujoco.mj_forward(model, data)
    berry_r = Rotation.from_quat(berry["quaternion_xyzw"][k]).as_matrix()
    grasp = {}
    for g, points in samples:
        name = model.geom(g).name
        if "hand" not in name:
            continue
        # Full mesh sweep above; bounded downsampling here avoids an enormous
        # triangle-proximity query. This contact check is explicitly sampled.
        world = points[:: max(1, len(points) // 500)] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        distances = trimesh.proximity.signed_distance(mesh, (world - berry["position_w"][k]) @ berry_r)
        grasp[name] = {
            "max_sample_penetration_m": float(max(0, distances.max())),
            "nearest_sample_surface_distance_m": float(abs(distances).min()),
        }
    final_rot = Rotation.from_quat(berry["quaternion_xyzw"][-1]).as_matrix()
    final_vertices = np.asarray(mesh.vertices) @ final_rot.T + berry["position_w"][-1]
    final_local = (final_vertices - basket["position_w"][-1]) @ basket_rot[-1]
    task = times < 98
    urdf = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/body.urdf")
    limits = {
        j.get("name"): float(j.find("limit").get("velocity", "20"))
        for j in ET.parse(urdf).findall("joint")
        if j.find("limit") is not None
    }
    speed_report = {}
    for i, name in enumerate(motion.joint_names):
        limit = limits.get(name, 3.0)
        bad = abs(motion.joint_vel[:, i]) > limit + 1e-5
        speed_report[name] = {
            "limit_rad_s": limit,
            "source": "body.urdf" if name in limits else "retarget3 3rad/s fallback; hardware limit not identified",
            "maximum_rad_s": float(abs(motion.joint_vel[:, i]).max()),
            "new_08_violation_frames": np.flatnonzero(bad[:400]).tolist(),
            "protected_09_violation_frames": (400 + np.flatnonzero(bad[400:])).tolist(),
        }
    relative_positions, relative_rotations = [], []
    palm = model.geom("right_hand_palm_collision").id
    release_event = folder / "release_event.json"
    release_time = json.loads(release_event.read_text())["release_time_s"] if release_event.exists() else 96.7
    for k in range(120, round((release_time - 90) * 50)):
        data.qpos[addresses] = motion.joint_pos[k]
        data.qpos[:3], data.qpos[3:7] = motion.body_pos_w[k, 0], motion.body_quat_w[k, 0]
        mujoco.mj_forward(model, data)
        palm_r = data.geom_xmat[palm].reshape(3, 3)
        relative_positions.append((berry["position_w"][k] - data.geom_xpos[palm]) @ palm_r)
        relative_rotations.append(palm_r.T @ Rotation.from_quat(berry["quaternion_xyzw"][k]).as_matrix())
    relative_positions = np.array(relative_positions)
    relative_rotation = Rotation.from_matrix(relative_rotations)
    report = {
        "scope": "Right hand, wrist and near forearm mesh vertices at 100Hz interpolated joint states; not triangle-intersection or full-body collision proof.",
        "mesh_vertices_per_sample": sum(len(points) for _, points in samples),
        "new_08_basket_minimum_m": float(minimum[task].min()),
        "new_08_basket_violation_times_s": times[task & (minimum < -0.002)].tolist(),
        "inherited_09_preload_minimum_m": float(minimum[~task].min()),
        "table_proxy_minimum_m": float(table_minimum.min()),
        "table_proxy_violation_times_s": times[table_minimum < -0.002].tolist(),
        "canonical_grasp_sampled_surface": grasp,
        "final_mesh_bottom_above_basket_base_m": float(final_local[:, 2].min()),
        "final_basket_inner_xy_margin_m": float(
            np.minimum(0.182 - abs(final_local[:, 0]), 0.122 - abs(final_local[:, 1])).min()
        ),
        "wrist_angles_rad": {
            name: {"min": float(motion.joint_pos[:, i].min()), "max": float(motion.joint_pos[:, i].max())}
            for i, name in enumerate(motion.joint_names)
            if name.startswith("right_wrist")
        },
        "peak_new_08_joint_speed_rad_s": float(abs(motion.joint_vel[50:399]).max()),
        "peak_inherited_09_joint_speed_rad_s": float(abs(motion.joint_vel[400:]).max()),
        "peak_joint_acceleration_rad_s2": float(abs(np.gradient(motion.joint_vel, 0.02, axis=0)).max()),
        "accepted": False,
        "joint_speed_limits": speed_report,
        "fixed_held_transform_position_max_drift_m": float(
            np.linalg.norm(relative_positions - relative_positions[0], axis=1).max()
        ),
        "fixed_held_transform_orientation_max_drift_rad": float(
            (relative_rotation[0].inv() * relative_rotation).magnitude().max()
        ),
    }
    (folder / "dense_geometry_review.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
