"""Sample the v6 whole-hand sweep against the inherited basket and tabletop."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = example_dir(__file__)
ASSETS = ROOT.parent / "brush_pick_1012/assets"
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
URDF = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/body.urdf")
MOTION = ROOT / "retarget/brush_sweep_kinematic_v12/motion.npz"
PRIOR = ROOT / "retarget/body_prior_8_28/motion.npz"
OBJECT = ROOT / "retarget/brush_sweep_kinematic_v12/object_estimate.npz"


def mesh_points(model, body_name):
    points = []
    body_id = model.body(body_name).id
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id or model.geom_group[g] == 3 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        points.append(Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]]).apply(vertices) + model.geom_pos[g])
    if not points:
        return np.empty((0, 3))
    vertices = np.unique(np.concatenate(points), axis=0)
    return vertices[ConvexHull(vertices).vertices] if len(vertices) >= 4 else vertices


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    order = json.loads(ORDER.read_text())["robot_joint_names"]
    addresses = np.asarray([model.joint(name).qposadr[0] for name in order])
    hand = [(model.body(i).name, i, mesh_points(model, model.body(i).name)) for i in range(model.nbody)
            if model.body(i).name.startswith(("right_hand_", "right_wrist_"))]
    hand = [(name, bid, vertices) for name, bid, vertices in hand if len(vertices)]
    basket_quat = Rotation.from_quat([0, 0, 0.7071067812, 0.7071067812])
    basket_pos = np.array([0.944999933, -0.569999993, 0.400999993])
    basket = []
    for path in sorted((ASSETS / "white_storage_box/collision_hulls").glob("*.obj")):
        vertices = basket_quat.apply(trimesh.load(path, force="mesh", process=False).vertices) + basket_pos
        hull = ConvexHull(vertices)
        basket.append((path.name, vertices.min(axis=0), vertices.max(axis=0), hull.equations))
    grip_vertices = trimesh.load(ASSETS / "green_hand_brush/collision_hulls/Hull_grip.obj", force="mesh").vertices
    with np.load(MOTION) as motion, np.load(PRIOR) as prior, np.load(OBJECT) as obj:
        q = motion["joint_pos"]
        old = prior["joint_pos"]
        obj_pos = obj["position_w"]
        obj_rot = obj["quaternion_xyzw"]
        limits = {j.get("name"): float(j.find("limit").get("velocity", "20"))
                  for j in ET.parse(URDF).findall("joint") if j.find("limit") is not None}
        speed_limits = np.array([limits.get(name, 3.0) for name in order])
        speed_ratio = np.max(abs(np.diff(q, axis=0)) * 50 / speed_limits)
        changed = [name for j, name in enumerate(order) if np.max(abs(q[:, j] - old[:, j])) > 1e-6]
        basket_hits = []
        table_hits = []
        grip_gaps = []
        for k in range(round((11 - 8) * 50), round((26.5 - 8) * 50) + 1, 2):
            data.qpos[:3] = motion["body_pos_w"][k, 0]
            data.qpos[3:7] = motion["body_quat_w"][k, 0]
            data.qpos[addresses] = q[k]
            mujoco.mj_forward(model, data)
            for name, bid, local in hand:
                points = local @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
                inside_table = ((points[:, 0] >= 0.76) & (points[:, 0] <= 1.36)
                                & (points[:, 1] >= -0.71) & (points[:, 1] <= 0.69)
                                & (points[:, 2] < 0.400))
                if inside_table.any():
                    table_hits.append({"time_s": 8 + k / 50, "body": name,
                                       "max_depth_m": float(0.400 - points[inside_table, 2].min())})
                for basket_name, lower, upper, equations in basket:
                    near = points[np.all((points >= lower - 0.01) & (points <= upper + 0.01), axis=1)]
                    if len(near):
                        depth = np.max(near @ equations[:, :3].T + equations[:, 3], axis=1)
                        inside = depth < -0.001
                        if inside.any():
                            basket_hits.append({"time_s": 8 + k / 50, "body": name,
                                                "basket_hull": basket_name,
                                                "max_depth_m": float(-depth[inside].min())})
            if k in (round((12.8 - 8) * 50), round((25.6 - 8) * 50)):
                grip_world = Rotation.from_quat(obj_rot[k]).apply(grip_vertices) + obj_pos[k]
                for name, bid, local in hand:
                    if not name.startswith(("right_hand_thumb_2", "right_hand_index_1", "right_hand_middle_1")):
                        continue
                    points = local @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
                    distance = np.linalg.norm(points[:, None] - grip_world[None, :], axis=2).min()
                    grip_gaps.append({"time_s": 8 + k / 50, "body": name, "sampled_vertex_gap_m": float(distance)})
    report = {
        "status": "sampled_kinematic_audit_not_contact_validation",
        "changed_joints": changed,
        "root_legs_torso_left_arm_unchanged": all(name.startswith("right_") for name in changed),
        "max_speed_limit_ratio": float(speed_ratio),
        "sampled_hand_table_penetrations": table_hits[:50],
        "sampled_hand_table_penetration_count": len(table_hits),
        "sampled_hand_basket_penetrations": basket_hits[:50],
        "sampled_hand_basket_penetration_count": len(basket_hits),
        "grip_sampled_vertex_gaps": grip_gaps,
        "sampling_limit": "Hand visual mesh convex vertices every 2 frames; no continuous edge sweep or physical force measured.",
    }
    (ROOT / "review/hand_audit_v12.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if "penetrations" not in k}, indent=2))
    print("table hits", len(table_hits), "basket hits", len(basket_hits))


if __name__ == "__main__":
    main()
