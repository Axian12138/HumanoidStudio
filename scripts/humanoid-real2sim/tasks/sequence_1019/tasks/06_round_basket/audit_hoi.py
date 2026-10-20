"""50 Hz and half-frame geometry audit of the round-basket HOI candidate."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


HERE = example_dir(__file__)
OUT = HERE / "hoi_v2"
MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/"
    "robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)
ORDER = HERE.parents[1] / "retarget/ground_support_27_end/sidecars/motion_order.json"
URDF = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/body.urdf")
PALM_LOCAL = np.array([0.10629904, 0.03630376, 0.0])


def geometry(path):
    parts = {}
    name = None
    for line in path.open():
        if line.startswith("o "):
            name = line[2:].strip()
        elif name is not None and line.startswith("v "):
            parts.setdefault(name, []).append(tuple(map(float, line.split()[1:4])))
    result = []
    for name, points in parts.items():
        vertices = np.asarray(points)
        if len(vertices) < 4:
            continue
        hull = ConvexHull(vertices)
        result.append((name, vertices.min(axis=0), vertices.max(axis=0), hull.equations, vertices[hull.vertices]))
    return result


def hand_meshes(model):
    result = []
    for bid in range(model.nbody):
        name = model.body(bid).name
        if not name.startswith(("right_hand_", "right_wrist_")):
            continue
        meshes = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != bid or model.geom_group[geom] == 3 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[geom]
            vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            meshes.append(Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).apply(vertices) + model.geom_pos[geom])
        if meshes:
            vertices = np.unique(np.concatenate(meshes), axis=0)
            result.append((name, bid, vertices[ConvexHull(vertices).vertices]))
    return result


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    order = json.loads(ORDER.read_text())
    names = order["robot_joint_names"]
    addresses = np.asarray([model.joint(name).qposadr[0] for name in names])
    wrist = model.body("right_wrist_yaw_link").id
    hand = hand_meshes(model)
    basket = geometry(HERE / "round_laundry_basket_collision.obj")
    handles = [entry for entry in basket if "handle" in entry[0].lower()]
    handle_vertices = np.concatenate([entry[-1] for entry in handles])
    limits = {j.get("name"): float(j.find("limit").get("velocity", "20"))
              for j in ET.parse(URDF).findall("joint") if j.find("limit") is not None}
    speed_limit = np.asarray([limits.get(name, 20.0) for name in names])
    with np.load(OUT / "motion.npz") as motion, np.load(HERE / "body_prior.npz") as prior, np.load(OUT / "object_estimate.npz") as obj:
        q = motion["joint_pos"]
        old = prior["joint_pos"]
        obj_pos = obj["position_w"]
        changed = [name for j, name in enumerate(names) if np.max(np.abs(q[:, j] - old[:, j])) > 1e-6]
        speed_ratio = np.max(abs(np.diff(q, axis=0)) * 50 / speed_limit)
        collisions = []
        grip_gaps = []
        palm = {}
        samples = sorted(set(np.arange(0, len(q), dtype=float).tolist() +
                             np.arange(round((72.4-69)*50), round((74.9-69)*50), dtype=float).tolist() +
                             (np.arange(round((72.4-69)*50), round((74.9-69)*50)) + 0.5).tolist()))
        for sample in samples:
            i = min(int(sample), len(q) - 1)
            blend = sample - i
            i2 = min(i + 1, len(q) - 1)
            root = (1 - blend) * motion["body_pos_w"][i, 0] + blend * motion["body_pos_w"][i2, 0]
            root_q = Rotation.from_quat(motion["body_quat_w"][i, 0, [1, 2, 3, 0]])
            next_q = Rotation.from_quat(motion["body_quat_w"][i2, 0, [1, 2, 3, 0]])
            root_q = root_q * Rotation.from_rotvec((root_q.inv() * next_q).as_rotvec() * blend)
            data.qpos[:3] = root
            data.qpos[3:7] = root_q.as_quat()[[3, 0, 1, 2]]
            data.qpos[addresses] = (1 - blend) * q[i] + blend * q[i2]
            mujoco.mj_forward(model, data)
            rotation = data.xmat[wrist].reshape(3, 3)
            palm[sample] = (data.xpos[wrist] + rotation @ PALM_LOCAL, rotation[:, 1].copy())
            object_position = (1 - blend) * obj_pos[i] + blend * obj_pos[i2]
            grip_tree = cKDTree(handle_vertices + object_position)
            for name, bid, local in hand:
                points = local @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
                if sample in (round((73.5-69)*50), round((74.3-69)*50)) and name.startswith(("right_hand_thumb_2", "right_hand_index_1", "right_hand_middle_1")):
                    grip_gaps.append(dict(time_s=69 + sample / 50, body=name,
                                          sampled_vertex_gap_m=float(grip_tree.query(points)[0].min())))
                local_points = points - object_position
                for part, lower, upper, equations, _ in basket:
                    near = local_points[np.all((local_points >= lower - 0.01) & (local_points <= upper + 0.01), axis=1)]
                    if len(near) == 0:
                        continue
                    depth = np.max(near @ equations[:, :3].T + equations[:, 3], axis=1)
                    inside = -depth > 0.001
                    if inside.any():
                        collisions.append(dict(time_s=69 + sample / 50, body=name, basket_part=part,
                                               depth_m=float((-depth[inside]).max())))

    def angle(first, last):
        a, _ = palm[round((first-69)*50)]
        b, _ = palm[round((last-69)*50)]
        direction = b - a
        direction /= np.linalg.norm(direction)
        normal = palm[round((first-69)*50)][1] + palm[round((last-69)*50)][1]
        normal /= np.linalg.norm(normal)
        return float(np.degrees(np.arccos(np.clip(direction @ normal, -1, 1))))

    right_wrist = [names.index("right_wrist_" + axis + "_joint") for axis in ("roll", "pitch", "yaw")]
    active = slice(round((71.4 - 69) * 50), round((74.8 - 69) * 50) + 1)
    joint_limit_excess = {}
    for col, name in enumerate(names):
        joint = model.joint(name)
        if joint.limited:
            exceed = max(float(joint.range[0] - np.min(q[:, col])),
                         float(np.max(q[:, col]) - joint.range[1]), 0.0)
            if exceed > 1e-4:
                joint_limit_excess[name] = exceed
    forbidden = [hit for hit in collisions if "handle" not in hit["basket_part"].lower()]
    top_forbidden = sorted(forbidden, key=lambda hit: hit["depth_m"], reverse=True)[:20]
    report = dict(
        status="sampled_kinematic_audit_not_physical_grasp",
        sample_count=len(samples),
        changed_joints=changed,
        only_right_arm_and_hand_changed=all(name.startswith("right_") for name in changed),
        seam_joint_max_rad=float(max(np.max(np.abs(q[0]-old[0])), np.max(np.abs(q[-1]-old[-1])))),
        max_joint_speed_limit_ratio=float(speed_ratio),
        right_wrist_axis_abs_max_deg=dict(zip(("roll", "pitch", "yaw"),
                                          np.max(np.abs(np.rad2deg(q[:, right_wrist])), axis=0).tolist())),
        active_right_wrist_axis_abs_max_deg=dict(zip(("roll", "pitch", "yaw"),
                                          np.max(np.abs(np.rad2deg(q[active][:, right_wrist])), axis=0).tolist())),
        joint_limit_excess_rad=joint_limit_excess,
        approach_palm_velocity_angle_deg={"72.0-72.5": angle(72.0, 72.5),
                                          "72.5-72.9": angle(72.5, 72.9),
                                          "72.9-73.25": angle(72.9, 73.25)},
        sampled_all_collision_count=len(collisions),
        sampled_forbidden_collision_count=len(forbidden),
        sampled_forbidden_max_depth_m=max((hit["depth_m"] for hit in forbidden), default=0.0),
        sampled_forbidden_time_range_s=[min((hit["time_s"] for hit in forbidden), default=None),
                                        max((hit["time_s"] for hit in forbidden), default=None)],
        sampled_forbidden_deepest=top_forbidden,
        sampled_forbidden_collisions=forbidden[:30],
        sampled_handle_contacts=collisions[:30],
        finger_handle_vertex_gaps_m=grip_gaps,
        limits="Hand hull vertices at 50 Hz and interleaved half frames; static 1012 collision hulls, no physical forces or continuous-time certificate.",
    )
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "only_right_arm_and_hand_changed", "seam_joint_max_rad", "max_joint_speed_limit_ratio",
        "active_right_wrist_axis_abs_max_deg", "joint_limit_excess_rad", "approach_palm_velocity_angle_deg",
        "sampled_forbidden_collision_count", "finger_handle_vertex_gaps_m",
    )}, indent=2))


if __name__ == "__main__":
    main()
