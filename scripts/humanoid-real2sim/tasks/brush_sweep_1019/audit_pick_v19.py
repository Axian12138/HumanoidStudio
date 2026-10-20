"""Check the final Dex3 palm against the animated brush at every 50 Hz frame."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = example_dir(__file__)
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
BRUSH_HULLS = ROOT.parent / "brush_pick_1012/assets/green_hand_brush/collision_hulls"
MOTION = ROOT / "retarget/brush_sweep_kinematic_v19/motion.npz"
OBJECT = ROOT / "retarget/brush_sweep_kinematic_v19/object_estimate.npz"


def frame(seconds):
    return round((seconds - 8.0) * 50)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    names = json.loads(ORDER.read_text())["robot_joint_names"]
    addresses = np.asarray([model.joint(name).qposadr[0] for name in names])
    palm_geom = model.geom("right_hand_palm_collision").id
    mesh = model.geom_dataid[palm_geom]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    vertices = Rotation.from_quat(model.geom_quat[palm_geom][[1, 2, 3, 0]]).apply(vertices) + model.geom_pos[palm_geom]
    palm_body = model.geom_bodyid[palm_geom]
    wrist_body = model.body("right_wrist_yaw_link").id
    hulls = []
    for path in sorted(BRUSH_HULLS.glob("*.obj")):
        local = trimesh.load(path, force="mesh").vertices
        hulls.append((path.name, ConvexHull(local).equations))

    with np.load(MOTION) as motion, np.load(OBJECT) as brush:
        joints = motion["joint_pos"]
        roots = motion["body_pos_w"][:, 0]
        root_quats = motion["body_quat_w"][:, 0]
        brush_pos = brush["position_w"]
        brush_rot = Rotation.from_quat(brush["quaternion_xyzw"])
        wrist_cols = [names.index("right_wrist_" + axis + "_joint") for axis in ("roll", "pitch", "yaw")]
        wrist_max_deg = np.max(np.abs(np.rad2deg(joints[frame(11):frame(27.2) + 1, wrist_cols])), axis=0)
        inside = []
        approach_points = {}
        approach_normals = {}
        for k in range(frame(11), frame(27.2) + 1):
            data.qpos[:3] = roots[k]
            data.qpos[3:7] = root_quats[k]
            data.qpos[addresses] = joints[k]
            mujoco.mj_forward(model, data)
            palm_world = vertices @ data.xmat[palm_body].reshape(3, 3).T + data.xpos[palm_body]
            palm_object = brush_rot[k].inv().apply(palm_world - brush_pos[k])
            for name, equations in hulls:
                depth = np.max(palm_object @ equations[:, :3].T + equations[:, 3], axis=1)
                penetration = -float(np.min(depth))
                if penetration > 0.0005:
                    inside.append((float(8 + k / 50), name, penetration))
            if k in (frame(11.8), frame(12.15), frame(12.4)):
                rotation = data.xmat[wrist_body].reshape(3, 3)
                approach_points[k] = data.xpos[wrist_body] + rotation @ np.array([0.1062990399, 0.0483037578, 0])
                approach_normals[k] = rotation[:, 1]
    approach_angles = {}
    for first, second in ((11.8, 12.15), (12.15, 12.4)):
        start, stop = frame(first), frame(second)
        velocity = approach_points[stop] - approach_points[start]
        velocity /= np.linalg.norm(velocity)
        normal = approach_normals[start] + approach_normals[stop]
        normal /= np.linalg.norm(normal)
        approach_angles[f"{first:.2f}-{second:.2f}s"] = float(np.degrees(np.arccos(np.clip(velocity @ normal, -1, 1))))
    report = {
        "status": "50_hz_vertex_audit_not_continuous_collision_or_physical_grasp",
        "palm_vs_brush_penetration_over_0_5mm_count": len(inside),
        "palm_vs_brush_max_penetration_m": max((item[2] for item in inside), default=0.0),
        "palm_vs_brush_first_hits": inside[:20],
        "palm_approach_direction_error_deg": approach_angles,
        "right_wrist_axis_abs_max_deg_11_to_27_2s": dict(zip(("roll", "pitch", "yaw"), map(float, wrist_max_deg))),
    }
    (ROOT / "review/pick_audit_v19.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
