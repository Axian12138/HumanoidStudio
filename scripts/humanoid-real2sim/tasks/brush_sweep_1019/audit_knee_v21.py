"""Audit the composed right-knee contact, moving toe, and brush sweep."""

import argparse
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from retarget3.common import FEET
from retarget3.ground_stage import contact_geometry_values
from retarget3.robot import MJCasadiRobot


ROOT = example_dir(__file__)
MOTION = ROOT / "retarget/brush_sweep_kinematic_v21_knee/motion.npz"
OBJECT = ROOT / "retarget/brush_sweep_kinematic_v21_knee/object_estimate.npz"
HUMAN = ROOT / "retarget/body_prior_8_28/intermediate/targets.npz"
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
BRISTLES = ROOT.parent / "brush_pick_1012/assets/green_hand_brush/collision_hulls"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=MOTION)
    parser.add_argument("--object", type=Path, default=OBJECT)
    parser.add_argument("--report", type=Path, default=ROOT / "review/knee_audit_v21.json")
    args = parser.parse_args()
    robot = MJCasadiRobot()
    names = json.loads(ORDER.read_text())["robot_joint_names"]
    with np.load(args.motion, allow_pickle=False) as z:
        motion = {key: z[key] for key in z.files}
    with np.load(args.object, allow_pickle=False) as z:
        obj = {key: z[key] for key in z.files}
    with np.load(HUMAN, allow_pickle=False) as z:
        human = z["human"]
    q = np.empty((len(motion["joint_pos"]), robot.nq))
    q[:, :3] = motion["body_pos_w"][:, 0]
    q[:, 3:7] = motion["body_quat_w"][:, 0][:, [1, 2, 3, 0]]
    for col, name in enumerate(names):
        q[:, robot.dof_name_to_index[name]] = motion["joint_pos"][:, col]
    geometry = contact_geometry_values(robot, q)
    pos, quat = robot.forward(q, ["right_knee_link", FEET[0], FEET[1]])
    core = slice(420, 709)
    knee_xy_error = np.linalg.norm(pos[core, 0, :2] - human[core, 4, :2], axis=1)
    right_foot_rotation = Rotation.from_quat(quat[:, 2][:, [1, 2, 3, 0]])
    toe_xy = pos[:, 2, :2] + right_foot_rotation.apply(np.array([.132, 0, -.025]))[:, :2]
    toe_path = np.linalg.norm(np.diff(toe_xy[core], axis=0), axis=1)
    left_foot_slip = np.linalg.norm(pos[core, 1] - pos[core.start, 1], axis=1)

    vertices = np.concatenate([
        trimesh.load(path, force="mesh").vertices
        for path in sorted(BRISTLES.glob("Hull_bristles*.obj"))
    ])
    rotation = Rotation.from_quat(obj["quaternion_xyzw"])
    clearances = []
    for k in range(round((15 - 8) * 50), round((22 - 8) * 50) + 1):
        world = rotation[k].apply(vertices) + obj["position_w"][k]
        over = ((world[:, 0] >= .76) & (world[:, 0] <= 1.36)
                & (world[:, 1] >= -.71) & (world[:, 1] <= .69))
        if over.any():
            clearances.append(float((world[over, 2] - .401).min()))
    joint_speed = np.abs(np.diff(motion["joint_pos"], axis=0)) * 50
    report = {
        "status": "kinematic_geometry_audit_not_physical_contact",
        "human_right_knee_contact_window_global_s": [16.4, 22.18],
        "human_right_knee_mean_xy_m": human[core, 4, :2].mean(axis=0).tolist(),
        "robot_right_knee_mean_xy_m": pos[core, 0, :2].mean(axis=0).tolist(),
        "robot_to_human_right_knee_xy_error_mm_max": float(knee_xy_error.max() * 1000),
        "robot_to_human_right_knee_xy_error_mm_mean": float(knee_xy_error.mean() * 1000),
        "right_knee_mesh_gap_mm_range": [(geometry[core, 1, 0] * 1000).min().item(),
                                          (geometry[core, 1, 0] * 1000).max().item()],
        "right_knee_mesh_gap_mm_max": (geometry[core, 1, 0] * 1000).max().item(),
        "right_toe_mesh_gap_mm_range": [(geometry[core, 1, 1] * 1000).min().item(),
                                         (geometry[core, 1, 1] * 1000).max().item()],
        "right_heel_height_mm_min": float(geometry[core, 1, 2].min() * 1000),
        "right_toe_xy_excursion_mm": float(np.linalg.norm(toe_xy[core] - toe_xy[core.start], axis=1).max() * 1000),
        "right_toe_xy_path_length_mm": float(toe_path.sum() * 1000),
        "left_stance_ankle_slip_mm_max": float(left_foot_slip.max() * 1000),
        "minimum_visible_foot_knee_mesh_floor_clearance_mm": float(geometry[:, :, 3:].min() * 1000),
        "sweep_bristle_table_clearance_mm_min": float(min(clearances) * 1000),
        "max_joint_speed_rad_s": float(joint_speed.max()),
        "max_root_translation_step_mm": float(np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1).max() * 1000),
    }
    failures = []
    if report["right_knee_mesh_gap_mm_max"] > 1.1:
        failures.append("right knee did not reach floor")
    if report["right_toe_mesh_gap_mm_range"][1] > 1.1:
        failures.append("right toe lost floor contact")
    if report["right_heel_height_mm_min"] < 80:
        failures.append("heel did not lift")
    if report["minimum_visible_foot_knee_mesh_floor_clearance_mm"] < -.55:
        failures.append("leg mesh penetrated floor")
    if report["sweep_bristle_table_clearance_mm_min"] < -3:
        failures.append("brush bristles penetrated table")
    report["failures"] = failures
    path = args.report
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError(", ".join(failures))


if __name__ == "__main__":
    main()
