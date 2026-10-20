"""Measure kinematic brush/table clearance and joints for candidate review."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


ROOT = example_dir(__file__)
ASSETS = ROOT.parent / "brush_pick_1012/assets/green_hand_brush/collision_hulls"
MODEL_ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
TABLE_TOP_Z = 0.401
TABLE_X = (0.76, 1.36)
TABLE_Y = (-0.71, 0.69)


def main():
    report = {}
    order = json.loads(MODEL_ORDER.read_text())["robot_joint_names"]
    brush_vertices = np.concatenate([
        trimesh.load(path, force="mesh").vertices
        for path in sorted(ASSETS.glob("Hull_bristles*.obj"))
    ])
    for version in ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v19"):
        folder = ROOT / f"retarget/brush_sweep_kinematic_{version}"
        with np.load(folder / "motion.npz") as motion, np.load(folder / "object_estimate.npz") as obj:
            q = motion["joint_pos"]
            joint_vel = np.gradient(q, 1 / 50, axis=0)
            joint_acc = np.gradient(joint_vel, 1 / 50, axis=0)
            positions = obj["position_w"]
            rotations = Rotation.from_quat(obj["quaternion_xyzw"])
            clearance = []
            overlap = []
            for t in np.arange(15.0, 22.01, 0.02):
                k = round((t - 8) * 50)
                world = rotations[k].apply(brush_vertices) + positions[k]
                on_table = ((world[:, 0] >= TABLE_X[0]) & (world[:, 0] <= TABLE_X[1])
                            & (world[:, 1] >= TABLE_Y[0]) & (world[:, 1] <= TABLE_Y[1]))
                overlap.append(float(np.mean(on_table)))
                if on_table.any():
                    clearance.append([float(t), float(np.min(world[on_table, 2] - TABLE_TOP_Z))])
            report[version] = {
                "sweep_min_bristle_clearance_m": float(np.min(np.asarray(clearance)[:, 1])),
                "sweep_max_bristle_clearance_m": float(np.max(np.asarray(clearance)[:, 1])),
                "sweep_fraction_frames_any_bristles_over_table": float(np.mean(np.asarray(overlap) > 0)),
                "sweep_mean_bristle_vertex_fraction_over_table": float(np.mean(overlap)),
                "sweep_fraction_frames_bristle_penetrating_table": float(np.mean(np.asarray(clearance)[:, 1] < -0.003)),
                "sweep_bristle_clearance_samples_s_m": clearance[::25],
                "max_joint_speed_rad_s": float(np.max(abs(joint_vel))),
                "max_joint_acceleration_rad_s2": float(np.max(abs(joint_acc))),
                "wrist_axis_ranges_rad": {
                    name: [float(np.min(q[:, order.index(name)])), float(np.max(q[:, order.index(name)]))]
                    for name in ("right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
                },
            }
    (ROOT / "review/kinematic_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "sweep_bristle_clearance_samples_s_m"} for key, value in report.items()}, indent=2))


if __name__ == "__main__":
    main()
