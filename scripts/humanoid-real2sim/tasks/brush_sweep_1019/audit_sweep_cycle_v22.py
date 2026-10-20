"""Audit the human-timed lift/place/stroke/lift kinematic sweep candidate."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


ROOT = example_dir(__file__)
OLD = ROOT / "retarget/brush_sweep_kinematic_v21_knee"
NEW = ROOT / "retarget/brush_sweep_kinematic_v22_cycle"
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
BRISTLES = ROOT.parent / "brush_pick_1012/assets/green_hand_brush/collision_hulls"
TABLE = (.76, 1.36, -.71, .69, .401)


def main():
    with np.load(OLD / "motion.npz", allow_pickle=False) as z:
        old = {name: z[name] for name in z.files}
    with np.load(NEW / "motion.npz", allow_pickle=False) as z:
        new = {name: z[name] for name in z.files}
    with np.load(OLD / "object_estimate.npz", allow_pickle=False) as z:
        old_object = {name: z[name] for name in z.files}
    with np.load(NEW / "object_estimate.npz", allow_pickle=False) as z:
        obj = {name: z[name] for name in z.files}
    names = json.loads(ORDER.read_text())["robot_joint_names"]
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    joint_addr = np.asarray([model.joint(name).qposadr[0] for name in names])
    arm_cols = [i for i, name in enumerate(names) if name.startswith("right_") and
                ("shoulder" in name or "elbow" in name or "wrist" in name)]
    other_cols = [i for i in range(len(names)) if i not in arm_cols]
    start, stop = 500, 760
    vertices = np.concatenate([trimesh.load(path, force="mesh").vertices
                               for path in sorted(BRISTLES.glob("Hull_bristles*.obj"))])
    rotation = Rotation.from_quat(obj["quaternion_xyzw"])
    clearances = np.empty(len(new["joint_pos"]))
    brush_x = obj["position_w"][:, 0]
    hand_meshes = []
    for geom in range(model.ngeom):
        body = model.body(model.geom_bodyid[geom]).name
        if not body.startswith("right_hand_") or model.geom_group[geom] == 3 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[geom]
        verts = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].copy()
        hand_meshes.append((geom, verts))
    hand_table_penetration = 0.0
    for k in range(round((18 - 8) * 50), round((23.2 - 8) * 50) + 1):
        world = rotation[k].apply(vertices) + obj["position_w"][k]
        clearances[k] = world[:, 2].min() - TABLE[4]
        data.qpos[:3] = new["body_pos_w"][k, 0]
        data.qpos[3:7] = new["body_quat_w"][k, 0]
        data.qpos[joint_addr] = new["joint_pos"][k]
        mujoco.mj_forward(model, data)
        for geom, verts in hand_meshes:
            points = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            inside = ((points[:, 0] >= TABLE[0]) & (points[:, 0] <= TABLE[1])
                      & (points[:, 1] >= TABLE[2]) & (points[:, 1] <= TABLE[3]))
            if inside.any():
                hand_table_penetration = max(hand_table_penetration,
                                             float((TABLE[4] - points[inside, 2]).max()))
    phases = {"lift_and_far_reach": (18.4, 19.4), "far_placement": (19.4, 20.0),
              "table_stroke": (20.0, 21.0), "final_lift": (21.2, 21.8)}
    phase_report = {}
    for phase, (a, b) in phases.items():
        sl = slice(round((a - 8) * 50), round((b - 8) * 50) + 1)
        phase_report[phase] = {
            "brush_x_start_m": float(brush_x[sl.start]),
            "brush_x_end_m": float(brush_x[sl.stop - 1]),
            "bristle_clearance_mm_range": [float(clearances[sl].min() * 1000),
                                             float(clearances[sl].max() * 1000)],
        }
    arm_speed = np.abs(np.diff(new["joint_pos"][:, arm_cols], axis=0)) * 50
    wrist_cols = [names.index("right_wrist_" + part + "_joint") for part in ("roll", "pitch", "yaw")]
    report = {
        "status": "kinematic_geometry_not_physical_contact",
        "phases": phase_report,
        "far_brush_x_m_at_19_6s": float(brush_x[round((19.6 - 8) * 50)]),
        "old_brush_x_m_at_19_6s": float(old_object["position_w"][round((19.6 - 8) * 50), 0]),
        "stroke_brush_x_displacement_m": float(brush_x[round((21.0 - 8) * 50)] - brush_x[round((20.0 - 8) * 50)]),
        "stroke_clearance_mm_max": float(clearances[round((20.0 - 8) * 50):round((21.0 - 8) * 50) + 1].max() * 1000),
        "hand_table_penetration_mm_max": hand_table_penetration * 1000,
        "max_right_arm_speed_rad_s": float(arm_speed.max()),
        "max_wrist_axis_abs_deg": float(np.max(np.abs(new["joint_pos"][:, wrist_cols])) * 180 / np.pi),
        "root_legs_torso_left_arm_unchanged": bool(np.array_equal(new["joint_pos"][:, other_cols], old["joint_pos"][:, other_cols])
                                                     and np.array_equal(new["body_pos_w"][:, 0], old["body_pos_w"][:, 0])
                                                     and np.allclose(new["body_quat_w"][:, 0], old["body_quat_w"][:, 0], atol=1e-7)),
        "arm_outside_sweep_unchanged": bool(np.array_equal(new["joint_pos"][:start, arm_cols], old["joint_pos"][:start, arm_cols])
                                             and np.array_equal(new["joint_pos"][stop + 1:, arm_cols], old["joint_pos"][stop + 1:, arm_cols])),
    }
    failures = []
    if report["far_brush_x_m_at_19_6s"] < 1.08:
        failures.append("brush did not reach far table")
    if report["stroke_brush_x_displacement_m"] > -.2:
        failures.append("brush did not sweep back")
    if report["stroke_clearance_mm_max"] > 10:
        failures.append("brush did not stay near tabletop")
    if report["hand_table_penetration_mm_max"] > .5:
        failures.append("hand penetrated tabletop")
    if report["max_right_arm_speed_rad_s"] > 6.05:
        failures.append("arm velocity jump")
    if not report["root_legs_torso_left_arm_unchanged"] or not report["arm_outside_sweep_unchanged"]:
        failures.append("previously accepted motion changed outside sweep")
    report["failures"] = failures
    path = ROOT / "review/sweep_cycle_audit_v22.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError(", ".join(failures))


if __name__ == "__main__":
    main()
