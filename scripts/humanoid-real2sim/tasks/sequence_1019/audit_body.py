"""Audit the Stage 3.1 candidate and task slices against geometric supports."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from scipy.spatial.transform import Rotation as R

from retarget3.common import FEET, segments
from retarget3.ground_stage import contact_geometry_values
from retarget3.motion_io import validate_motion
from retarget3.robot import MJCasadiRobot, mujoco


ROOT = example_dir(__file__)
OUT = ROOT / "retarget/ground_support_27_end"
PRIOR = ROOT / "retarget/body_prior_27_end"
VIZ = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization")
FPS = 50


def idx(time_s):
    return round((time_s - 27.0) * FPS)


def hand_collision_heights(robot, q, start, end):
    model, data = robot.m, robot.d
    meshes = []
    for g in range(model.ngeom):
        name = model.geom(g).name
        if not name.startswith("right_hand_") or model.geom_group[g] != 3:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        mesh = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        meshes.append((g, mesh))
    heights = []
    for k in range(start, end):
        row = q[k]
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]
        data.qpos[7:] = row[7:]
        mujoco.mj_forward(model, data)
        heights.append(min(float((data.geom_xpos[g, 2] + mesh @ data.geom_xmat[g].reshape(3, 3)[2]).min())
                           for g, mesh in meshes))
    return np.asarray(heights)


def main():
    robot = MJCasadiRobot()
    q = np.load(OUT / "intermediate/body_prior_supported_q.npy")
    with np.load(PRIOR / "intermediate/targets.npz") as z:
        targets = {key: z[key] for key in z.files}
    with np.load(OUT / "intermediate/ground_targets.npz") as z:
        ground = {key: z[key] for key in z.files}
    positions, quats = robot.forward(q, FEET + ["torso_link"])
    stance = targets["contacts"] & (ground["gate"] < .001)
    slip = []
    for side in range(2):
        slip.append(max((np.linalg.norm(positions[a:b, side]-positions[a, side], axis=1).max()
                         for a, b in segments(stance[:, side])), default=0))
    torso_up = R.from_quat(quats[:, 2][:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    torso_angle = np.rad2deg(np.arccos(np.clip(np.sum(torso_up*targets["torso"][:, :, 2], axis=1), -1, 1)))
    contacts = contact_geometry_values(robot, q)
    knees = []
    for start, end in [(28.9, 31.8), (71.0, 74.7)]:
        a, b = idx(start), idx(end)
        knees.append({
            "seconds": [start, end],
            "left_knee_gap_mm": (contacts[a:b, 0, 0]*1000).min().item(),
            "left_knee_gap_mm_max": (contacts[a:b, 0, 0]*1000).max().item(),
            "left_toe_gap_mm": (contacts[a:b, 0, 1]*1000).min().item(),
            "left_toe_gap_mm_max": (contacts[a:b, 0, 1]*1000).max().item(),
            "left_heel_height_mm_min": (contacts[a:b, 0, 2]*1000).min().item(),
        })
    hand_core = hand_collision_heights(robot, q, idx(84.5), idx(88.8))
    hand_exit = hand_collision_heights(robot, q, idx(88.8), idx(90.0))
    task_manifest = json.loads((ROOT / "segments.json").read_text())
    task_validation = {}
    task_overlap_checks = []
    for task in task_manifest["tasks"]:
        if task["status"] == "existing":
            continue
        path = ROOT / task["body_prior"]
        task_validation[task["id"]] = validate_motion(path)["frames"]
    candidate_tasks = [task for task in task_manifest["tasks"] if task["status"] != "existing"]
    for i, first in enumerate(candidate_tasks):
        for second in candidate_tasks[i+1:]:
            overlap_start = max(first["start_s"], second["start_s"])
            overlap_end = min(first["end_s"], second["end_s"])
            if overlap_end <= overlap_start:
                continue
            count = round((overlap_end-overlap_start)*FPS)
            a = round((overlap_start-first["start_s"])*FPS)
            b = round((overlap_start-second["start_s"])*FPS)
            with np.load(ROOT / first["body_prior"]) as f, np.load(ROOT / second["body_prior"]) as s:
                mismatch = float(np.max(abs(f["joint_pos"][a:a+count]-s["joint_pos"][b:b+count])))
            task_overlap_checks.append({"tasks": [first["id"], second["id"]],
                                        "seconds": [overlap_start, overlap_end],
                                        "max_joint_mismatch_rad": mismatch})
    with np.load(ROOT / "tasks/03_floor_strawberry_to_basket/source_sync.npz") as z:
        camera_sync_ms = {key: float(np.max(abs(z[f"camera_{key}_time_error_s"]))*1000)
                          for key in ("head", "left", "right")}
    with np.load(ROOT.parent / "brush_sweep_1019/retarget/body_prior_8_28/motion.npz") as old, np.load(OUT / "supported_motion.npz") as new:
        seam_root_mm = float(np.linalg.norm(old["body_pos_w"][950, 0]-new["body_pos_w"][0, 0])*1000)
        seam_nonhand_rad = float(np.max(abs(old["joint_pos"][950, :29]-new["joint_pos"][0, :29])))
    report = {
        "stage": "3.1 body prior candidate",
        "frames": len(q), "fps": FPS, "source_range_s_half_open": [27.0, 27.0+len(q)/FPS],
        "task_frames": task_validation,
        "task_overlap_checks": task_overlap_checks,
        "max_stance_foot_slip_mm": [float(v*1000) for v in slip],
        "min_visible_leg_floor_clearance_mm": float(contacts[:, :, 3:].min()*1000),
        "torso_up_error_deg_max": float(torso_angle.max()),
        "joint_velocity_ratio_max": float((abs(np.diff(q[:, 7:], axis=0))*FPS/robot.v_abs[6:]).max()),
        "knee_supports": knees,
        "right_hand_table_core_gap_mm": [float((hand_core.min()-.754)*1000), float((hand_core.max()-.754)*1000)],
        "right_hand_table_exit_gap_mm": [float((hand_exit.min()-.754)*1000), float((hand_exit.max()-.754)*1000)],
        "right_hand_surface_model": "1012 dining tabletop plane z=0.754 m; exact G1 right hand collision meshes; tabletop footprint check remains visual",
        "max_camera_sync_error_ms_task03": camera_sync_ms,
        "existing_brush_to_new_overlap_at_27s": {
            "root_position_gap_mm": seam_root_mm,
            "maximum_nonhand_joint_gap_rad": seam_nonhand_rad,
            "status": "separate accepted Stage 2 and Stage 3.1 candidates retain 1 s overlap; splice optimization remains pending",
        },
        "physics_validated": False,
    }
    failures = []
    if max(slip)*1000 > .1: failures.append("stance slip")
    if any(item["max_joint_mismatch_rad"] > 0 for item in task_overlap_checks): failures.append("task overlap mismatch")
    if contacts[:, :, 3:].min()*1000 < -.53: failures.append("leg floor penetration")
    if report["joint_velocity_ratio_max"] > 1.001: failures.append("joint speed")
    if any(item["left_knee_gap_mm_max"] > 1.1 for item in knees): failures.append("knee contact")
    if report["right_hand_table_core_gap_mm"][0] < 0: failures.append("hand tabletop penetration")
    report["failures"] = failures
    (OUT / "review/body_prior_quality.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("Stage 3.1 quality failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
