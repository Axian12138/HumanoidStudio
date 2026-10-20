"""Apply existing ground solver to both observed left-knee contacts.

The upstream detector only uses knee height < 7 cm. The second support has a
measured human knee height of 8–10 cm, so it is added explicitly here.
"""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from retarget3.common import FEET
from retarget3.ground_stage import GroundOptimizer, contact_geometry_values, mesh_points, smooth_gate
from retarget3.motion_io import export_motion
from retarget3.robot import MJCasadiRobot
from retarget3.robot import ca


ROOT = example_dir(__file__)
PRIOR = ROOT / "retarget/body_prior_27_end"
OUT = ROOT / "retarget/ground_support_27_end"
START_S = 27.0
FPS = 50


def index(time_s):
    return round((time_s - START_S) * FPS)


def load_npz(path):
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def add_second_knee(robot, data):
    start, stop = index(70.92), index(74.78)
    n = len(data["prior"])
    gate = smooth_gate(n, start, stop, ramp=40)
    data["gate"][:, 0] = np.maximum(data["gate"][:, 0], gate)
    data["core"][start:stop, 0] = True
    with np.load(PRIOR / "intermediate/canonical.npz") as stored:
        offset = stored["ankle_to_sole_offsets"][0]
    source_toe = data["human"][:, 5] + np.einsum("tij,j->ti", data["human_rot"][:, 5], offset)
    old_pos, old_quat = robot.forward(data["prior"], FEET)
    old_rot = R.from_quat(old_quat[:, 0][:, [1, 2, 3, 0]]).as_matrix()
    baseclear = contact_geometry_values(robot, data["prior"])
    for k in np.flatnonzero(gate > 0):
        amount = gate[k]
        desired = data["human_rot"][k, 5]
        data["feet_rot"][k, 0] = Slerp(
            [0, 1], R.from_matrix(np.stack([old_rot[k], desired]))
        )([amount]).as_matrix()[0]
        wanted = np.r_[source_toe[k, :2], .01] - desired @ np.array([.132, 0, -.025])
        data["feet"][k, 0] = (1 - amount) * old_pos[k, 0] + amount * wanted
    data["knee_upper"][:, 0] = np.where(gate > 0,
        (1 - gate) * np.maximum(baseclear[:, 0, 0], 0) + gate * .001 + .00001,
        data["knee_upper"][:, 0])
    data["toe_upper"][:, 0] = np.where(gate > 0,
        (1 - gate) * np.maximum(baseclear[:, 0, 1], 0) + gate * .001 + .00001,
        data["toe_upper"][:, 0])
    data["heel_lower"][:, 0] = np.maximum(data["heel_lower"][:, 0], gate * .08 - .0005)
    data["toe_xy"][:, 0] = source_toe[:, :2]
    np.savez_compressed(OUT / "intermediate/ground_targets.npz", **data)
    schedule_path = OUT / "review/contact_schedule.json"
    schedule = json.loads(schedule_path.read_text())
    schedule["intervals"].append({"side": "left", "start_frame": start, "end_frame": stop,
        "start_s": 70.92, "end_s": 74.78, "source": "video + human knee < 0.11 m"})
    schedule_path.write_text(json.dumps(schedule, indent=2) + "\n")
    return data


def project_right_stance(robot, output, prior, seam_frames):
    names = ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
             "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
    ids = np.asarray([robot.dof_name_to_index[name] for name in names])
    x = ca.SX.sym("right_leg", len(ids))
    row = ca.SX.sym("row", robot.nq)
    target_p = ca.SX.sym("target_p", 3)
    target_r = ca.SX.sym("target_r", 9)
    pose = ca.SX(row)
    pose[ids.tolist()] = x
    foot = FEET[1]
    sole_points = mesh_points(robot.m, foot)
    sole_world = robot.world_pos(foot, pose) + robot.world_rotmat(foot, pose) @ ca.DM(sole_points.T)
    sole_min = -1e-5*ca.logsumexp(-sole_world[2, :].T/1e-5)
    cost = 1e3*ca.sumsqr(ca.vec(robot.world_rotmat(foot, pose))-target_r)
    cost += .1*ca.sumsqr(x-row[ids.tolist()])
    solver = ca.nlpsol("stance_seam", "ipopt", {"x": x,
        "p": ca.vertcat(row, target_p, target_r), "f": cost,
        "g": ca.vertcat(robot.world_pos(foot, pose)-target_p, sole_min)},
        {"print_time": False, "ipopt": {"print_level": 0, "max_iter": 300, "tol": 1e-8}})
    target_pos, target_quat = robot.forward(prior[seam_frames], [foot])
    target_rot = R.from_quat(target_quat[:, 0][:, [1, 2, 3, 0]]).as_matrix()
    for j, k in enumerate(seam_frames):
        params = np.r_[output[k], target_pos[j, 0], target_rot[j].ravel(order="F")]
        sol = solver(x0=output[k, ids], p=params,
                     lbx=robot.q_lower[ids], ubx=robot.q_upper[ids],
                     lbg=[0, 0, 0, -.00049], ubg=[0, 0, 0, np.inf])
        if not solver.stats()["success"]:
            raise RuntimeError(f"Right-foot seam projection failed at {k}")
        output[k, ids] = np.asarray(sol["x"]).ravel()


def main():
    robot = MJCasadiRobot()
    data = add_second_knee(robot, load_npz(OUT / "intermediate/ground_targets.npz"))
    cfg = {"fps": FPS}
    optimizer = GroundOptimizer(robot, cfg, OUT)
    full = data["seed"].copy()
    reports = []
    seam_frames = []
    for start_s, stop_s in [(28.0, 32.7), (70.1, 75.6)]:
        start, end = index(start_s), index(stop_s)
        result = optimizer.solve(data, start, end)
        if result is None:
            raise RuntimeError("Unexpected solver checkpoint")
        if len(result) != end - start:
            raise RuntimeError("Partial ground result is incomplete")
        # Use 0.2 s feather only outside the core knee-contact interval.
        weight = np.ones(len(result))
        ramp = min(10, len(result) // 4)
        weight[:ramp] = np.linspace(0, 1, ramp)
        weight[-ramp:] = np.linspace(1, 0, ramp)
        seam_frames.extend(k for k in range(start, end)
                           if 0 < weight[k-start] < 1 and data["contacts"][k, 1])
        full[start:end, :3] = (1-weight[:, None])*full[start:end, :3] + weight[:, None]*result[:, :3]
        q_old = R.from_quat(full[start:end, 3:7])
        q_new = R.from_quat(result[:, 3:7])
        full[start:end, 3:7] = (q_old * R.from_rotvec((q_old.inv()*q_new).as_rotvec()*weight[:, None])).as_quat()
        full[start:end, 7:] = (1-weight[:, None])*full[start:end, 7:] + weight[:, None]*result[:, 7:]
        reports.append({"window_s": [start_s, stop_s], "frames": len(result)})
    project_right_stance(robot, full, data["prior"], np.asarray(seam_frames, dtype=int))
    np.save(OUT / "intermediate/support_refined_q.npy", full)
    export_motion(OUT / "motion.npz", robot, full, FPS)
    geometry = contact_geometry_values(robot, full)
    for item in [(28.9, 31.8), (71.0, 74.7)]:
        a,b = map(index, item)
        reports.append({"knee_core_s": list(item), "left_knee_gap_mm": [
            float(geometry[a:b, 0, 0].min()*1000), float(geometry[a:b, 0, 0].max()*1000)],
            "left_toe_gap_mm": [float(geometry[a:b, 0, 1].min()*1000), float(geometry[a:b, 0, 1].max()*1000)]})
    (OUT / "review/support_refinement.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
