"""Refine the left-toe-to-knee support transition before the basket return.

Keep the existing 71–75 s knee/toe ground solution and the basket motion.
Only the left leg is solved in the entry window; the right stance leg, root,
arms, and fingers retain the current candidate.
"""

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


HERE = example_dir(__file__)
SOURCE = HERE / "hoi_v4_59_76"
OUTPUT = HERE / "hoi_v5_kneel_59_76"
ORDER = HERE.parents[1] / "retarget/ground_support_27_end/sidecars/motion_order.json"
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
             "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
FPS = 50
LEG = tuple("left_" + name + "_joint" for name in (
    "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
))


def frame(t: float) -> int:
    return round((t - 59.0) * FPS)


def mesh_points(model: mujoco.MjModel, name: str) -> tuple[int, np.ndarray]:
    body = model.body(name).id
    pieces = []
    for geom in range(model.ngeom):
        if (model.geom_bodyid[geom] != body or model.geom_group[geom] == 3
                or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = model.geom_dataid[geom]
        mesh = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pieces.append(Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).apply(mesh)
                      + model.geom_pos[geom])
    return body, np.concatenate(pieces)


def main() -> None:
    source = MotionFields.load(SOURCE / "motion.npz", ORDER)
    prior = MotionFields.load(HERE / "body_prior.npz", ORDER)
    with np.load(HERE / "review_59_76/human_motion_world.npz", allow_pickle=False) as stored:
        human_names = list(map(str, stored["joint_names"]))
        human_sole = stored["positions_w"][:, human_names.index("left_sole")].astype(float)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = np.asarray([model.joint(name).qposadr[0] for name in source.joint_names])
    cols = np.asarray([source.joint_names.index(name) for name in LEG])
    leg_addr = addresses[cols]
    lower = np.asarray([model.joint(name).range[0] for name in LEG]) + 1e-5
    upper = np.asarray([model.joint(name).range[1] for name in LEG]) - 1e-5
    foot, foot_mesh = mesh_points(model, "left_ankle_roll_link")
    knee, knee_mesh = mesh_points(model, "left_knee_link")
    toe_vertices = foot_mesh[:, 0] > .105
    heel_vertices = foot_mesh[:, 0] < -.035
    knee_vertices = knee_mesh[:, 2] > -.07
    toe_local = np.array([.132, 0.0, -.025])

    def pose(k: int, leg: np.ndarray | None = None) -> None:
        data.qpos[:3] = source.body_pos_w[k, 0]
        data.qpos[3:7] = source.body_quat_w[k, 0]
        data.qpos[addresses] = source.joint_pos[k]
        if leg is not None:
            data.qpos[leg_addr] = leg
        mujoco.mj_forward(model, data)

    def geometry() -> tuple[np.ndarray, float, float, float, float]:
        rotation = data.xmat[foot].reshape(3, 3)
        world_foot = foot_mesh @ rotation.T + data.xpos[foot]
        world_knee = knee_mesh @ data.xmat[knee].reshape(3, 3).T + data.xpos[knee]
        toe_xy = data.xpos[foot, :2] + (rotation @ toe_local)[:2]
        return (toe_xy, float(world_foot[toe_vertices, 2].min()),
                float(world_foot[heel_vertices, 2].min()),
                float(world_knee[knee_vertices, 2].min()),
                float(min(world_foot[:, 2].min(), world_knee[:, 2].min())))

    # Calibrate the source sole's horizontal marker offset at a frame whose
    # existing robot knee and toe are already simultaneously grounded.
    pose(frame(71.0))
    toe_bias = geometry()[0] - human_sole[frame(71.0), :2]
    joint = source.joint_pos.astype(float).copy()
    start, stop = frame(69.0), frame(71.0)
    solve_frames = np.arange(start, stop)
    pose(start)
    start_toe_xy = geometry()[0]
    fit_log = []
    previous = source.joint_pos[start - 1, cols].astype(float)
    for k in solve_frames:
        t = 59 + k / FPS
        baseline = prior.joint_pos[k, cols].astype(float)
        pose(int(k), baseline)
        knee_xy = data.xpos[knee, :2].copy()
        toe_mix = np.clip((t - 69.0) / .5, 0, 1)
        toe_target_xy = ((1 - toe_mix) * start_toe_xy
                         + toe_mix * (human_sole[k, :2] + toe_bias))
        # The toe touches before the knee. The desired knee gap falls
        # monotonically, while heel clearance rises as the ankle folds.
        toe_target_z = np.interp(t, [69.0, 69.4, 70.9], [.035, .0005, .0005])
        heel_target = np.interp(t, [69.0, 69.5, 69.8, 70.9],
                                [.036, .07, .09, .165])
        knee_target_z = np.interp(t, [69.0, 69.4, 69.8, 70.3, 70.9],
                                  [.246, .215, .14, .07, .001])

        def residual(q: np.ndarray) -> np.ndarray:
            data.qpos[leg_addr] = q
            mujoco.mj_forward(model, data)
            toe_xy, toe_z, heel_z, knee_z, floor_z = geometry()
            return np.r_[
                90 * (toe_xy - toe_target_xy),
                400 * (toe_z - toe_target_z),
                130 * (knee_z - knee_target_z),
                110 * (heel_z - heel_target),
                250 * min(floor_z + .0005, 0),
                5 * (data.xpos[knee, :2] - knee_xy),
                .3 * (q - baseline),
                3 * (q - previous),
            ]

        frame_lower = np.maximum(lower, previous - .12)
        frame_upper = np.minimum(upper, previous + .12)
        frame_lower[2] = max(frame_lower[2], baseline[2] - .2)
        frame_upper[2] = min(frame_upper[2], baseline[2] + .2)
        if k >= stop - 5:
            join_target = source.joint_pos[stop, cols]
            remaining = stop - k
            frame_lower = np.maximum(frame_lower, join_target - .115 * remaining)
            frame_upper = np.minimum(frame_upper, join_target + .115 * remaining)
        result = least_squares(residual, np.clip(previous, frame_lower + 1e-5, frame_upper - 1e-5),
                               bounds=(frame_lower, frame_upper), max_nfev=100)
        residual(result.x)
        toe_xy, toe_z, heel_z, knee_z, floor_z = geometry()
        joint[k, cols] = result.x
        previous = result.x
        fit_log.append({"time_s": float(t), "toe_xy_m": toe_xy.tolist(),
                        "toe_target_xy_m": toe_target_xy.tolist(), "toe_gap_m": toe_z,
                        "heel_gap_m": heel_z, "knee_gap_m": knee_z,
                        "minimum_leg_gap_m": floor_z})

    # The contact core from 71 s onward is copied exactly. Feather the first
    # few solved frames into the unchanged 68–69 s action.
    for k in range(start, start + 12):
        amount = (k - start + 1) / 12
        amount = amount * amount * (3 - 2 * amount)
        joint[k, cols] = (1 - amount) * source.joint_pos[k, cols] + amount * joint[k, cols]
    active = np.arange(start, stop + 1)
    candidate = refresh_fk(replace(source, joint_pos=joint), MODEL)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate.save(OUTPUT / "motion.npz")
    with np.load(SOURCE / "object_estimate.npz", allow_pickle=False) as stored:
        np.savez_compressed(OUTPUT / "object_estimate.npz", **{key: stored[key] for key in stored.files})

    dense = []
    for k in active:
        pose(int(k), joint[k, cols])
        toe_xy, toe_z, heel_z, knee_z, floor_z = geometry()
        dense.append((59 + k / FPS, *toe_xy, toe_z, heel_z, knee_z, floor_z))
    dense = np.asarray(dense)
    core = dense[(dense[:, 0] >= 69.6) & (dense[:, 0] <= 70.9)]
    report = {
        "status": "stage35_candidate_pending_user_review",
        "source": str(SOURCE / "motion.npz"),
        "modified_window_s": [69.0, 71.0],
        "human_sole_to_robot_toe_xy_bias_m": toe_bias.tolist(),
        "toe_gap_69_6_70_9_m_range": [float(core[:, 3].min()), float(core[:, 3].max())],
        "heel_gap_70_9_m": float(dense[-6, 4]),
        "knee_gap_70_9_m": float(dense[-6, 5]),
        "minimum_leg_gap_69_0_71_0_m": float(dense[:, 6].min()),
        "toe_xy_69_6_m": core[0, 1:3].tolist(),
        "toe_xy_70_9_m": core[-1, 1:3].tolist(),
        "max_left_leg_joint_speed_rad_s": float(np.max(np.abs(np.diff(joint[:, cols], axis=0)) * FPS)),
        "fit_log": fit_log,
        "dense_contact_log": dense.tolist(),
    }
    (OUTPUT / "kneel_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("fit_log", "dense_contact_log")}, indent=2))


if __name__ == "__main__":
    main()
