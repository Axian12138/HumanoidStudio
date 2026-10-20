"""Compose right-knee ground contact with the accepted v20 brush interaction.

The ground solver changes root, legs and waist. Right-arm IK preserves the
v20 world wrist path so the brush sweep does not drift with the kneeling body.
No physics or contact force is inferred from this kinematic operation.
"""

from dataclasses import replace
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


ROOT = example_dir(__file__)
SOURCE = ROOT / "retarget/brush_sweep_kinematic_v20"
GROUND = ROOT / "retarget/right_knee_ground_v3/intermediate/debug_370_830_q.npy"
OUT = ROOT / "retarget/brush_sweep_kinematic_v21_knee"
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
START, STOP = 370, 830  # local 8–28 s clip; global 15.4–24.6 s


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def main():
    original = MotionFields.load(SOURCE / "motion.npz", ORDER)
    ground = np.load(GROUND)
    assert ground.shape == (STOP - START, 50)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    joint_addr = np.asarray([model.joint(name).qposadr[0] for name in original.joint_names])
    lower_cols = np.asarray([
        i for i, name in enumerate(original.joint_names)
        if any(part in name for part in ("_hip_", "_knee_", "_ankle_"))
    ])
    arm_cols = np.asarray([
        original.joint_names.index("right_" + part + "_joint")
        for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                     "wrist_roll", "wrist_pitch", "wrist_yaw")
    ])
    arm_addr = joint_addr[arm_cols]
    bounds = np.asarray([model.joint(original.joint_names[i]).range for i in arm_cols]).T
    bounds[0, 4:] = np.maximum(bounds[0, 4:], -np.deg2rad(35))
    bounds[1, 4:] = np.minimum(bounds[1, 4:], np.deg2rad(35))
    wrist_body = model.body("right_wrist_yaw_link").id
    wrist_motion = original.body_names.index("right_wrist_yaw_link")

    roots = original.body_pos_w[:, 0].astype(float).copy()
    quats = original.body_quat_w[:, 0].astype(float).copy()
    joints = original.joint_pos.astype(float).copy()
    weights = np.ones(STOP - START)
    ramp = 20
    weights[:ramp] = smooth(np.arange(ramp) / (ramp - 1))
    weights[-ramp:] = smooth(np.arange(ramp)[::-1] / (ramp - 1))
    for index, k in enumerate(range(START, STOP)):
        weight = weights[index]
        row = ground[index]
        roots[k] = (1 - weight) * roots[k] + weight * row[:3]
        old = Rotation.from_quat(quats[k, [1, 2, 3, 0]])
        new = Rotation.from_quat(row[3:7])
        quats[k] = (old * Rotation.from_rotvec((old.inv() * new).as_rotvec() * weight)).as_quat()[[3, 0, 1, 2]]
        for col in lower_cols:
            addressed = joint_addr[col]
            joints[k, col] = (1 - weight) * joints[k, col] + weight * row[addressed]

    # Joint-space blending can lower a toe a millimetre below the plane even
    # when both endpoint poses satisfy the floor constraint. Correct only the
    # entry/exit transitions; the knee-contact core remains the solver output.
    floor_meshes = []
    for geom in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom]).name
        if body_name not in ("left_ankle_roll_link", "right_ankle_roll_link",
                             "left_knee_link", "right_knee_link"):
            continue
        if model.geom_group[geom] == 3 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        floor_meshes.append((geom, vertices.copy()))
    floor_lifts = []
    for k in list(range(START, 420)) + list(range(709, STOP)):
        data.qpos[:3] = roots[k]
        data.qpos[3:7] = quats[k]
        data.qpos[joint_addr] = joints[k]
        mujoco.mj_forward(model, data)
        minimum = min(float((vertices @ data.geom_xmat[geom].reshape(3, 3)[2]
                             + data.geom_xpos[geom, 2]).min())
                      for geom, vertices in floor_meshes)
        lift = max(0.0, -0.00045 - minimum)
        if lift:
            roots[k, 2] += lift
            floor_lifts.append((k, lift))

    wrist_errors = []
    previous = original.joint_pos[START, arm_cols].astype(float)
    for k in range(START, STOP):
        if weights[k - START] == 0:
            continue
        target_pos = original.body_pos_w[k, wrist_motion]
        target_rot = Rotation.from_quat(original.body_quat_w[k, wrist_motion, [1, 2, 3, 0]])
        data.qpos[:3] = roots[k]
        data.qpos[3:7] = quats[k]
        data.qpos[joint_addr] = joints[k]
        prior = original.joint_pos[k, arm_cols].astype(float)

        def residual(angles):
            data.qpos[arm_addr] = angles
            mujoco.mj_forward(model, data)
            current_rot = Rotation.from_matrix(data.xmat[wrist_body].reshape(3, 3))
            return np.r_[100 * (data.xpos[wrist_body] - target_pos),
                         10 * (target_rot.inv() * current_rot).as_rotvec(),
                         .08 * (angles - prior)]

        lower = np.maximum(bounds[0], previous - 0.12)
        upper = np.minimum(bounds[1], previous + 0.12)
        solution = least_squares(
            residual, np.clip(previous, lower + 1e-7, upper - 1e-7),
            bounds=(lower, upper), max_nfev=120,
        )
        joints[k, arm_cols] = solution.x
        previous = solution.x
        data.qpos[arm_addr] = solution.x
        mujoco.mj_forward(model, data)
        current_rot = Rotation.from_matrix(data.xmat[wrist_body].reshape(3, 3))
        wrist_errors.append((k, float(np.linalg.norm(data.xpos[wrist_body] - target_pos)),
                             float(np.linalg.norm((target_rot.inv() * current_rot).as_rotvec()))))
    body_pos = original.body_pos_w.copy()
    body_quat = original.body_quat_w.copy()
    body_pos[:, 0] = roots
    body_quat[:, 0] = quats
    candidate = refresh_fk(replace(original, joint_pos=joints, body_pos_w=body_pos,
                                   body_quat_w=body_quat), MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    candidate.save(OUT / "motion.npz")

    with np.load(SOURCE / "object_estimate.npz", allow_pickle=False) as source_object:
        position = source_object["position_w"].copy()
        orientation = source_object["quaternion_xyzw"].copy()
        extra = {name: source_object[name] for name in source_object.files
                 if name not in ("position_w", "quaternion_xyzw")}
    old_wrist = Rotation.from_quat(original.body_quat_w[START:STOP, wrist_motion][:, [1, 2, 3, 0]])
    new_wrist = Rotation.from_quat(candidate.body_quat_w[START:STOP, wrist_motion][:, [1, 2, 3, 0]])
    local_position = old_wrist.inv().apply(position[START:STOP] - original.body_pos_w[START:STOP, wrist_motion])
    local_orientation = old_wrist.inv() * Rotation.from_quat(orientation[START:STOP])
    position[START:STOP] = candidate.body_pos_w[START:STOP, wrist_motion] + new_wrist.apply(local_position)
    orientation[START:STOP] = (new_wrist * local_orientation).as_quat()
    np.savez_compressed(OUT / "object_estimate.npz", position_w=position,
                        quaternion_xyzw=orientation, **extra)

    errors = np.asarray(wrist_errors)
    report = {
        "status": "kinematic_knee_and_hoi_composed_candidate",
        "source_motion_sha256": sha256(SOURCE / "motion.npz"),
        "ground_q_sha256": sha256(GROUND),
        "motion_sha256": sha256(OUT / "motion.npz"),
        "ground_window_global_s": [8 + START / 50, 8 + STOP / 50],
        "right_knee_core_global_s": [16.4, 22.18],
        "wrist_position_error_m_max": float(errors[:, 1].max()),
        "wrist_orientation_error_deg_max": float(np.rad2deg(errors[:, 2].max())),
        "wrist_error_max_frame_local": int(errors[np.argmax(errors[:, 1]), 0]),
        "right_wrist_axis_abs_max_deg": {
            part: float(np.max(np.abs(np.rad2deg(joints[START:STOP, col]))))
            for part, col in zip(("roll", "pitch", "yaw"), arm_cols[4:])
        },
        "transition_floor_lift_m_max": max((item[1] for item in floor_lifts), default=0.0),
        "stage4_stage5": "skipped_by_user",
    }
    (OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
