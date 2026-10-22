"""Export executed generalized states, never the reference targets, as motion."""

from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields


def unique_suffix(model, kind, suffix):
    count = model.njnt if kind == "joint" else model.nbody
    matches = [i for i in range(count) if getattr(model, kind)(i).name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"ambiguous {kind} {suffix}: {matches}")
    return matches[0]


def export_motion(model, qpos, qvel, joint_names, body_names, output, *, fps=50):
    import mujoco

    qpos, qvel = np.asarray(qpos), np.asarray(qvel)
    if qpos.shape != (len(qpos), model.nq) or qvel.shape != (len(qpos), model.nv):
        raise ValueError("inconsistent executed-state shape")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("nonfinite executed states")
    joints = [unique_suffix(model, "joint", n) for n in joint_names]
    bodies = [unique_suffix(model, "body", n) for n in body_names]
    data = mujoco.MjData(model)
    positions, quaternions, linear, angular = [], [], [], []
    for q, v in zip(qpos, qvel, strict=True):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[bodies].copy())
        quaternions.append(data.xquat[bodies].copy())
        velocities = np.zeros((len(bodies), 6))
        for i, body in enumerate(bodies):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocities[i], 0)
        angular.append(velocities[:, :3])
        linear.append(velocities[:, 3:])
    motion = MotionFields(
        tuple(joint_names),
        tuple(body_names),
        qpos[:, model.jnt_qposadr[joints]],
        qvel[:, model.jnt_dofadr[joints]],
        np.array(positions),
        np.array(quaternions),
        np.array(linear),
        np.array(angular),
        fps,
    )
    motion.save(Path(output))
    return motion
