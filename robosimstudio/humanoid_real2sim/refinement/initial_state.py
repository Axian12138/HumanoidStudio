"""Bounded collision projection of an explicitly selected initial posture.

Only selected actuated joints move. Free roots, objects, physical parameters,
and all other joints remain fixed. This repairs an initialization candidate;
it does not establish contact equilibrium or dynamic feasibility.
"""

import numpy as np


def project_initial(model, qpos, joint_ids, *, max_change=0.35, max_evaluations=100):
    import mujoco
    from scipy.optimize import least_squares

    joint_ids = np.asarray(joint_ids, dtype=int)
    if not len(joint_ids) or len(np.unique(joint_ids)) != len(joint_ids):
        raise ValueError("select distinct scalar joints")
    if not np.all(np.isin(model.jnt_type[joint_ids], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE])):
        raise ValueError("cannot project a free or ball joint")
    adr = model.jnt_qposadr[joint_ids]
    qpos = np.asarray(qpos, dtype=float)
    initial = qpos[adr].copy()
    data = mujoco.MjData(model)

    def residual(values):
        data.qpos[:] = qpos
        data.qpos[adr] = values
        mujoco.mj_forward(model, data)
        depth = np.zeros(model.ngeom)
        for c in range(data.ncon):
            contact = data.contact[c]
            for geom in contact.geom:
                depth[geom] = max(depth[geom], -float(contact.dist))
        return np.r_[depth, 0.0001 * (values - initial)]

    before = residual(initial)[: model.ngeom].max()
    lower = np.maximum(model.jnt_range[joint_ids, 0], initial - max_change)
    upper = np.minimum(model.jnt_range[joint_ids, 1], initial + max_change)
    result = least_squares(
        residual,
        np.clip(initial, lower, upper),
        bounds=(lower, upper),
        max_nfev=max_evaluations,
        diff_step=1e-4,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    after = residual(result.x)[: model.ngeom].max()
    candidate = qpos.copy()
    if after < before:
        candidate[adr] = result.x
    return candidate, {
        "max_penetration_before_m": float(before),
        "max_penetration_after_m": float(min(before, after)),
        "max_joint_change_rad": float(np.max(np.abs(candidate[adr] - initial))),
        "evaluations": int(result.nfev),
        "solver_success": bool(result.success),
        "dynamic_equilibrium_verified": False,
    }
