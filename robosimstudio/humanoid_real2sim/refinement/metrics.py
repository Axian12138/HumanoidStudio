"""Contact diagnostics reconstructed at recorded physical states.

Sampling at control frames can miss substep peaks. Force reconstruction uses
MuJoCo forward dynamics at the saved qpos/qvel and saved controls; it is not a
sensor measurement and is explicitly distinguished from geometry-only checks.
"""

import numpy as np


def contact_diagnostics(model, qpos, qvel, controls, object_body, foot_bodies):
    import mujoco

    data = mujoco.MjData(model)
    rows = []
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    wrench = np.zeros(6)
    for frame, (q, v) in enumerate(zip(qpos, qvel, strict=True)):
        data.qpos[:], data.qvel[:] = q, v
        data.ctrl[:] = controls[min(max(frame - 1, 0), len(controls) - 1)]
        mujoco.mj_forward(model, data)
        depth, slip, left, right, support, other_robot = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        loaded_foot = False
        unintended = 0.0
        for c in range(data.ncon):
            contact = data.contact[c]
            depth = max(depth, -float(contact.dist))
            a, b = (int(model.geom_bodyid[g]) for g in contact.geom)
            mujoco.mj_contactForce(model, data, c, wrench)
            normal = max(float(wrench[0]), 0.0)
            if object_body in (a, b):
                other = b if a == object_body else a
                name = model.body(other).name
                if other != 0 and not any(part in name for part in ("_elbow_", "_wrist_", "_hand_")):
                    unintended += normal
                if any(f"left_{part}" in name for part in ("shoulder", "elbow", "wrist", "hand")):
                    left += normal
                elif any(f"right_{part}" in name for part in ("shoulder", "elbow", "wrist", "hand")):
                    right += normal
                elif other == 0:
                    support += normal
                else:
                    other_robot += normal
            if normal > 1.0 and contact.pos[2] < 0.05:
                for foot in foot_bodies:
                    if foot in (a, b):
                        loaded_foot = True
                        mujoco.mj_jac(model, data, jp, jr, contact.pos, foot)
                        slip = max(slip, float(np.linalg.norm((jp @ v)[:2])))
        rows.append([depth, slip, left, right, support, other_robot, float(loaded_foot), unintended])
    values = np.array(rows)
    return {
        "penetration_m": values[:, 0],
        "loaded_foot_contact_speed_m_s": values[:, 1],
        "left_object_normal_force_N": values[:, 2],
        "right_object_normal_force_N": values[:, 3],
        "environment_object_normal_force_N": values[:, 4],
        "other_robot_object_normal_force_N": values[:, 5],
        "loaded_foot_contact": values[:, 6].astype(bool),
        "unintended_object_normal_force_N": values[:, 7],
    }
