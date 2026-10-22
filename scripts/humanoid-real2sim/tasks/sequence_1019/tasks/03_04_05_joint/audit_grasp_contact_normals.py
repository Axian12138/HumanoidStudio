"""Report contact patches and opposed normals; no force-closure claim."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import linprog

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--all-hold", action="store_true")
    a = p.parse_args()
    r = a.run
    c = r / "retarget/candidates" / a.candidate
    motion = np.load(c / "motion.npz")
    order = json.loads((c / "motion_order.json").read_text())
    hat = np.load(c / "black_ribbon_hat.npz")
    model = mujoco.MjModel.from_xml_path(str(r / "scene/03_04_05/robot_props_drawer_with_rack_early_registered.xml"))
    d = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in order["robot_joint_names"]])
    ha = model.joint("black_ribbon_hat_free").qposadr[0]
    hb = model.body("black_ribbon_hat").id
    rows = []
    times = np.arange(38.3, 47.641, 0.02) if a.all_hold else [38.3, 39, 40, 42, 44, 46, 47.64]
    for t in times:
        k = round((t - 27) * 50)
        d.qpos[:3] = motion["body_pos_w"][k, 0]
        d.qpos[3:7] = motion["body_quat_w"][k, 0]
        d.qpos[addr] = motion["joint_pos"][k]
        d.qpos[ha : ha + 3] = hat["position_w"][k]
        d.qpos[ha + 3 : ha + 7] = hat["quaternion_xyzw"][k][[3, 0, 1, 2]]
        mujoco.mj_forward(model, d)
        patches = []
        for contact in d.contact:
            b1, b2 = map(int, model.geom_bodyid[[contact.geom1, contact.geom2]])
            if hb not in (b1, b2):
                continue
            other = b2 if b1 == hb else b1
            name = model.body(other).name
            if not name.startswith("right_hand"):
                continue
            normal = np.array(contact.frame[:3]) * (1 if b2 == hb else -1)
            patches.append(
                {
                    "hand_body": name,
                    "position_w": contact.pos.tolist(),
                    "hand_to_hat_normal_w": normal.tolist(),
                    "signed_distance_m": float(contact.dist),
                    "sliding_friction": float(contact.friction[0]),
                }
            )
        thumb = [p for p in patches if "thumb" in p["hand_body"]]
        index = [p for p in patches if "index" in p["hand_body"]]
        dots = [float(np.dot(p["hand_to_hat_normal_w"], q["hand_to_hat_normal_w"])) for p in thumb for q in index]
        wrench = []
        for patch in patches:
            n = np.array(patch["hand_to_hat_normal_w"])
            tangent = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.8 else [0, 1, 0])
            tangent /= np.linalg.norm(tangent)
            other = np.cross(n, tangent)
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                f = n + patch["sliding_friction"] * (np.cos(angle) * tangent + np.sin(angle) * other)
                wrench.append(np.r_[f, np.cross(np.array(patch["position_w"]) - d.xipos[hb], f)])
        target = np.r_[-model.body_mass[hb] * model.opt.gravity, np.zeros(3)]
        lp = (
            linprog(np.ones(len(wrench)), A_eq=np.array(wrench).T, b_eq=target, bounds=(0, 10), method="highs")
            if wrench
            else None
        )
        rows.append(
            {
                "source_s": t,
                "static_gravity_wrench_feasible": bool(lp is not None and lp.success),
                "static_required_sum_normal_N": float(lp.fun) if lp is not None and lp.success else None,
                "static_wrench_mass_kg": float(model.body_mass[hb]),
                "patches": patches,
                "thumb_index_min_normal_dot": min(dots) if dots else None,
                "has_both_thumb_index_contacts": bool(thumb and index),
            }
        )
    result = {
        "status": "contact_geometry_diagnostic_not_force_closure",
        "candidate": a.candidate,
        "records": rows,
        "limitations": "Kinematic patches do not establish frictional force closure or robot-driven physical grasp. Source grip geometry and full enabled self-collision still require review.",
    }
    (c / ("grasp_contact_patches_all_hold.json" if a.all_hold else "grasp_contact_patches.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        {
            "frames": len(rows),
            "gravity_infeasible": sum(not x["static_gravity_wrench_feasible"] for x in rows),
            "missing_thumb_index": sum(not x["has_both_thumb_index_contacts"] for x in rows),
        }
    )


if __name__ == "__main__":
    main()
