"""Compare a local torso refinement and native MuJoCo self contacts at 100 Hz."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

    p = argparse.ArgumentParser()
    for key in ["run", "motion", "seed", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    motion, seed = [MotionFields.load(path, order) for path in [a.motion, a.seed]]
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    data = mujoco.MjData(model)
    addr = np.array([model.joint(n).qposadr[0] for n in motion.joint_names])
    waist = [i for i, n in enumerate(motion.joint_names) if n.startswith("waist_")]
    legs = [i for i, n in enumerate(motion.joint_names) if any(x in n for x in ["hip_", "knee_", "ankle_"])]
    records = []
    for sample in range(2 * len(motion.joint_pos) - 1):
        frame, alpha = sample // 2, (sample % 2) / 2
        nxt = min(frame + 1, len(motion.joint_pos) - 1)
        contacts = []
        for current in [seed, motion]:
            data.qpos[:3] = (1 - alpha) * current.body_pos_w[frame, 0] + alpha * current.body_pos_w[nxt, 0]
            qa, qb = current.body_quat_w[frame, 0], current.body_quat_w[nxt, 0]
            if qa @ qb < 0:
                qb = -qb
            quat = (1 - alpha) * qa + alpha * qb
            data.qpos[3:7] = quat / np.linalg.norm(quat)
            data.qpos[addr] = (1 - alpha) * current.joint_pos[frame] + alpha * current.joint_pos[nxt]
            mujoco.mj_forward(model, data)
            pairs = {}
            for c in data.contact[: data.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if model.geom_bodyid[g1] == 0 or model.geom_bodyid[g2] == 0:
                    continue
                pair = tuple(sorted([model.geom(g1).name or str(g1), model.geom(g2).name or str(g2)]))
                pairs[pair] = min(pairs.get(pair, 0), float(c.dist))
            contacts.append(pairs)
        for pair, distance in contacts[1].items():
            if distance < -0.002:
                old = contacts[0].get(pair, 0)
                records.append(
                    {
                        "source_s": 108 + sample / 100,
                        "pair": pair,
                        "distance_m": distance,
                        "seed_distance_m": old,
                        "new_or_deeper_over_2mm": old >= -0.002 or distance < old - 0.002,
                    }
                )
    seams = []
    for t in [119, 125.2]:
        k = round((t - 108) * 50)
        seams.append(
            {
                "source_s": t,
                "max_position_delta_to_seed_rad": float(np.max(abs(motion.joint_pos[k] - seed.joint_pos[k]))),
                "max_incoming_step_rad": float(np.max(abs(motion.joint_pos[k] - motion.joint_pos[k - 1]))),
                "max_outgoing_step_rad": float(np.max(abs(motion.joint_pos[k + 1] - motion.joint_pos[k]))),
            }
        )
    out = {
        "scope": "native MuJoCo convex collision geoms and authored filters; not a triangle mesh containment proof",
        "waist_names": [motion.joint_names[i] for i in waist],
        "max_waist_delta_rad": np.max(abs(motion.joint_pos[:, waist] - seed.joint_pos[:, waist]), axis=0).tolist(),
        "root_position_max_delta_m": float(np.max(abs(motion.body_pos_w[:, 0] - seed.body_pos_w[:, 0]))),
        "root_quaternion_max_delta": float(np.max(abs(motion.body_quat_w[:, 0] - seed.body_quat_w[:, 0]))),
        "leg_joint_max_delta_rad": float(np.max(abs(motion.joint_pos[:, legs] - seed.joint_pos[:, legs]))),
        "first50_joint_max_delta_rad": float(np.max(abs(motion.joint_pos[:50] - seed.joint_pos[:50]))),
        "seams": seams,
        "self_contact_records_over_2mm": len(records),
        "new_or_deeper_records": sum(x["new_or_deeper_over_2mm"] for x in records),
        "records": records,
    }
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "records"}))


if __name__ == "__main__":
    main()
