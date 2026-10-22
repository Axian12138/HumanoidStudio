"""Audit the bootstrap with actual authored collision meshes; never infer grasp success."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", default="03_04_05_joint_v0")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start", type=float, default=27)
    p.add_argument("--end", type=float, default=76)
    p.add_argument("--with-rack", action="store_true")
    p.add_argument("--registered-rack", action="store_true")
    a = p.parse_args()
    base = a.run / "retarget/candidates" / a.candidate
    if a.registered_rack:
        a.with_rack = True
    scene_name = (
        "robot_props_drawer_with_rack_early_registered.xml"
        if a.registered_rack
        else ("robot_props_drawer_with_rack.xml" if a.with_rack else "robot_props_drawer.xml")
    )
    model = mujoco.MjModel.from_xml_path(str(a.run / "scene/03_04_05" / scene_name))
    data = mujoco.MjData(model)
    motion = np.load(base / "motion.npz")
    order = json.loads((base / "motion_order.json").read_text())
    addresses = np.array([model.joint(n).qposadr[0] for n in order["robot_joint_names"]])
    names = ["floor_strawberry", "black_ribbon_hat", "round_laundry_basket"]
    objects = {n: np.load(base / (n + ".npz")) for n in names}
    body_ids = {model.body(n).id: n for n in names}
    robot_ids = {model.body(n).id for n in order["robot_body_names"]}
    free = {n: model.joint(n + "_free").qposadr[0] for n in names}
    drawer = np.load(base / "drawer_slide.npz")["position"]
    da = model.joint("drawer_slide").qposadr[0]
    contact_bodies = {n: {} for n in names}
    records = []
    rack_records = []
    rack_id = model.body("coat_rack_support").id if a.with_rack else -1
    rack_minimum = {}
    counts = dict.fromkeys(names, 0)
    minimum = dict.fromkeys(names, 0.0)
    sampled_frames = range(
        max(0, round((a.start - 27) * 50)), min(len(motion["joint_pos"]), round((a.end - 27) * 50)), a.stride
    )
    for k in sampled_frames:
        data.qpos[:3] = motion["body_pos_w"][k, 0]
        data.qpos[3:7] = motion["body_quat_w"][k, 0]
        data.qpos[addresses] = motion["joint_pos"][k]
        data.qpos[da] = drawer[k]
        for n, z in objects.items():
            i = free[n]
            data.qpos[i : i + 3] = z["position_w"][k]
            data.qpos[i + 3 : i + 7] = z["quaternion_xyzw"][k][[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        touched = set()
        for contact in data.contact:
            b1, b2 = map(int, model.geom_bodyid[[contact.geom1, contact.geom2]])
            if rack_id in (b1, b2):
                other_rack = b2 if b1 == rack_id else b1
                rack_name = model.body(other_rack).name
                rack_minimum[rack_name] = min(rack_minimum.get(rack_name, 0), float(contact.dist))
                if contact.dist < -0.005:
                    rack_records.append(
                        {
                            "frame": k,
                            "source_s": 27 + k / 50,
                            "other_body": rack_name,
                            "distance_m": float(contact.dist),
                        }
                    )
            obj = body_ids.get(b1, body_ids.get(b2))
            other = b2 if b1 in body_ids else b1
            if obj is None or other not in robot_ids:
                continue
            contact_bodies[obj][model.body(other).name] = contact_bodies[obj].get(model.body(other).name, 0) + 1
            touched.add(obj)
            minimum[obj] = min(minimum[obj], float(contact.dist))
            if contact.dist < -0.005:
                records.append(
                    {
                        "frame": k,
                        "source_s": 27 + k / 50,
                        "object": obj,
                        "robot_body": model.body(other).name,
                        "distance_m": float(contact.dist),
                    }
                )
        for n in touched:
            counts[n] += 1
    result = {
        "status": "diagnostic_not_passed",
        "frames": len(sampled_frames),
        "total_motion_frames": len(motion["joint_pos"]),
        "window_s": [a.start, a.end],
        "sample_stride": a.stride,
        "geometry": "authored robot/prop collisions, MuJoCo FK only",
        "contact_frames": counts,
        "contact_body_samples": contact_bodies,
        "minimum_contact_distance_m": minimum,
        "deep_intersections_over_5mm": records,
        "rack_minimum_contact_distance_m": rack_minimum,
        "rack_deep_intersections_over_5mm": rack_records,
        "limitations": [
            "static room collision not in this audit",
            "no between-frame sweep",
            "contact does not prove enclosure or stable grasp",
            "grasps use unreviewed fixed transforms",
        ],
    }
    (
        base
        / ("collision_diagnostic.json" if a.start == 27 and a.end == 76 else f"collision_{a.start:g}_{a.end:g}.json")
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k not in ("deep_intersections_over_5mm", "rack_deep_intersections_over_5mm")
        }
    )
    print("deep intersections", len(records))


if __name__ == "__main__":
    main()
