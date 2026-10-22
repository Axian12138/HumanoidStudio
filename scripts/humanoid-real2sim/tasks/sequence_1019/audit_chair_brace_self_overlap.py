"""Compare remaining bridge model contacts with both accepted source motions."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    report = json.loads((args.candidate / "numeric_clearance_audit.json").read_text())
    names = json.loads((args.candidate / "motion_order.json").read_text())["robot_joint_names"]
    pairs = sorted({(c["a"], c["b"]) for c in report["self_contacts_below_minus2mm"]})
    model = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    data = mujoco.MjData(model)
    adr = [int(model.joint(n).qposadr[0]) for n in names]
    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test")
    specs = [
        ("candidate", args.candidate / "motion.npz", 75),
        ("accepted_push", source / "push-armchair-index-middle-kinematic-v1/motion.npz", 75),
        ("accepted_brace", source / "brace-dining-table-kinematic-v1/motion.npz", 83.5),
    ]
    measurements = {}
    for label, path, offset in specs:
        with np.load(path) as z:
            q, pos, quat = z["joint_pos"], z["body_pos_w"][:, 0], z["body_quat_w"][:, 0]
        distances = [[] for _ in pairs]
        times = []
        for step in range(111):
            time = 83.4 + step / 100
            frame = (time - offset) * 50
            if frame < -1e-7 or frame > len(q) - 1 + 1e-7:
                continue
            frame = np.clip(frame, 0, len(q) - 1)
            k, j = int(frame), min(int(frame) + 1, len(q) - 1)
            w = frame - k
            data.qpos[adr] = (1 - w) * q[k] + w * q[j]
            data.qpos[:3] = (1 - w) * pos[k] + w * pos[j]
            orientation = (1 - w) * quat[k] + w * quat[j]
            data.qpos[3:7] = orientation / np.linalg.norm(orientation)
            mujoco.mj_forward(model, data)
            times.append(time)
            for i, (a, b) in enumerate(pairs):
                distances[i].append(
                    float(mujoco.mj_geomDistance(model, data, model.geom(a).id, model.geom(b).id, 0.2, None))
                )
        measurements[label] = {"times_s": times, "distances_m": distances}
    rows = []
    for i, pair in enumerate(pairs):
        rows.append(
            {
                "pair": pair,
                "minimum_gap_m": {name: min(value["distances_m"][i]) for name, value in measurements.items()},
                "negative_sample_counts": {
                    name: sum(x < -0.002 for x in value["distances_m"][i]) for name, value in measurements.items()
                },
            }
        )
    output = {
        "scope": "model collision proxies in candidate and accepted references; inherited overlap is not proof of harmless physical contact",
        "pairs": rows,
        "measurements": measurements,
        "accepted": False,
    }
    (args.candidate / "self_overlap_source_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
