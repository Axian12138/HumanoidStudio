"""Compare authored, enabled self contacts at matching interpolated source times."""

import argparse
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", nargs=2, type=float, required=True)
    parser.add_argument("--source-start-s", type=float, default=27)
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument("--worsening-m", type=float, default=0.001)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
        ),
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    times = np.arange(round(args.window[0] * args.rate), round(args.window[1] * args.rate) + 1) / args.rate
    measurements, hashes = {}, {str(args.model): sha256(args.model)}
    for label, folder in (("baseline", args.baseline), ("candidate", args.candidate)):
        motion_path, order_path = folder / "motion.npz", folder / "motion_order.json"
        motion = MotionFields.load(motion_path, order_path)
        source = args.source_start_s + np.arange(len(motion.joint_pos)) / motion.fps
        if times[0] < source[0] - 1e-7 or times[-1] > source[-1] + 1e-7:
            raise ValueError("Audit interval lies outside motion")
        q = np.column_stack([np.interp(times, source, motion.joint_pos[:, i]) for i in range(len(motion.joint_names))])
        addresses = [int(model.joint(n).qposadr[0]) for n in motion.joint_names]
        robot = {model.body(n).id for n in motion.body_names}
        entries = []
        for coordinates in q:
            data.qpos[:] = model.qpos0
            data.qpos[addresses] = coordinates
            mujoco.mj_forward(model, data)
            pairs = {}
            for contact in data.contact:
                g, h = int(contact.geom1), int(contact.geom2)
                if int(model.geom_bodyid[g]) not in robot or int(model.geom_bodyid[h]) not in robot:
                    continue
                key = " / ".join(sorted([model.geom(g).name, model.geom(h).name]))
                pairs[key] = min(pairs.get(key, 0.0), float(contact.dist))
            entries.append(pairs)
        measurements[label] = entries
        hashes.update({str(p): sha256(p) for p in (motion_path, order_path)})
    worsened = []
    for time, baseline, candidate in zip(times, measurements["baseline"], measurements["candidate"], strict=True):
        for pair, distance in candidate.items():
            previous = baseline.get(pair, 0.0)
            if distance < previous - args.worsening_m:
                worsened.append(
                    {
                        "source_s": float(time),
                        "pair": pair,
                        "baseline_gap_m": previous,
                        "candidate_gap_m": distance,
                        "worsening_m": previous - distance,
                    }
                )
    report = {
        "scope": "Actual enabled authored robot self contacts only. Root is fixed because pairwise self distances are invariant to common rigid transforms. No stability or full physical acceptance.",
        "threshold_m": args.worsening_m,
        "samples": len(times),
        "rate_hz": args.rate,
        "new_or_worsened_over_threshold": worsened,
        "measurements": measurements,
        "times_s": times.tolist(),
        "input_sha256": hashes,
        "script_sha256": sha256(Path(__file__)),
        "limitations": [
            "Inherited proxy overlaps remain measured; a comparison pass does not establish collision-free physical geometry."
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("new or worsened contacts:", len(worsened))


if __name__ == "__main__":
    main()
