"""Check authored prop/robot and rack contacts between the exported 50 Hz frames."""

import argparse
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256


def main():
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", nargs=2, type=float, required=True)
    parser.add_argument("--rate", type=int, default=100)
    args = parser.parse_args()
    motion = MotionFields.load(args.candidate / "motion.npz", args.candidate / "motion_order.json")
    source = 27 + np.arange(len(motion.joint_pos)) / motion.fps
    times = np.arange(round(args.window[0] * args.rate), round(args.window[1] * args.rate) + 1) / args.rate
    if times[0] < source[0] or times[-1] > source[-1]:
        raise ValueError("Requested sweep must stay inside recorded candidate")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    coordinates = np.repeat(data.qpos[None], len(times), axis=0)

    def vector(values):
        return np.column_stack([np.interp(times, source, values[:, axis]) for axis in range(values.shape[1])])

    def pose(position, quat_xyzw):
        quaternion = Slerp(source, Rotation.from_quat(quat_xyzw))(times).as_quat()
        return np.column_stack((vector(position), quaternion[:, [3, 0, 1, 2]]))

    coordinates[:, :7] = pose(motion.body_pos_w[:, 0], motion.body_quat_w[:, 0, [1, 2, 3, 0]])
    addresses = [int(model.joint(name).qposadr[0]) for name in motion.joint_names]
    coordinates[:, addresses] = vector(motion.joint_pos)
    names = ["floor_strawberry", "black_ribbon_hat", "round_laundry_basket"]
    input_paths = [args.model, args.candidate / "motion.npz", args.candidate / "motion_order.json"]
    for name in names:
        path = args.candidate / f"{name}.npz"
        input_paths.append(path)
        with np.load(path) as track:
            start = int(model.joint(name + "_free").qposadr[0])
            coordinates[:, start : start + 7] = pose(track["position_w"], track["quaternion_xyzw"])
    drawer_path = args.candidate / "drawer_slide.npz"
    input_paths.append(drawer_path)
    with np.load(drawer_path) as track:
        coordinates[:, int(model.joint("drawer_slide").qposadr[0])] = np.interp(times, source, track["position"])
    props = {model.body(name).id for name in names}
    robot = {model.body(name).id for name in motion.body_names}
    rack = model.body("coat_rack_support").id
    minima, violations = {}, []
    for sample, time in enumerate(times):
        data.qpos[:] = coordinates[sample]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            first, second = map(int, model.geom_bodyid[[contact.geom1, contact.geom2]])
            if not (
                rack in (first, second) or (first in props and second in robot) or (second in props and first in robot)
            ):
                continue
            pair = " / ".join(sorted([model.body(first).name, model.body(second).name]))
            distance = float(contact.dist)
            if pair not in minima or distance < minima[pair]["distance_m"]:
                minima[pair] = {"distance_m": distance, "source_s": float(time)}
            if distance < -0.005:
                violations.append({"pair": pair, "source_s": float(time), "distance_m": distance})
        if sample % 100 == 0:
            print(f"checked {sample + 1}/{len(times)}", flush=True)
    report = {
        "status": "passed_5mm_contact_depth_gate" if not violations else "failed_contact_depth_gate",
        "scope": "Authored collision contacts at interpolated samples; not a continuous collision proof, source fit, grasp stability or full room acceptance.",
        "interpolation": "Linear joint coordinates and translations; quaternion SLERP.",
        "rate_hz": args.rate,
        "samples": len(times),
        "source_interval_inclusive_s": [float(times[0]), float(times[-1])],
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "pair_minima": minima,
        "contacts_deeper_than_5mm": violations,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], "violations", len(violations))


if __name__ == "__main__":
    main()
