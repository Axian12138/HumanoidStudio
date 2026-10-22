"""Compare accepted06 basket contacts with a new combined candidate in one model."""

import argparse
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import sha256


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    accepted = Path("/home/galbot/workspace/ScaleTrack/data/1019test/round-basket-foot-push-kinematic-v1")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    order = json.loads((args.candidate / "motion_order.json").read_text())
    original_order = json.loads((accepted / "metadata/motion_order.json").read_text())
    for key in ("robot_joint_names", "robot_body_names"):
        if order[key] != original_order[key]:
            raise ValueError(f"Incompatible {key}")
    addresses = [int(model.joint(n).qposadr[0]) for n in order["robot_joint_names"]]
    robot_ids = {model.body(n).id for n in order["robot_body_names"]}
    basket_id = model.body("round_laundry_basket").id
    basket_address = int(model.joint("round_laundry_basket_free").qposadr[0])
    datasets = []
    for directory, basket_file, offset, quat_key in (
        (accepted, "round_laundry_basket_trajectory.npz", 0, "quaternion_wxyz"),
        (args.candidate, "round_laundry_basket.npz", 1600, "quaternion_xyzw"),
    ):
        sync_path = directory / ("metadata/source_sync.npz" if directory == accepted else "source_sync.npz")
        with np.load(sync_path) as sync:
            if not np.allclose(
                sync["recording_time_s"][offset : offset + 850], 59 + np.arange(850) / 50, rtol=0, atol=1e-6
            ):
                raise ValueError("Motion source clock differs from the protected interval.")
        with np.load(directory / "motion.npz") as z:
            motion = {k: z[k][offset : offset + 850] for k in ("joint_pos", "body_pos_w", "body_quat_w")}
        with np.load(directory / basket_file) as z:
            position = z["position_w"][offset : offset + 850]
            quaternion = z[quat_key][offset : offset + 850]
            if quat_key.endswith("xyzw"):
                quaternion = quaternion[:, [3, 0, 1, 2]]
        if len(position) != 850 or len(motion["joint_pos"]) != 850:
            raise ValueError("Expected all 850 protected06 frames, source59-76s.")
        datasets.append((motion, position, quaternion))
    contact_sets = []
    for label, (motion, position, quaternion) in zip(("accepted06", "candidate"), datasets, strict=True):
        data = mujoco.MjData(model)
        rows = []
        for k in range(850):
            data.qpos[:3] = motion["body_pos_w"][k, 0]
            data.qpos[3:7] = motion["body_quat_w"][k, 0]
            data.qpos[addresses] = motion["joint_pos"][k]
            data.qpos[basket_address : basket_address + 3] = position[k]
            data.qpos[basket_address + 3 : basket_address + 7] = quaternion[k]
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                first, second = map(int, model.geom_bodyid[[contact.geom1, contact.geom2]])
                other = second if first == basket_id else first
                if basket_id not in (first, second) or other not in robot_ids:
                    continue
                rows.append(
                    {
                        "source_s": 59 + k / 50,
                        "geom_ids": [int(contact.geom1), int(contact.geom2)],
                        "robot_body": model.body(other).name,
                        "distance_m": float(contact.dist),
                    }
                )
            if k % 200 == 0:
                print(label, k, len(rows), flush=True)
        contact_sets.append(rows)
    first, second = datasets
    report = {
        "scope": "50Hz59-76s basket/robot contacts, same authored collision model; other props remain at model defaults and are excluded. Not a continuous sweep or acceptance of original06 penetration.",
        "model_sha256": sha256(args.model),
        "accepted_motion_sha256": sha256(accepted / "motion.npz"),
        "candidate_motion_sha256": sha256(args.candidate / "motion.npz"),
        "candidate": str(args.candidate),
        "input_max_deltas": {
            **{k: float(abs(first[0][k] - second[0][k]).max()) for k in first[0]},
            "basket_position_m": float(abs(first[1] - second[1]).max()),
            "basket_quaternion_components": float(abs(first[2] - second[2]).max()),
        },
        "per_joint_max_delta_rad": dict(
            zip(
                order["robot_joint_names"],
                abs(first[0]["joint_pos"] - second[0]["joint_pos"]).max(axis=0).tolist(),
                strict=True,
            )
        ),
        "root_position_max_delta_m": float(
            abs(first[0]["body_pos_w"][:, 0] - second[0]["body_pos_w"][:, 0]).max()
        ),
        "root_quaternion_max_delta": float(
            abs(first[0]["body_quat_w"][:, 0] - second[0]["body_quat_w"][:, 0]).max()
        ),
        "contact_records_exact_equal": contact_sets[0] == contact_sets[1],
        "contacts": dict(zip(("accepted06", "candidate"), contact_sets, strict=True)),
        "accepted": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("exact contact equality", report["contact_records_exact_equal"], flush=True)


if __name__ == "__main__":
    main()
