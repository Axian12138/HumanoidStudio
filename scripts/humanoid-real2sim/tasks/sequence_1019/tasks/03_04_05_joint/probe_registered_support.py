"""Test alternative cavity yaw support without changing geometry or physics parameters."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    m = mujoco.MjModel.from_xml_path(str(r / "scene/03_04_05/hat_rack_physics_early_registered.xml"))
    target = np.array([2.150866718016237, -1.5736, 0.9271032539115094]) + np.array(
        json.loads((r / "scene/03_04_05/early_rack_registration.json").read_text())["translation_world_m"]
    )
    records = []
    for yaw in [-180, -90, -45, 0]:
        d = mujoco.MjData(m)
        R = Rotation.from_euler("y", -70, degrees=True) * Rotation.from_euler("z", yaw, degrees=True)
        d.qpos[:3] = target
        d.qpos[3:7] = R.as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(m, d)
        initial = min([c.dist for c in d.contact] or [0])
        samples = []
        contacts = []
        for k in range(14000):
            mujoco.mj_step(m, d)
            if k >= 11500 and k % 50 == 0:
                samples.append(d.qpos.copy())
                contacts.append(d.ncon > 0)
        samples = np.array(samples)
        rot = Rotation.from_quat(samples[:, 3:7][:, [1, 2, 3, 0]])
        record = {
            "yaw_deg": yaw,
            "initial_penetration_m": float(initial),
            "minimum_last5s_z": float(samples[:, 2].min()),
            "last5s_root_max_displacement_m": float(np.linalg.norm(samples[:, :3] - samples[0, :3], axis=1).max()),
            "last5s_rotation_max_rad": float((rot * rot[0].inv()).magnitude().max()),
            "contact_fraction": float(np.mean(contacts)),
            "final_qpos": d.qpos.tolist(),
            "final_velocity": d.qvel.tolist(),
        }
        records.append(record)
        print(record, flush=True)
    (r / "review/scene/03_04_05_registered_support_yaw.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
