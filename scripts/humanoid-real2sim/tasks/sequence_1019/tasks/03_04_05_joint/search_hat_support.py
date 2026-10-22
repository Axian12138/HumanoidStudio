"""Test nearby release poses against the authored rack segments under gravity."""

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    scene = a.run / "scene/03_04_05"
    source = ET.parse(scene / "robot_props_drawer_with_rack.xml").getroot()
    root = ET.Element("mujoco", model="hat_rack_physics")
    ET.SubElement(root, "option", timestep=".002", gravity="0 0 -9.81", iterations="100")
    asset = ET.SubElement(root, "asset")
    for item in source.find("asset"):
        if item.tag == "mesh" and (
            item.attrib["name"].startswith("black_ribbon_hat") or item.attrib["name"].startswith("rack_")
        ):
            asset.append(copy.deepcopy(item))
    world = ET.SubElement(root, "worldbody")
    world.append(copy.deepcopy(source.find("worldbody/body[@name='black_ribbon_hat']")))
    world.append(copy.deepcopy(source.find("worldbody/body[@name='coat_rack_support']")))
    ET.SubElement(world, "geom", type="plane", size="10 10 .1")
    path = scene / "hat_rack_physics.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    z = np.load(a.run / "retarget/candidates/03_04_05_joint_v14/black_ribbon_hat.npz")
    initial = z["position_w"][1050]
    quat = z["quaternion_xyzw"][1050][[3, 0, 1, 2]]
    results = []
    for dx in [0, 0.02]:
        for dy in [-0.035, -0.055, -0.075]:
            for dz in [0, 0.03, 0.06]:
                mujoco.mj_resetData(m, d)
                pos = initial + np.array([dx, dy, dz])
                d.qpos[:3] = pos
                d.qpos[3:7] = quat
                mujoco.mj_forward(m, d)
                penetration = min([float(c.dist) for c in d.contact] or [0])
                snap = []
                for step in range(1500):
                    mujoco.mj_step(m, d)
                    if step in [499, 999, 1499]:
                        snap.append(
                            {
                                "time_s": float(d.time),
                                "position_m": d.qpos[:3].tolist(),
                                "quaternion_wxyz": d.qpos[3:7].tolist(),
                                "speed": float(np.linalg.norm(d.qvel)),
                            }
                        )
                results.append(
                    {
                        "offset_m": [dx, dy, dz],
                        "initial_position_m": pos.tolist(),
                        "initial_quaternion_wxyz": quat.tolist(),
                        "initial_penetration_m": penetration,
                        "samples": snap,
                        "final_displacement_m": float(np.linalg.norm(d.qpos[:3] - pos)),
                        "final_orientation_change_rad": float(
                            (
                                Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]])
                                * Rotation.from_quat(quat[[1, 2, 3, 0]]).inv()
                            ).magnitude()
                        ),
                        "held_above_floor": bool(d.qpos[2] > 0.6),
                        "last_second_drift_m": float(np.linalg.norm(d.qpos[:3] - np.array(snap[-2]["position_m"]))),
                    }
                )
                print(
                    results[-1]["offset_m"],
                    results[-1]["held_above_floor"],
                    results[-1]["final_displacement_m"],
                    results[-1]["initial_penetration_m"],
                    flush=True,
                )
    (a.run / "review/scene/03_04_05_hat_support_search.json").write_text(
        json.dumps(
            {
                "scope": "MuJoCo isolated source hat and source rack collision segments, gravity3s; RSS interaction not yet verified",
                "source_release_candidate": "v14 at48s",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
