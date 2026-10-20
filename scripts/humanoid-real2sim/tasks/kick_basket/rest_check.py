"""Check basket mass properties and free-rest stability without the robot."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import newton
import numpy as np

import robosimstudio as rss
from robosimstudio.core.tuning import notify
from robosimstudio.scene.entity import label_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--friction", type=float, required=True)
    parser.add_argument("--com-z", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    path = Path(__file__).with_name("scene.py")
    spec = importlib.util.spec_from_file_location("kick_basket_scene", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = rss.Scene(cell="void", name="kick_basket_free_rest", physics=rss.MUJOCO_500HZ)
    scene.add(module.basket_spec(mass=args.mass, friction=args.friction))
    sim = scene.build()
    body = label_index(list(sim.model.body_label), "round_laundry_basket")
    com = np.asarray(sim.model.body_com.numpy()).copy()
    com[body, 2] = args.com_z
    sim.model.body_com.assign(com)
    notify(sim.pipeline, int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))
    sim.reset()
    poses = [sim.pipeline.snapshot.body_q[body].copy()]
    for _ in range(150):
        sim.step_teleop()
        poses.append(sim.pipeline.snapshot.body_q[body].copy())
    poses = np.asarray(poses)
    initial, final = poses[0], poses[-1]
    tilt_deg = np.degrees(np.arccos(np.clip(1 - 2 * (poses[:, 3] ** 2 + poses[:, 4] ** 2), -1, 1)))
    report = {
        "mass_kg": float(sim.model.body_mass.numpy()[body]),
        "friction": args.friction,
        "com_local_m": com[body].tolist(),
        "duration_s": 3.0,
        "initial_pose_xyzw": initial.tolist(),
        "final_pose_xyzw": final.tolist(),
        "xy_drift_m": float(np.linalg.norm(final[:2] - initial[:2])),
        "z_change_m": float(final[2] - initial[2]),
        "max_tilt_deg": float(tilt_deg.max()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
