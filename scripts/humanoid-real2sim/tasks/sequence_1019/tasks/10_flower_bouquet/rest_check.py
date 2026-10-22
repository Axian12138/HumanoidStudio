"""Independent RSS free rest for both initial objects; no tracker or robot."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--scene-module", default="scene")
a = p.parse_args()
sim = importlib.import_module(a.scene_module).make_scene(room=False).build()
names = ["flower_bouquet", "small_wicker_basket"]
poses = {n: [] for n in names}
for i in range(151):
    for n in names:
        poses[n].append(sim[n].pose.tolist())
    if i < 150:
        sim.run(1).sync()
report = {
    "duration_s": 3.0,
    "scope": "initial independent free rest, no robot and no tracker",
    "representation": "explicit approximate separate convex collision solids",
    "objects": {},
}
for n in names:
    v = np.array(poses[n])
    report["objects"][n] = {
        "initial_pose": v[0].tolist(),
        "final_pose": v[-1].tolist(),
        "translation_m": float(np.linalg.norm(v[-1, :3] - v[0, :3])),
        "last_second_position_range_m": np.ptp(v[-50:, :3], axis=0).tolist(),
        "poses": poses[n],
    }
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(report, indent=2))
print({n: report["objects"][n]["translation_m"] for n in names})
