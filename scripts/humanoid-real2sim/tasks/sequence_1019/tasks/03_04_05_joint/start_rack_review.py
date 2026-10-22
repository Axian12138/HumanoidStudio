"""Prepare the live RSS page to test the supported hat on the actual rack."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate")
    p.add_argument("--release-time", type=float, default=47.64)
    a = p.parse_args()
    run = a.run.resolve()
    base = run / "scene/03_04_05"
    root = ET.parse(
        base / ("hat_rack_physics_early_registered.xml" if a.candidate else "hat_rack_physics.xml")
    ).getroot()
    hat = root.find("worldbody/body[@name='black_ribbon_hat']")
    hat.set("pos", "2.150866718016237 -1.5736 .9271032539115094")
    hat.set("quat", ".8191520442889918 0 -.573576436351046 0")
    if a.candidate:
        track = np.load(run / "retarget/candidates" / a.candidate / "black_ribbon_hat.npz")
        frame = round((a.release_time - 27) * 50)
        hat.set("pos", " ".join(map(str, track["position_w"][frame])))
        hat.set("quat", " ".join(map(str, track["quaternion_xyzw"][frame][[3, 0, 1, 2]])))
    path = base / (f"hat_rack_rest_{a.candidate}.xml" if a.candidate else "hat_rack_rest_candidate.xml")
    ET.ElementTree(root).write(path, encoding="unicode")
    live = run / "review/scene/03_04_05_live"
    room = run / "scene/unified/per_furniture_v4.obj" if a.candidate else base / "room_without_drawer.obj"
    (live / "scene.py").write_text(f"""import robosimstudio as rss
scene = rss.Scene(cell="void", name="hat_rack_support_review", physics=rss.MUJOCO_500HZ)
scene.add(rss.robot(rss.EmbodimentSpec(name="hat_rack",asset=rss.AssetSpec(path={str(path)!r}),preserve_free_bodies=True,enable_self_collisions=True)))
scene.add(rss.StaticSpec(name="room",asset=rss.AssetSpec(path={str(room)!r}),world_xform=True,collide=False))
scene.add(rss.CameraSpec(name="review",width=960,height=540,eye=(.8,-.4,1.6),look_at=(2.1,-1.57,1.05)))
""")
    print(live)


if __name__ == "__main__":
    main()
