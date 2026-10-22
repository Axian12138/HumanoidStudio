"""Prepare an isolated live scene for source-geometry review, without G1."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    base = run / "scene/03_04_05"
    root = ET.parse(base / "props.xml").getroot()
    drawer = ET.parse(base / "drawer.xml").getroot()
    for key in ("asset", "worldbody"):
        for element in drawer.find(key):
            root.find(key).append(element)
    ET.SubElement(
        root.find("worldbody"), "geom", name="ground_support", type="plane", size="5 5 .1", rgba="0 0 0 0", group="3"
    )
    ET.SubElement(
        root.find("worldbody"),
        "geom",
        name="sofa_support_proxy",
        type="box",
        size=".4 .4 .05",
        pos="2.005 .82 .3931",
        rgba="0 0 0 0",
        group="3",
    )
    ET.ElementTree(root).write(base / "props_drawer.xml", encoding="unicode")
    live = run / "review/scene/03_04_05_live"
    live.mkdir(parents=True, exist_ok=True)
    code = f"""import robosimstudio as rss
scene = rss.Scene(cell="void", name="03_04_05_scene_review", physics=rss.MUJOCO_500HZ)
scene.add(rss.robot(rss.EmbodimentSpec(name="props_drawer", asset=rss.AssetSpec(path={str(base / "props_drawer.xml")!r}), preserve_free_bodies=True, enable_self_collisions=True, arms=(rss.ArmSpec(name="slide", joint_names=("drawer_slide",), ee_body_name="drawer", home_q=(.19,)),))))
scene.add(rss.StaticSpec(name="room", asset=rss.AssetSpec(path={str(base / "room_without_drawer.obj")!r}), world_xform=True, collide=False))
scene.add(rss.CameraSpec(name="review", width=960, height=540, eye=(.0,-.7,2.3), look_at=(.8,-.6,.4)))
"""
    (live / "scene.py").write_text(code)
    print(live)


if __name__ == "__main__":
    main()
