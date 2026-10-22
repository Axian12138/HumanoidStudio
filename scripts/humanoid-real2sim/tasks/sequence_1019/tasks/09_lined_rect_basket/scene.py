"""Stage 3.5 kinematic review of the lined basket interaction."""

import os
from pathlib import Path

import robosimstudio as rss

from robosimstudio.humanoid_real2sim.paths import example_dir


TASK = example_dir(__file__)
ROOM = TASK.parents[1] / "assets/ikea_1012_room.obj"


def build_sim(asset, motion):
    basket_mesh = Path(os.environ["RSS_BASKET_VISUAL_MESH"])
    robot = rss.EmbodimentSpec(
        name="g1_lined_basket_1019",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(rss.ArmSpec(
            name="whole_body",
            joint_names=motion.joint_names,
            ee_body_name="right_wrist_yaw_link",
            home_q=tuple(float(value) for value in motion.joint_pos[0]),
        ),),
    )
    scene = rss.Scene(cell="void", name="lined_basket_1019_stage35", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    if os.environ.get("RSS_BASKET_SHOW_ROOM") == "1":
        scene.add(rss.StaticSpec(
            name="ikea_1012_room", asset=rss.AssetSpec(path=str(ROOM)),
            world_xform=True, collide=False,
        ))
    scene.add(rss.body(
        name="lined_rect_basket",
        asset=rss.AssetSpec(path=str(basket_mesh), approximate="bounding_box", keep_visual=True),
        at=(-0.875, -2.6, 0.754), quat=(0.0, 0.0, 0.7071, 0.7071),
        mass=1.0, friction=0.6,
    ))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(1.5, -1.2, 1.7), look_at=(0.0, -1.3, 0.8),
    ))
    return scene.build(control_mode="joint_position")
