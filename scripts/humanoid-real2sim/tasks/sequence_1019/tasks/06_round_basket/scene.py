"""Stage 3.5 round-basket review in the aligned 1012 IKEA room."""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import robosimstudio as rss


HERE = example_dir(__file__)
ROOM = HERE.parents[1] / "assets/ikea_1012_room.obj"
BASKET = HERE / "round_laundry_basket_visual.obj"
INITIAL_POSITION = (0.25, -1.78, 0.001)  # provisional 1019 foot/shin contact fit


def build_sim(asset, motion):
    robot = rss.EmbodimentSpec(
        name="g1_round_basket_1019",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(rss.ArmSpec(
            name="whole_body", joint_names=motion.joint_names,
            ee_body_name="right_wrist_yaw_link",
            home_q=tuple(float(value) for value in motion.joint_pos[0]),
        ),),
    )
    scene = rss.Scene(cell="void", name="round_basket_1019_stage35", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    scene.add(rss.StaticSpec(
        name="ikea_1012_room", asset=rss.AssetSpec(path=str(ROOM)),
        world_xform=True, collide=False,
    ))
    scene.add(rss.body(
        name="round_laundry_basket",
        asset=rss.AssetSpec(path=str(BASKET), approximate="bounding_box", keep_visual=True),
        at=INITIAL_POSITION, mass=0.8, friction=0.6,
    ))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(0.8, -0.7, 1.3), look_at=(0.0, -1.65, 0.3),
    ))
    return scene.build(control_mode="joint_position")
