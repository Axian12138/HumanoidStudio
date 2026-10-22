"""Kinematic basket scene; the aligned table is a distinct review overlay."""

from robosimstudio.humanoid_real2sim.paths import example_dir

import robosimstudio as rss


HERE = example_dir(__file__)
BASKET = HERE / "round_laundry_basket_visual.obj"


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
    scene = rss.Scene(cell="void", name="round_basket_1019_rigid_review", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    scene.add(rss.body(
        name="round_laundry_basket",
        asset=rss.AssetSpec(path=str(BASKET), approximate="bounding_box", keep_visual=True),
        at=(0.16, -1.92, 0.001), mass=0.8, friction=0.6,
    ))
    return scene.build(control_mode="joint_position")
