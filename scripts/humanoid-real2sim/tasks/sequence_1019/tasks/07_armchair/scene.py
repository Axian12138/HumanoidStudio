"""Stage 3.5 chair interaction review in the aligned 1012 room."""

from pathlib import Path

import robosimstudio as rss

from robosimstudio.humanoid_real2sim.paths import example_dir


HERE = example_dir(__file__)
ROOM = HERE.parents[1] / "assets/ikea_1012_room.obj"
CHAIR = HERE / "grey_armchair_visual.obj"
INITIAL_POSITION = (-0.737, -1.082, 0.001)  # replaced by the fitted track each frame
INITIAL_QUAT_XYZW = (0.0, 0.0, 0.151, 0.988)


def build_sim(asset, motion):
    robot = rss.EmbodimentSpec(
        name="g1_armchair_1019",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(rss.ArmSpec(
            name="whole_body", joint_names=motion.joint_names,
            ee_body_name="right_wrist_yaw_link",
            home_q=tuple(float(value) for value in motion.joint_pos[0]),
        ),),
    )
    scene = rss.Scene(cell="void", name="armchair_1019_stage35", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    scene.add(rss.StaticSpec(
        name="ikea_1012_room", asset=rss.AssetSpec(path=str(ROOM)),
        world_xform=True, collide=False,
    ))
    scene.add(rss.body(
        name="grey_armchair",
        asset=rss.AssetSpec(path=str(CHAIR), approximate="bounding_box", keep_visual=True),
        at=INITIAL_POSITION, quat=INITIAL_QUAT_XYZW, mass=7.0, friction=0.6,
    ))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(0.9, -0.35, 1.65), look_at=(-0.55, -1.55, 0.65),
    ))
    return scene.build(control_mode="joint_position")
