"""Source-derived room, three separate objects and a passive drawer slider.

Set SEQUENCE_1019_RUN to the output run root. This is a kinematic review scene;
its static room is visual-only until per-furniture collision checks complete.
"""

import os
from pathlib import Path

import robosimstudio as rss


def build_sim(asset, motion):
    run = Path(os.environ["SEQUENCE_1019_RUN"]).resolve()
    base = run / "scene/03_04_05"
    model_name = (
        "robot_props_drawer_with_rack.xml"
        if os.environ.get("SEQUENCE_1019_WITH_RACK") == "1"
        else "robot_props_drawer.xml"
    )
    robot = rss.EmbodimentSpec(
        name="g1_sequence_1019_joint",
        asset=rss.AssetSpec(path=str(base / model_name)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=motion.joint_names,
                ee_body_name="right_wrist_yaw_link",
                home_q=tuple(float(v) for v in motion.joint_pos[0]),
            ),
        ),
    )
    scene = rss.Scene(cell="void", name="sequence_1019_joint_candidate", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    scene.add(
        rss.StaticSpec(
            name="room",
            asset=rss.AssetSpec(path=str(base / "room_without_drawer.obj")),
            world_xform=True,
            collide=False,
        )
    )
    scene.add(rss.CameraSpec(name="review", width=960, height=540, eye=(2.8, -4.0, 2.1), look_at=(0.8, -0.8, 0.8)))
    return scene.build(control_mode="joint_position")
