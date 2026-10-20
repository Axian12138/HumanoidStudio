"""Stage 3.5 scene: the inherited IKEA room and the 1019 G1 body prior.

Furniture is the authored 1012 USD in the shared world frame.  Dynamic
object paths have not yet been reconstructed for these later interactions.
"""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import robosimstudio as rss


ROOM = example_dir(__file__) / "assets" / "ikea_1012_room.obj"


def build_sim(asset, motion):
    robot = rss.EmbodimentSpec(
        name="g1_sequence_1019",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(rss.ArmSpec(
            name="whole_body", joint_names=motion.joint_names,
            ee_body_name="right_wrist_yaw_link",
            home_q=tuple(float(value) for value in motion.joint_pos[0]),
        ),),
    )
    scene = rss.Scene(cell="void", name="sequence_1019_stage35", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    scene.add(rss.StaticSpec(
        name="ikea_1012_room", asset=rss.AssetSpec(path=str(ROOM)),
        world_xform=True, collide=False,
    ))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(2.2, -1.8, 1.7), look_at=(0.2, -0.5, 0.65),
    ))
    return scene.build(control_mode="joint_position")
