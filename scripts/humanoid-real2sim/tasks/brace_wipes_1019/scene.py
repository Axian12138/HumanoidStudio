"""Reviewed brace room with the specified wipes package added on the table."""

import json
import os
from pathlib import Path

import robosimstudio as rss


def build_sim(asset, motion):
    run = Path(os.environ["RSS_WIPES_RUN"])
    cfg = json.loads((run / "scene/config.json").read_text())
    snap = Path(
        "/home/galbot/workspace/robosimstudio/outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot"
    )
    robot = rss.EmbodimentSpec(
        name="g1_brace_wipes",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=motion.joint_names,
                ee_body_name="left_wrist_yaw_link",
                home_q=tuple(float(x) for x in motion.joint_pos[0]),
            ),
        ),
    )
    scene = rss.Scene(cell="void", name="brace_left_fist_wipes", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    scene.add(
        rss.StaticSpec(
            name="room",
            asset=rss.AssetSpec(path=str(snap / "assets/ikea_1012_room.obj")),
            world_xform=True,
            collide=False,
        )
    )
    scene.add(
        rss.body(
            name="grey_armchair",
            asset=rss.AssetSpec(
                path=str(snap / "tasks/07_armchair/grey_armchair_visual.obj"),
                approximate="bounding_box",
                keep_visual=True,
            ),
            at=(-0.5523157716, -2.1457662582, 0.001),
            quat=(0, 0, -0.0494105928, 0.9987785220),
            mass=7.0,
            friction=0.6,
        )
    )
    scene.add(
        rss.body(
            name="wet_wipes",
            asset=rss.AssetSpec(
                path=str(run / "scene/wet_wipes_visual.obj"),
                approximate="convex_hull",
                keep_visual=True,
                material_colors=True,
            ),
            at=tuple(cfg["position_w"]),
            quat=tuple(cfg["quaternion_xyzw"]),
            mass=0.2,
            friction=0.7,
        )
    )
    scene.add(rss.CameraSpec(name="review", width=960, height=640, eye=(-0.5, -1.0, 1.9), look_at=(0, -2.2, 0.8)))
    return scene.build(control_mode="joint_position")
