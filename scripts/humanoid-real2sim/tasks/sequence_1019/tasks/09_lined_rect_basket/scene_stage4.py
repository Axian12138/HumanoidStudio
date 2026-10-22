"""Physical Stage 4 scene for the lined-basket retarget candidate."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio.humanoid_real2sim.paths import example_dir


TASK = example_dir(__file__)
SNAPSHOT = TASK.parents[1]
SOURCE_MESHES = Path("/home/galbot/workspace/ScaleTrack/data/1012test/hoi_stage2/scene/assets")
ROOM = SNAPSHOT / "assets/ikea_1012_room.obj"
DINING_TABLE = SNAPSHOT / "assets/morbylanga_table_1012.obj"
COFFEE_TABLE = Path(
    "/home/galbot/workspace/robosimstudio/outputs/humanoid-real2sim/2026-10-20/"
    "push_bin_1019/curated_snapshot/assets/coffee_table_visual.obj"
)


def build_sim(asset: Path, motion):
    candidate = Path(os.environ["RSS_BASKET_STAGE4_CANDIDATE"])
    with np.load(candidate / "object_estimate.npz", allow_pickle=False) as track:
        if len(track["position_w"]) != motion.frames or int(track["fps"]) != motion.fps:
            raise ValueError("basket sidecar and tracker reference must share frames and FPS")
        initial = tuple(float(value) for value in track["position_w"][0])
        initial_quat = tuple(float(value) for value in track["quaternion_xyzw"][0])

    pieces = tuple(SOURCE_MESHES / f"object_lined_rect_basket_{part}_0.obj" for part in range(1, 6))
    if not all(path.is_file() for path in pieces):
        raise FileNotFoundError("the basket base and four independent wall meshes are required")

    robot = rss.EmbodimentSpec(
        name="g1_lined_basket_1019_stage4",
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
    scene = rss.Scene(cell="void", name="lined_basket_1019_stage4", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    # The scene meshes are already in world coordinates. Their simple static
    # collider proxies provide the measured 0.754m and 0.400m tabletops.
    for name, path in (("dining_table", DINING_TABLE), ("coffee_table", COFFEE_TABLE)):
        scene.add(rss.StaticSpec(
            name=name,
            asset=rss.AssetSpec(path=str(path), approximate="bounding_box"),
            world_xform=True, collide=True,
        ))
    if os.environ.get("RSS_BASKET_SHOW_ROOM") == "1":
        scene.add(rss.StaticSpec(
            name="ikea_1012_room", asset=rss.AssetSpec(path=str(ROOM)),
            world_xform=True, collide=False,
        ))
    scene.add(rss.body(
        name="lined_rect_basket",
        asset=rss.AssetSpec(
            path=str(candidate / "basket_visual.obj"),
            collision_paths=tuple(str(path) for path in pieces),
            keep_visual=True,
        ),
        at=initial, quat=initial_quat, mass=1.0, friction=1.2,
    ))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(1.5, -1.2, 1.7), look_at=(0.0, -1.3, 0.8),
    ))
    return scene.build(control_mode="joint_position")
