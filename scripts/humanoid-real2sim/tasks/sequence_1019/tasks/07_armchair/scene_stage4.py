"""Physical chair-push scene shared by both Stage 4 hand-contact candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio.humanoid_real2sim.paths import example_dir
from robosimstudio.scene.registry import register_cell
from robosimstudio.specs import CellSpec


TASK = example_dir(__file__)
SNAPSHOT = TASK.parents[1]
ROOM = SNAPSHOT / "assets/ikea_1012_room.obj"
TABLE = SNAPSHOT / "assets/morbylanga_table_1012.obj"
CHAIR = TASK / "grey_armchair_visual.obj"
CELL = register_cell(CellSpec(name="armchair_1019_stage4_floor", ground_friction=0.3))


def build_sim(asset: Path, motion):
    candidate = Path(os.environ["RSS_ARMCHAIR_STAGE4_CANDIDATE"])
    assets = Path(os.environ["RSS_ARMCHAIR_STAGE4_ASSETS"])
    manifest = json.loads((assets / "collision_manifest.json").read_text())
    with np.load(candidate / "object_estimate.npz", allow_pickle=False) as track:
        if len(track["position_w"]) != motion.frames or int(track["fps"]) != int(motion.fps):
            raise ValueError("chair reference and robot motion have different clocks")
        initial = tuple(float(value) for value in track["position_w"][0])
        initial_quat = tuple(float(value) for value in track["quaternion_xyzw"][0])
    chair_hulls = tuple(assets / name for name in manifest["objects"]["chair"]["hulls"])
    table_hulls = tuple(assets / name for name in manifest["objects"]["table"]["hulls"])
    if len(chair_hulls) < 38 or len(table_hulls) < 4:
        raise ValueError("chair/table collision decomposition is incomplete")
    if len(manifest["objects"]["chair"].get("semantic_hulls", {})) != 6:
        raise ValueError("chair collider needs four legs and two backrest halves")
    if not all(path.is_file() for path in (*chair_hulls, *table_hulls)):
        raise FileNotFoundError("missing Stage 4 collision hull")

    robot = rss.EmbodimentSpec(
        name="g1_armchair_1019_stage4",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(rss.ArmSpec(
            name="whole_body", joint_names=motion.joint_names,
            ee_body_name="right_wrist_yaw_link",
            home_q=tuple(float(value) for value in motion.joint_pos[0]),
        ),),
    )
    scene = rss.Scene(cell=CELL.name, name="armchair_1019_stage4", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    scene.add(rss.StaticSpec(
        name="ikea_1012_room", asset=rss.AssetSpec(path=str(ROOM)),
        world_xform=True, collide=False,
    ))
    scene.add(rss.StaticSpec(
        name="morbylanga_table",
        asset=rss.AssetSpec(path=str(TABLE), collision_paths=tuple(map(str, table_hulls))),
        world_xform=True, friction=0.6,
    ))
    scene.add(rss.body(
        name="grey_armchair",
        asset=rss.AssetSpec(path=str(CHAIR), collision_paths=tuple(map(str, chair_hulls)),
                            keep_visual=True),
        at=initial, quat=initial_quat, mass=7.0, friction=0.3,
    ))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(0.9, -0.35, 1.65), look_at=(-0.55, -1.55, 0.65),
    ))
    return scene.build(control_mode="joint_position")
