"""Build one scene from explicitly versioned room, robot/prop and extra-body assets.

The configuration belongs in the task run. A successful build verifies bindings,
not source registration or collision acceptance.
"""

import json
import os
from pathlib import Path

import robosimstudio as rss


def build_sim(asset, motion):
    config_path = Path(os.environ["RSS_SEQUENCE_SCENE_CONFIG"]).resolve()
    config = json.loads(config_path.read_text())
    scene = rss.Scene(cell="void", name="sequence_1019_unified", physics=rss.MUJOCO_500HZ)
    robot = rss.EmbodimentSpec(
        name="g1_sequence_1019",
        asset=rss.AssetSpec(path=config["robot_and_early_props"]),
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
    scene.add(rss.robot(robot))
    scene.add(
        rss.StaticSpec(
            name="room",
            asset=rss.AssetSpec(path=config["room_mesh"]),
            world_xform=True,
            collide=False,
        )
    )
    for static in config.get("additional_static_assets", []):
        scene.add(
            rss.StaticSpec(
                name=static["name"],
                asset=rss.AssetSpec(path=static["path"]),
                world_xform=True,
                collide=static.get("collide", True),
            )
        )
    names = set()
    for body in config["additional_bodies"]:
        name = body["name"]
        if name in names:
            raise ValueError(f"duplicate additional body: {name}")
        names.add(name)
        asset_args = {
            "path": body["visual_mesh"],
            "material_colors": body.get("material_colors", False),
        }
        if body.get("collision_meshes"):
            asset_args["collision_paths"] = tuple(body["collision_meshes"])
        else:
            asset_args.update(approximate=body["approximate_collision"], keep_visual=True)
        scene.add(
            rss.body(
                name=name,
                asset=rss.AssetSpec(**asset_args),
                at=tuple(body["position"]),
                quat=tuple(body["quaternion_xyzw"]),
                mass=body["mass"],
                friction=body["friction"],
                color=tuple(body.get("color", [0.5, 0.5, 0.5])),
            )
        )
    scene.add(rss.CameraSpec(name="review", width=960, height=540, eye=(2.8, -4.0, 2.1), look_at=(0.8, -0.8, 0.8)))
    return scene.build(control_mode="joint_position")
