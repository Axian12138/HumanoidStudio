"""Local RSS scene for the 1012 white-basket brush pickup."""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import robosimstudio as rss

ASSETS = example_dir(__file__) / "assets"
BASKET_POSITION = (0.944999933, -0.569999993, 0.400999993)
BASKET_QUAT_XYZW = (0.0, 0.0, 0.7071067812, 0.7071067812)
# RSS free-rest result after 5 s from a 13 mm basketward placement correction.
BRUSH_POSITION = (0.8716025949, -0.5871267319, 0.4387029409)
BRUSH_QUAT_XYZW = (0.002078, 0.14354791, -0.00853049, 0.98960447)


def body_spec(name, position, quat, mass, friction):
    directory = ASSETS / name
    return rss.body(
        name=name,
        asset=rss.AssetSpec(
            path=str(directory / "visual.obj"),
            collision_paths=tuple(str(p) for p in sorted((directory / "collision_hulls").glob("*.obj"))),
        ),
        at=position,
        quat=quat,
        mass=mass,
        friction=friction,
    )


def add_contact_region(scene, *, brush_position=BRUSH_POSITION, brush_quat=BRUSH_QUAT_XYZW):
    # The white basket's source origin is at its bottom. The local coffee-table
    # support top is 0.401 m in the shared IKEA world frame.
    scene.add(rss.slab(name="coffee_table_support", half=(0.32, 0.30, 0.02), at=(0.945, -0.57, 0.381)))
    scene.add(body_spec("white_storage_box", BASKET_POSITION, BASKET_QUAT_XYZW, 0.8, 0.6))
    scene.add(body_spec("green_hand_brush", brush_position, brush_quat, 0.15, 0.6))


def build_rest_sim(*, brush_position=BRUSH_POSITION, brush_quat=BRUSH_QUAT_XYZW):
    scene = rss.Scene(cell="void", name="brush_pick_rest", physics=rss.MUJOCO_500HZ)
    add_contact_region(scene, brush_position=brush_position, brush_quat=brush_quat)
    return scene.build()


def build_sim(asset, motion, *, brush_position=BRUSH_POSITION, brush_quat=BRUSH_QUAT_XYZW):
    robot = rss.EmbodimentSpec(
        name="g1_brush_pick", asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True, enable_self_collisions=False,
        arms=(rss.ArmSpec(
            name="whole_body", joint_names=motion.joint_names,
            ee_body_name="right_wrist_yaw_link",
            home_q=tuple(float(value) for value in motion.joint_pos[0]),
        ),),
    )
    scene = rss.Scene(cell="void", name="brush_pick_1012", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    add_contact_region(scene, brush_position=brush_position, brush_quat=brush_quat)
    scene.add(rss.CameraSpec(name="review", width=960, height=540,
                             eye=(1.7, -1.4, 1.1), look_at=(0.91, -0.57, 0.5)))
    return scene.build(control_mode="joint_position")
