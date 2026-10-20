"""RSS brush-sweep contact region in the inherited IKEA 1012 world frame.

The basket and brush assets/poses are the existing 1012 candidates. Their
placement against this new take still requires same-camera review.
"""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir, EXAMPLES

import newton
import numpy as np

import robosimstudio as rss
from robosimstudio.core.tuning import notify
from robosimstudio.scene.entity import label_index


ASSETS = EXAMPLES / "brush_pick_1012" / "assets"
BIN_ASSETS = EXAMPLES / "push_bin_1019" / "assets"
BASKET_POSITION = (0.944999933, -0.569999993, 0.400999993)
BASKET_QUAT_XYZW = (0.0, 0.0, 0.7071067812, 0.7071067812)
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
    # Existing 1012 support top is 0.401 m. Broaden the fixed proxy to cover
    # the sweep, while preserving the basket/brush placements and collisions.
    # Bounds measured from static hoi_batch_30 of the inherited 1012 scene:
    # x=0.76..1.36, y=-0.71..0.69, top z=0.400 m.
    scene.add(rss.slab(name="coffee_table_support", half=(0.30, 0.70, 0.02), at=(1.06, -0.01, 0.381)))
    scene.add(body_spec("white_storage_box", BASKET_POSITION, BASKET_QUAT_XYZW, 0.8, 0.6))
    scene.add(body_spec("green_hand_brush", brush_position, brush_quat, 0.15, 0.6))


def add_bin(scene):
    hulls = tuple(sorted((BIN_ASSETS / "collision_hulls").glob("*.obj")))
    mouth_guard = BIN_ASSETS / "play_only_mouth_guard.obj"
    if len(hulls) != 25 or not mouth_guard.is_file():
        raise FileNotFoundError("The reviewed 1019 bin collision set is incomplete")
    scene.add(rss.body(
        name="white_waste_bin",
        asset=rss.AssetSpec(
            path=str(BIN_ASSETS / "white_waste_bin_v2.obj"),
            collision_paths=tuple(str(path) for path in (*hulls, mouth_guard)),
        ),
        at=(0.180000007, 0.970000029, 0.001),
        mass=0.3,
        friction=0.10,
        color=(0.88, 0.88, 0.85),
    ))


def configure_bin_physics(sim):
    """Apply the existing low-COM, contact-priority 1019 bin variant."""
    body = label_index(list(sim.model.body_label), "white_waste_bin")
    com = np.asarray(sim.model.body_com.numpy()).copy()
    com[body] = (0.0, 0.0, 0.03)
    sim.model.body_com.assign(com)
    notify(sim.pipeline, int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))

    solver = sim.pipeline.program.mujoco_solver
    if solver is None:
        raise RuntimeError("expected MuJoCo Warp for the bin contacts")
    shape_indices = np.array([
        i for i, label in enumerate(sim.model.shape_label)
        if str(label).startswith("white_waste_bin_collision")
    ], dtype=np.int32)
    if len(shape_indices) != 26:
        raise ValueError(f"expected 26 bin collision shapes, found {len(shape_indices)}")
    authored_priority = sim.model.mujoco.geom_priority.numpy().copy()
    authored_priority[shape_indices] = 1
    sim.model.mujoco.geom_priority.assign(authored_priority)
    geom_shape = solver.mjc_geom_to_newton_shape.numpy()
    priority = solver.mjw_model.geom_priority.numpy().copy()
    mask = np.isin(geom_shape[0], shape_indices)
    if np.count_nonzero(mask) != 26:
        raise ValueError(f"expected 26 active bin geoms, found {np.count_nonzero(mask)}")
    priority[mask] = 1
    solver.mjw_model.geom_priority.assign(priority)
    solver.mj_model.geom_priority[mask] = 1


def build_rest_sim(*, brush_position=BRUSH_POSITION, brush_quat=BRUSH_QUAT_XYZW):
    scene = rss.Scene(cell="void", name="brush_sweep_1019_rest", physics=rss.MUJOCO_500HZ)
    add_contact_region(scene, brush_position=brush_position, brush_quat=brush_quat)
    return scene.build()


def build_sim(asset, motion):
    robot = rss.EmbodimentSpec(
        name="g1_brush_sweep_1019",
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
    scene = rss.Scene(cell="void", name="brush_sweep_1019", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(robot))
    add_bin(scene)
    add_contact_region(scene)
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(1.7, -1.4, 1.1), look_at=(0.52, -0.57, 0.5),
    ))
    sim = scene.build(control_mode="joint_position")
    configure_bin_physics(sim)
    return sim
