"""Stage 5 object trial: IKEA coffee table and effective bin friction 0.15.

The table is extracted from the 1012 living room USD, including its eight
authored collision meshes. The bin keeps the low 30 mm center of mass.
"""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import newton
import numpy as np

import robosimstudio as rss
from robosimstudio.core.tuning import notify
from robosimstudio.scene.entity import label_index


ASSETS = example_dir(__file__) / "assets"
COLLISION_HULLS = tuple(sorted((ASSETS / "collision_hulls").glob("*.obj")))
MOUTH_GUARD = ASSETS / "play_only_mouth_guard.obj"
TABLE_COLLISION = tuple(sorted(ASSETS.glob("coffee_table_collision_*.obj")))


def build_sim(asset, motion):
    if len(COLLISION_HULLS) != 25:
        raise ValueError(f"expected 25 waste-bin collision hulls, found {len(COLLISION_HULLS)}")
    if not MOUTH_GUARD.is_file():
        raise FileNotFoundError(MOUTH_GUARD)
    if len(TABLE_COLLISION) != 8:
        raise ValueError(f"expected eight table collision meshes, found {len(TABLE_COLLISION)}")
    g1 = rss.EmbodimentSpec(
        name="g1_push_bin_1019",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(rss.ArmSpec(
            name="whole_body",
            joint_names=motion.joint_names,
            ee_body_name="left_wrist_yaw_link",
            home_q=tuple(float(value) for value in motion.joint_pos[0]),
        ),),
    )
    scene = rss.Scene(cell="void", name="push_bin_1019", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(g1))
    scene.add(rss.StaticSpec(
        name="coffee_table",
        asset=rss.AssetSpec(
            path=str(ASSETS / "coffee_table_visual.obj"),
            collision_paths=tuple(map(str, TABLE_COLLISION)),
        ),
        world_xform=True,
        friction=0.60,
        color=(0.55, 0.38, 0.25),
    ))
    scene.add(rss.body(
        name="white_waste_bin",
        asset=rss.AssetSpec(
            path=str(ASSETS / "white_waste_bin_v2.obj"),
            collision_paths=tuple(map(str, (*COLLISION_HULLS, MOUTH_GUARD))),
        ),
        at=(0.180000007, 0.970000029, 0.001),
        mass=0.3,
        # RSS currently has one Coulomb coefficient per shape. Use the
        # Stage 5 trial raises the effective contact coefficient from 0.10.
        friction=0.15,
        color=(0.88, 0.88, 0.85),
    ))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(1.8, -1.1, 1.55), look_at=(0.3, 0.75, 0.55),
    ))
    sim = scene.build(control_mode="joint_position")
    body = label_index(list(sim.model.body_label), "white_waste_bin")
    com = np.asarray(sim.model.body_com.numpy()).copy()
    com[body] = (0.0, 0.0, 0.03)
    sim.model.body_com.assign(com)
    notify(sim.pipeline, int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))

    # MuJoCo combines equally ranked geom frictions by taking their maximum.
    # The floor has mu=1.0, so merely setting the bin shapes to 0.15
    # leaves bin-floor contacts at 1.0. Give the bin's 26 collision shapes a
    # higher MuJoCo geom priority; this keeps foot-floor contacts unchanged.
    # Newton's current SHAPE_PROPERTIES refresh does not update geom_priority,
    # so synchronize the active MJWarp model as well as the authored model.
    solver = sim.pipeline.program.mujoco_solver
    if solver is None:
        raise RuntimeError("expected the MuJoCo Warp rigid solver")
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
        raise ValueError(f"expected 26 MuJoCo bin geoms, found {np.count_nonzero(mask)}")
    priority[mask] = 1
    solver.mjw_model.geom_priority.assign(priority)
    solver.mj_model.geom_priority[mask] = 1
    return sim
