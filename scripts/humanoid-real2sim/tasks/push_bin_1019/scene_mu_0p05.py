"""RSS contact scene for the first 1019 white waste bin push, friction 0.05.

The bin mesh and 25 hulls come from the 1012 asset. The additional collision
guard follows the trashbin-v2 runtime USD. This branch halves the RSS bin
friction again, from 0.10 to 0.05. The void cell supplies a z=0 floor.
"""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import robosimstudio as rss


ASSETS = example_dir(__file__) / "assets"
COLLISION_HULLS = tuple(sorted((ASSETS / "collision_hulls").glob("*.obj")))
MOUTH_GUARD = ASSETS / "play_only_mouth_guard.obj"


def build_sim(asset, motion):
    if len(COLLISION_HULLS) != 25:
        raise ValueError(f"expected 25 waste-bin collision hulls, found {len(COLLISION_HULLS)}")
    if not MOUTH_GUARD.is_file():
        raise FileNotFoundError(MOUTH_GUARD)
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
    scene.add(rss.body(
        name="white_waste_bin",
        asset=rss.AssetSpec(
            path=str(ASSETS / "white_waste_bin_v2.obj"),
            collision_paths=tuple(map(str, (*COLLISION_HULLS, MOUTH_GUARD))),
        ),
        at=(0.180000007, 0.970000029, 0.001),
        mass=0.3,
        # RSS currently has one Coulomb coefficient per shape. The local USD
        # variant records the corresponding static/dynamic 0.05/0.04 values.
        friction=0.05,
        color=(0.88, 0.88, 0.85),
    ))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(1.8, -1.1, 1.55), look_at=(0.3, 0.75, 0.55),
    ))
    return scene.build(control_mode="joint_position")
