"""Local RSS contact scene for the 1012 kick-basket motion.

The basket visual mesh is exported from round_laundry_basket_v2.blend. Its
mass and friction are provisional until the USD physics values are imported.
The void cell supplies the ground at world z=0.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import newton
import numpy as np

import robosimstudio as rss
from robosimstudio.core.tuning import notify
from robosimstudio.scene.entity import label_index

ASSETS = example_dir(__file__) / "assets"
COLLISION_HULLS = tuple(sorted((ASSETS / "collision_hulls").glob("Hull_*.obj")))


def basket_spec(*, mass=0.8, friction=0.8):
    return rss.body(
        name="round_laundry_basket",
        asset=rss.AssetSpec(
            path=str(ASSETS / "round_laundry_basket_v2.obj"),
            collision_paths=tuple(map(str, COLLISION_HULLS)),
        ),
        at=(0.159999996, -1.920000076, 0.001),
        mass=mass,
        friction=friction,
        color=(0.60, 0.43, 0.29),
    )


def build_sim(asset, motion, *, basket_mass=0.8, basket_friction=0.8, basket_com_z=None):
    if len(COLLISION_HULLS) != 65:
        raise ValueError(f"expected 65 separate laundry basket collision hulls, found {len(COLLISION_HULLS)}")
    gain_scale = float(os.environ.get("RSS_KICK_DIAGNOSTIC_GAIN_SCALE", "1"))
    if gain_scale != 1:
        if gain_scale <= 0:
            raise ValueError("diagnostic gain scale must be positive")
        tree = ET.parse(asset)
        for actuator in tree.findall("./actuator/position"):
            actuator.set("kp", str(float(actuator.get("kp")) * gain_scale))
            actuator.set("kv", str(float(actuator.get("kv")) * gain_scale ** 0.5))
        asset = Path(asset).with_name("g1_diagnostic_gain.xml")
        tree.write(asset, encoding="unicode")
    g1 = rss.EmbodimentSpec(
        name="g1_kick_basket_repair",
        asset=rss.AssetSpec(path=str(asset)),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        gravcomp=os.environ.get("RSS_KICK_DIAGNOSTIC_GRAVCOMP") == "1",
        arms=(rss.ArmSpec(
            name="whole_body",
            joint_names=motion.joint_names,
            ee_body_name="right_wrist_yaw_link",
            home_q=tuple(float(value) for value in motion.joint_pos[0]),
        ),),
    )
    scene = rss.Scene(cell="void", name="kick_basket_repair", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot(g1))
    scene.add(basket_spec(mass=basket_mass, friction=basket_friction))
    scene.add(rss.CameraSpec(
        name="review", width=960, height=540,
        eye=(1.65, -3.15, 1.55), look_at=(0.25, -1.8, 0.45),
    ))
    sim = scene.build(control_mode="joint_position")
    if basket_com_z is not None:
        body = label_index(list(sim.model.body_label), "round_laundry_basket")
        com = np.asarray(sim.model.body_com.numpy()).copy()
        com[body, 2] = basket_com_z
        sim.model.body_com.assign(com)
        notify(sim.pipeline, int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))
    return sim
