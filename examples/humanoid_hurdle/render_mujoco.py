# SPDX-License-Identifier: Apache-2.0
"""Render the hurdle setup with the G1's full visual meshes using MuJoCo EGL."""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(
    os.environ.get(
        "G1_MJCF_PATH",
        ROOT / "ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml",
    )
).expanduser().resolve()
OUTPUT = Path(__file__).with_name("hurdle_full_visual.png")


def main() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.find("visual/global").set("offwidth", "800")
    root.find("visual/global").set("offheight", "450")
    skybox = root.find("./asset/texture[@type='skybox']")
    skybox.set("rgb1", "0.12 0.14 0.15")
    skybox.set("rgb2", "0.02 0.02 0.02")
    robot_world = root.find("worldbody")
    robot_world.find("body").set("pos", "-0.9 0 0.763")
    stage = root.findall("worldbody")[-1]
    floor = next(geom for geom in stage.findall("geom") if geom.get("name") == "floor")
    floor.attrib.pop("material", None)
    floor.set("rgba", "0.16 0.18 0.19 1")
    ET.SubElement(
        stage,
        "geom",
        name="hurdle",
        type="box",
        size="0.0325 0.675 0.04",
        pos="0 0 0.04",
        rgba="0.48 0.50 0.51 1",
        friction="0.8 0.005 0.0001",
    )
    ET.SubElement(
        stage,
        "geom",
        name="goal_marker",
        type="box",
        size="0.025 0.32 0.002",
        pos="0.98 0 0.002",
        rgba="0.19 0.73 0.42 1",
        contype="0",
        conaffinity="0",
    )
    with tempfile.TemporaryDirectory(prefix="rss_hurdle_render_") as folder:
        render_dir = Path(folder)
        (render_dir / "meshes").symlink_to(SOURCE.parent / "meshes", target_is_directory=True)
        xml = render_dir / "scene.xml"
        tree.write(xml, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        home = {
            "left_hip_pitch_joint": -0.312,
            "left_knee_joint": 0.669,
            "left_ankle_pitch_joint": -0.363,
            "right_hip_pitch_joint": -0.312,
            "right_knee_joint": 0.669,
            "right_ankle_pitch_joint": -0.363,
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_elbow_joint": 0.6,
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_elbow_joint": 0.6,
        }
        for name, value in home.items():
            data.joint(name).qpos[0] = value
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (-0.02, 0.0, 0.61)
        camera.distance = 3.30
        camera.azimuth = 124
        camera.elevation = -22
        with mujoco.Renderer(model, height=450, width=800) as renderer:
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
