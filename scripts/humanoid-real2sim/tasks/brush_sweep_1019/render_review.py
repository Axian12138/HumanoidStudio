"""Offline source/body/candidate panels with the exact offsethand G1 geometry."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir
import tempfile
import xml.etree.ElementTree as ET

import cv2
from retarget3 import runtime  # noqa: F401 - activates bundled MuJoCo wheel
import mujoco
import numpy as np


ROOT = example_dir(__file__)
ROBOT = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
ASSETS = ROOT.parent / "brush_pick_1012/assets"
VIDEO = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization/camera_head_color_image_raw_compressed.mp4")
TIMES = (11.5, 12.5, 16.5, 17.5, 19.5, 20.5, 25.5, 26.0)


def camera(eye, look):
    cam = mujoco.MjvCamera()
    offset = np.asarray(eye) - np.asarray(look)
    cam.lookat[:] = look
    cam.distance = np.linalg.norm(offset)
    cam.azimuth = np.degrees(np.arctan2(-offset[1], -offset[0]))
    cam.elevation = -np.degrees(np.arcsin(offset[2] / cam.distance))
    return cam


def label(image, title):
    cv2.rectangle(image, (0, 0), (image.shape[1], 35), (20, 25, 30), -1)
    cv2.putText(image, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def build_review_model(path):
    tree = ET.parse(ROBOT)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(ROBOT.parent / "meshes"))
    asset = root.find("asset")
    world = root.find("worldbody")
    for name in ("white_storage_box", "green_hand_brush"):
        ET.SubElement(asset, "mesh", name=name + "_review_mesh", file=str(ASSETS / name / "visual.obj"))
    ET.SubElement(world, "geom", name="coffee_table_review", type="box", size="0.30 0.70 0.02",
                  pos="1.06 -0.01 0.381", rgba="0.55 0.43 0.32 1", contype="0", conaffinity="0")
    box = ET.SubElement(world, "body", name="white_storage_box_review", pos="0.944999933 -0.569999993 0.400999993",
                        quat="0.7071067812 0 0 0.7071067812")
    ET.SubElement(box, "geom", type="mesh", mesh="white_storage_box_review_mesh", contype="0", conaffinity="0",
                  rgba="0.9 0.9 0.9 1")
    brush = ET.SubElement(world, "body", name="green_hand_brush_review")
    ET.SubElement(brush, "freejoint", name="green_hand_brush_review_joint")
    ET.SubElement(brush, "inertial", pos="0 0 0", mass="0.15", diaginertia="0.001 0.001 0.001")
    ET.SubElement(brush, "geom", type="mesh", mesh="green_hand_brush_review_mesh", contype="0", conaffinity="0",
                  rgba="0.38 0.64 0.31 1")
    tree.write(path)
    return mujoco.MjModel.from_xml_path(str(path))


def main():
    with np.load(ROOT / "retarget/body_prior_8_28/motion.npz") as stored:
        prior = {key: stored[key] for key in stored.files}
    with np.load(ROOT / "retarget/brush_sweep_kinematic_v8/motion.npz") as stored:
        candidate = {key: stored[key] for key in stored.files}
    with np.load(ROOT / "retarget/brush_sweep_kinematic_v8/object_estimate.npz") as stored:
        brush_position = stored["position_w"]
        brush_rotation = stored["quaternion_xyzw"]
    with tempfile.TemporaryDirectory() as temporary:
        model = build_review_model(Path(temporary) / "review.xml")
        model.vis.global_.offwidth = 1200
        model.vis.global_.offheight = 800
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        joint_names = json.loads(ORDER.read_text())["robot_joint_names"]
        joint_addresses = np.asarray([model.joint(name).qposadr[0] for name in joint_names])
        brush_address = model.joint("green_hand_brush_review_joint").qposadr[0]
        overview = camera((1.8, 0.65, 1.45), (0.85, -0.10, 0.48))
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        cap = cv2.VideoCapture(str(VIDEO))
        for t in TIMES:
            cam = overview
            k = round((t - 8.0) * 50)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, source = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read video at {t} s")
            source = cv2.resize(source, (640, 480))

            def render(motion, candidate_object):
                data.qpos[:] = model.qpos0
                data.qpos[:3] = motion["body_pos_w"][k, 0]
                data.qpos[3:7] = motion["body_quat_w"][k, 0]
                data.qpos[joint_addresses] = motion["joint_pos"][k]
                if candidate_object:
                    data.qpos[brush_address:brush_address + 3] = brush_position[k]
                    data.qpos[brush_address + 3:brush_address + 7] = brush_rotation[k][[3, 0, 1, 2]]
                else:
                    data.qpos[brush_address:brush_address + 3] = (0.8716025949, -0.5871267319, 0.4387029409)
                    data.qpos[brush_address + 3:brush_address + 7] = (0.98960447, 0.002078, 0.14354791, -0.00853049)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                return cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

            panels = [label(source, f"source head | {t:.1f} s"),
                      label(render(prior, False), "body prior | kinematic"),
                      label(render(candidate, True), "brush candidate | kinematic estimate")]
            cv2.imwrite(str(ROOT / "review" / f"candidate_{t:04.1f}s.jpg"), np.hstack(panels))
            look = brush_position[k] + np.array([0.0, 0.0, 0.05])
            cam = camera(look + np.array([0.48, 0.36, 0.42]), look)
            close_panels = [label(source.copy(), f"source head | {t:.1f} s"),
                            label(render(prior, False), "body prior | close view"),
                            label(render(candidate, True), "brush candidate | close view")]
            cv2.imwrite(str(ROOT / "review" / f"candidate_close_{t:04.1f}s.jpg"), np.hstack(close_panels))
        cap.release()
        renderer.close()


if __name__ == "__main__":
    main()
