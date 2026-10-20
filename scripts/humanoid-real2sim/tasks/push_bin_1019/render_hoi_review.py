"""Render same-frame source, body prior and HOI candidate in the reused 1012 room.

Run with ScaleTrack's Python environment and PYTHONPATH=ScaleTrack/retarget3/src.
The bin path is an estimated kinematic sidecar; no physics is stepped here.
"""

import os
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
from retarget3.motion_io import order
from retarget3.robot import mujoco


ROOT = example_dir(__file__)
SCALETRACK = Path("/home/galbot/workspace/ScaleTrack")
SCENE = SCALETRACK / "data/1012test/1012-1/stage9-bin-approach/scene/hoi_render.xml"
SOURCE_VIDEO = (
    SCALETRACK
    / "data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization"
    / "camera_head_color_image_raw_compressed.mp4"
)
TIMES = (3.2, 3.8, 4.3, 4.7, 5.5, 6.5, 7.5, 8.3, 9.0)


def camera(eye, look):
    result = mujoco.MjvCamera()
    delta = np.asarray(eye) - np.asarray(look)
    result.lookat[:] = look
    result.distance = np.linalg.norm(delta)
    result.azimuth = np.degrees(np.arctan2(-delta[1], -delta[0]))
    result.elevation = -np.degrees(np.arcsin(delta[2] / result.distance))
    return result


def label(image, title):
    cv2.rectangle(image, (0, 0), (640, 36), (24, 27, 32), -1)
    cv2.putText(image, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 2)
    return image


def main():
    with np.load(ROOT / "retarget/body_prior_scene.npz") as source:
        prior = {key: source[key] for key in source.files}
    with np.load(ROOT / "retarget/hoi_candidate_scene.npz") as source:
        candidate = {key: source[key] for key in source.files}
    with np.load(ROOT / "sidecars/white_waste_bin_estimated.npz") as source:
        bin_pos = source["position_w"]
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 960
    model.vis.quality.offsamples = 0
    renderer = mujoco.Renderer(model, 480, 640)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    _, names = order()
    joint_ids = np.asarray([model.joint(name).qposadr[0] for name in names])
    bin_id = model.joint("white_waste_bin_freejoint").qposadr[0]
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))

    def render(motion, frame, cam):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = motion["body_pos_w"][frame, 0]
        data.qpos[3:7] = motion["body_quat_w"][frame, 0]
        data.qpos[joint_ids] = motion["joint_pos"][frame]
        data.qpos[bin_id : bin_id + 3] = bin_pos[frame]
        data.qpos[bin_id + 3 : bin_id + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        return cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

    for seconds in TIMES:
        frame = min(round(seconds * 50), len(bin_pos) - 1)
        center = bin_pos[frame]
        close = camera(center + [-0.54, 0.58, 0.38], center + [0, 0, 0.16])
        room = camera([-2.65, 0.428, 1.65], [0.30, 0.55, 0.55])
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, source = cap.read()
        if not ok:
            raise RuntimeError(f"source video read failed at {seconds} s")
        source = cv2.resize(source, (640, 480))
        panels = (
            label(source, f"source head | {seconds:.1f} s"),
            label(render(prior, frame, close), "approved body prior | kinematic"),
            label(render(candidate, frame, close), "HOI candidate | kinematic"),
            label(render(candidate, frame, room), "HOI candidate | 1012 room"),
        )
        mosaic = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
        cv2.imwrite(str(ROOT / "review" / f"hoi_{seconds:04.1f}s.jpg"), mosaic)
        if seconds in (4.7, 5.5, 6.5, 7.5):
            top = camera(center + [-0.20, 0.0, 1.0], center + [0, 0, 0.15])
            overhead = np.hstack((
                label(render(prior, frame, top), "body prior | top view"),
                label(render(candidate, frame, top), "L-shaped hand candidate | top view"),
            ))
            cv2.imwrite(str(ROOT / "review" / f"hoi_top_{seconds:04.1f}s.jpg"), overhead)
    cap.release()
    renderer.close()


if __name__ == "__main__":
    main()
