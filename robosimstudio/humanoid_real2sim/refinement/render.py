"""Render recorded dynamics and reference as a single side-by-side review video."""

from pathlib import Path

import numpy as np


def render_comparison(model, reference, baseline, optimized, output, *, fps=50):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    output = Path(output)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.3, -1.4, 0.65]
    camera.distance = 4.3
    camera.azimuth = 135
    camera.elevation = -25
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1  # RSS places robot collision geometry in group 3
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.8
    model.vis.headlight.specular[:] = 0.1
    robot_geoms = [i for i in range(model.ngeom) if model.geom_group[i] == 3]
    basket = model.body("lined_rect_basket").id
    model.geom_rgba[model.geom_bodyid == basket] = [0.95, 0.55, 0.15, 1.0]
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[i] = [0.18, 0.22, 0.27, 1.0]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    paths = [reference, baseline, optimized]
    labels = ["Kinematic reference (not physics)", "Direct joint-target execution", "Optimized control execution"]
    count = min(map(len, paths))
    try:
        with imageio.get_writer(output / "comparison.mp4", fps=fps // 2, codec="libx264", quality=7) as video:
            for frame in range(0, count, 2):
                panels = []
                colors = ([0.3, 0.8, 0.95, 1.0], [0.7, 0.7, 0.7, 1.0], [0.3, 0.85, 0.5, 1.0])
                for trajectory, label, color in zip(paths, labels, colors, strict=True):
                    model.geom_rgba[robot_geoms] = color
                    data.qpos[:] = trajectory[frame]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    picture = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(picture)
                    draw.rectangle((0, 0, 640, 36), fill=(10, 16, 26))
                    draw.text((12, 10), f"{label} | {frame / fps:.2f} s", fill="white")
                    panels.append(np.asarray(picture))
                combined = np.concatenate(panels, axis=1)
                video.append_data(combined)
                if frame in (0, 100, 250, 400):
                    imageio.imwrite(output / f"comparison_{frame:04d}.png", combined)
    finally:
        renderer.close()
