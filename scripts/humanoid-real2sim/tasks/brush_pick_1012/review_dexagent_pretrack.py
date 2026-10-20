"""Render the pre-tracker DexAgent-style hand proposal against its input motion."""
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from audit_handle import mesh_points
from dexagent_hand_only import MODEL, ORDER, OUT, SOURCE
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields
from scene import ASSETS, BASKET_POSITION, BASKET_QUAT_XYZW, BRUSH_POSITION, BRUSH_QUAT_XYZW


def main():
    old = MotionFields.load(SOURCE, ORDER)
    new = MotionFields.load(OUT / "projected_motion.npz", ORDER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    qaddr = np.array([model.joint(n).qposadr[0] for n in old.joint_names])
    parts = [(b, mesh_points(model, model.body(b).name)) for b in range(model.nbody)
             if model.body(b).name.startswith(("right_hand_", "right_wrist_"))]
    parts = [(b, v[::max(1, len(v) // 110)]) for b, v in parts if len(v)]
    brush = []
    for p in sorted((ASSETS / "green_hand_brush/collision_hulls").glob("*.obj")):
        v = trimesh.load(p, force="mesh", process=False).vertices
        brush.append(Rotation.from_quat(BRUSH_QUAT_XYZW).apply(v) + BRUSH_POSITION)
    brush = np.concatenate(brush)
    basket = []
    for p in sorted((ASSETS / "white_storage_box/collision_hulls").glob("*.obj")):
        v = trimesh.load(p, force="mesh", process=False).vertices
        basket.append(Rotation.from_quat(BASKET_QUAT_XYZW).apply(v) + BASKET_POSITION)
    basket = np.concatenate(basket)
    font = ImageFont.load_default()

    def hand_points(motion, frame):
        data.qpos[:3] = motion.body_pos_w[frame, 0]
        data.qpos[3:7] = motion.body_quat_w[frame, 0]
        data.qpos[qaddr] = motion.joint_pos[frame]
        mujoco.mj_forward(model, data)
        return np.concatenate([v @ data.xmat[b].reshape(3, 3).T + data.xpos[b] for b, v in parts])

    def panel(draw, box, axes, bounds, old_hand, new_hand, title):
        x, y, width, height = box
        draw.rectangle((x, y, x + width, y + height), fill="#fafafa", outline="#333333", width=2)
        draw.text((x + 10, y + 8), title, fill="#111111", font=font)
        left, right, bottom, top = bounds

        def project(v):
            px = x + (v[:, axes[0]] - left) * width / (right - left)
            py = y + height - (v[:, axes[1]] - bottom) * height / (top - bottom)
            return np.stack([px, py], axis=1)

        for points, color, radius in ((basket[::2], "#d4d4d8", 1),
                                      (brush, "#16a34a", 2),
                                      (old_hand, "#efb0b0", 2),
                                      (new_hand, "#2563eb", 2)):
            for px, py in project(points):
                if x <= px <= x + width and y <= py <= y + height:
                    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    def render(frame, *, large=False):
        if large:
            image = Image.new("RGB", (1320, 550), "white")
            boxes = [(25, 90, 620, 420), (675, 90, 620, 420)]
        else:
            image = Image.new("RGB", (1220, 390), "white")
            boxes = [(15, 65, 585, 305), (620, 65, 585, 305)]
        draw = ImageDraw.Draw(image)
        draw.text((20, 13), f"PRE-TRACKER FK  |  frame {frame} ({frame / old.fps:.2f}s)", fill="#111111", font=font)
        draw.text((20, 36), "pink: prior hand   blue: optimized hand   green: fixed brush   grey: basket", fill="#333333", font=font)
        old_hand = hand_points(old, frame)
        new_hand = hand_points(new, frame)
        panel(draw, boxes[0], (0, 1), (0.65, 1.03, -0.76, -0.45), old_hand, new_hand, "TOP  X/Y")
        panel(draw, boxes[1], (0, 2), (0.65, 1.03, 0.37, 0.78), old_hand, new_hand, "SIDE  X/Z")
        return image

    selected = (500, 525, 545, 560)
    sheet = Image.new("RGB", (1320, 2200), "white")
    for row, frame in enumerate(selected):
        sheet.paste(render(frame, large=True), (0, row * 550))
    sheet.save(OUT / "pretrack_review.png")
    frames = [render(f) for f in range(500, 581, 4)]
    frames[0].save(OUT / "pretrack_motion.gif", save_all=True, append_images=frames[1:],
                   duration=120, loop=0, optimize=True)
    print(OUT / "pretrack_review.png")
    print(OUT / "pretrack_motion.gif")


if __name__ == "__main__":
    main()
