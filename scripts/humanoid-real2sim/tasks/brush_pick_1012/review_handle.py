"""Draw synchronized kinematic grasp close-ups and reference trajectories."""
from pathlib import Path

from PIL import Image, ImageDraw
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from audit_handle import mesh_points
from retarget_handle import BASE, MODEL, ORDER, OUT
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields
from scene import ASSETS, BASKET_POSITION, BASKET_QUAT_XYZW


def main():
    motion = MotionFields.load(OUT / 'motion.npz', ORDER)
    old = MotionFields.load(BASE, ORDER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = [model.joint(n).qposadr[0] for n in motion.joint_names]
    hands = [(b, mesh_points(model, model.body(b).name)) for b in range(model.nbody)
             if model.body(b).name.startswith('right_hand_') or model.body(b).name.startswith('right_wrist_')]
    hands = [(b, v) for b, v in hands if len(v)]
    basket = np.concatenate([trimesh.load(p, force='mesh', process=False).vertices
                             for p in sorted((ASSETS / 'white_storage_box/collision_hulls').glob('*.obj'))])
    basket = Rotation.from_quat(BASKET_QUAT_XYZW).apply(basket) + BASKET_POSITION
    brush_local = np.stack([
        np.linspace(-.135, .09, 100), np.zeros(100), np.full(100, .0609027427)
    ], axis=1)
    object_estimate = np.load(OUT / 'object_estimate.npz')
    fit = __import__('json').loads(Path('/home/galbot/workspace/ScaleTrack/data/1012test/1012-1/stage4-brush/review/handle_fit.json').read_text())
    center = np.asarray(fit['center_in_wrist_m'])
    w = motion.body_names.index('right_wrist_yaw_link')
    image = Image.new('RGB', (1600, 1000), 'white')
    draw = ImageDraw.Draw(image)
    def point(p, bounds, panel, axes):
        x0, x1, y0, y1 = bounds
        px, py, width, height = panel
        return (int(px + (p[axes[0]] - x0) / (x1 - x0) * width),
                int(py + height - (p[axes[1]] - y0) / (y1 - y0) * height))
    for row, frame in enumerate((525, 565)):
        brush = object_estimate['position_w'][frame] + Rotation.from_quat(
            object_estimate['quaternion_xyzw'][frame]).apply(brush_local)
        data.qpos[:3] = motion.body_pos_w[frame, 0]
        data.qpos[3:7] = motion.body_quat_w[frame, 0]
        data.qpos[addresses] = motion.joint_pos[frame]
        mujoco.mj_forward(model, data)
        hand = np.concatenate([v @ data.xmat[b].reshape(3, 3).T + data.xpos[b] for b, v in hands])
        new_eye = motion.body_pos_w[frame, w] + Rotation.from_quat(
            motion.body_quat_w[frame, w, [1, 2, 3, 0]]).apply(center)
        old_eye = old.body_pos_w[frame, w] + Rotation.from_quat(
            old.body_quat_w[frame, w, [1, 2, 3, 0]]).apply(center)
        for col, (axes, bounds, title) in enumerate(((
            (0, 1), (.65, 1.03, -.76, -.45), 'top X/Y'),
            ((0, 2), (.65, 1.03, .37, .78), 'side X/Z'))):
            panel = (70 + 790 * col, 85 + 475 * row, 680, 360)
            px, py, width, height = panel
            draw.rectangle((px, py, px + width, py + height), outline='#444444', width=2)
            draw.text((px, py - 30), f'{frame/50:.2f}s  {title}', fill='black')
            for array, color, radius in ((basket, '#dddddd', 1), (hand, '#2563eb', 1),
                                         (brush, '#16a34a', 2)):
                for p in array:
                    x, y = point(p, bounds, panel, axes)
                    if px <= x <= px + width and py <= y <= py + height:
                        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
            for p, color in ((new_eye, '#1d4ed8'), (old_eye, '#dc2626')):
                x, y = point(p, bounds, panel, axes)
                draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline='white', width=2)
    draw.text((70, 12), 'Brush handle grasp: basket grey, handle green, new hand blue, old fist eye red', fill='black')
    out = OUT / 'review_closeup.png'
    image.save(out)
    print(out)


if __name__ == '__main__':
    main()
