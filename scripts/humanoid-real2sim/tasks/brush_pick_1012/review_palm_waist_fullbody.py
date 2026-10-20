"""Draw same-frame human, body prior and new HOI candidate in shared world axes."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields
from retarget_palm_waist import BASE, ORDER, OUT

HUMAN = Path('/home/galbot/workspace/ScaleTrack/data/1012test/furniture_stage2_straight_wrist/1-2026-10-12-17-32-36/sidecars/human.npz')
MAP = ('pelvis', 'left_hip_pitch_link', 'right_hip_pitch_link', 'left_knee_link',
       'right_knee_link', 'left_ankle_roll_link', 'right_ankle_roll_link', None, None,
       'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_elbow_link',
       'right_elbow_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link')
EDGES = ((0, 1), (1, 3), (3, 5), (0, 2), (2, 4), (4, 6),
         (0, 9), (9, 11), (11, 13), (0, 10), (10, 12), (12, 14), (9, 10))


def main():
    old = MotionFields.load(BASE, ORDER)
    new = MotionFields.load(OUT / 'motion.npz', ORDER)
    human = np.load(HUMAN)['keypoints_w']
    image = Image.new('RGB', (1900, 950), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((25, 12), '6-12 s: human grey, full-body prior red, handle retarget blue; same world frame', fill='black')
    for col, frame in enumerate((480, 525, 575, 600)):
        for row, (axes, bounds, label) in enumerate((((0, 2), (-.1, 1.3, -.05, 1.25), 'X/Z'),
                                                      ((0, 1), (-.1, 1.3, -1., .2), 'X/Y'))):
            panel = (25 + col * 470, 70 + row * 435, 430, 350)
            px, py, width, height = panel
            draw.rectangle((px, py, px + width, py + height), outline='#444444', width=2)
            draw.text((px + 4, py + 4), f'{frame/50:.2f}s  {label}', fill='black')
            def project(point):
                x0, x1, y0, y1 = bounds
                return (int(px + (point[axes[0]] - x0) / (x1 - x0) * width),
                        int(py + height - (point[axes[1]] - y0) / (y1 - y0) * height))
            for pose, color, thickness in ((human[frame, :15], '#999999', 3),
                                           (np.stack([old.body_pos_w[frame, old.body_names.index(name)]
                                                      if name else np.full(3, np.nan) for name in MAP]), '#dc2626', 2),
                                           (np.stack([new.body_pos_w[frame, new.body_names.index(name)]
                                                      if name else np.full(3, np.nan) for name in MAP]), '#2563eb', 2)):
                for a, b in EDGES:
                    if np.isfinite(pose[[a, b]]).all():
                        draw.line((*project(pose[a]), *project(pose[b])), fill=color, width=thickness)
                for index in (0, 5, 6, 13, 14):
                    if np.isfinite(pose[index]).all():
                        x, y = project(pose[index])
                        draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
    out = OUT / 'review_fullbody.png'
    image.save(out)
    print(out)


if __name__ == '__main__':
    main()
