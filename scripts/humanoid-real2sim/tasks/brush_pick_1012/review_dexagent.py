"""Summarize a saved hand-only proposal and its actual physical replay."""
from __future__ import annotations

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from PIL import Image, ImageDraw

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256

ROOT = example_dir(__file__)
OUT = ROOT / "retarget/dexagent_hand_only_v1"
REPLAY = OUT / "physical_replay"
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")


def main():
    motion = MotionFields.load(OUT / "projected_motion.npz", ORDER)
    labels = json.loads((REPLAY / "scene_labels.json").read_text())
    with np.load(REPLAY / "rollout.npz") as trace:
        pos = trace["achieved_body_q"][:, :, :3]
        achieved_q = trace["achieved_joint_q"]
        commanded_q = trace["commanded_joint_target_q"]
        source_frame = trace["source_frame"]
    body_names = labels["body_names"]
    brush = body_names.index("green_hand_brush")
    wrist = next(i for i, n in enumerate(body_names) if n.endswith("/right_wrist_yaw_link"))
    pelvis = next(i for i, n in enumerate(body_names) if n.endswith("/pelvis"))
    ref_wrist = motion.body_names.index("right_wrist_yaw_link")
    target_wrist = motion.body_pos_w[source_frame, ref_wrist]
    wrist_err = np.linalg.norm(pos[:, wrist] - target_wrist, axis=1)
    root_err = np.linalg.norm(pos[:, pelvis] - motion.body_pos_w[source_frame, 0], axis=1)
    brush_z = pos[:, brush, 2]
    names = labels["joint_names"]
    starts = labels["joint_q_start"]
    fingers = {}
    for short in ("right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                  "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_index_0_joint", "right_hand_index_1_joint"):
        index = next(i for i, n in enumerate(names) if n.endswith("/" + short))
        coord = starts[index]
        fingers[short] = {"commanded_at_560_rad": float(commanded_q[560, coord]),
                          "achieved_at_560_rad": float(achieved_q[560, coord])}
    report = {
        "status": "physical_replay_rejected_for_training",
        "motion_sha256": sha256(OUT / "projected_motion.npz"),
        "rollout_sha256": sha256(REPLAY / "rollout.npz"),
        "brush_height_change_500_to_599_m": float(brush_z[599] - brush_z[500]),
        "brush_xy_displacement_525_to_599_m": float(np.linalg.norm(pos[599, brush, :2] - pos[525, brush, :2])),
        "max_brush_rise_545_to_599_over_500_m": float(np.max(brush_z[545:600]) - brush_z[500]),
        "wrist_error_at_frames_m": {str(f): float(wrist_err[f]) for f in (500, 525, 545, 560, 599)},
        "pelvis_error_at_frames_m": {str(f): float(root_err[f]) for f in (500, 525, 545, 560, 599)},
        "finger_joint_tracking": fingers,
        "training_data_ready": False,
        "finding": "The hand-only target is kinematically feasible but the tracked wrist does not reach the brush; the brush is pushed out and downward.",
    }
    (OUT / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    time = (source_frame + 1) / motion.fps
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    panels = [(80, 65, 1120, 300, -60, 70), (80, 395, 1120, 635, 0, 180)]
    for x0, y0, x1, y1, y_min, y_max in panels:
        draw.rectangle((x0, y0, x1, y1), outline="#444444", width=2)
        for tick in (10, 11, 12):
            x = x0 + (tick - 9.5) / 2.6 * (x1 - x0)
            draw.line((x, y0, x, y1), fill="#dddddd")
            draw.text((x - 10, y1 + 8), f"{tick}s", fill="#444444")
        for value in (y_min, 0, y_max):
            y = y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)
            draw.line((x0, y, x1, y), fill="#dddddd")
            draw.text((10, y - 7), str(value), fill="#444444")
    def plot(panel, values, color):
        x0, y0, x1, y1, y_min, y_max = panel
        pts = [(x0 + (float(t) - 9.5) / 2.6 * (x1 - x0),
                y1 - (float(v) - y_min) / (y_max - y_min) * (y1 - y0))
               for t, v in zip(time, values, strict=True) if 9.5 <= t <= 12.1]
        draw.line(pts, fill=color, width=3)
    plot(panels[0], (brush_z - brush_z[500]) * 1000, "#15803d")
    plot(panels[1], wrist_err * 1000, "#dc2626")
    plot(panels[1], root_err * 1000, "#2563eb")
    draw.text((80, 18), "Brush height relative to 10 s (green); 50 mm lift required", fill="#111111")
    draw.text((80, 345), "Tracking error: wrist (red), pelvis (blue)", fill="#111111")
    image.save(OUT / "acceptance.png")
    print(json.dumps({k: report[k] for k in ("status", "brush_height_change_500_to_599_m",
                                                "brush_xy_displacement_525_to_599_m", "wrist_error_at_frames_m")}, indent=2))


if __name__ == "__main__":
    main()
