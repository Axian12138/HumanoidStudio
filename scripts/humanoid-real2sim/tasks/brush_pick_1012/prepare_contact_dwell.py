"""Probe a 0.8 s contact-pose dwell while the right hand closes."""

from dataclasses import replace
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

ROOT = example_dir(__file__)
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
MODEL = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/robot.xml")


def main():
    source = MotionFields.load(ROOT / "retarget/palm_waist_v3/motion.npz", ORDER)
    frames = np.arange(len(source.joint_pos))
    mapped = np.where(frames <= 522, frames, np.maximum(522, frames - 40))
    joint = source.joint_pos[mapped].copy()
    finger = [i for i, name in enumerate(source.joint_names) if name.startswith("right_hand_")]
    joint[:, finger] = source.joint_pos[:, finger]
    root_position = source.body_pos_w[mapped].copy()
    root_quaternion = source.body_quat_w[mapped].copy()
    candidate = refresh_fk(replace(
        source, joint_pos=joint, body_pos_w=root_position, body_quat_w=root_quaternion,
    ), MODEL)
    output = ROOT / "retarget/palm_waist_v3_dwell40/motion.npz"
    candidate.save(output)
    print(output)


if __name__ == "__main__":
    main()
