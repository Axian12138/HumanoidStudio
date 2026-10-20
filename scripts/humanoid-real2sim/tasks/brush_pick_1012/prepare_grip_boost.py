"""Make an isolated stronger-closure reference from the approved v3 motion."""

from dataclasses import replace
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

ROOT = example_dir(__file__)
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
MODEL = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/robot.xml")
SOURCE = ROOT / "retarget/palm_waist_v3/motion.npz"
OUTPUT = ROOT / "retarget/palm_waist_v3_grip18/motion.npz"
SIGNS = {
    "right_hand_thumb_1_joint": -1,
    "right_hand_thumb_2_joint": -1,
    "right_hand_middle_0_joint": 1,
    "right_hand_middle_1_joint": 1,
    "right_hand_index_0_joint": 1,
    "right_hand_index_1_joint": 1,
}


def smooth(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def main():
    source = MotionFields.load(SOURCE, ORDER)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    joint = source.joint_pos.copy()
    frames = np.arange(len(joint))
    envelope = smooth((frames - 522) / 38) * smooth((660 - frames) / 60)
    for name, sign in SIGNS.items():
        column = source.joint_names.index(name)
        lo, hi = model.joint(name).range
        joint[:, column] = np.clip(joint[:, column] + sign * 0.18 * envelope, lo, hi)
    candidate = refresh_fk(replace(source, joint_pos=joint), MODEL)
    candidate.save(OUTPUT)
    changed = np.max(np.abs(candidate.joint_pos - source.joint_pos), axis=0)
    assert all(changed[source.joint_names.index(name)] == 0 for name in source.joint_names if name not in SIGNS)
    print(f"{OUTPUT} source_sha256={sha256(SOURCE)} motion_sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
