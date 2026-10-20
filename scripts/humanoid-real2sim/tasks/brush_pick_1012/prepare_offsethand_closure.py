"""Make a brush closure candidate on the ScaleTrack offsethand geometry."""

from dataclasses import replace
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import mujoco
import numpy as np

import retarget_palm_waist as ret
from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

ROOT = example_dir(__file__)
SOURCE = ROOT / 'retarget/palm_waist_v3/motion.npz'
OUTPUT = ROOT / 'retarget/palm_waist_v3_offsethand_clamp/motion.npz'


def main():
    source = MotionFields.load(SOURCE, ret.ORDER)
    model = mujoco.MjModel.from_xml_path(str(ret.MODEL))
    columns = np.array([source.joint_names.index('right_hand_' + name + '_joint') for name in ret.FINGERS])
    seed = source.joint_pos[560, columns]
    closed, gaps = ret.fit_closed_fingers(model, mujoco.MjData(model), source, 560,
                                          source.joint_pos, seed, ret.FINGERS,
                                          target_gap=.002)
    joint = source.joint_pos.copy()
    open_pose = source.joint_pos[500, columns]
    for frame in range(500, 661):
        closure = ret.smooth((frame - 518) / 42)
        return_to_prior = ret.smooth((frame - 600) / 60)
        grasp = open_pose * (1 - closure) + closed * closure
        joint[frame, columns] = (1 - return_to_prior) * grasp + return_to_prior * source.joint_pos[frame, columns]
    fitted = refresh_fk(replace(source, joint_pos=joint), ret.MODEL)
    fitted.save(OUTPUT)
    print(OUTPUT)
    print('closed angles', dict(zip(ret.FINGERS, closed, strict=True)))
    print('signed distal mesh gaps at source 560', gaps)


if __name__ == '__main__':
    main()
