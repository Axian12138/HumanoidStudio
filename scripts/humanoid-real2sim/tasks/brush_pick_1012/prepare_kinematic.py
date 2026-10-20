"""Prepare a fixed Dex3 command schedule on the reviewed stage8 body prior."""

import json
from dataclasses import replace
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256

ROOT = example_dir(__file__)
SCALETRACK = Path("/home/galbot/workspace/ScaleTrack")
SOURCE = SCALETRACK / "data/1012test/1012-1/stage8-bin-brush-natural/motion_candidate.npz"
ORDER = SCALETRACK / "retarget3/assets/robot/motion_order.json"
MODEL = SCALETRACK / "retarget3/assets/robot/robot.xml"
RIGHT_HAND = (
    "right_hand_thumb_0_joint", "right_hand_middle_0_joint", "right_hand_index_0_joint",
    "right_hand_thumb_1_joint", "right_hand_middle_1_joint", "right_hand_index_1_joint",
    "right_hand_thumb_2_joint",
)


def blend(joint, indices, start, stop, start_pose, end_pose):
    phase = np.linspace(0, 1, stop - start + 1)
    eased = phase * phase * (3 - 2 * phase)
    joint[start:stop + 1, indices] = (1 - eased[:, None]) * start_pose + eased[:, None] * end_pose


def main():
    source = MotionFields.load(SOURCE, ORDER)
    joint = source.joint_pos.astype(np.float64).copy()
    indices = [source.joint_names.index(name) for name in RIGHT_HAND]
    closed_before = joint[445, indices].copy()
    open_pose = joint[500, indices].copy()
    closed_after = joint[560, indices].copy()
    # Estimated command timing: opening starts well before the 10.5 s first
    # contact prior; closing starts after approach and finishes during lift.
    blend(joint, indices, 445, 490, closed_before, open_pose)
    joint[491:521, indices] = open_pose
    blend(joint, indices, 520, 560, open_pose, closed_after)
    prepared = replace(source, joint_pos=joint)
    motion = refresh_fk(prepared, MODEL)
    out = ROOT / "retarget"
    out.mkdir(exist_ok=True)
    motion.save(out / "kinematic_motion.npz")
    speed = np.max(np.abs(motion.joint_vel[300:600, indices]), axis=0)
    manifest = {
        "status": "kinematic_candidate_unreviewed",
        "source_recording": "1-2026-10-12-17-32-36",
        "source_motion": str(SOURCE), "source_sha256": sha256(SOURCE),
        "source_window_frames": [300, 600], "fps": 50,
        "finger_schedule_is_fixed_not_optimization_parameter": True,
        "right_hand_joint_names": RIGHT_HAND,
        "open_command_start_s": 8.90, "open_target_reached_s": 9.80,
        "close_command_start_s": 10.40, "close_target_reached_s": 11.20,
        "first_contact_prior_s": 10.50,
        "measured_open_close_timing": None,
        "maximum_commanded_hand_joint_speed_rad_s": speed.tolist(),
        "root_and_body_prior": "stage8-bin-brush-natural; unchanged outside fixed hand schedule",
        "output_sha256": sha256(out / "kinematic_motion.npz"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
