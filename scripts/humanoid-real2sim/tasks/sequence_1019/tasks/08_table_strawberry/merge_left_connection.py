"""Reproduce v22: preserve the audited v20 left arm while delaying right release."""

import argparse
from dataclasses import replace
from pathlib import Path

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release_candidate", type=Path)
    parser.add_argument("left_candidate", type=Path)
    parser.add_argument("output_motion", type=Path)
    args = parser.parse_args()
    if args.output_motion.exists():
        raise FileExistsError(args.output_motion)
    base = MotionFields.load(args.release_candidate / "motion.npz", args.release_candidate / "motion_order.json")
    left = MotionFields.load(args.left_candidate / "motion.npz", args.left_candidate / "motion_order.json")
    assert base.joint_names == left.joint_names
    ids = [
        i
        for i, name in enumerate(base.joint_names)
        if name.startswith("left_") and any(part in name for part in ("shoulder", "elbow", "wrist"))
    ]
    q = base.joint_pos.copy()
    q[:, ids] = left.joint_pos[:, ids]
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    refresh_fk(replace(base, joint_pos=q), model).save(args.output_motion)


if __name__ == "__main__":
    main()
