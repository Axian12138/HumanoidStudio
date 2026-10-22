"""Compose independently fitted arms without changing the shared body prior."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--right-checkpoint", type=Path, required=True)
    p.add_argument("--left-motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    order = a.run.parents[1] / "curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json"
    prior = MotionFields.load(a.run / "retarget/body_prior/10_flower_bouquet/body_prior.npz", order)
    left = MotionFields.load(a.left_motion, order)
    right = np.load(a.right_checkpoint)
    q = prior.joint_pos.copy()
    left_columns = [
        i
        for i, name in enumerate(prior.joint_names)
        if name.startswith("left_") and ("hand_" in name or any(x in name for x in ["shoulder", "elbow", "wrist"]))
    ]
    q[:, left_columns] = left.joint_pos[:, left_columns]
    right_columns = [prior.joint_names.index(name) for name in right["joint_names"].tolist()]
    keys = right["keys"]
    if not np.all(np.diff(keys) == 1):
        raise ValueError("Only dense 50Hz right checkpoints may be composed")
    q[np.ix_(keys, right_columns)] = right["solutions"]
    if not np.array_equal(q[:50], prior.joint_pos[:50]):
        raise ValueError("09 overlap changed")
    robot = "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    a.out.mkdir(parents=True, exist_ok=False)
    refresh_fk(replace(prior, joint_pos=q), Path(robot)).save(a.out / "motion.npz")
    (a.out / "composition.json").write_text(
        json.dumps(
            {
                "status": "unreviewed composed candidate",
                "right": str(a.right_checkpoint),
                "left": str(a.left_motion),
                "root_and_legs": "exact original body prior",
                "overlap": "first50 q exact",
                "required": [
                    "independent joint contract",
                    "dense full hand collision",
                    "actual opposing contacts",
                    "live visual review",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
