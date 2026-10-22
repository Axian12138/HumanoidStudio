"""Connect a new crown grasp to the audited prior withdrawal without IK branch switches."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk

MODEL = Path(
    "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--base", default="03_04_05_joint_v62")
    p.add_argument("--output", default="03_04_05_joint_v64")
    a = p.parse_args()
    r = a.run
    c = r / "retarget/candidates" / a.base
    out = r / "retarget/candidates" / a.output
    out.mkdir(exist_ok=False)
    motion = MotionFields.load(c / "motion.npz", c / "motion_order.json")
    prior = MotionFields.load(
        r / "retarget/candidates/03_04_05_joint_v56/motion.npz",
        r / "retarget/candidates/03_04_05_joint_v56/motion_order.json",
    )
    arm = [
        "right_" + n + "_joint"
        for n in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    ]
    cols = [motion.joint_names.index(n) for n in arm] + [
        i for i, n in enumerate(motion.joint_names) if n.startswith("right_hand")
    ]
    q = motion.joint_pos.copy().astype(float)
    release = 1032
    delta = q[release, cols] - prior.joint_pos[release, cols]
    for k in range(release + 1, round((49.7 - 27) * 50) + 1):
        t = 27 + k / 50
        x = np.clip((t - 47.64) / (49.7 - 47.64), 0, 1)
        fade = 1 - (6 * x**5 - 15 * x**4 + 10 * x**3)
        q[k, cols] = prior.joint_pos[k, cols] + fade * delta
    result = refresh_fk(replace(motion, joint_pos=q), MODEL)
    for f in c.iterdir():
        if (
            f.is_file()
            and f.suffix in [".npz", ".json", ".xml"]
            and f.name not in ["motion.npz", "review.json", "manifest.json"]
            and not f.name.startswith(("collision_", "grasp_"))
        ):
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    (out / "review.json").write_text(
        json.dumps(
            {
                "status": "withdrawal_connection_pending_collision_source_speed_audit",
                "base": str(c),
                "prior": "03_04_05_joint_v56",
                "window_s": [47.64, 49.7],
                "release_and_hold_exact": True,
                "right_arm_finger_release_delta": delta.tolist(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
