"""Apply a local measured carry-clearance correction without changing the grasp."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk


def main():
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    a = p.parse_args()
    src = a.run / "retarget/candidates/03_04_05_joint_v9"
    out = a.run / "retarget/candidates/03_04_05_joint_v10"
    out.mkdir(parents=True, exist_ok=True)
    m = MotionFields.load(src / "motion.npz", src / "motion_order.json")
    q = m.joint_pos.copy()
    time = 27 + np.arange(len(q)) / 50

    def smooth(start, end):
        u = np.clip((time - start) / (end - start), 0, 1)
        return u * u * (3 - 2 * u)

    q[:, m.joint_names.index("right_shoulder_pitch_joint")] -= 0.2 * smooth(39.5, 40) * (1 - smooth(41, 41.5))
    model = Path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    result = refresh_fk(replace(m, joint_pos=q), model)
    old = np.load(src / "black_ribbon_hat.npz")
    pos = old["position_w"].copy()
    quat = old["quaternion_xyzw"].copy()
    wi = m.body_names.index("right_wrist_yaw_link")
    g = 600
    r = 1050
    before = Rotation.from_quat(m.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    after = Rotation.from_quat(result.body_quat_w[:, wi][:, [1, 2, 3, 0]])
    local = before[g : r + 1].inv().apply(pos[g : r + 1] - m.body_pos_w[g : r + 1, wi])
    relative = before[g : r + 1].inv() * Rotation.from_quat(quat[g : r + 1])
    pos[g : r + 1] = result.body_pos_w[g : r + 1, wi] + after[g : r + 1].apply(local)
    quat[g : r + 1] = (after[g : r + 1] * relative).as_quat()
    for f in src.iterdir():
        if f.is_file() and f.name not in [
            "motion.npz",
            "black_ribbon_hat.npz",
            "review.json",
            "manifest.json",
            "protection_check.json",
            "collision_diagnostic.json",
        ]:
            shutil.copy2(f, out / f.name)
    result.save(out / "motion.npz")
    np.savez_compressed(
        out / "black_ribbon_hat.npz", position_w=pos, quaternion_xyzw=quat, fps=np.array(50), source_time_s=time
    )
    review = json.loads((src / "review.json").read_text())
    review["status"] = "carry_clearance_refined_not_passed"
    review["hat_carry_clearance"] = {
        "joint": "right_shoulder_pitch_joint",
        "delta_rad": -0.2,
        "window_s": [39.5, 41.5],
        "probe_s": 40.6,
        "before_hip_gap_m": -0.0401318,
        "after_hip_gap_m": 0.0411403,
        "full_sweep_pending": True,
    }
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")


if __name__ == "__main__":
    main()
