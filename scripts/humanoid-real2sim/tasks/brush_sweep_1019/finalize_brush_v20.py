"""Cap the residual carry-window wrist bend and preserve the held brush transform."""

from dataclasses import replace
import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, refresh_fk, sha256


ROOT = example_dir(__file__)
SOURCE = ROOT / "retarget/brush_sweep_kinematic_v19"
OUT = ROOT / "retarget/brush_sweep_kinematic_v20"
ORDER = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json")
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")


def main():
    original = MotionFields.load(SOURCE / "motion.npz", ORDER)
    q = original.joint_pos.copy()
    column = original.joint_names.index("right_wrist_pitch_joint")
    bound = np.deg2rad(35.0)
    original_wrist = q[:, column].copy()
    q[:, column] = np.clip(q[:, column], -bound, bound)
    changed = np.flatnonzero(abs(q[:, column] - original_wrist) > 1e-8)
    candidate = refresh_fk(replace(original, joint_pos=q), MODEL)
    wrist = candidate.body_names.index("right_wrist_yaw_link")
    with np.load(SOURCE / "object_estimate.npz") as source_object:
        position = source_object["position_w"].copy()
        orientation = source_object["quaternion_xyzw"].copy()
        extra = {name: source_object[name] for name in source_object.files
                 if name not in ("position_w", "quaternion_xyzw")}
    before = Rotation.from_quat(original.body_quat_w[changed, wrist][:, [1, 2, 3, 0]])
    after = Rotation.from_quat(candidate.body_quat_w[changed, wrist][:, [1, 2, 3, 0]])
    local_position = before.inv().apply(position[changed] - original.body_pos_w[changed, wrist])
    local_orientation = before.inv() * Rotation.from_quat(orientation[changed])
    position[changed] = candidate.body_pos_w[changed, wrist] + after.apply(local_position)
    orientation[changed] = (after * local_orientation).as_quat()
    OUT.mkdir(parents=True, exist_ok=True)
    candidate.save(OUT / "motion.npz")
    np.savez_compressed(OUT / "object_estimate.npz", position_w=position,
                        quaternion_xyzw=orientation, **extra)
    report = json.loads((SOURCE / "review.json").read_text())
    report.update({
        "status": "kinematic_candidate_unreviewed",
        "parent_motion_sha256": sha256(SOURCE / "motion.npz"),
        "motion_sha256": sha256(OUT / "motion.npz"),
        "wrist_pitch_limit_deg": 35.0,
        "wrist_pitch_capped_frames": changed.tolist(),
        "object_track_changed_max_m": float(np.max(np.linalg.norm(position[changed] -
            np.load(SOURCE / "object_estimate.npz")["position_w"][changed], axis=1))),
    })
    (OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("wrist_pitch_limit_deg", "wrist_pitch_capped_frames",
                                                    "object_track_changed_max_m")}, indent=2))


if __name__ == "__main__":
    main()
