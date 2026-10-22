"""Audit the two hand revisions against the chair surface and MJCF limits."""

from __future__ import annotations

import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields

from fit_hoi import HERE, MODEL, ORDER, frame


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mesh = trimesh.load_mesh(HERE / "grey_armchair_visual.obj")
    for folder in ("hoi_v3_thumb_contact", "hoi_v3_palm_push"):
        path = HERE / folder
        motion = MotionFields.load(path / "motion.npz", ORDER)
        with np.load(path / "object_estimate.npz", allow_pickle=False) as chair:
            chair_pos = chair["position_w"]
            chair_rot = Rotation.from_quat(chair["quaternion_xyzw"]).as_matrix()
        addresses = [model.joint(name).qposadr[0] for name in motion.joint_names]
        limits = np.asarray([model.joint(name).range for name in motion.joint_names])
        outside = np.maximum(limits[:, 0] - motion.joint_pos, 0) + np.maximum(
            motion.joint_pos - limits[:, 1], 0
        )
        distances = {side: {part: [] for part in ("thumb", "index", "middle")}
                     for side in ("left", "right")}
        for k in range(frame(79.2), frame(83.4) + 1, 5):
            data.qpos[:3] = motion.body_pos_w[k, 0]
            data.qpos[3:7] = motion.body_quat_w[k, 0]
            data.qpos[addresses] = motion.joint_pos[k]
            mujoco.mj_forward(model, data)
            for side in distances:
                for part in distances[side]:
                    tip = data.xpos[model.body(f"{side}_hand_{part}_finger_tip").id]
                    local = chair_rot[k].T @ (tip - chair_pos[k])
                    _, distance, _ = trimesh.proximity.closest_point(mesh, local[None])
                    distances[side][part].append(float(distance[0]))
        report = json.loads((path / "review.json").read_text())
        report["contact_audit"] = {
            "sample_interval_frames": 5,
            "sample_window_s": [79.2, 83.4],
            "fingertip_to_chair_mesh_distance_mean_m": {
                side: {part: float(np.mean(values)) for part, values in parts.items()}
                for side, parts in distances.items()
            },
            "fingertip_to_chair_mesh_distance_max_m": {
                side: {part: float(max(values)) for part, values in parts.items()}
                for side, parts in distances.items()
            },
            "joint_limit_max_violation_rad": float(outside.max()),
        }
        (path / "review.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"candidate": folder, **report["contact_audit"]}, indent=2))


if __name__ == "__main__":
    main()
