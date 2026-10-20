"""Compare the Stage 5 bin trajectory with the fixed-motion reference."""

from __future__ import annotations

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = example_dir(__file__)
TRIAL = ROOT / "rollouts/stage5_coffee_mu_0p15"
SECOND_TRIAL = ROOT / "rollouts/stage5_coffee_mu_0p20"
BASELINE = ROOT / "rollouts/stage4_mu_0p10_com_0p03_contactfix/motion_repair/rollout.npz"


def main():
    poses = np.load(TRIAL / "bin_pose_xyzw.npy")
    second_poses = np.load(SECOND_TRIAL / "bin_pose_xyzw.npy")
    rollout = np.load(TRIAL / "rollout.npz")
    inputs = json.loads((TRIAL / "inputs.json").read_text())
    labels = inputs["body_labels"]
    thumb = next(i for i, label in enumerate(labels) if label.endswith("left_hand_thumb_finger_tip"))
    index = next(i for i, label in enumerate(labels) if label.endswith("left_hand_index_finger_tip"))
    tip = rollout["achieved_body_q"][:, [thumb, index], :3]
    estimated = np.load(ROOT / "sidecars/white_waste_bin_estimated.npz")["position_w"]
    old = np.load(BASELINE)
    old_bin = old["achieved_body_q"][:, -1, :3]
    tilt = np.degrees(np.arccos(np.clip(Rotation.from_quat(poses[:, 3:]).apply([0, 0, 1])[:, 2], -1, 1)))

    fig, ax = plt.subplots(figsize=(8, 7))
    for path in sorted((ROOT / "assets").glob("coffee_table_collision_*.obj")):
        mesh = trimesh.load(path, force="mesh")
        xy = np.asarray(mesh.vertices)[:, :2]
        hull = ConvexHull(xy)
        polygon = xy[hull.vertices]
        ax.fill(polygon[:, 0], polygon[:, 1], color="#b28761", alpha=0.22)
        ax.plot(np.r_[polygon[:, 0], polygon[0, 0]], np.r_[polygon[:, 1], polygon[0, 1]], color="#8c6748", lw=0.7)
    ax.plot(estimated[:, 0], estimated[:, 1], "--", color="#777777", label="Estimated bin path", lw=1.6)
    ax.plot(old_bin[:, 0], old_bin[:, 1], color="#9bbaeb", label="0.10, no table", lw=1.6)
    ax.plot(poses[:, 0], poses[:, 1], color="#d26723", label="0.15, table", lw=2.3)
    ax.plot(second_poses[:, 0], second_poses[:, 1], color="#9855a0", label="0.20, table", lw=1.6)
    for frame in (250, 300, 350, 375, 400):
        ax.plot(poses[frame, 0], poses[frame, 1], "o", color="#d26723", ms=4)
        ax.annotate(str(frame), poses[frame, :2], xytext=(3, 4), textcoords="offset points", fontsize=8)
    ax.scatter(tip[349:401, 0, 0], tip[349:401, 0, 1], s=4, color="#2577ac", label="Left thumb tip, frames 350–400")
    ax.scatter(tip[349:401, 1, 0], tip[349:401, 1, 1], s=4, color="#39a49c", label="Left index tip, frames 350–400")
    ax.set(xlabel="World x (m)", ylabel="World y (m)", title="Waste bin and coffee table, Stage 5 physical replay")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    image = ROOT / "review/stage5_coffee_mu_0p15_top.png"
    fig.savefig(image, dpi=180)
    metrics = {
        "frames": int(len(rollout["source_frame"])),
        "initial_bin_pos_m": poses[0, :3].tolist(),
        "final_bin_pos_m": poses[-1, :3].tolist(),
        "estimated_final_bin_pos_m": estimated[-1].tolist(),
        "final_xy_error_to_kinematic_estimate_m": float(np.linalg.norm(poses[-1, :2] - estimated[-1, :2])),
        "mu_0p20_final_bin_pos_m": second_poses[-1, :3].tolist(),
        "mu_0p20_final_xy_error_to_kinematic_estimate_m": float(np.linalg.norm(second_poses[-1, :2] - estimated[-1, :2])),
        "horizontal_travel_m": float(np.linalg.norm(poses[-1, :2] - poses[0, :2])),
        "maximum_tilt_deg": float(np.max(tilt)),
        "static_before_push": bool(np.max(np.linalg.norm(poses[:200, :2] - poses[0, :2], axis=1)) < 0.001),
        "thumb_index_xy_distance_to_bin_center_m_at_frame_375": np.linalg.norm(tip[374, :, :2] - poses[375, :2], axis=1).tolist(),
        "solver_note": "MuJoCo Warp repeatedly reported line-search iteration limit; pose results require visual/operator review.",
    }
    (ROOT / "review/stage5_coffee_mu_0p15_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(image)


if __name__ == "__main__":
    main()
