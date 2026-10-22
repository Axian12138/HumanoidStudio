"""Geometry and continuity audit for the armchair backrest-hook candidate."""

from __future__ import annotations

import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields
from fit_hoi_v2 import HERE, MODEL, ORDER, OUT, SIDES, TABLE_NEAR_EDGE_Y, frame


def main() -> None:
    prior = MotionFields.load(HERE / "body_prior.npz", ORDER)
    motion = MotionFields.load(OUT / "motion.npz", ORDER)
    with np.load(HERE / "human.npz", allow_pickle=False) as source:
        human = source["keypoints_w"]
        contact = source["foot_contact"]
    with np.load(OUT / "human_motion_world.npz", allow_pickle=False) as overlay:
        np.testing.assert_array_equal(human, overlay["positions_w"])
    with np.load(OUT / "object_estimate.npz", allow_pickle=False) as obj:
        chair_pos = obj["position_w"]
        chair_quat = obj["quaternion_xyzw"]
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addr = np.asarray([model.joint(n).qposadr[0] for n in motion.joint_names])
    bodies = {side: {part: model.body(f"{side}_hand_{part}_finger_tip").id
                     for part in ("thumb", "index", "middle")} for side in SIDES}
    foot_bodies = [model.body(f"{side}_ankle_roll_link").id for side in SIDES]
    local = np.empty((frame(83.4) - frame(79.2) + 1, 2, 3, 3))
    human_hand_gap = np.empty((len(local), 2))
    foot_w = np.empty((len(motion.joint_pos), 2, 3))
    for k in range(len(motion.joint_pos)):
        data.qpos[:3] = motion.body_pos_w[k, 0]
        data.qpos[3:7] = motion.body_quat_w[k, 0]
        data.qpos[addr] = motion.joint_pos[k]
        mujoco.mj_forward(model, data)
        foot_w[k] = data.xpos[foot_bodies]
        if frame(79.2) <= k <= frame(83.4):
            rot = Rotation.from_quat(chair_quat[k]).as_matrix()
            for side_i, side in enumerate(SIDES):
                local[k - frame(79.2), side_i] = (
                    rot.T @ (data.xpos[list(bodies[side].values())] - chair_pos[k]).T
                ).T
                human_hand_gap[k - frame(79.2), side_i] = np.linalg.norm(
                    data.xpos[bodies[side]["index"]] - human[k, 15 + side_i]
                )

    diff = motion.joint_pos - prior.joint_pos
    assert np.max(np.abs(diff[:frame(78.35)])) < 1e-6
    assert np.max(np.abs(diff[frame(84.55):])) < 1e-6
    assert np.max(np.abs(motion.body_pos_w[frame(84.56):, 0] - prior.body_pos_w[frame(84.56):, 0])) < 1e-6
    limits = np.asarray([model.joint(n).range for n in motion.joint_names])
    violation = np.maximum(limits[:, 0] - motion.joint_pos, motion.joint_pos - limits[:, 1])
    assert np.max(violation) < 1e-4
    velocity = np.gradient(motion.joint_pos, .02, axis=0)
    # Absolute grounded-foot speed also includes the intended new step landing;
    # report only consecutive frames where the source marks contact on both.
    planted = contact[1:] & contact[:-1]
    foot_speed = np.linalg.norm(np.diff(foot_w, axis=0) * 50, axis=2)
    planted_speed = foot_speed[planted]
    verts = np.asarray([[float(x) for x in line.split()[1:4]]
                        for line in (HERE / "grey_armchair_visual.obj").open()
                        if line.startswith("v ")])
    high_verts = verts[verts[:, 2] > .753]
    final_world = Rotation.from_quat(chair_quat[-1]).apply(high_verts) + chair_pos[-1]
    seat_world = Rotation.from_quat(chair_quat[-1]).apply(verts[(verts[:, 2] > .4) & (verts[:, 2] < .6)]) + chair_pos[-1]
    closest_high_y = float(final_world[:, 1].min())
    assert closest_high_y > TABLE_NEAR_EDGE_Y, "chair backrest penetrates the table top"
    # The mesh tapers toward its crown. Finger tips must be in front of its
    # front face or above its top, rather than in the material as in v1.
    fingers = local[:, :, 1:, :]
    thumb = local[:, :, 0, :]
    audit = {
        "status": "kinematic_candidate_pending_visual_and_physical_review",
        "source_window_s": [75, 91],
        "contact_window_s": [79.2, 83.4],
        "fingertips_chair_local": {
            "thumb_min_height_m": float(thumb[:, :, 2].min()),
            "index_middle_max_forward_y_m": float(fingers[:, :, :, 1].max()),
            "index_middle_height_range_m": [float(fingers[:, :, :, 2].min()), float(fingers[:, :, :, 2].max())],
        },
        "table_gap_at_chair_high_mesh_m": closest_high_y - TABLE_NEAR_EDGE_Y,
        "seat_front_under_table_edge_depth_m": TABLE_NEAR_EDGE_Y - float(seat_world[:, 1].min()),
        "chair_final_position_m": chair_pos[-1].tolist(),
        "index_to_source_human_hand_aux_at_83_4_m": human_hand_gap[-1].tolist(),
        "root_max_tableward_adjustment_m": float(np.max(prior.body_pos_w[:, 0, 1] - motion.body_pos_w[:, 0, 1])),
        "planted_foot_speed_mean_m_s": float(np.mean(planted_speed)),
        "planted_foot_speed_p95_m_s": float(np.percentile(planted_speed, 95)),
        "planted_foot_speed_max_m_s": float(np.max(planted_speed)),
        "max_joint_speed_rad_s": float(np.max(np.abs(velocity))),
        "max_joint_limit_violation_rad": float(max(0, np.max(violation))),
        "limitations": [
            "Finger face clearance is checked in the local chair frame; full hand mesh sweep still needs visual review.",
            "The chair route is estimated from human pose and table clearance, not measured motion or dynamics.",
            "At the final contact frame, robot fingertips differ from source human hand auxiliary points by 0.16-0.21 m; check scene calibration and the exact human fingertip locations.",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
