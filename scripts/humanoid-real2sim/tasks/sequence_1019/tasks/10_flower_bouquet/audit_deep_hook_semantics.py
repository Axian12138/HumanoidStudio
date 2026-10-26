"""Audit palm seating, finger insertion, curl and shoulder/elbow direction independently."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    parent = json.load(open(a.run / "inputs/parent.json"))
    order = parent["parent"]["order"]
    s = MotionFields.load(a.motion, order)
    old = MotionFields.load(parent["motion"], order)
    b = np.load(Path(parent["parent"]["tracks"]) / "small_wicker_basket.npz")
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    ad = [m.joint(n).qposadr[0] for n in s.joint_names]
    wb = m.body("left_wrist_yaw_link").id
    el = m.body("left_elbow_link").id
    sh = m.body("left_shoulder_roll_link").id
    palm = m.geom("left_hand_palm_collision").id
    mi = m.geom_dataid[palm]
    vertices = m.mesh_vert[m.mesh_vertadr[mi] : m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
    out = {}
    for label, motion in [("parent", old), ("candidate", s)]:
        track_folder = (
            parent.get("original_tracks", parent["parent"]["tracks"])
            if label == "parent"
            else parent["parent"]["tracks"]
        )
        b = np.load(Path(track_folder) / "small_wicker_basket.npz")
        rows = []
        for k in range(535, 601):
            d.qpos[:3] = motion.body_pos_w[k, 0]
            d.qpos[3:7] = motion.body_quat_w[k, 0]
            d.qpos[ad] = motion.joint_pos[k]
            mujoco.mj_forward(m, d)
            br = R.from_quat(b["quaternion_xyzw"][k]).as_matrix()
            bp = b["position_w"][k]
            wr = d.xmat[wb].reshape(3, 3)
            rim = bp + br @ np.array([0, -0.16872634, 0.20841585])
            rimlocal = wr.T @ (rim - d.xpos[wb])
            v = (vertices @ d.geom_xmat[palm].reshape(3, 3).T + d.geom_xpos[palm] - bp) @ br
            top = v.copy()
            top[:, 0] = np.clip(top[:, 0], -0.10499038, 0.10499038)
            top[:, 1] = np.clip(top[:, 1], -0.17472634, -0.16272634)
            top[:, 2] = 0.20841585
            gap = float(np.linalg.norm(v - top, axis=1).min())
            tips = np.array(
                [br.T @ (d.xpos[m.body("left_hand_" + f + "_finger_tip").id] - bp) for f in ["index", "middle"]]
            )
            rootr = R.from_quat(motion.body_quat_w[k, 0, [1, 2, 3, 0]]).as_matrix()
            elbow = rootr.T @ (d.xpos[el] - d.xpos[sh])
            rows.append(
                {
                    "time_s": 108 + k / 50,
                    "palm_top_gap_m": gap,
                    "rim_in_wrist": rimlocal.tolist(),
                    "finger_depth_m": (0.20841585 - tips[:, 2]).tolist(),
                    "finger_inside_wall_m": (tips[:, 1] + 0.16272634).tolist(),
                    "elbow_relative_shoulder_robot": elbow.tolist(),
                }
            )
        out[label] = {
            "palm_top_gap_max_m": max(x["palm_top_gap_m"] for x in rows),
            "finger_depth_min_m": min(min(x["finger_depth_m"]) for x in rows),
            "finger_inside_wall_min_m": min(min(x["finger_inside_wall_m"]) for x in rows),
            "elbow_outward_y_min_m": min(x["elbow_relative_shoulder_robot"][1] for x in rows),
            "elbow_vertical_z_max_m": max(x["elbow_relative_shoulder_robot"][2] for x in rows),
            "records": rows,
        }
    diff = abs(s.joint_pos - old.joint_pos)
    changed = np.argwhere(diff > 1e-7)
    out["changed_joint_names"] = [s.joint_names[c] for c in np.unique(changed[:, 1])]
    out["change_interval_s"] = [108 + int(changed[:, 0].min()) / 50, 108 + int(changed[:, 0].max()) / 50]
    out["root_position_max_delta"] = float(abs(s.body_pos_w[:, 0] - old.body_pos_w[:, 0]).max())
    out["sha256"] = sha256(a.motion)
    out["scope"] = (
        "Palm native mesh vertices to finite rim top, fingertip FK insertion and arm direction; geometric evidence only, not friction or load support."
    )
    a.out.write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                n: {k: v for k, v in row.items() if k != "records"}
                for n, row in out.items()
                if n in ["parent", "candidate"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
