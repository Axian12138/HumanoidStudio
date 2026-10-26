"""Measure full visible hand/forearm clearance and the accepted-chair seam."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--version", default="v2")
    a = p.parse_args()
    r = a.run
    c = r / "retarget" / a.version
    model = "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    m = mujoco.MjModel.from_xml_path(model)
    d = mujoco.MjData(m)
    z = np.load(c / "motion.npz")
    order = json.loads((c / "motion_order.json").read_text())
    names = order["robot_joint_names"]
    adr = [m.joint(n).qposadr[0] for n in names]
    geoms = []
    for g in range(m.ngeom):
        name = m.body(m.geom_bodyid[g]).name
        if (
            (
                name.startswith(("right_hand", "left_hand"))
                or name
                in ["right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link", "right_elbow_link"]
            )
            and not m.geom_contype[g]
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            i = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]]
            geoms.append((g, name, v))
    ts = 83.5 + np.arange(len(z["joint_pos"])) / 50
    tt = np.arange(161) / 100 + 83.5
    rot = Slerp(ts, Rotation.from_quat(z["body_quat_w"][:, 0, [1, 2, 3, 0]]))(tt).as_quat()
    rows = []
    for k, t in enumerate(tt):
        d.qpos[:3] = [np.interp(t, ts, z["body_pos_w"][:, 0, j]) for j in range(3)]
        d.qpos[3:7] = rot[k, [3, 0, 1, 2]]
        d.qpos[adr] = [np.interp(t, ts, z["joint_pos"][:, j]) for j in range(len(adr))]
        mujoco.mj_kinematics(m, d)
        for g, name, v in geoms:
            world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            inside = (
                (world[:, 0] >= -1.03) & (world[:, 0] <= 0.57) & (world[:, 1] >= -2.97) & (world[:, 1] <= -2.11905405)
            )
            if inside.any():
                rows.append(
                    {"source_s": float(t), "body": name, "minimum_clearance_m": float(world[inside, 2].min() - 0.754)}
                )
    src = Path("/home/galbot/workspace/ScaleTrack/data/1019test")
    ch = np.load(src / "push-armchair-thumb-kinematic-v1/motion.npz")
    old = np.load(src / "brace-dining-table-kinematic-v1/motion.npz")
    thumb = [names.index(s + "_hand_thumb_0_joint") for s in ["left", "right"]]
    others = [i for i in range(len(names)) if i not in thumb]
    cross = [x for x in rows if x["body"].startswith("right") and 84.10 <= x["source_s"] <= 84.62]
    result = {
        "sampling": "100Hz interpolation with root SLERP; all visible mesh vertices, plus separate bidirectional room edge checks",
        "first_frame_non_thumb_joint_error_rad": float(
            abs(z["joint_pos"][0, others] - ch["joint_pos"][425, others]).max()
        ),
        "first_frame_thumb_root_change_rad": (z["joint_pos"][0, thumb] - ch["joint_pos"][425, thumb]).tolist(),
        "root_exact": bool(np.array_equal(z["body_pos_w"][:, 0], old["body_pos_w"][:, 0])),
        "support_after_85p1_non_thumb_joints_exact": bool(
            np.array_equal(z["joint_pos"][80:, others], old["joint_pos"][80:, others])
        ),
        "high_crossing_window_s": [84.10, 84.62],
        "high_crossing_minimum_mesh_clearance_m": min([x["minimum_clearance_m"] for x in cross], default=None),
        "approach_minimum_mesh_clearance_m": min([x["minimum_clearance_m"] for x in rows], default=None),
        "penetration_samples_over_1mm": [x for x in rows if x["minimum_clearance_m"] < -0.001],
        "records": rows,
    }
    (r / "review" / f"clearance_{a.version}.json").write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k not in ["records", "penetration_samples_over_1mm"]})
    print("penetration samples", len(result["penetration_samples_over_1mm"]))


if __name__ == "__main__":
    main()
