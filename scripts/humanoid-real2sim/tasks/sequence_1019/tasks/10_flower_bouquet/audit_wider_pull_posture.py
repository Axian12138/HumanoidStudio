"""Independent geometry and posture evidence for the wider basket pull candidate."""


def main():
    import argparse
    import json
    from pathlib import Path

    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    r = parser.parse_args().run
    p = json.load(open(r / "inputs/parent.json"))
    s = MotionFields.load(r / "retarget/both_v5/motion.npz", p["parent"]["order"])
    old = MotionFields.load(p["motion"], p["parent"]["order"])
    out = {}
    m = mujoco.MjModel.from_xml_path(
        "/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
    )
    d = mujoco.MjData(m)
    ad = [m.joint(n).qposadr[0] for n in s.joint_names]
    for label, mo in [("parent", old), ("candidate", s)]:
        row = {}
        for side in ["left", "right"]:
            for axis in ["pitch", "yaw", "roll"]:
                q = mo.joint_pos[:, mo.joint_names.index(side + "_wrist_" + axis + "_joint")]
                row[side + "_wrist_" + axis] = {
                    "all_max_abs_deg": float(np.rad2deg(abs(q).max())),
                    "hold_max_abs_deg": float(np.rad2deg(abs(q[260:761]).max())),
                }
        vals = []
        for k in range(250, 651):
            d.qpos[:3] = mo.body_pos_w[k, 0]
            d.qpos[3:7] = mo.body_quat_w[k, 0]
            d.qpos[ad] = mo.joint_pos[k]
            mujoco.mj_forward(m, d)
            vals.append(
                (d.xpos[m.body("right_elbow_link").id] - d.xpos[m.body("right_shoulder_roll_link").id]).tolist()
            )
        row["right_elbow_shoulder_z_range_m"] = [
            float(np.min(np.array(vals)[:, 2])),
            float(np.max(np.array(vals)[:, 2])),
        ]
        out[label] = row
    out["root_position_max_delta_m"] = float(abs(s.body_pos_w[:, 0] - old.body_pos_w[:, 0]).max())
    legs = [c for c, n in enumerate(s.joint_names) if any(x in n for x in ["hip", "knee", "ankle"])]
    out["leg_joint_max_delta_rad"] = float(abs(s.joint_pos[:, legs] - old.joint_pos[:, legs]).max())
    out["waist_delta_deg"] = {
        n: float(
            np.rad2deg(abs(s.joint_pos[:, s.joint_names.index(n)] - old.joint_pos[:, s.joint_names.index(n)]).max())
        )
        for n in ["waist_pitch_joint", "waist_roll_joint"]
    }
    b = np.load(r / "tracks_wider/small_wicker_basket.npz")
    v = np.array(
        [
            [float(x) for x in l.split()[1:4]]
            for l in (Path(p["parent"]["assets"]) / "small_wicker_basket.obj").read_text().splitlines()
            if l.startswith("v ")
        ]
    )
    w = v @ R.from_quat(b["quaternion_xyzw"][0]).as_matrix().T + b["position_w"][0]
    out["basket_initial_world_bounds"] = [w.min(0).tolist(), w.max(0).tolist()]
    out["basket_initial_backwall_clearance_m"] = float(w[:, 0].min() + 3.02237009)
    out["basket_initial_base_above_support_m"] = float(w[:, 2].min() - 0.661)
    out["flower_track_unchanged"] = sha256(r / "tracks_wider/flower_bouquet.npz") == sha256(
        Path(p["original_tracks"]) / "flower_bouquet.npz"
    )
    out["motion_sha256"] = sha256(r / "retarget/both_v5/motion.npz")
    (r / "review/posture_support_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
