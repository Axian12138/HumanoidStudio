"""Audit straight-thumb face contact on the chair back's rear vertical face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from fit_hoi import HERE, MODEL, ORDER


def inspect(clip: Path):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    names = json.loads(ORDER.read_text())["robot_joint_names"]
    addresses = [model.joint(name).qposadr[0] for name in names]
    chair_mesh = trimesh.load_mesh(HERE / "grey_armchair_visual.obj")
    result = {}
    with np.load(clip / "motion.npz") as motion, np.load(clip / "object_estimate.npz") as chair:
        for sec in (79.2, 81.0, 83.0, 83.3):
            k = round((sec - 75.0) * 50)
            data.qpos[:3] = motion["body_pos_w"][k, 0]
            data.qpos[3:7] = motion["body_quat_w"][k, 0]
            data.qpos[addresses] = motion["joint_pos"][k]
            mujoco.mj_forward(model, data)
            chair_p = chair["position_w"][k]
            chair_r = Rotation.from_quat(chair["quaternion_xyzw"][k]).as_matrix()
            sample = {}
            for side, sign in (("left", 1), ("right", -1)):
                digit_report = {}
                centers = []
                for digit in ("thumb_1", "thumb_2"):
                    geom = model.geom(f"{side}_hand_{digit}_collision").id
                    mesh_id = model.geom_dataid[geom]
                    start = model.mesh_vertadr[mesh_id]
                    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
                    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                    local = (world - chair_p) @ chair_r
                    nearest, distances, faces = trimesh.proximity.closest_point(chair_mesh, local)
                    normals = chair_mesh.face_normals[faces]
                    signed = np.einsum("ij,ij->i", local - nearest, normals)
                    rear_face = (normals[:, 1] > .5) & (np.abs(normals[:, 2]) < .2)
                    near = rear_face & (distances <= .008) & (signed >= -.002)
                    digit_report[digit] = {
                        "minimum_signed_clearance_mm": float(signed.min() * 1000),
                        "rear_face_near_vertices": int(near.sum()),
                        "rear_face_contact_z_span_mm": float(np.ptp(local[near, 2]) * 1000)
                        if near.any() else 0.0,
                    }
                    centers.append(chair_r.T @ (data.geom_xpos[geom] - chair_p))
                thumb_axis = centers[1] - centers[0]
                thumb_axis /= np.linalg.norm(thumb_axis)
                distal = model.geom(f"{side}_hand_thumb_2_collision").id
                thumb_x = chair_r.T @ data.geom_xmat[distal].reshape(3, 3)[:, 0]
                chair_b_normal = np.array([sign * .53, .85, 0])
                chair_b_normal /= np.linalg.norm(chair_b_normal)
                digit_report["downward_angle_deg"] = float(np.degrees(np.arccos(
                    np.clip(-thumb_axis[2], -1, 1))))
                digit_report["thumb_plus_x_toward_b_dot"] = float(-thumb_x @ chair_b_normal)
                digit_report["thumb_minus_x_toward_b_dot"] = float(thumb_x @ chair_b_normal)
                palm = model.geom(f"{side}_hand_palm_collision").id
                palm_x = chair_r.T @ data.geom_xmat[palm].reshape(3, 3)[:, 0]
                digit_report["palm_x_chair_local"] = palm_x.tolist()
                digit_report["wrist_joints_rad"] = {
                    name: float(data.qpos[model.joint(f"{side}_wrist_{name}_joint").qposadr[0]])
                    for name in ("roll", "pitch", "yaw")
                }
                sample[side] = digit_report
            result[str(sec)] = sample
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {
        "criterion": "Thumb mesh within 8 mm of chair rear face (normal Y > 0.5, |normal Z| < 0.2); contact length is chair-local vertical Z span",
        "limitation": "Mesh is nonwatertight; finite kinematic samples do not measure contact force or prove full-frame collision freedom",
        "candidate": inspect(args.clip),
    }
    if args.comparison:
        report["comparison"] = inspect(args.comparison)
    out = args.out or args.clip.parent / "backface_contact_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
