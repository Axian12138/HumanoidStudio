"""Measure whole-thumb chair contact, rather than tip proximity alone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from fit_hoi import HERE, MODEL, ORDER


def inspect(clip: Path, seconds: tuple[float, ...], threshold_m: float):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    names = json.loads(ORDER.read_text())["robot_joint_names"]
    addresses = [model.joint(name).qposadr[0] for name in names]
    chair_mesh = trimesh.load_mesh(HERE / "grey_armchair_visual.obj")
    report = {}
    with np.load(clip / "motion.npz") as motion, np.load(clip / "object_estimate.npz") as chair:
        for sec in seconds:
            k = round((sec - 75.0) * 50)
            data.qpos[:3] = motion["body_pos_w"][k, 0]
            data.qpos[3:7] = motion["body_quat_w"][k, 0]
            data.qpos[addresses] = motion["joint_pos"][k]
            mujoco.mj_forward(model, data)
            chair_p = chair["position_w"][k]
            chair_r = Rotation.from_quat(chair["quaternion_xyzw"][k]).as_matrix()
            sample = {}
            for side in ("left", "right"):
                digit_report = {}
                centers = []
                for digit in ("thumb_1", "thumb_2"):
                    geom = model.geom(f"{side}_hand_{digit}_collision").id
                    mesh_id = model.geom_dataid[geom]
                    start = model.mesh_vertadr[mesh_id]
                    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
                    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                    local = (world - chair_p) @ chair_r
                    nearest, dist, faces = trimesh.proximity.closest_point(chair_mesh, local)
                    signed = np.einsum("ij,ij->i", local - nearest,
                                       chair_mesh.face_normals[faces])
                    contact = (dist <= threshold_m) & (signed >= -0.002) & (local[:, 2] > .69)
                    digit_report[digit] = {
                        "minimum_signed_clearance_mm": float(signed.min() * 1000),
                        "near_surface_vertices": int(contact.sum()),
                        "near_surface_y_span_mm": float(np.ptp(local[contact, 1]) * 1000)
                        if contact.any() else 0.0,
                    }
                    centers.append(chair_r.T @ (data.geom_xpos[geom] - chair_p))
                axis = centers[1] - centers[0]
                axis /= np.linalg.norm(axis)
                digit_report["axis_angle_to_chair_top_edge_deg"] = float(
                    np.degrees(np.arccos(np.clip(abs(axis[0]), 0, 1))))
                digit_report["axis_chair_local"] = axis.tolist()
                sample[side] = digit_report
            report[str(sec)] = sample
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seconds = (79.2, 81.0, 83.0)
    threshold_m = .008
    result = {
        "criterion": "thumb collision mesh within 8 mm of chair mesh and no more than 2 mm inside; chair-local y span measures length across the top edge",
        "limitations": "visual mesh is nonwatertight and face normals are not globally consistent; this is kinematic geometry, not force or friction",
        "source_seconds": seconds,
        "candidate": inspect(args.clip, seconds, threshold_m),
    }
    if args.comparison:
        result["comparison"] = inspect(args.comparison, seconds, threshold_m)
    output = args.out or args.clip.parent / "contact_patch_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
