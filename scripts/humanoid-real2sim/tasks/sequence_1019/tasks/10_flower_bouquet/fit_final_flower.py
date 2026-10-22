"""Fit source 128 s wrapper landmarks with measured floor/side/back support constraints."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run / "scene/10_flower_bouquet"
    cal = json.loads((folder / "head_marker_diagnostic_v3_header.json").read_text())
    raw = json.loads((args.run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"][
        "128.0"
    ]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    H = np.eye(4)
    H[:3, :3] = C @ Rotation.from_quat(raw["quaternion_xyzw_raw"]).as_matrix()
    H[:3, 3] = C @ raw["position_raw"]
    H[2, 3] += 0.08245383203029633
    camera = H @ cal["T_head_rigid_camera_optical"]
    room = np.array(json.loads((folder / "shared_level_room_floor_v3.json").read_text())["T_room_correction"])
    head = trimesh.load(folder / "registered_assets/flower_bouquet_hull1.obj", force="mesh", process=False).vertices
    initial = Rotation.from_quat(
        json.loads((folder / "placed_stem_supported_multiview.json").read_text())["quaternion_xyzw"]
    )

    def unpack(x):
        axis = np.r_[x[2:], np.sqrt(1 - np.dot(x[2:], x[2:]))]
        rotation = Rotation.align_vectors(axis[None], initial.as_matrix()[:, 2][None])[0] * initial
        return np.r_[x[:2], 0.680], rotation, axis

    def project(world):
        v = (world - camera[:3, 3]) @ camera[:3, :3]
        uvw = v @ np.array(cal["K"]).T
        return uvw[:, :2] / uvw[:, 2:]

    def residual(x):
        base, rotation, axis = unpack(x)
        vertices = rotation.apply(head) + base
        curated = (vertices - room[:3, 3]) @ room[:3, :3]
        return np.r_[
            (project(np.array([base, base + 0.36 * axis])) - [[636, 366], [560, 185]]).ravel() / 15,
            300 * (curated[:, 0].min() + 3.04),
            300 * (curated[:, 1].min() + 0.624),
        ]

    fit = least_squares(
        residual, [-2.70, -0.29, -0.40, -0.15], bounds=([-2.9, -0.43, -0.65, -0.5], [-2.5, -0.15, 0.2, 0.1])
    )
    base, rotation, axis = unpack(fit.x)
    result = {
        "status": "estimated source128 pose with explicit measured two-wall support assumptions, independent rest pending",
        "source_s": 128.0,
        "base": base.tolist(),
        "quaternion_xyzw": rotation.as_quat().tolist(),
        "axis": axis.tolist(),
        "observed_pixels": [[636, 366], [560, 185]],
        "projected_pixels": project(np.array([base, base + 0.36 * axis])).tolist(),
        "residuals": residual(fit.x).tolist(),
        "camera_annotation_sigma_px": 15,
    }
    (folder / "final_flower_supported_source128.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
