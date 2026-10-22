"""Independent multi-view visible wrapper landmarks after release, before the push."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    folder = a.run / "scene/10_flower_bouquet"
    cal = json.loads((folder / "head_marker_diagnostic_v3_header.json").read_text())
    raw = json.loads((a.run / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    K = np.array(cal["K"])
    cameras = []
    times = [123.4, 124.0, 124.6]
    for t in times:
        row = raw[str(t)]
        H = np.eye(4)
        H[:3, :3] = C @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
        H[:3, 3] = C @ row["position_raw"]
        H[2, 3] += 0.08245383203029633
        cameras.append(H @ cal["T_head_rigid_camera_optical"])

    def project(x):
        uv = []
        for T in cameras:
            v = (x - T[:3, 3]) @ T[:3, :3]
            u = K @ v
            uv.append(u[:2] / u[2])
        return np.array(uv)

    result = {
        "status": "estimated manual landmark triangulation, no exact geometry claim",
        "times": times,
        "sigma_px": 6.0,
        "points": {},
    }
    for name, points, initial in [
        ("stem_tip", [[749, 435], [710, 408], [707, 390]], [-2.62, -0.31, 0.68]),
        ("wrapper_top", [[749, 220], [688, 190], [667, 169]], [-2.70, -0.32, 1.02]),
    ]:
        fit = least_squares(lambda x, points=points: (project(x) - points).ravel(), initial)
        result["points"][name] = {
            "world": fit.x.tolist(),
            "pixels": points,
            "projected": project(fit.x).tolist(),
            "errors_px": np.linalg.norm(project(fit.x) - points, axis=1).tolist(),
            "jacobian_singular_values": np.linalg.svd(fit.jac, compute_uv=False).tolist(),
        }
    vector = np.array(result["points"]["wrapper_top"]["world"]) - result["points"]["stem_tip"]["world"]
    result["wrapper_length_m"] = float(np.linalg.norm(vector))
    result["axis_world"] = (vector / np.linalg.norm(vector)).tolist()
    (folder / "placed_stem_multiview.json").write_text(json.dumps(result, indent=2))
    print(result)
    import trimesh

    hull = trimesh.load(folder / "registered_assets/flower_bouquet_hull1.obj", force="mesh", process=False).vertices
    old = Rotation.from_quat(json.loads((folder / "flower_registered_candidate.json").read_text())["quaternion_xyzw"])
    observed_bottom = np.array([[749, 435], [710, 408], [707, 390]])
    observed_top = np.array([[749, 220], [688, 190], [667, 169]])

    def unpack(x):
        axis = np.r_[x[2:], np.sqrt(max(0.001, 1 - np.dot(x[2:], x[2:])))]
        rot = Rotation.align_vectors(axis[None], old.as_matrix()[:, 2][None])[0] * old
        return np.r_[x[:2], 0.680], rot, axis

    def residual(x):
        base, rot, axis = unpack(x)
        head = rot.apply(hull) + base
        return np.r_[
            (project(base) - observed_bottom).ravel() / 20,
            (project(base + 0.36 * axis) - observed_top).ravel() / 20,
            500 * (head[:, 1].min() + 0.486),
        ]

    fit = least_squares(
        residual, [-2.63, -0.30, -0.2, -0.14], bounds=([-2.78, -0.42, -0.5, -0.5], [-2.50, -0.24, 0.3, 0.1])
    )
    base, rot, axis = unpack(fit.x)
    supported = {
        "status": "multi-view plus explicit side-support candidate, not accepted",
        "base": base.tolist(),
        "quaternion_xyzw": rot.as_quat().tolist(),
        "axis": axis.tolist(),
        "bottom_errors_px": np.linalg.norm(project(base) - observed_bottom, axis=1).tolist(),
        "wrapper_top_errors_px": np.linalg.norm(project(base + 0.36 * axis) - observed_top, axis=1).tolist(),
        "foliage_side_min_y": float((rot.apply(hull) + base)[:, 1].min()),
        "camera_annotation_sigma_px": 20,
        "reference_times": times,
        "wrapper_length_m": 0.36,
    }
    (folder / "placed_stem_supported_multiview.json").write_text(json.dumps(supported, indent=2))
    print(supported)


if __name__ == "__main__":
    main()
