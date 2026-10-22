"""Estimate supported bouquet pose and length from two views, with explicit priors."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    folder = r / "scene/10_flower_bouquet"
    h = json.loads((r / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    cal = json.loads((folder / "head_marker_diagnostic_v3_header.json").read_text())
    E = np.array(cal["T_head_rigid_camera_optical"])
    K = np.array(cal["K"])
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    cams = []
    support = (
        0.661
        + np.array(json.loads((folder / "shared_level_room_floor_v3.json").read_text())["T_room_correction"])[2, 3]
    )
    for t in ["111.0", "112.0"]:
        q = h[t]
        T = np.eye(4)
        T[:3, :3] = C @ Rotation.from_quat(q["quaternion_xyzw_raw"]).as_matrix()
        T[:3, 3] = C @ q["position_raw"]
        T[2, 3] += 0.08245383203029633
        cams.append(T @ E)

    def project(p):
        out = []
        for T in cams:
            v = (p - T[:3, 3]) @ T[:3, :3]
            uv = K @ v
            out.append(uv[:2] / uv[2])
        return np.array(out)

    estimates = {}
    for name, pix, z, x0 in [
        ("stem_tip", [[914, 125], [935, 259]], support + 0.04, [-2.65, 0.6]),
        ("flower_center", [[714, 199], [645, 342]], support + 0.13, [-2.65, 0.2]),
    ]:
        bounds = ([-3.0, -0.4], [-2.60, 1.2]) if name == "flower_center" else ([-3.0, -0.4], [-2.54, 1.2])
        fit = least_squares(lambda xy, z=z, pix=pix: (project(np.r_[xy, z]) - pix).ravel(), x0, bounds=bounds)
        point = np.r_[fit.x, z]
        estimates[name] = {
            "position": point.tolist(),
            "source_pixels": pix,
            "projected_pixels": project(point).tolist(),
            "errors_px": np.linalg.norm(project(point) - pix, axis=1).tolist(),
            "height_prior_m": z - support,
        }
    start = np.array(estimates["stem_tip"]["position"])
    vector = np.array(estimates["flower_center"]["position"]) - start
    length = np.linalg.norm(vector)
    axis = vector / length
    old = Rotation.from_quat([0.6412218182, -0.0448385959, -0.0534365622, 0.7641783915])
    delta = Rotation.align_vectors(axis[None], old.as_matrix()[:, 2][None])[0]
    quat = (delta * old).as_quat()
    scale = [1.0, 1.0, float(length / 0.405)]
    result = {
        "position_m": start.tolist(),
        "quaternion_xyzw": quat.tolist(),
        "native_scale": scale,
        "status": "supported proxy geometry estimate; not shape ground truth",
        "source_times": [111, 112],
        "landmarks": estimates,
        "camera_lateral_uncertainty_m": 0.02,
        "geometry_uncertainty_m": 0.055,
    }
    (folder / "flower_registered_candidate.json").write_text(json.dumps(result, indent=2))
    for path in (folder / "assets").glob("flower_bouquet_hull*.obj"):
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh.apply_scale(scale)
        mesh.export(folder / "registered_assets" / path.name)
    print(result)


if __name__ == "__main__":
    main()
