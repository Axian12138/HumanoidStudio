"""Bounded single-view pose diagnostic; does not establish support or acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    folder = r / "review/scene/03_04_05_shared_room_validation"
    old = json.loads((r / "review/scene/03_04_05_hat_source_pose_comparison.json").read_text())
    obs = np.array(old["source_observations_px"])
    base = np.array(old["variants"]["mujoco_at_image_header"]["pose_xyzw"])
    T = np.array(
        next(x for x in json.loads((folder / "projections.json").read_text())["records"] if x["source_s"] == 48)[
            "T_world_camera"
        ]
    )
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    v = np.array(
        [
            list(map(float, line.split()[1:]))
            for line in (r / "scene/03_04_05/black_ribbon_hat.obj").read_text().splitlines()
            if line.startswith("v ")
        ]
    )

    def project(x):
        rotation = Rotation.from_rotvec(x[3:]) * Rotation.from_quat(base[3:])
        w = rotation.apply(v) + base[:3] + x[:3]
        c = (w - T[:3, 3]) @ T[:3, :3]
        uv = c @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        return uv[ConvexHull(uv).vertices]

    def errors(x):
        poly = project(x)
        delta = np.roll(poly, -1, axis=0) - poly
        u = np.clip(((obs[:, None] - poly) * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
        return np.linalg.norm(obs[:, None] - (poly[None] + u[:, :, None] * delta), axis=2).min(axis=1)

    def residual(x):
        return np.r_[errors(x) / 5, x[:3] / 0.04, x[3:] / 0.4]

    fit = least_squares(
        residual, np.zeros(6), bounds=(np.r_[[-0.06] * 3, [-0.6] * 3], np.r_[[0.06] * 3, [0.6] * 3]), max_nfev=150
    )
    pose = np.r_[base[:3] + fit.x[:3], (Rotation.from_rotvec(fit.x[3:]) * Rotation.from_quat(base[3:])).as_quat()]
    report = {
        "status": "single_view_fit_diagnostic_not_accepted",
        "base_pose_xyzw": base.tolist(),
        "pose_xyzw": pose.tolist(),
        "delta_translation_m": fit.x[:3].tolist(),
        "delta_rotation_world_rad": fit.x[3:].tolist(),
        "errors_px": errors(fit.x).tolist(),
        "median_error_px": float(np.median(errors(fit.x))),
        "max_error_px": float(errors(fit.x).max()),
        "solver_success": bool(fit.success),
        "limitations": [
            "Single view contour fit has depth ambiguity",
            "No support or IK check",
            "Independent source holdout required",
            "Camera early marker uncertainty about21px retained",
        ],
    }
    im = Image.open(r / "inputs/03_04_05/head_48_header.jpg")
    draw = ImageDraw.Draw(im)
    poly = project(fit.x)
    draw.line([tuple(x) for x in np.vstack([poly, poly[0]])], fill="red", width=2)
    for x, y in obs:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="lime", width=2)
    im.save(folder / "head_48_hat_silhouette_fit.png")
    (r / "review/scene/03_04_05_hat_silhouette_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
