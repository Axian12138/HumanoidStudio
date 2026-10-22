"""Fit rack XY from early base/axis observations; reserve47s axis as holdout."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.optimize import least_squares

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    folder = r / "review/scene/03_04_05_shared_room_validation"
    source = json.loads((folder / "projections.json").read_text())
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    cameras = {x["source_s"]: np.array(x["T_world_camera"]) for x in source["records"]}
    base_pixels = np.array(
        [[451, 378], [470, 346], [508, 339], [546, 355], [558, 384], [545, 410], [510, 430], [476, 423]]
    )
    axis_pixels = {
        44.0: np.array([[315, 145], [400, 270], [478, 374]]),
        47.0: np.array([[853, 21], [830, 111], [807, 205], [797, 245]]),
    }
    original = np.array([2.239263954, -1.573561914])
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    ring = np.c_[0.17 * np.cos(theta), 0.17 * np.sin(theta), np.full(len(theta), 0.026)]
    axis = np.array([[0, 0, 0.05], [0, 0, 1.4]])

    def project(points, xy, t):
        w = points + np.r_[xy, 0]
        c = cameras[t]
        v = (w - c[:3, 3]) @ c[:3, :3]
        uv = v @ K.T
        return uv[:, :2] / uv[:, 2:]

    def line_error(xy, t):
        uv = project(axis, xy, t)
        line = np.cross(np.r_[uv[0], 1], np.r_[uv[1], 1])
        line /= np.linalg.norm(line[:2])
        return np.c_[axis_pixels[t], np.ones(len(axis_pixels[t]))] @ line

    def residual(xy):
        uv = project(ring, xy, 44.0)
        distance = np.linalg.norm(base_pixels[:, None] - uv[None], axis=2).min(axis=1)
        return np.r_[distance / 5, line_error(xy, 44.0) / 7]

    fit = least_squares(residual, original, bounds=(original - 0.35, original + 0.35))
    records = {}
    for t in [44.0, 47.0]:
        records[str(t)] = {
            "original_axis_error_px": abs(line_error(original, t)).tolist(),
            "fit_axis_error_px": abs(line_error(fit.x, t)).tolist(),
            "split": "fit" if t == 44 else "holdout",
        }
        im = Image.open(folder / f"head_{t:g}.png")
        d = ImageDraw.Draw(im)
        for xy, color in [(original, "blue"), (fit.x, "red")]:
            uv = project(ring, xy, t)
            d.line([tuple(v) for v in np.vstack([uv, uv[0]])], fill=color, width=2)
            ax = project(axis, xy, t)
            d.line([tuple(v) for v in ax], fill=color, width=3)
        for x, y in axis_pixels[t]:
            d.ellipse((x - 4, y - 4, x + 4, y + 4), outline="lime", width=2)
        if t == 44:
            for x, y in base_pixels:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), outline="lime", width=2)
        im.save(folder / f"head_{t:g}_rack_fit.png")
    uv = project(ring, fit.x, 44)
    report = {
        "status": "local_rack_registration_candidate",
        "original_axis_xy_m": original.tolist(),
        "fitted_axis_xy_m": fit.x.tolist(),
        "translation_world_m": np.r_[fit.x - original, 0].tolist(),
        "fixed_dimensions": {"base_radius_m": 0.17, "floor_z_m": 0, "base_top_z_m": 0.026},
        "observations": {
            "base_contour44": base_pixels.tolist(),
            "axis_pixels": {str(k): v.tolist() for k, v in axis_pixels.items()},
        },
        "base_contour_error_px": np.linalg.norm(base_pixels[:, None] - uv[None], axis=2).min(axis=1).tolist(),
        "axis_checks": records,
        "camera_uncertainty": "independent early marker max21.32px; this XY is candidate, not surveyed truth",
        "solver_success": bool(fit.success),
    }
    (r / "scene/03_04_05/early_rack_registration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
