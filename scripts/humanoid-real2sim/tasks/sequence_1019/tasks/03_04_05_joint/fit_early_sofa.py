"""Compare an XY-only sofa silhouette fit against independent camera frames."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    f = r / "review/scene/03_04_05_shared_room_validation"
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    cameras = {
        x["source_s"]: np.array(x["T_world_camera"])
        for x in json.loads((f / "projections.json").read_text())["records"]
    }
    points = np.array(
        [
            list(map(float, l.split()[1:]))
            for l in (r / "scene/03_04_05/sofa_world.obj").read_text().splitlines()
            if l.startswith("v ")
        ]
    )
    obs = {
        44: np.array([[200, 400], [300, 412], [450, 432], [610, 450], [650, 550], [666, 600], [649, 690]]),
        47: np.array([[100, 317], [250, 300], [400, 283], [510, 278], [553, 345], [588, 450], [596, 600]]),
    }

    def boundary(shift, t):
        T = cameras[t]
        v = (points + np.r_[shift, 0] - T[:3, 3]) @ T[:3, :3]
        v = v[v[:, 2] > 0.01]
        uv = v @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        return uv[ConvexHull(uv).vertices]

    def errors(shift, t):
        poly = boundary(shift, t)
        start = poly
        end = np.roll(poly, -1, axis=0)
        delta = end - start
        u = np.clip(((obs[t][:, None] - start) * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
        nearest = start[None] + u[:, :, None] * delta
        return np.linalg.norm(obs[t][:, None] - nearest, axis=2).min(axis=1)

    fit = least_squares(lambda x: errors(x, 44), [0, 0], bounds=(-0.25, 0.25))
    record = {
        "status": "diagnostic_not_accepted",
        "fit_time_s": 44,
        "holdout_time_s": 47,
        "translation_world_m": [*fit.x, 0],
        "uncertainty": "Approximate visible silhouette annotations; source upholstery may deform; fixed calibrated camera has early marker error up to21px.",
        "frames": {},
    }
    for t, observed in obs.items():
        record["frames"][str(t)] = {
            "pixels": obs[t].tolist(),
            "original_errors_px": errors([0, 0], t).tolist(),
            "fitted_errors_px": errors(fit.x, t).tolist(),
        }
        im = (
            Image.open(f / f"head_{t}_header.jpg")
            if (f / f"head_{t}_header.jpg").exists()
            else Image.open(f / f"head_{t}.png")
        )
        d = ImageDraw.Draw(im)
        for shift, color in [([0, 0], "blue"), (fit.x, "red")]:
            poly = boundary(shift, t)
            d.line([tuple(v) for v in np.vstack([poly, poly[0]])], fill=color, width=2)
        for x, y in observed:
            d.ellipse((x - 3, y - 3, x + 3, y + 3), outline="lime", width=2)
        im.save(f / f"head_{t}_sofa_fit.png")
    (r / "scene/03_04_05/early_sofa_registration.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record)


if __name__ == "__main__":
    main()
