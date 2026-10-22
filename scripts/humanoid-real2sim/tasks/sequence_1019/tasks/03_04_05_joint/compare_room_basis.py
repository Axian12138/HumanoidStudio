"""Compare raw and curated-composed registrations with early independent image evidence."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    f = r / "review/scene/03_04_05_shared_room_validation"
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    camera = {
        x["source_s"]: np.array(x["T_world_camera"])
        for x in json.loads((f / "projections.json").read_text())["records"]
    }
    late = np.array(
        json.loads((r / "scene/10_flower_bouquet/shared_level_room_floor_v3.json").read_text())["T_room_correction"]
    )
    base = np.eye(4)
    base[:3, 3] = [-0.04, -0.17, 0]
    transforms = {"raw": np.eye(4), "direct_floor_v3_incorrect_basis": late, "curated_composed": late @ base}
    colors = {"raw": "blue", "direct_floor_v3_incorrect_basis": "orange", "curated_composed": "red"}
    report = {
        "status": "early_comparison_not_acceptance",
        "T_raw_to_curated": base.tolist(),
        "transforms": {k: v.tolist() for k, v in transforms.items()},
        "drawer": {},
        "sofa": {},
    }

    def project(points, t, T, positive_only=False):
        w = points @ T[:3, :3].T + T[:3, 3]
        C = camera[t]
        v = (w - C[:3, 3]) @ C[:3, :3]
        v = v[v[:, 2] > 0.01] if positive_only else v
        uv = v @ K.T
        return uv[:, :2] / uv[:, 2:]

    edge = np.array([[1.45, -1.69, 0.882], [0.653, -1.69, 0.882]])
    for t, obs in {
        44: np.array([[861, 74], [1040, 78], [1220, 81], [1279, 83]]),
        51: np.array([[466, 431], [660, 412], [850, 391], [1040, 369], [1240, 350]]),
    }.items():
        im = Image.open(f / f"head_{t}.png")
        d = ImageDraw.Draw(im)
        record = {"observations_px": obs.tolist(), "variants": {}}
        for name, T in transforms.items():
            uv = project(edge, t, T)
            line = np.cross(np.r_[uv[0], 1], np.r_[uv[1], 1])
            line /= np.linalg.norm(line[:2])
            error = np.c_[obs, np.ones(len(obs))] @ line
            record["variants"][name] = {
                "signed_errors_px": error.tolist(),
                "median_abs_px": float(np.median(abs(error))),
                "max_abs_px": float(abs(error).max()),
            }
            d.line([tuple(x) for x in uv], fill=colors[name], width=3)
        for x, y in obs:
            d.ellipse((x - 3, y - 3, x + 3, y + 3), outline="lime", width=2)
        im.save(f / f"head_{t}_drawer_basis_comparison.png")
        report["drawer"][str(t)] = record
    sofa = np.array(
        [
            list(map(float, l.split()[1:]))
            for l in (r / "scene/03_04_05/sofa_world.obj").read_text().splitlines()
            if l.startswith("v ")
        ]
    )
    annotations = json.loads((r / "scene/03_04_05/early_sofa_registration.json").read_text())["frames"]
    for ts, row in annotations.items():
        t = int(ts)
        obs = np.array(row["pixels"])
        record = {}
        im = Image.open(f / f"head_{t}.png")
        d = ImageDraw.Draw(im)
        for name, T in transforms.items():
            uv = project(sofa, t, T, positive_only=True)
            poly = uv[ConvexHull(uv).vertices]
            start = poly
            end = np.roll(poly, -1, axis=0)
            delta = end - start
            u = np.clip(((obs[:, None] - start) * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
            errors = np.linalg.norm(obs[:, None] - (start[None] + u[:, :, None] * delta), axis=2).min(axis=1)
            record[name] = {
                "errors_px": errors.tolist(),
                "median_px": float(np.median(errors)),
                "max_px": float(errors.max()),
            }
            d.line([tuple(v) for v in np.vstack([poly, poly[0]])], fill=colors[name], width=2)
        for x, y in obs:
            d.ellipse((x - 3, y - 3, x + 3, y + 3), outline="lime", width=2)
        im.save(f / f"head_{t}_sofa_basis_comparison.png")
        report["sofa"][ts] = record
    (r / "scene/03_04_05/early_room_basis_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
