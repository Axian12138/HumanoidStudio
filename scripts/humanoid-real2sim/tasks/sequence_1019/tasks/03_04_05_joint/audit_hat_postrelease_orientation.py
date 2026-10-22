"""Fixed-marker-camera, band-only postrelease pose sensitivity diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    r = a.run.resolve()
    d = r / "inputs/03_04_05/hat_final_visible_multiview"
    output = r / "review/scene/04_postrelease_band_orientation"
    output.mkdir(exist_ok=True)
    seed = json.loads((d / "parent_seed.json").read_text())
    marker = json.loads((d / "parent_bilateral_marker_fit.json").read_text())
    annotation = json.loads((d / "head1466_band_annotations.json").read_text())
    band = json.loads((r / "scene/03_04_05/hat_visual_shell_collision_v1/source_band_observations.json").read_text())
    base = np.array(
        json.loads((r / "review/scene/04_ribbon_bow_v2_release_pose.json").read_text())["variants"]["rigid"][
            "pose_xyzw"
        ]
    )
    k = np.array(seed["K"])
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ring = np.column_stack(
        [
            0.095 * 1.1086393424796992 * np.cos(angles),
            0.095 * 1.1086393424796992 * 0.235 / 0.28 * np.sin(angles),
            np.full(720, (0.031071428 + 0.047642857) / 2),
        ]
    )
    records = []
    for time in [47.7, 48, 48.3]:
        camera = np.array(
            json.loads((r / f"review/scene/parent_head{time:g}_marker_camera_pose_diagnostic.json").read_text())[
                "fitted_T_world_camera"
            ]
        )
        image = (
            r
            / "inputs/03_04_05"
            / ("head_48_header.jpg" if time == 48 else f"hat_pose_holdout/head_{time:g}_header.jpg")
        )
        records.append(
            {
                "time": time,
                "camera": camera,
                "points": np.array(band["observations_px"][f"{time:g}"]),
                "sigma": 3,
                "image": image,
            }
        )
    last = seed["rows"]["48.86666666666667"]
    result = {
        "status": "band_only_orientation_diagnostic_not_static_acceptance",
        "source_times_clock": "Firstthree requestedHeader-near times; final exactHeader48.839008424, encodedPTS48.86666666666667",
        "geometry": "Same fixed elliptical band ofcandidate v2, not source-measured metric ring",
        "limits": [
            "Band centerline alone doesnot determine fullhat material-axis orientation uniquely.",
            "Camera fits are independentmarkers but uncertain; seed/candidate sensitivity isnot statisticalconfidence.",
            "No static conclusion from finalvisibleframe; no observations afterhat leavesview.",
            "Bounds +/-100mm and +/-0.7rad are numerical search limits, not measurementconfidence.",
        ],
        "variants": {},
    }

    def rotation(x):
        return Rotation.from_rotvec(x[3:]) * Rotation.from_quat(base[3:])

    def project(x, rec):
        world = rotation(x).apply(ring) + base[:3] + x[:3]
        camera = rec["camera"]
        uv = ((world - camera[:3, 3]) @ camera[:3, :3]) @ k.T
        return uv[:, :2] / uv[:, 2:]

    def errors(x, rec):
        uv = project(x, rec)
        return np.linalg.norm(rec["points"][:, None] - uv[None], axis=2).min(axis=1)

    def fit(rows, initial=None):
        starts = [np.zeros(6) if initial is None else initial]
        if initial is None:
            starts += [np.r_[np.zeros(3), offset] for offset in [[0.2, 0, 0], [-0.2, 0, 0], [0, 0.2, 0], [0, -0.2, 0]]]
        fits = [
            least_squares(
                lambda x: np.r_[np.concatenate([errors(x, z) / z["sigma"] for z in rows]), x[:3] / 0.2, x[3:] / 1.0],
                start,
                bounds=(np.r_[[-0.1] * 3, [-0.7] * 3], np.r_[[0.1] * 3, [0.7] * 3]),
                max_nfev=200,
            )
            for start in starts
        ]
        return min(fits, key=lambda f: f.cost)

    def summary(x, rows):
        return {
            "pose_xyzw": np.r_[base[:3] + x[:3], rotation(x).as_quat()].tolist(),
            "normal_world": rotation(x).apply([0, 0, 1]).tolist(),
            "rows": [
                {
                    "time": z["time"],
                    "median_px": float(np.median(errors(x, z))),
                    "max_px": float(errors(x, z).max()),
                    "rms_px": float(np.sqrt(np.mean(errors(x, z) ** 2))),
                }
                for z in rows
            ],
        }

    for name, mount in [("seed", seed["T_head_camera"]), ("candidate", marker["T_head_camera_candidate"])]:
        last_rec = {
            "time": annotation["header"]["header_relative_s"],
            "camera": np.array(last["T_world_head"]) @ np.array(mount),
            "points": np.array(annotation["band_centerline_px"]),
            "sigma": 6,
            "image": d / "head_mcap_1466.jpg",
        }
        rows = [*records, last_rec]
        common = fit(rows)
        individual = []
        common_normal = rotation(common.x).apply([0, 0, 1])
        first_normal = None
        for index, rec in enumerate(rows):
            single = fit([rec], common.x)
            normal = rotation(single.x).apply([0, 0, 1])
            if first_normal is None:
                first_normal = normal
            info = summary(single.x, [rec])
            info["normal_angle_from_first_deg"] = float(np.degrees(np.arccos(np.clip(normal @ first_normal, -1, 1))))
            info["normal_angle_from_common_deg"] = float(np.degrees(np.arccos(np.clip(normal @ common_normal, -1, 1))))
            individual.append(info)
            im = Image.open(rec["image"])
            draw = ImageDraw.Draw(im)
            for x, color in [(common.x, "cyan"), (single.x, "orange")]:
                uv = project(x, rec)
                draw.line([tuple(v) for v in np.vstack([uv, uv[0]])], fill=color, width=2)
            for uv in rec["points"]:
                u, v = uv
                draw.ellipse((u - 2, v - 2, u + 2, v + 2), outline="yellow")
            draw.text(
                (15, 15),
                "cyan common pose / orange independent pose / yellow observed band; not visibility-rendered",
                fill="white",
            )
            im.save(output / f"{name}_{index}.png")
        result["variants"][name] = {
            "common_pose": summary(common.x, rows),
            "independent_poses": individual,
            "common_fit_cost": float(common.cost),
        }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
