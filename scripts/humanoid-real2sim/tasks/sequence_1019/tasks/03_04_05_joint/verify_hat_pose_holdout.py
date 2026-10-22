"""Independent source-time contour checks with fixed marker camera extrinsics."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    folder = r / "inputs/03_04_05/hat_pose_holdout"
    raw = json.loads((folder / "head_raw_pose_audit_header.json").read_text())["rows"]
    timing = json.loads((folder / "image_header_timestamp_audit.json").read_text())
    cal = json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())
    K = np.array(cal["K"])
    E = np.array(cal["T_head_rigid_camera_optical"])
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    mesh = np.array(
        [
            list(map(float, l.split()[1:]))
            for l in (r / "scene/03_04_05/black_ribbon_hat.obj").read_text().splitlines()
            if l.startswith("v ")
        ]
    )
    track = np.load(r / "retarget/candidates/03_04_05_joint_v39/black_ribbon_hat.npz")
    fitted = np.array(json.loads((r / "review/scene/03_04_05_hat_silhouette_fit.json").read_text())["pose_xyzw"])
    observations = {
        47.7: [
            [870, 184],
            [906, 192],
            [935, 220],
            [952, 263],
            [960, 302],
            [937, 395],
            [908, 422],
            [870, 426],
            [815, 416],
            [772, 394],
            [750, 360],
            [743, 319],
            [750, 281],
            [774, 237],
            [816, 204],
        ],
        48.3: [
            [737, 138],
            [770, 147],
            [795, 171],
            [813, 206],
            [822, 250],
            [818, 294],
            [801, 326],
            [768, 345],
            [726, 352],
            [682, 341],
            [650, 313],
            [637, 279],
            [638, 239],
            [653, 199],
            [681, 162],
            [711, 145],
        ],
    }
    report = {
        "status": "independent_source_holdout_not_acceptance",
        "annotation_uncertainty_px": 5,
        "excluded": "48.7 motion blurred; no contour annotation",
        "records": [],
    }
    out = r / "review/scene/03_04_05_hat_pose_holdout"
    out.mkdir(exist_ok=True)
    for t, points in observations.items():
        row = raw[str(t)]
        T = np.eye(4)
        T[:3, :3] = C @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
        T[:3, 3] = C @ row["position_raw"]
        T[2, 3] += 0.08245383203029633
        camera = T @ E
        st = (timing["samples"][str(t)]["header_time_ns"] - timing["origin_log_time_ns"]) / 1e9
        k = round((st - 27) * 50)
        variants = {
            "track_at_header": np.r_[track["position_w"][k], track["quaternion_xyzw"][k]],
            "fixed_48_fit": fitted,
        }
        obs = np.array(points)
        record = {
            "requested_s": t,
            "source_header_s": st,
            "observations_px": points,
            "T_world_camera": camera.tolist(),
            "variants": {},
        }
        for label, pose in variants.items():
            world = Rotation.from_quat(pose[3:]).apply(mesh) + pose[:3]
            pc = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = pc @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            poly = uv[ConvexHull(uv).vertices]
            d = np.roll(poly, -1, axis=0) - poly
            u = np.clip(((obs[:, None] - poly) * d).sum(axis=2) / (d * d).sum(axis=1), 0, 1)
            err = np.linalg.norm(obs[:, None] - (poly[None] + u[:, :, None] * d), axis=2).min(axis=1)
            record["variants"][label] = {
                "pose_xyzw": pose.tolist(),
                "median_error_px": float(np.median(err)),
                "max_error_px": float(err.max()),
                "errors_px": err.tolist(),
            }
            im = Image.open(folder / f"head_{t:g}_header.jpg")
            draw = ImageDraw.Draw(im)
            draw.line([tuple(x) for x in np.vstack([poly, poly[0]])], fill="red", width=2)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="lime", width=2)
            im.save(out / f"head_{t:g}_{label}.png")
        report["records"].append(record)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print([{x["requested_s"]: x["variants"]} for x in report["records"]])


if __name__ == "__main__":
    main()
