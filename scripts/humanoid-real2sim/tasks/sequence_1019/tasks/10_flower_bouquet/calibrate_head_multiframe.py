"""Fit a fixed head rigid-to-camera calibration to two cabinet observations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    cal = json.loads(
        Path(
            "/home/galbot/workspace/ScaleTrack/data/1019test/lined-rect-basket-kinematic-v1/metadata/camera_calibration.json"
        ).read_text()
    )["cameras"]["head"]
    k = np.array(cal["K"])
    world = np.array([[-2.578, -0.624, 0.661], [-2.578, 1.14, 0.661], [-2.578, 1.14, 0.17], [-2.578, -0.624, 0.17]])
    images = np.array(
        [
            [[628.0, 139.0], [1164.0, 55.0], [1120.0, 210.0], [647.0, 266.0]],
            [[432.0, 299.0], [1150.0, 104.0], [1060.0, 270.0], [504.0, 418.0]],
        ]
    )
    heads = json.loads((r / "inputs/10_flower_bouquet/head_keyframes.json").read_text())
    transforms = []
    for i in range(2):
        row = heads[str(i)]
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat(row[2]).as_matrix()
        t[:3, 3] = row[1]
        transforms.append(t)
    prior = np.array(
        json.loads((r / "scene/10_flower_bouquet/head_camera_rigid_position_candidate.json").read_text())[
            "camera_world_transform"
        ]
    )
    ext = np.linalg.inv(transforms[0]) @ prior
    initial = np.r_[Rotation.from_matrix(ext[:3, :3]).as_rotvec(), [0.0, 0.0, 0.0]]
    reports = []
    for label, d in [("pinhole", np.zeros(14)), ("raw_distortion", np.array(cal["D_raw"]))]:

        def pose(x):
            out = np.eye(4)
            out[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
            out[:3, 3] = x[3:]
            return out

        def projections(x, distortion=d):
            projected = []
            for t in transforms:
                w = np.linalg.inv(t @ pose(x))
                projected.append(
                    cv2.projectPoints(world, Rotation.from_matrix(w[:3, :3]).as_rotvec(), w[:3, 3], k, distortion)[0][
                        :, 0
                    ]
                )
            return np.array(projected)

        def fun(x):
            return np.r_[(projections(x) - images).ravel(), x[3:] / 0.1 * 3]

        fit = least_squares(fun, initial, bounds=([-6] * 3 + [-0.20] * 3, [6] * 3 + [0.20] * 3), max_nfev=500)
        pred = projections(fit.x)
        err = np.linalg.norm(pred - images, axis=2)
        reports.append(
            {
                "model": label,
                "T_head_rigid_camera_optical": pose(fit.x).tolist(),
                "camera_world_transforms": [(t @ pose(fit.x)).tolist() for t in transforms],
                "residuals_px": err.tolist(),
                "max_error_px": float(err.max()),
                "translation_norm_m": float(np.linalg.norm(fit.x[3:])),
                "source_times_s": [110.0, 111.0],
                "world_landmarks_m": world.tolist(),
                "source_landmarks_px": images.tolist(),
                "status": "pending_independent_validation",
            }
        )
    (r / "scene/10_flower_bouquet/head_multiframe_candidates.json").write_text(json.dumps(reports, indent=2))
    print([(z["model"], z["max_error_px"], z["translation_norm_m"]) for z in reports])


if __name__ == "__main__":
    main()
