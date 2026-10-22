"""Estimate a head-camera candidate from cabinet corners; never mark calibration accepted.

The corners are manually read from head_110.png. Four coplanar correspondences
fit a pose but cannot independently establish correct world registration.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
a = p.parse_args()
cal = json.loads(
    Path(
        "/home/galbot/workspace/ScaleTrack/data/1019test/lined-rect-basket-kinematic-v1/metadata/camera_calibration.json"
    ).read_text()
)
k = np.array(cal["cameras"]["head"]["K"])
world = np.array([[-2.554, -0.624, 0.661], [-2.554, 1.14, 0.661], [-2.554, 1.14, 0.17], [-2.554, -0.624, 0.17]])
pixels = np.array([[628.0, 139.0], [1164.0, 55.0], [1120.0, 210.0], [647.0, 266.0]])
ok, rv, tv = cv2.solvePnP(world, pixels, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
assert ok
rot = cv2.Rodrigues(rv)[0]
t = np.eye(4)
t[:3, :3] = rot.T
t[:3, 3] = -rot.T @ tv[:, 0]
proj = cv2.projectPoints(world, rv, tv, k, None)[0][:, 0]
report = {
    "status": "pending_independent_validation",
    "source_time_s": 110.0,
    "world_points_m": world.tolist(),
    "source_pixels": pixels.tolist(),
    "pixel_selection_uncertainty": 8.0,
    "camera_world_transform": t.tolist(),
    "fit_reprojection_errors_px": np.linalg.norm(proj - pixels, axis=1).tolist(),
    "warning": "fit residuals are in-sample; cabinet plane/edges uncertain; not calibrated by residual alone",
}
(a.run / "scene/10_flower_bouquet/head_camera_candidate.json").write_text(json.dumps(report, indent=2))
print(report)
