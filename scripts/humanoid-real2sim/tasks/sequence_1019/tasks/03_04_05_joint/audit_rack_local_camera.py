"""Fit empty-rack three-tip source evidence and hold out later visible tips."""

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
    r = a.run
    source = json.loads((r / "review/scene/03_04_05_rack_support_hypotheses/report.json").read_text())["records"]
    cal = json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())
    K = np.array(cal["K"])
    axis = np.array([2.409627927, -1.469792849, 0.0])
    obs = {
        45: {"rack_30": [358, 210], "rack_55": [553, 178], "rack_80": [465, 94]},
        47: {"rack_30": [687, 210], "rack_80": [893, 136]},
        48: {"rack_30": [636, 183], "rack_80": [819, 106]},
    }
    records = {t: next(x for x in source if x["source_s"] == t and x["variant"] == "early_registered") for t in obs}

    records[48]["T_world_camera"] = json.loads(
        (r / "review/scene/parent_head48_marker_camera_pose_diagnostic.json").read_text()
    )["fitted_T_world_camera"]

    def projection(x, t, n):
        rec = records[t]
        T = np.array(rec["T_world_camera"])
        point = np.array(next(p for p in rec["points"] if p["geom"] == n)["world_m"])
        point = Rotation.from_euler("z", x[0]).apply((point - axis) * [x[1], x[1], 1]) + axis + [0, 0, x[2]]
        cam = (point - T[:3, 3]) @ T[:3, :3]
        uv = K @ cam
        return uv[:2] / uv[2]

    def residual(x):
        return np.r_[
            *[projection(x, 45, n) - uv for n, uv in obs[45].items()], x[0] / 0.6, (x[1] - 1) / 0.2, x[2] / 0.1
        ]

    fit = least_squares(residual, [0, 1, 0], bounds=([-0.8, 0.7, -0.15], [0.8, 1.3, 0.15]))
    report = {
        "status": "empty_rack_fit_pending_camera_error_budget_and_source_holdout",
        "fit_source_s": 45,
        "holdout_s": [47, 48],
        "fixed_axis_world_m": axis.tolist(),
        "camera48_source": "parent_head48_marker_camera_pose_diagnostic.json",
        "camera_fitted_with_rack": False,
        "fit": {"yaw_rad": float(fit.x[0]), "radial_scale": float(fit.x[1]), "height_offset_m": float(fit.x[2])},
        "observations_px": obs,
        "records": {},
    }
    for t, observations in obs.items():
        report["records"][t] = {
            n: {
                "predicted_px": projection(fit.x, t, n).tolist(),
                "observation_px": uv,
                "error_px": float(np.linalg.norm(projection(fit.x, t, n) - uv)),
            }
            for n, uv in observations.items()
        }
    (r / "review/scene/03_04_05_rack_empty45_local48_camera_holdout.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
