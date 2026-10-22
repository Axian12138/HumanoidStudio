"""Sweep the original strawberry mesh against basket solids at100Hz."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from scipy.spatial.transform import Rotation, Slerp

    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    a = ap.parse_args()
    root = Path("/home/galbot/workspace/ScaleTrack/data/1012test/hoi_stage2/scene/assets")
    vertices = np.asarray(trimesh.load_mesh(root / "object_strawberry_existing_0_0.obj").vertices)
    spec = importlib.util.spec_from_file_location("geo", Path(__file__).parents[1] / "09_lined_rect_basket/fit_hoi.py")
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    report = {
        "scope": "All original berry mesh vertices at100Hz against five inherited basket solid SDFs; vertex sampling is not triangle intersection or dynamics proof.",
        "threshold_penetration_m": 0.001,
        "threshold_source": "same predeclared RSS free-rest penetration threshold1mm",
        "vertices": len(vertices),
        "tracks": {},
    }
    for name, bpath in [
        ("table_strawberry.npz", a.candidate / "lined_rect_basket.npz"),
        (
            "table_strawberry_successor_09.npz",
            Path(
                "/home/galbot/workspace/ScaleTrack/data/1019test/lined-rect-basket-kinematic-v1/lined_rect_basket_trajectory.npz"
            ),
        ),
    ]:
        x = np.load(a.candidate / name)
        b = np.load(bpath)
        n = len(x["position_w"])
        t = np.arange(n) / 50
        tt = np.arange(2 * n - 1) / 100

        def interp(z, tt=tt, t=t):
            return np.stack([np.interp(tt, t, z[:, i]) for i in range(3)], axis=1)

        xp = interp(x["position_w"])
        bp = interp(b["position_w"])
        xr = Slerp(t, Rotation.from_quat(x["quaternion_xyzw"]))(tt)
        bq = b["quaternion_xyzw"] if "quaternion_xyzw" in b else b["quaternion_wxyz"][:, [1, 2, 3, 0]]
        br = Slerp(t, Rotation.from_quat(bq))(tt)
        minima = []
        for i in range(len(tt)):
            points = br[i].inv().apply(xr[i].apply(vertices) + xp[i] - bp[i])
            minima.append(float(geo.basket_sdf(points).min()))
        minima = np.asarray(minima)
        offset = float(x["source_time_s"][0])
        report["tracks"][name] = {
            "frames_100hz": len(tt),
            "minimum_m": float(minima.min()),
            "worst_source_s": float(offset + tt[minima.argmin()]),
            "violation_source_s": (offset + tt[minima < -0.001]).tolist(),
            "post_release_minimum_m": float(minima[offset + tt >= 97.0].min()),
        }
    x = np.load(a.candidate / "table_strawberry.npz")
    y = np.load(a.candidate / "table_strawberry_successor_09.npz")
    report["successor_overlap_position_max_m"] = float(abs(x["position_w"][400:] - y["position_w"][:50]).max())
    report["successor_overlap_quaternion_max"] = float(
        abs(x["quaternion_xyzw"][400:] - y["quaternion_xyzw"][:50]).max()
    )
    (a.candidate / "object_sweep_review.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
