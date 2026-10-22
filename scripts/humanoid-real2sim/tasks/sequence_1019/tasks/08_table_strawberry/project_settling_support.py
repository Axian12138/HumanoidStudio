"""Explicitly project measured settling estimates above the original basket base."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    import trimesh
    from scipy.ndimage import maximum_filter1d
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=False)
    for p in a.candidate.iterdir():
        if p.is_file():
            shutil.copy2(p, a.output / p.name)
    vertices = np.asarray(
        trimesh.load_mesh(
            "/home/galbot/workspace/ScaleTrack/data/1012test/hoi_stage2/scene/assets/object_strawberry_existing_0_0.obj"
        ).vertices
    )
    report = {}
    for name, bp in [
        ("table_strawberry.npz", a.output / "lined_rect_basket.npz"),
        (
            "table_strawberry_successor_09.npz",
            Path(
                "/home/galbot/workspace/ScaleTrack/data/1019test/lined-rect-basket-kinematic-v1/lined_rect_basket_trajectory.npz"
            ),
        ),
    ]:
        z = np.load(a.output / name)
        d = {k: z[k].copy() for k in z.files}
        b = np.load(bp)
        br = Rotation.from_quat(
            b["quaternion_xyzw"] if "quaternion_xyzw" in b else b["quaternion_wxyz"][:, [1, 2, 3, 0]]
        )
        rr = br.inv() * Rotation.from_quat(d["quaternion_xyzw"])
        local = br.inv().apply(d["position_w"] - b["position_w"])
        delta = np.array(
            [max(0, 0.0082 - (rr[i].apply(vertices)[:, 2].min() + local[i, 2])) for i in range(len(local))]
        )
        delta[d["source_time_s"] < 97.12] = 0
        delta = maximum_filter1d(delta, size=7)
        delta[d["source_time_s"] < 97.12] = 0
        d["position_w"] += br.apply(np.stack([np.zeros(len(delta)), np.zeros(len(delta)), delta], axis=1))
        np.savez_compressed(a.output / name, **d)
        report[name] = {
            "max_vertical_correction_m": float(delta.max()),
            "nonzero_frames": np.flatnonzero(delta).tolist(),
        }
    x = np.load(a.output / "table_strawberry.npz")
    y = np.load(a.output / "table_strawberry_successor_09.npz")
    successor = {k: y[k].copy() for k in y.files}
    for k in ["position_w", "quaternion_xyzw"]:
        successor[k][:50] = x[k][400:]
    np.savez_compressed(a.output / "table_strawberry_successor_09.npz", **successor)
    report["scope"] = (
        "Original-size mesh support projection of independent RSS settling trace. This alters measured trace by a bounded upward correction; hybrid kinematic output, not physical rollout."
    )
    (a.output / "settling_support_projection.json").write_text(json.dumps(report, indent=2))
    shutil.copy2(__file__, a.output / "project_settling_support_source.py")
    print(json.dumps({k: v.get("max_vertical_correction_m") for k, v in report.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
