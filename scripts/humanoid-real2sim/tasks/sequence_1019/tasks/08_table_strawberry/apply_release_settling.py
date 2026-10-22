"""Map an independently simulated rest transient into the moving basket frame."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--start-source-time", type=float, default=96.82)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=False)
    for p in a.candidate.iterdir():
        if p.is_file():
            shutil.copy2(p, a.output / p.name)
    shutil.copy2(__file__, a.output / "apply_release_settling_source.py")
    shutil.copy2(a.trace, a.output / "release_settling_trace.json")
    trace = json.loads(a.trace.read_text())
    poses = np.array(trace["poses"])
    berry_r = Rotation.from_quat(poses[:, 0, 3:])
    basket_r = Rotation.from_quat(poses[:, 1, 3:])
    local_p = basket_r.inv().apply(poses[:, 0, :3] - poses[:, 1, :3])
    local_r = basket_r.inv() * berry_r
    start = a.start_source_time
    for name, basket_path, time0 in [
        ("table_strawberry.npz", a.output / "lined_rect_basket.npz", 90),
        (
            "table_strawberry_successor_09.npz",
            Path(
                "/home/galbot/workspace/ScaleTrack/data/1019test/lined-rect-basket-kinematic-v1/lined_rect_basket_trajectory.npz"
            ),
            98,
        ),
    ]:
        old = np.load(a.output / name)
        data = {k: old[k].copy() for k in old.files}
        b = np.load(basket_path)
        times = time0 + np.arange(len(data["position_w"])) / 50
        mask = times >= start
        indices = np.clip(np.rint((times[mask] - start) * 50).astype(int), 0, len(poses) - 1)
        q = b["quaternion_xyzw"] if "quaternion_xyzw" in b else b["quaternion_wxyz"][:, [1, 2, 3, 0]]
        R = Rotation.from_quat(q[mask])
        data["position_w"][mask] = b["position_w"][mask] + R.apply(local_p[indices])
        data["quaternion_xyzw"][mask] = (R * local_r[indices]).as_quat()
        np.savez_compressed(a.output / name, **data)
    report = {
        "source_candidate": str(a.candidate),
        "source_trace": str(a.trace),
        "start_source_time_s": start,
        "trace_duration_s": 2,
        "mapping": "measured relative berry/basket transform from a static-container RSS rest trial; mapped into inherited basket trajectory after estimated ballistic drop",
        "limitations": [
            "No robot interaction simulated",
            "The moving-basket mapping after98s is kinematic, not a carrying dynamics proof",
        ],
        "terminal_object_in_basket_position": local_p[-1].tolist(),
        "terminal_object_in_basket_quaternion_xyzw": local_r[-1].as_quat().tolist(),
        "accepted": False,
    }
    (a.output / "release_settling_review.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
