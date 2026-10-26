"""Decelerate the final push, then hold basket and contained flower during hand withdrawal."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    from scipy.interpolate import CubicHermiteSpline
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.run / "inputs/parent.json").read_text())
    source = Path(info["original_tracks"])
    out = a.run / "tracks_stopped"
    out.mkdir(exist_ok=False)
    for f in source.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)
    basket = dict(np.load(source / "small_wicker_basket.npz"))
    flower = dict(np.load(source / "flower_bouquet.npz"))
    bp = basket["position_w"].copy()
    bq = basket["quaternion_xyzw"].copy()
    fp = flower["position_w"].copy()
    fq = flower["quaternion_xyzw"].copy()
    times = 108 + np.arange(len(bp)) / 50
    begin, end = 126.9, 127.3
    i, j = [round((t - 108) * 50) for t in [begin, end]]
    verts = np.array(
        [
            [float(x) for x in line.split()[1:4]]
            for line in (Path(info["parent"]["assets"]) / "small_wicker_basket.obj").read_text().splitlines()
            if line.startswith("v ")
        ]
    )
    target = bp[j].copy()
    target[0] = -3.02237009 + 0.006 - (verts @ R.from_quat(bq[j]).as_matrix().T)[:, 0].min()
    velocity = (bp[i + 1] - bp[i - 1]) * 25
    spline = CubicHermiteSpline([begin, end], np.array([bp[i], target]), np.array([velocity, np.zeros(3)]))
    for k in range(i, len(bp)):
        t = times[k]
        u = float(np.clip((t - begin) / (end - begin), 0, 1))
        w = u**3 * (10 - 15 * u + 6 * u * u)
        basket["position_w"][k] = spline(min(t, end))
        newr = Slerp([0, 1], R.from_quat(bq[[i, j]]))([w])[0]
        basket["quaternion_xyzw"][k] = newr.as_quat()
        src = min(k, j)
        oldr = R.from_quat(bq[src])
        relative = oldr.inv().apply(fp[src] - bp[src])
        flower["position_w"][k] = basket["position_w"][k] + newr.apply(relative)
        flower["quaternion_xyzw"][k] = (newr * oldr.inv() * R.from_quat(fq[src])).as_quat()
    for n, data in [("small_wicker_basket", basket), ("flower_bouquet", flower)]:
        np.savez(out / (n + ".npz"), **data)
    motion = a.run / "retarget/stopped_v1"
    motion.mkdir(parents=True, exist_ok=False)
    shutil.copy2(info["motion"], motion / "motion.npz")
    shutil.copy2(info["parent"]["order"], motion / "motion_order.json")
    info["parent"]["tracks"] = str(out)
    (a.run / "inputs/parent.json").write_text(json.dumps(info, indent=2))
    gaps = []
    for k in range(i, len(bp)):
        world = verts @ R.from_quat(basket["quaternion_xyzw"][k]).as_matrix().T + basket["position_w"][k]
        gaps.append(world[:, 0].min() + 3.02237009)
    report = {
        "begin_s": begin,
        "stopped_at_s": end,
        "old_outward_return_m": float(bp[-1, 0] - bp[:, 0].min()),
        "new_final_position": target.tolist(),
        "new_outward_step_max_m": float(max(0, np.diff(basket["position_w"][i:, 0]).max())),
        "post_stop_basket_motion_m": float(
            np.linalg.norm(basket["position_w"][j:] - basket["position_w"][j], axis=1).max()
        ),
        "post_stop_flower_motion_m": float(
            np.linalg.norm(flower["position_w"][j:] - flower["position_w"][j], axis=1).max()
        ),
        "backwall_clearance_min_m": float(min(gaps)),
    }
    (a.run / "review/stop_report.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
