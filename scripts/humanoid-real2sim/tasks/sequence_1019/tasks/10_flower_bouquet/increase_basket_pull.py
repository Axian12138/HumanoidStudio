"""Move the basket start inward within the cabinet footprint and preserve its final placement."""


def main():

    import argparse
    import json
    import shutil
    from pathlib import Path

    import numpy as np
    from scipy.spatial.transform import Rotation as R

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    r = parser.parse_args().run
    p = json.load(open(r / "inputs/parent.json"))
    old = Path(p.get("original_tracks", p["parent"]["tracks"]))
    out = r / "tracks_wider"
    out.mkdir(exist_ok=True)
    for f in old.glob("*"):
        if f.is_file():
            shutil.copy2(f, out / f.name)
    b = dict(np.load(old / "small_wicker_basket.npz"))
    orig = b["position_w"].copy()
    br = R.from_quat(b["quaternion_xyzw"][0]).as_matrix()
    delta = br @ np.array([0, 0.10, 0])
    delta[2] = 0
    start = orig[535] + delta
    end = orig[620]
    t = 108 + np.arange(len(orig)) / 50
    u = np.clip((t - 118.7) / 1.3, 0, 1)
    u = u**3 * (10 - 15 * u + 6 * u * u)
    mask = t <= 120.4
    b["position_w"][mask] = (1 - u[mask, None]) * start + u[mask, None] * end
    np.savez(out / "small_wicker_basket.npz", **b)
    p["parent"]["tracks"] = str(out)
    p["original_tracks"] = str(old)
    (r / "inputs/parent.json").write_text(json.dumps(p, indent=2))
    print("delta", delta, "oldpull", np.linalg.norm(orig[620] - orig[535]), "newpull", np.linalg.norm(end - start))
    (r / "review/basket_shift.json").write_text(
        json.dumps(
            {
                "delta": delta.tolist(),
                "initial": start.tolist(),
                "end": end.tolist(),
                "old_pull_m": float(np.linalg.norm(orig[620] - orig[535])),
                "new_pull_m": float(np.linalg.norm(end - start)),
                "pull_interval_s": [118.7, 120],
                "unchanged_after_s": 120.4,
            },
            indent=2,
        )
    )
    for name in ["cabinet_support", "cabinet_side", "room_back_wall"]:
        f = Path(p["parent"]["assets"]) / (name + ".obj")
        if f.exists():
            v = np.array([[float(x) for x in l.split()[1:4]] for l in f.read_text().splitlines() if l.startswith("v ")])
            print(name, v.min(0), v.max(0))


if __name__ == "__main__":
    main()
