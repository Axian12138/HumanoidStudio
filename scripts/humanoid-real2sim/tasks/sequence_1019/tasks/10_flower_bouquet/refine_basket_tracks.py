"""Refine basket translation/yaw from source rim observations, preserving flower poses."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    folder = args.run / "scene/10_flower_bouquet"
    placed = json.loads((folder / "placed_basket_multiview.json").read_text())
    final = json.loads((folder / "placed_basket_final_source128.json").read_text())
    data = dict(np.load(args.source / "small_wicker_basket.npz"))
    t = data["source_time_s"]

    def smooth(begin, end):
        x = np.clip((t - begin) / (end - begin), 0, 1)
        return x * x * (3 - 2 * x)

    pull = smooth(119.0, 120.4)
    push = smooth(125.0, 127.8)
    plateau_index = np.argmin(abs(t - 124.6))
    a = np.array(placed["position"]) - data["position_w"][plateau_index]
    b = np.array(final["position"]) - data["position_w"][-1]
    data["position_w"] = data["position_w"] + pull[:, None] * a + push[:, None] * (b - a)
    yaw = Rotation.from_quat(data["quaternion_xyzw"]).as_euler("xyz")[:, 2]
    delta_a = placed["world_rectangle_yaw"] + np.pi / 2 - yaw[plateau_index]
    delta_b = final["world_rectangle_yaw"] + np.pi / 2 - yaw[-1]
    angle = yaw + pull * delta_a + push * (delta_b - delta_a)
    data["quaternion_xyzw"] = Rotation.from_euler("z", angle[:, None]).as_quat()
    np.savez(args.out / "small_wicker_basket.npz", **data)
    (args.out / "flower_bouquet.npz").write_bytes((args.source / "flower_bouquet.npz").read_bytes())
    (args.out / "tracks.json").write_bytes((args.source / "tracks.json").read_bytes())
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "status": "source rim refinement candidate, live/geometry/rest review pending",
                "source_tracks": str(args.source),
                "placement_evidence": str(folder / "placed_basket_multiview.json"),
                "final_evidence": str(folder / "placed_basket_final_source128.json"),
                "plateau_translation_correction": a.tolist(),
                "final_translation_correction": b.tolist(),
                "flower": "unchanged source-constrained v7; not retranslated using wrist displacement",
                "limitation": "rim corners rounded/partly occluded and camera has centimetre-scale uncertainty",
            },
            indent=2,
        )
    )
    print(args.out)


if __name__ == "__main__":
    main()
