"""Independent early-frame camera holdout using visible right-hand rod-end balls."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.optimize import linear_sum_assignment

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    out = r / "review/scene/03_04_05_shared_room_validation"
    raw = json.loads((r / "inputs/03_04_05/right_hand_marker_audit_header.json").read_text())["rows"]
    proj = json.loads((out / "projections.json").read_text())
    K = np.array(json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())["K"])
    annotations = {
        39.0: [[942, 234], [1045, 255], [882, 308]],
        47.0: [[1020, 356], [1139, 420], [975, 519]],
        51.0: [[853, 409], [946, 412], [810, 460]],
    }
    records = []
    for t, observed in annotations.items():
        camera = np.array(next(x for x in proj["records"] if x["source_s"] == t)["T_world_camera"])
        markers = [m for m in raw[str(t)]["markers"] if m["member_id"] in [2, 3, 4]]
        points = np.array([m["position_world"] for m in markers])
        v = (points - camera[:3, 3]) @ camera[:3, :3]
        uv = v @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        obs = np.array(observed)
        cost = np.linalg.norm(uv[:, None] - obs[None], axis=2)
        ii, jj = linear_sum_assignment(cost)
        record = {
            "source_s": t,
            "annotation_sigma_px": 4,
            "assignment": "three independently marked rod-end balls, minimum-distance one-to-one association, no extrinsic fit",
            "head_mount": "head_marker_diagnostic_v4_header",
            "hand_pose_time_error_ns": raw[str(t)]["target_error_ns"],
            "matches": [],
        }
        im = Image.open(out / f"head_{t:g}.png")
        draw = ImageDraw.Draw(im)
        for i, j in zip(ii, jj, strict=True):
            x, y = uv[i]
            u, w = obs[j]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="red", width=3)
            draw.text((x + 8, y), str(markers[i]["member_id"]), fill="red")
            draw.ellipse((u - 6, w - 6, u + 6, w + 6), outline="lime", width=3)
            draw.line((x, y, u, w), fill="yellow", width=2)
            record["matches"].append(
                {
                    "member_id": markers[i]["member_id"],
                    "projected_px": uv[i].tolist(),
                    "observed_px": obs[j].tolist(),
                    "error_px": float(cost[i, j]),
                    "marker_flags": markers[i]["flags"],
                }
            )
        im.save(out / f"head_{t:g}_marker_holdout.png")
        records.append(record)
    (out / "early_marker_holdout.json").write_text(
        json.dumps({"status": "independent_early_camera_holdout_no_refitting", "records": records}, indent=2) + "\n"
    )
    print([(x["source_s"], [(m["member_id"], m["error_px"]) for m in x["matches"]]) for x in records])


if __name__ == "__main__":
    main()
