"""Project fixed table and berry using independently fitted hand-marker camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    r = args.run
    cal = json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())
    poses = json.loads((r / "inputs/10_flower_bouquet/head_raw_pose_audit_header.json").read_text())["rows"]
    K = np.array(cal["K"])
    Tmount = np.array(cal["T_head_rigid_camera_optical"])
    basis = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    out = r / "review/scene/08_marker_camera"
    out.mkdir(exist_ok=True)
    variants = {
        "original": np.eye(4),
        "parent_shared": np.array(
            json.loads((r / "review/scene/parent_room_fixed_marker_mount_diagnostic.json").read_text())["results"][1][
                "T_room_correction"
            ]
        ),
        "level": np.array(
            json.loads((r / "scene/10_flower_bouquet/level_room_candidate.json").read_text())["T_room_correction"]
        ),
    }
    table = np.array(
        [[-1.03, -2.11905405, 0.754], [-1.03, -2.97, 0.754], [0.57, -2.97, 0.754], [0.57, -2.11905405, 0.754]]
    )
    points = np.r_[table, [[-0.822655201, -2.20, 0.7907203], [-0.822655201, -2.071139812, 0.803148925]]]
    result = {}
    for t in [90.0, 92.0, 96.0]:
        raw = poses[str(t)]
        head = np.eye(4)
        head[:3, :3] = basis @ Rotation.from_quat(raw["quaternion_xyzw_raw"]).as_matrix()
        head[:3, 3] = basis @ raw["position_raw"] + [0, 0, 0.082453832]
        camera = head @ Tmount
        row = {"T_world_camera": camera.tolist(), "variants": {}}
        for label, correction in variants.items():
            world = points @ correction[:3, :3].T + correction[:3, 3]
            local = (world - camera[:3, 3]) @ camera[:3, :3]
            uv = local @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            row["variants"][label] = {"pixels": uv.tolist(), "depth": local[:, 2].tolist()}
            im = Image.open(r / f"review/scene/08_camera_candidate/{t:.1f}.png").convert("RGB")
            draw = ImageDraw.Draw(im)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            polygon = np.round(uv[:4]).astype(int)
            for i in range(4):
                draw.line([tuple(polygon[i]), tuple(polygon[(i + 1) % 4])], fill="yellow", width=3)
                draw.text(tuple(polygon[i]), f"P{i}", fill="red", font=font)
            for i, name in [(4, "current berry"), (5, "1012 berry")]:
                pt = tuple(np.round(uv[i]).astype(int))
                draw.ellipse((pt[0] - 8, pt[1] - 8, pt[0] + 8, pt[1] + 8), outline="magenta", width=2)
                draw.text(pt, name, fill="magenta", font=font)
            draw.text((20, 35), label, fill="red", font=font)
            im.save(out / f"{t:.1f}_{label}.jpg")
        result[str(t)] = row
    (out / "projection.json").write_text(
        json.dumps(
            {"world_points": points.tolist(), "camera_source": "head_marker_diagnostic_v4_header.json", "rows": result},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
