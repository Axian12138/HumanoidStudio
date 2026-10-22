"""Cross-window projection of unchanged rack geometry using header-synchronized markers."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    out = r / "review/scene/03_04_05_shared_room_validation"
    out.mkdir(exist_ok=True)
    cal = json.loads((r / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())
    E = np.array(cal["T_head_rigid_camera_optical"])
    K = np.array(cal["K"])
    correction = np.array(
        json.loads((r / "scene/10_flower_bouquet/shared_level_room_v2.json").read_text())["T_room_correction"]
    )
    raw = json.loads((r / "inputs/03_04_05/head_raw_pose_audit_header.json").read_text())["rows"]
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    rack = json.loads((r / "scene/03_04_05/rack_support.json").read_text())["parts"]
    names = []
    points = []
    for part in rack:
        if "hook_ball" in part["source_prim"]:
            names.append(part["source_prim"].split("/")[-1])
            points.append(part["center_world_m"])
    points.extend([[0.653, -1.69, 0.685], [1.45, -1.69, 0.685], [1.45, -1.69, 0.882], [0.653, -1.69, 0.882]])
    names.extend(["drawer_bottom_xmin", "drawer_bottom_xmax", "drawer_top_xmax", "drawer_top_xmin"])
    points = np.array(points)
    sync = np.load(r / "retarget/candidates/03_04_05_joint_v0/source_sync.npz")
    report = []
    video = "/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization/camera_head_color_image_raw_compressed.mp4"
    for t in [35.0, 39.0, 44.0, 47.0, 48.0, 51.0]:
        row = raw[str(t)]
        T = np.eye(4)
        T[:3, :3] = C @ Rotation.from_quat(row["quaternion_xyzw_raw"]).as_matrix()
        T[:3, 3] = C @ row["position_raw"]
        T[2, 3] += 0.08245383203029633
        camera = T @ E
        frame = int(sync["head_frame_index"][round((t - 27) * 50)])
        file = out / f"head_{t:g}.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", "-y", str(file)],
            check=True,
        )
        base = Image.open(file)
        images = []
        versions = {}
        for label, transform, color in [("original", np.eye(4), "blue"), ("shared_v2", correction, "red")]:
            pworld = points @ transform[:3, :3].T + transform[:3, 3]
            pc = (pworld - camera[:3, 3]) @ camera[:3, :3]
            uv = pc @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            im = base.copy()
            draw = ImageDraw.Draw(im)
            for i, (x, y) in enumerate(uv):
                if pc[i, 2] > 0 and -30 < x < 1310 and -30 < y < 830:
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)
                    draw.text((x + 7, y), str(i), fill=color)
            draw.text((20, 20), label, fill=color)
            im.save(out / f"head_{t:g}_{label}.png")
            images.append(im)
            versions[label] = {"pixels": uv.tolist(), "depths": pc[:, 2].tolist()}
        sheet = Image.new("RGB", (2560, 800))
        sheet.paste(images[0], (0, 0))
        sheet.paste(images[1], (1280, 0))
        sheet.save(out / f"head_{t:g}_comparison.png")
        report.append(
            {
                "source_s": t,
                "source_frame": frame,
                "head_tracking_valid": row["tracking_valid"],
                "pose_time_error_ns": row["target_error_ns"],
                "T_world_camera": camera.tolist(),
                "versions": versions,
            }
        )
    (out / "projections.json").write_text(
        json.dumps(
            {
                "landmark_names": names,
                "landmarks_world": points.tolist(),
                "camera_mount": "head_marker_diagnostic_v4_header",
                "room_candidate": "shared_level_room_v2",
                "records": report,
                "status": "independent_early_window_projection_not_yet_annotated",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
