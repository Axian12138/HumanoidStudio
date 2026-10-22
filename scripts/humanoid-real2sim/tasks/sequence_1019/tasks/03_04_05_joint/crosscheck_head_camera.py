"""Independent cross-window diagnostic for the rejected head-camera candidate."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation

    sys.path[:0] = [
        "/home/galbot/workspace/ScaleTrack/retarget2/src",
        "/home/galbot/workspace/ScaleTrack/retarget2/scripts",
    ]
    from extract_right_camera_targets import parse_rigid_frame
    from mcap_skeleton_to_bvh import McapReader
    from processors import SOURCE_TO_CANONICAL

    r = Path("outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800")
    out = r / "review/scene/03_04_05_head_validation"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(r / "retarget/candidates/03_04_05_joint_v0/source_sync.npz")
    times = [31.0, 47.0, 51.0]
    target = z["receive_timestamp_ns"][[int((t - 27) * 50) for t in times]]
    world = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ SOURCE_TO_CANONICAL
    reader = McapReader(Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap"))
    best = {}
    for _, seq, ts, _, payload in reader.iter_messages({"/optitrack/head_rigid"}):
        for j, v in enumerate(target):
            if j not in best or abs(ts - v) < best[j]["error_ns"]:
                f = parse_rigid_frame(seq, ts, payload)
                T = np.eye(4)
                T[:3, :3] = world @ Rotation.from_quat(f.orientation_xyzw).as_matrix()
                T[:3, 3] = world @ f.position
                T[2, 3] += 0.08245383203029633
                best[j] = {"error_ns": int(abs(ts - v)), "matrix": T.tolist(), "valid": bool(f.tracking_valid)}
    cal = next(
        x
        for x in json.load(open(r / "scene/10_flower_bouquet/head_multiframe_candidates.json"))
        if x["model"] == "pinhole"
    )
    extr = np.array(cal["T_head_rigid_camera_optical"])
    K = np.array(
        json.load(
            open(
                "/home/galbot/workspace/ScaleTrack/data/1019test/lined-rect-basket-kinematic-v1/metadata/camera_calibration.json"
            )
        )["cameras"]["head"]["K"]
    )
    points = np.array(
        [
            [0.653, -1.69, 0.685],
            [1.45, -1.69, 0.685],
            [1.45, -1.69, 0.882],
            [0.653, -1.69, 0.882],
            [2.0693, -1.5736, 1.01],
            [0.65, -0.05, 0.03672],
        ]
    )
    for j, t in enumerate(times):
        frame = int(z["head_frame_index"][int((t - 27) * 50)])
        video = "/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization/camera_head_color_image_raw_compressed.mp4"
        f = out / f"head_{t:g}.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", "-y", str(f)],
            check=True,
        )
        T = np.array(best[j]["matrix"]) @ extr
        camera = (points - T[:3, 3]) @ T[:3, :3]
        uv = camera @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        im = Image.open(f)
        d = ImageDraw.Draw(im)
        for k, (x, y) in enumerate(uv):
            if camera[k, 2] > 0 and -100 < x < 1380 and -100 < y < 900:
                d.ellipse((x - 6, y - 6, x + 6, y + 6), outline="red", width=3)
                d.text((x + 9, y), str(k), fill="red")
        im.save(out / f"head_{t:g}_projected.png")
        best[j].update(
            source_s=t,
            source_frame=frame,
            projection_pixels=uv.tolist(),
            camera_depth_m=camera[:, 2].tolist(),
            world_camera=T.tolist(),
        )
    (out / "projection.json").write_text(
        json.dumps(
            {
                "source": "clip10 candidate cross-window validation, not refitted",
                "head": best,
                "landmarks_world": points.tolist(),
            },
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
