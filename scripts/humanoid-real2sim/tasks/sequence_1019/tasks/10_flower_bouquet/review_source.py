"""Decode every source video frame and export dense synchronized source review sheets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    from PIL import Image, ImageDraw

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    dest = a.run / "analysis/10_flower_bouquet"
    dest.mkdir(parents=True, exist_ok=True)
    s = np.load(a.run / "retarget/body_prior/10_flower_bouquet/source_sync.npz")
    base = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26-visualization")
    report = {}
    for cam, stem in [("head", "camera_head"), ("left", "camera_hand_left"), ("right", "camera_hand_right")]:
        c = cv2.VideoCapture(str(base / (stem + "_color_image_raw_compressed.mp4")))
        first, last = s[f"camera_{cam}_frame_index"][[0, -1]].astype(int)
        c.set(1, int(first))
        frames = {}
        fail = []
        for ix in range(first, last + 1):
            ok, f = c.read()
            if not ok:
                fail.append(ix)
            else:
                frames[ix] = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (320, 240))
        indices = np.arange(0, len(s["recording_time_s"]), 10)
        for part in range((len(indices) + 19) // 20):
            im = Image.new("RGB", (1600, 960))
            d = ImageDraw.Draw(im)
            for j, ix in enumerate(indices[part * 20 : (part + 1) * 20]):
                f = int(s[f"camera_{cam}_frame_index"][ix])
                im.paste(Image.fromarray(frames[f]), ((j % 5) * 320, (j // 5) * 240))
                d.text(((j % 5) * 320 + 4, (j // 5) * 240 + 3), f"{s['recording_time_s'][ix]:.2f} / {f}", fill="red")
            im.save(dest / f"{cam}_dense_{part}.jpg")
        report[cam] = {
            "decoded_count": len(frames),
            "expected_count": int(last - first + 1),
            "failed": fail,
            "dense_sampling_hz": 5.0,
            "visual_review": "pending_dense_sheet_inspection",
        }
    (dest / "continuous_decode.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
