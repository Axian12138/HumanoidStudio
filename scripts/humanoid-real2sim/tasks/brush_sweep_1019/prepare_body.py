"""Retarget the 8–28 s overlap of the 1019 recording without rewriting source data.

Run with ScaleTrack's Python environment and PYTHONPATH=ScaleTrack/retarget3/src.
This prepares an independent body prior; no brush contact is fitted here.
"""

from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir
import sys

import yaml

from retarget3.body_only import prepare_body
from retarget3.robot import MJCasadiRobot
from retarget3.runtime import BUNDLE


ROOT = example_dir(__file__)
START_S = 8.0
STOP_S = 28.0


def main():
    sys.path.insert(0, str(BUNDLE / "vendor/retarget2/src"))
    import mcap_skeleton_to_bvh as bvh

    cfg = yaml.safe_load((ROOT / "body_case.yaml").read_text())
    source = Path(cfg["human"]["mcap"])
    full = bvh.load_recording(source)
    begin = round(START_S / full.frame_time)
    end = round(STOP_S / full.frame_time)
    if end > len(full.frames):
        raise ValueError("Requested segment exceeds the Motive recording")
    chosen = full.frames[begin:end]
    if any(b.info.motive_frame_number - a.info.motive_frame_number != 1 for a, b in zip(chosen, chosen[1:])):
        raise ValueError("Motive frame discontinuity in segment")
    clip = bvh.Recording(
        source=source,
        model=full.model,
        frames=chosen,
        frame_time=full.frame_time,
        dropped_motive_frames=0,
        position_error_median_m=full.position_error_median_m,
        position_error_p95_m=full.position_error_p95_m,
        tracked_min=full.tracked_min,
        tracked_max=full.tracked_max,
    )
    bvh.load_recording = lambda path: clip if Path(path) == source else (_ for _ in ()).throw(ValueError(path))
    output = ROOT / "retarget/body_prior_8_28"
    for folder in ("intermediate", "sidecars", "review", "scene"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    prepare_body(MJCasadiRobot(), cfg, output)
    print(f"source motive frames: {chosen[0].info.motive_frame_number}..{chosen[-1].info.motive_frame_number}")


if __name__ == "__main__":
    main()
