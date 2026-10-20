"""Record observed source events and independent Dex3 command times."""

import json
from pathlib import Path
from robosimstudio.humanoid_real2sim.paths import example_dir

import numpy as np


ROOT = example_dir(__file__)
EVENTS = [
    ("post_bin_stand", 9.5, 10.6, "body", "head video", 0.25),
    ("brush_approach", 10.6, 11.9, "right hand", "head and right-wrist video", 0.2),
    ("brush_first_surface_contact", 11.9, 12.3, "right thumb/index around exposed green grip", "right-wrist video", 0.25),
    ("brush_lift", 12.8, 14.2, "right hand", "head and right-wrist video", 0.3),
    ("table_sweep_first_pass", 16.0, 18.5, "brush bristles on tabletop", "head and right-wrist video", 0.3),
    ("table_sweep_second_pass", 19.5, 20.9, "brush bristles on tabletop", "head and right-wrist video", 0.3),
    ("return_to_basket", 24.2, 25.5, "right hand holding brush", "head and right-wrist video", 0.3),
    ("brush_set_down", 25.3, 26.1, "white storage box", "head and right-wrist video", 0.25),
    ("hand_withdrawal", 26.0, 27.0, "right hand", "head and right-wrist video", 0.25),
]


def main():
    with np.load(ROOT / "source/source_sync_8_28.npz") as sync:
        motive = sync["source_motive_frame_number"]
        head = sync["camera_head_frame_index"]

    def source(t):
        index = np.clip(round((t - 8.0) * 50), 0, len(motive) - 1)
        return {"motion_frame_global_50hz": round(t * 50),
                "motion_frame_local_8s": int(index),
                "motive_frame": int(motive[index]),
                "head_video_frame": int(head[index])}

    events = [{"event": name, "source_time_s": [start, stop], "site": site,
               "evidence": evidence, "boundary_uncertainty_s": uncertainty,
               "source_start": source(start), "source_stop": source(stop)}
              for name, start, stop, site, evidence, uncertainty in EVENTS]
    result = {
        "recording_id": "1-2026-10-19-16-31-26",
        "coordinate_frame": "IKEA_1012_world, metres, +Z up",
        "observed_events": events,
        "designed_dex3_commands_s": {
            "right_open_start": 10.5, "right_open_complete": 11.6,
            "right_close_start": 11.9, "right_close_complete": 12.8,
            "right_release_start": 25.45, "right_release_complete": 26.25,
        },
        "estimated_object_attachment_interval_s": [12.8, 25.65],
        "attachment_note": "Kinematic review sidecar only. Physical grip, sweep contact, release and slip remain unverified.",
        "left_hand_policy": "inherited body prior; no brush grasp command",
    }
    output = ROOT / "sidecars/contact_schedule.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
