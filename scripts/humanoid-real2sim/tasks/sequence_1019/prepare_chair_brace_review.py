"""Prepare source-synchronous sidecars for the connection candidate's existing viewer."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test")
    push = source / "push-armchair-index-middle-kinematic-v1/metadata"
    brace = source / "brace-dining-table-kinematic-v1/metadata"
    with np.load(push / "source_sync.npz") as a, np.load(brace / "source_sync.npz") as b:
        sync = {}
        for key in a.files:
            if a[key].ndim and len(a[key]) == 475:
                if len(b[key]) != 375:
                    raise ValueError(f"invalid brace sidecar {key}")
                sync[key] = np.concatenate((a[key][:425], b[key]), axis=0)
            else:
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f"non-frame metadata differs for {key}")
                sync[key] = a[key]
    expected = 75 + np.arange(800) / 50
    if not np.allclose(sync["recording_time_s"], expected, atol=1e-7, rtol=0):
        raise ValueError("combined source timestamps differ")
    sync["motion_time_s"] = np.arange(800) / 50
    np.savez_compressed(args.candidate / "source_sync.npz", **sync)
    human = Path(
        "outputs/humanoid-real2sim/2026-10-20/sequence_1019/historical_examples/retarget/ground_support_27_end/sidecars/human_motion_world.npz"
    )
    with np.load(human) as z:
        values = {key: z[key][2400:3200] if z[key].ndim and len(z[key]) == 5104 else z[key] for key in z.files}
    np.savez_compressed(args.candidate / "human_motion_world.npz", **values)
    with (
        np.load(push / "object_trajectories/grey_armchair.npz") as a,
        np.load(brace / "object_trajectories/grey_armchair.npz") as b,
    ):
        pos = np.concatenate((a["position_w"][:425], b["position_w"]))
        quat = np.concatenate((a["quaternion_wxyz"][:425], b["quaternion_wxyz"]))[:, [1, 2, 3, 0]]
        overlap = float(abs(a["position_w"][425:475] - b["position_w"][:50]).max())
    np.savez_compressed(
        args.candidate / "grey_armchair.npz", position_w=pos, quaternion_xyzw=quat, fps=50, source_time_s=expected
    )
    (args.candidate / "tracks.json").write_text(
        json.dumps({"objects": [{"name": "grey_armchair", "path": "grey_armchair.npz"}]}, indent=2) + "\n"
    )
    (args.candidate / "review_sidecars.json").write_text(
        json.dumps(
            {
                "scope": "source-aligned viewer sidecars only",
                "human_source": str(human),
                "human_frames": [2400, 3200],
                "chair_overlap_position_max_delta_m": overlap,
                "accepted": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("chair overlap position difference", overlap)


if __name__ == "__main__":
    main()
