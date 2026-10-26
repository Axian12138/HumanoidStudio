"""Copy an immutable G1 candidate window, preserving its original source clock."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from retarget3.motion_io import export_motion
    from retarget3.quality import audit_quality
    from retarget3.robot import MJCasadiRobot

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--stop-frame", type=int, required=True)
    args = ap.parse_args()
    src, out = args.source, args.output
    full = np.load(src / "intermediate/body_prior_q.npy")
    a, b = args.start_frame, args.stop_frame
    if not 0 <= a < b <= len(full) or b - a < 2:
        raise ValueError("Invalid source window")
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("intermediate", "sidecars", "review"):
        (out / folder).mkdir()
    q = full[a:b]
    np.save(out / "intermediate/body_prior_q.npy", q)
    with np.load(src / "intermediate/targets.npz") as z:
        np.savez_compressed(out / "intermediate/targets.npz", **{k: z[k][a:b] for k in z.files})
    fields = {
        "human.npz": ("positions_w",),
        "object_0.npz": ("position_w", "quaternion_xyzw"),
        "timing.npz": ("source_frame", "motion_time_s", "t_wall", "t_sim"),
    }
    for name, keys in fields.items():
        with np.load(src / "sidecars" / name) as z:
            np.savez_compressed(out / "sidecars" / name, **{k: z[k][a:b] if k in keys else z[k] for k in z.files})
    robot = MJCasadiRobot()
    export_motion(out / "body_prior.npz", robot, q, 50)
    audit_quality({"fps": 50}, out, robot, "body_prior")
    provenance = json.loads((src / "review/provenance.json").read_text())
    offset = provenance.get("source_offset_s", 0) + a / 50
    provenance.update(
        frames=b - a,
        duration_s=(b - a - 1) / 50,
        source_offset_s=offset,
        parent_candidate=str(src.resolve()),
        parent_frames=[a, b],
        scope="kinematic window; independent acceptance required",
    )
    (out / "review/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
