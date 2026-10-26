"""Isolate drawer push with pure body prior and inherited solved contacts."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    r = parser.parse_args().run.resolve()
    src = OLD / "final/05_drawer/scoped_candidate_v3"
    for folder in ["retarget/baseline", "retarget/body_prior", "scene", "review", "logs"]:
        (r / folder).mkdir(parents=True, exist_ok=True)
    start, stop = 1125, 1426  # recording 49.5--55.5 seconds inclusive; old offset=27

    def cut(source, dest):
        z = np.load(source)
        n = len(z["joint_pos"]) if "joint_pos" in z.files else 2450
        np.savez_compressed(
            dest, **{k: z[k][start:stop] if z[k].ndim and z[k].shape[0] == n else z[k] for k in z.files}
        )

    out = r / "retarget/baseline"
    for name in ["motion.npz", "human_motion_world.npz", "source_sync.npz"]:
        cut(src / name, out / name)
    shutil.copy2(src / "motion_order.json", out / "motion_order.json")
    pure = (
        ROOT
        / "outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/retarget/body_prior_27_end/motion.npz"
    )
    cut(pure, r / "retarget/body_prior/motion.npz")
    shutil.copy2(src / "motion_order.json", r / "retarget/body_prior/motion_order.json")
    unified = OLD / "retarget/candidates/unified_diagnostic_v3"
    tracks = json.loads((unified / "scene_tracks/tracks.json").read_text())
    (out / "scene_tracks").mkdir(exist_ok=True)
    for item in tracks["objects"] + tracks.get("joints", []):
        p = unified / "scene_tracks" / item["path"]
        z = np.load(p)
        n = len(z[z.files[0]])
        data = {k: z[k][start:stop] if z[k].ndim and z[k].shape[0] == n else z[k] for k in z.files}
        if item["name"] == "drawer_slide":
            old = np.load(src / "drawer_slide.npz")
            data = {k: old[k][start:stop] if old[k].ndim and len(old[k]) == 2450 else old[k] for k in old.files}
        # Keep the final accepted hat pose after its release, rather than the old hat asset/track.
        if item["name"] == "black_ribbon_hat_free":
            hat = (
                Path(Path("/tmp/hat_natural_run_path").read_text().strip())
                / "retarget/natural_v4/scene_tracks/black_ribbon_hat_free.npz"
            )
            h = np.load(hat)
            times = 49.5 + np.arange(301) / 50
            idx = np.minimum(np.rint((times - 35) * 50).astype(int), 750)
            data = {k: h[k][idx] if h[k].ndim and len(h[k]) == 751 else h[k] for k in h.files}
            data["source_time_s"] = times
        np.savez_compressed(out / "scene_tracks" / item["path"], **data)
    (out / "scene_tracks/tracks.json").write_text(json.dumps(tracks, indent=2))
    hatrun = Path(Path("/tmp/hat_natural_run_path").read_text().strip())
    config = json.loads((hatrun / "scene/config.json").read_text())
    (r / "scene/config.json").write_text(json.dumps(config, indent=2))
    shutil.copy2(OLD / "repairs/05_drawer_event_timing/v5/review.json", r / "review/inherited_contact_fit.json")
    (r / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(src),
                "body_prior": str(pure),
                "slice": [start, stop],
                "source_offset_s": 49.5,
                "fps": 50,
                "scope": "right arm drawer HOI; preserve root, lower body, left hand berry",
                "source_event_uncertainty": "initial extension/contact onset and closure remain source-constrained estimates",
            },
            indent=2,
        )
    )
    print(r)


if __name__ == "__main__":
    main()
