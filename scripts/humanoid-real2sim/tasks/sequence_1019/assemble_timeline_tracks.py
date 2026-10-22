"""Join explicitly timed object fragments and record all stationary extensions.

Reject gaps and conflicting overlap states. Endpoint holds require an explicit
reason in the task spec; they remain estimates rather than observed motion.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from robosimstudio.humanoid_real2sim.retarget.motion_fields import sha256
from robosimstudio.humanoid_real2sim.visualization.tracks import load_scene_tracks


def fragment(entry, kind):
    path = Path(entry["path"])
    with np.load(path) as z:
        if float(z["fps"]) != 50:
            raise ValueError(f"{path}: expected 50 Hz")
        if "source_time_s" in z:
            times = z["source_time_s"]
        else:
            with np.load(entry["source_sync"]) as sync:
                times = sync["recording_time_s"]
        if kind == "objects":
            quat = z["quaternion_xyzw"] if "quaternion_xyzw" in z else z["quaternion_wxyz"][:, [1, 2, 3, 0]]
            if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-5, rtol=0):
                raise ValueError(f"{path}: quaternion norm")
            values = np.concatenate((z["position_w"], quat), axis=1)
        else:
            values = z["position"][:, None]
    ticks = np.rint(times * 50).astype(int)
    if not np.allclose(times, ticks / 50, atol=1e-7, rtol=0) or np.any(np.diff(ticks) != 1):
        raise ValueError(f"{path}: invalid source time grid")
    if len(ticks) != len(values) or not np.isfinite(values).all():
        raise ValueError(f"{path}: invalid frame values")
    return ticks, values, {"path": str(path), "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    with np.load(args.candidate / "source_time.npz") as z:
        times = z["recording_time_s"]
    ticks = np.rint(times * 50).astype(int)
    products, manifest, reports = {}, {"objects": [], "joints": []}, []
    for kind, manifest_entries in manifest.items():
        for item in spec.get(kind, []):
            name = item["name"]
            if not name or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in name
            ):
                raise ValueError(f"unsafe scene name: {name}")
            if name in products:
                raise ValueError(f"duplicate track: {name}")
            values = np.full((len(ticks), 7 if kind == "objects" else 1), np.nan)
            seen, sources, overlaps = np.zeros(len(ticks), dtype=bool), [], []
            for entry in item["fragments"]:
                ft, fv, provenance = fragment(entry, kind)
                mask = (ft >= ticks[0]) & (ft <= ticks[-1])
                ft, fv = ft[mask], fv[mask]
                if not len(ft):
                    raise ValueError(f"{name}: fragment outside timeline")
                at = ft - ticks[0]
                repeated = seen[at]
                if repeated.any():
                    a, b = values[at[repeated]], fv[repeated]
                    position_delta = float(abs(a[:, :3] - b[:, :3]).max())
                    quat_delta = (
                        float(
                            np.minimum(
                                np.linalg.norm(a[:, 3:] - b[:, 3:], axis=1), np.linalg.norm(a[:, 3:] + b[:, 3:], axis=1)
                            ).max()
                        )
                        if kind == "objects"
                        else 0.0
                    )
                    if position_delta > 1e-6 or quat_delta > 1e-6:
                        raise ValueError(f"{name}: conflicting overlap {position_delta}, {quat_delta}")
                    overlaps.append(
                        {
                            "frames": int(repeated.sum()),
                            "max_position_delta": position_delta,
                            "max_sign_invariant_quaternion_delta": quat_delta,
                        }
                    )
                # Preserve the first fragment exactly where the overlap agrees.
                fresh = ~repeated
                values[at[fresh]] = fv[fresh]
                seen[at] = True
                sources.append(provenance)
            known = np.flatnonzero(seen)
            if not len(known) or not seen[known[0] : known[-1] + 1].all():
                raise ValueError(f"{name}: uncovered internal state interval")
            extensions = []
            for side, mask, index in (
                ("before", np.arange(len(ticks)) < known[0], known[0]),
                ("after", np.arange(len(ticks)) > known[-1], known[-1]),
            ):
                if mask.any():
                    reason = item.get(f"hold_{side}")
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError(f"{name}: explicit hold_{side} reason required")
                    values[mask] = values[index]
                    extensions.append(
                        {
                            "side": side,
                            "source_interval_inclusive_s": times[mask][[0, -1]].tolist(),
                            "assumption": reason,
                            "observed_motion": False,
                        }
                    )
            products[name] = (kind, values)
            manifest_entries.append({"name": name, "path": f"{name}.npz"})
            reports.append(
                {"name": name, "sources": sources, "overlaps": overlaps, "stationary_extensions": extensions}
            )
    out = args.candidate / "scene_tracks"
    out.mkdir(exist_ok=False)
    for name, (kind, values) in products.items():
        data = (
            {"position_w": values[:, :3], "quaternion_xyzw": values[:, 3:]}
            if kind == "objects"
            else {"position": values[:, 0]}
        )
        np.savez_compressed(out / f"{name}.npz", **data, fps=50, source_time_s=times)
    (out / "tracks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tracks = load_scene_tracks(out / "tracks.json", frames=len(times), fps=50, source_offset_s=times[0])
    report = {
        "status": "diagnostic_not_accepted",
        "tracks": reports,
        "strict_loader_count": len(tracks),
        "missing_task_states": spec.get("missing_task_states", []),
        "scope": "Time-aligned estimated states; no scene binding, support or interaction acceptance.",
    }
    (out / "assembly.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Loaded {len(tracks)} complete-time tracks; remaining states: {report['missing_task_states']}")


if __name__ == "__main__":
    main()
