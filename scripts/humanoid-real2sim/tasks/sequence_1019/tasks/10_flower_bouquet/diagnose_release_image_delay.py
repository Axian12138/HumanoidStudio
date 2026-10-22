"""Fit one common stream delay against independent visible left-hand markers."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    from scipy.optimize import minimize_scalar
    from scipy.spatial.transform import Rotation, Slerp

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, "/home/galbot/workspace/ScaleTrack/retarget2/src")
    from mcap_skeleton_to_bvh import CdrReader, McapReader, parse_frame_info

    source = Path("/home/galbot/workspace/ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap")
    headers = json.loads((args.run / "inputs/10_flower_bouquet/image_header_timestamp_audit.json").read_text())[
        "samples"
    ]
    # Visible sphere centers, original 1280x800 Header images. Fourth sphere
    # is deliberately omitted where hidden by the glove.
    observations = {
        123.4: [[435, 570], [394, 606], [338, 661]],
        124.0: [[469, 475], [429, 495], [371, 553]],
        124.6: [[590, 430], [535, 414], [469, 481]],
        125.0: [[604, 430], [544, 412], [493, 483]],
        128.0: [[509, 394], [460, 402], [393, 489]],
    }
    targets = {t: headers[str(t)]["header_time_ns"] for t in observations}
    epoch = min(targets.values())
    dense = {"head": [], "left_hand": []}
    topics = {f"/optitrack/{key}_rigid" for key in dense}
    for topic, _, timestamp, _, payload in McapReader(source).iter_messages(topics):
        if min(abs(timestamp - t) for t in targets.values()) > 350_000_000:
            continue
        c = CdrReader(payload)
        frame = parse_frame_info(c)
        name, resolved, present, rigid_id = c.string(), c.boolean(), c.boolean(), c.i32()
        position = [c.f64() for _ in range(3)]
        quaternion = [c.f64() for _ in range(4)]
        mean_error, params, valid, fused = c.f32(), c.i16(), c.boolean(), c.boolean()
        markers = []
        for _ in range(c.u32()):
            marker_id, model_id, member_id = c.i32(), c.i32(), c.i32()
            xyz = [c.f64() for _ in range(3)]
            size, marker_params, residual = c.f32(), c.i16(), c.f32()
            flags = [c.boolean() for _ in range(8)]
            markers.append(
                {
                    "member_id": member_id,
                    "marker_id": marker_id,
                    "model_id": model_id,
                    "size": size,
                    "params": marker_params,
                    "residual": residual,
                    "position_raw": xyz,
                    "flags": flags,
                }
            )
        key = topic.removeprefix("/optitrack/").removesuffix("_rigid")
        dense[key].append(
            {
                "log_time_ns": timestamp,
                "motive_frame": frame.motive_frame_number,
                "name": name,
                "rigid_id": rigid_id,
                "mean_error": mean_error,
                "params": params,
                "position": position,
                "quaternion": quaternion,
                "resolved": resolved,
                "present": present,
                "valid": valid,
                "fused": fused,
                "markers": markers,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (args.out / "raw_samples.json").write_text(json.dumps(dense, indent=2))
    usable = {key: [r for r in rows if r["resolved"] and r["present"] and r["valid"]] for key, rows in dense.items()}
    usable["left_hand"] = [r for r in usable["left_hand"] if {1, 2, 3}.issubset({m["member_id"] for m in r["markers"]})]
    ht = np.array([(r["log_time_ns"] - epoch) / 1e9 for r in usable["head"]])
    hp = np.array([r["position"] for r in usable["head"]])
    hs = Slerp(ht, Rotation.from_quat([r["quaternion"] for r in usable["head"]]))
    lt = np.array([(r["log_time_ns"] - epoch) / 1e9 for r in usable["left_hand"]])
    lp = np.array(
        [
            [next(m["position_raw"] for m in r["markers"] if m["member_id"] == i) for i in [1, 2, 3]]
            for r in usable["left_hand"]
        ]
    )
    cal = json.loads((args.run / "scene/10_flower_bouquet/head_marker_diagnostic_v4_header.json").read_text())
    mount, K = np.array(cal["T_head_rigid_camera_optical"]), np.array(cal["K"])
    C = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    def project(time, delay):
        target = (targets[time] - epoch) / 1e9 + delay
        H = np.eye(4)
        H[:3, :3] = C @ hs(target).as_matrix()
        H[:3, 3] = C @ np.array([np.interp(target, ht, hp[:, j]) for j in range(3)])
        H[2, 3] += 0.08245383203029633
        camera = H @ mount
        raw = np.array([[np.interp(target, lt, lp[:, i, j]) for j in range(3)] for i in range(3)])
        world = raw @ C.T
        world[:, 2] += 0.08245383203029633
        uvw = ((world - camera[:3, 3]) @ camera[:3, :3]) @ K.T
        return uvw[:, :2] / uvw[:, 2:]

    training = [123.4, 124.6]

    def cost(delay):
        return sum(np.sum((project(t, delay) - observations[t]) ** 2) for t in training)

    grid = np.linspace(-0.2, 0.2, 401)
    best = int(np.argmin([cost(x) for x in grid]))
    fit = minimize_scalar(cost, bounds=(grid[max(0, best - 2)], grid[min(len(grid) - 1, best + 2)]), method="bounded")
    rows = []
    for t, observed in observations.items():
        rows.append(
            {
                "source_s": t,
                "partition": "fit" if t in training else "holdout",
                "observed_uv": observed,
                "member_ids": [1, 2, 3],
                "annotation_sigma_px": 3,
                "zero_delay_uv": project(t, 0).tolist(),
                "fit_delay_uv": project(t, fit.x).tolist(),
                "zero_delay_errors_px": np.linalg.norm(project(t, 0) - observed, axis=1).tolist(),
                "fit_delay_errors_px": np.linalg.norm(project(t, fit.x) - observed, axis=1).tolist(),
            }
        )
    report = {
        "scope": "One shared delay applied to both head and marker OptiTrack receive timestamps relative to image Header; fixed mount/K/world. Diagnostic only.",
        "fit_delay_s": float(fit.x),
        "fit_times": training,
        "rows": rows,
        "limitations": "Manual sphere centers; short late window. Does not establish root cause or authorize changing source sync.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
