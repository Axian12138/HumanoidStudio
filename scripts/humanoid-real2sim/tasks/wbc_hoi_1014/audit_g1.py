"""Audit full G1 body-prior motion, source synchronization and geometry scope."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit_room_chunk(arguments):
    from retarget3.collision import TriangleScene, audit_motion
    from retarget3.robot import MJCasadiRobot

    candidate, triangles, begin, end = arguments
    q = np.load(candidate / "intermediate/body_prior_q.npy")[begin:end]
    path = candidate / f"review/room_chunk_{begin}_{end}.json"
    result = audit_motion(MJCasadiRobot(), q, TriangleScene(triangles), path)
    for item in result["frames_exceeding_tolerance"]:
        item["frame"] += begin
    result["start_frame"] = begin
    result["stop_frame"] = end
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    from retarget3.common import CORNERS, FEET, segments
    from retarget3.motion_io import order, validate_motion
    from retarget3.robot import MJCasadiRobot
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--room-triangles", type=Path)
    ap.add_argument("--room-workers", type=int, default=1)
    args = ap.parse_args()
    p = args.candidate
    robot = MJCasadiRobot()
    q = np.load(p / "intermediate/body_prior_q.npy")
    targets = np.load(p / "intermediate/targets.npz")
    timing = np.load(p / "sidecars/timing.npz")
    motion = np.load(p / "body_prior.npz")
    bn, jn = order()
    position, quat = robot.forward(q, bn)
    feet = {}
    for s, name in enumerate(FEET):
        idx = bn.index(name)
        pos = position[:, idx]
        rot = Rotation.from_quat(quat[:, idx][:, [1, 2, 3, 0]]).as_matrix()
        corners = pos[:, None] + np.einsum("tij,kj->tki", rot, CORNERS)
        stance = []
        for a, b in segments(targets["contacts"][:, s]):
            stance.append(
                {
                    "start_frame": int(a),
                    "stop_frame": int(b),
                    "max_anchor_error_m": float(np.linalg.norm(pos[a:b] - targets["feet"][a:b, s], axis=1).max()),
                    "max_slip_m": float(np.linalg.norm(pos[a:b] - pos[a], axis=1).max()),
                }
            )
        feet[name] = {"stance": stance, "min_sole_corner_z_m": float(corners[:, :, 2].min())}
    vel = np.diff(q[:, 7:], axis=0) * 50
    acc = np.diff(vel, axis=0) * 50
    lim = np.maximum(robot.q_lower[7:] - q[:, 7:], q[:, 7:] - robot.q_upper[7:])
    wrists = {}
    for side in ("left", "right"):
        wrists[side] = {}
        for axis in ("roll", "pitch", "yaw"):
            name = f"{side}_wrist_{axis}_joint"
            j = robot.dof_name_to_index[name]
            wrists[side][axis] = {
                "min_rad": float(q[:, j].min()),
                "max_rad": float(q[:, j].max()),
                "max_speed_rad_s": float(np.abs(np.diff(q[:, j]) * 50).max()),
            }
    report = {
        "schema": validate_motion(p / "body_prior.npz"),
        "frames": len(q),
        "duration_s": float(timing["motion_time_s"][-1] - timing["motion_time_s"][0]),
        "joint_names": jn,
        "body_names": bn,
        "feet": feet,
        "max_joint_limit_violation_rad": float(max(0, lim.max())),
        "max_joint_speed_ratio": float((np.abs(vel) / robot.v_abs[6:]).max()),
        "max_joint_acceleration_rad_s2": float(np.abs(acc).max()),
        "acceleration_acceptance": "reported, no universal acceptance threshold established",
        "max_fk_position_export_error_m": float(np.max(np.abs(position - motion["body_pos_w"]))),
        "max_root_xy_deviation_from_human_m": float(
            np.linalg.norm(q[:, :2] - targets["human"][:, 0, :2], axis=1).max()
        ),
        "wrists": wrists,
        "status": "body_prior_candidate_pending_visual_and_scene_review",
        "unverified": [
            "room collision",
            "full mesh self collision",
            "physical balance",
            "fruit contact",
            "hand swept contact",
        ],
        "clock": "wall clock resampled to 50Hz; source frame indices saved; no display translation",
    }
    if args.room_triangles:
        from concurrent.futures import ProcessPoolExecutor

        workers = max(1, min(args.room_workers, 4, len(q)))
        boundaries = np.linspace(0, len(q), workers + 1, dtype=int)
        jobs = [(p, args.room_triangles, int(boundaries[i]), int(boundaries[i + 1])) for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunks = list(executor.map(audit_room_chunk, jobs))
        room = dict(chunks[0])
        room["start_frame"] = 0
        room["stop_frame"] = len(q)
        room["minimum_clearance_m"] = min(c["minimum_clearance_m"] for c in chunks)
        room["frames_exceeding_tolerance"] = [item for c in chunks for item in c["frames_exceeding_tolerance"]]
        (p / "review/room_collision.json").write_text(json.dumps(room, indent=2) + "\n")
        report["room_collision"] = {
            "minimum_clearance_m": room["minimum_clearance_m"],
            "frames_exceeding_tolerance": len(room["frames_exceeding_tolerance"]),
            "coverage": room["coverage"],
        }
        report["unverified"].remove("room collision")
        report["unverified"].append("continuous swept room collision; sampled audit is framewise")
    (p / "review/kinematic_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("feet", "joint_names", "body_names", "wrists")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
