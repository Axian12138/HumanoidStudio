"""Build the actual mixed-joint scene and test kinematic track writes at task times."""

import argparse
import json
import os
from pathlib import Path

import newton
import numpy as np
from unified_scene import build_sim

from robosimstudio.humanoid_real2sim.retarget.motion_fields import MotionFields, sha256
from robosimstudio.humanoid_real2sim.retarget.root_joint import root_joint_index
from robosimstudio.humanoid_real2sim.visualization.tracks import bind_scene_tracks, load_scene_tracks
from robosimstudio.scene.entity import label_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("scene_config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ["RSS_SEQUENCE_SCENE_CONFIG"] = str(args.scene_config.resolve())
    m = MotionFields.load(args.candidate / "motion.npz", args.candidate / "motion_order.json")
    motion = m.reference(joint_names=m.joint_names, body_names=m.body_names)
    sim = build_sim(None, motion)
    labels = list(sim.model.joint_label)
    starts = sim.model.joint_q_start.numpy()
    root = int(starts[root_joint_index(sim.model, m.body_names[0])])
    robot = np.asarray([starts[label_index(labels, name)] for name in m.joint_names])
    tracks = load_scene_tracks(
        args.candidate / "scene_tracks/tracks.json", frames=len(m.joint_pos), fps=m.fps, source_offset_s=27
    )
    bindings = bind_scene_tracks(tracks, labels, starts, root, robot)
    children = sim.model.joint_child.numpy()
    base = sim.pipeline.snapshot.joint_q.copy()
    zeros = np.zeros_like(sim.pipeline.snapshot.joint_qd)
    results = []
    for time in [
        27,
        31,
        39,
        48,
        53,
        67,
        75.5,
        83.94,
        90.5,
        96.82,
        98.22,
        108.5,
        112.6,
        114.4,
        118,
        120.6,
        123.4,
        124.2,
        128,
        129.06,
    ]:
        frame = round((time - 27) * m.fps)
        q = base.copy()
        q[root : root + 3] = m.body_pos_w[frame, 0]
        q[root + 3 : root + 7] = m.body_quat_w[frame, 0, [1, 2, 3, 0]]
        q[robot] = m.joint_pos[frame]
        for track, coord in bindings:
            q[coord : coord + track.values.shape[1]] = track.values[frame]
        pose_errors = {}
        for state in (sim.pipeline.state_0, sim.pipeline.state_1):
            state.joint_q.assign(q)
            state.joint_qd.assign(zeros)
            newton.eval_fk(sim.model, state.joint_q, state.joint_qd, state)
            if not np.isfinite(state.body_q.numpy()).all():
                raise ValueError(f"nonfinite FK at {time}")
            actual = state.joint_q.numpy()
            for track, coord in bindings:
                if not np.allclose(
                    actual[coord : coord + track.values.shape[1]], track.values[frame], rtol=0, atol=1e-7
                ):
                    raise ValueError(f"failed write {track.name} at {time}")
                if track.values.shape[1] == 7:
                    # Fixed joints can share the next joint's coordinate start.
                    # Select the seven-coordinate free joint, not its fixed predecessor.
                    joint_index = int(np.flatnonzero((starts[:-1] == coord) & (np.diff(starts) == 7))[0])
                    body_pose = state.body_q.numpy()[int(children[joint_index])]
                    expected = track.values[frame]
                    position_error = float(np.linalg.norm(body_pose[:3] - expected[:3]))
                    quaternion_error = float(
                        min(
                            np.linalg.norm(body_pose[3:] - expected[3:]),
                            np.linalg.norm(body_pose[3:] + expected[3:]),
                        )
                    )
                    if position_error > 1e-6 or quaternion_error > 1e-6:
                        raise ValueError(
                            f"free-body FK mismatch {track.name} at {time}: {position_error}, {quaternion_error}"
                        )
                    pose_errors[track.name] = {
                        "position_error_m": position_error,
                        "sign_invariant_quaternion_error": quaternion_error,
                    }
        if sim.pipeline.step_count != 0:
            raise ValueError("binding check must not step physics")
        results.append(
            {
                "source_s": time,
                "frame": frame,
                "all_track_writes_match": True,
                "free_body_world_pose_errors": pose_errors,
            }
        )
    report = {
        "scope": "Actual RSS scene import, unique coordinate binding and FK writes; no scene or motion acceptance.",
        "scene_config_sha256": sha256(args.scene_config),
        "motion_sha256": sha256(args.candidate / "motion.npz"),
        "robot_root_coordinate": root,
        "bindings": [{"name": t.name, "coordinate": int(i), "width": t.values.shape[1]} for t, i in bindings],
        "checks": results,
        "sim_steps": sim.pipeline.step_count,
        "passed": True,
    }
    config = json.loads(args.scene_config.read_text())
    asset_paths = {config["robot_and_early_props"], config["room_mesh"]}
    asset_paths.update(item["path"] for item in config.get("additional_static_assets", []))
    for body in config["additional_bodies"]:
        asset_paths.add(body["visual_mesh"])
        asset_paths.update(body.get("collision_meshes", []))
    track_manifest_path = args.candidate / "scene_tracks/tracks.json"
    track_manifest = json.loads(track_manifest_path.read_text())
    report["direct_scene_asset_sha256"] = {path: sha256(Path(path)) for path in sorted(asset_paths)}
    report["track_manifest_sha256"] = sha256(track_manifest_path)
    report["track_file_sha256"] = {
        entry["name"]: sha256(track_manifest_path.parent / entry["path"])
        for kind in ("objects", "joints")
        for entry in track_manifest.get(kind, [])
    }
    report["provenance_scope"] = (
        "Direct assets and tracks; MJCF include/mesh and OBJ material dependencies are not recursively hashed."
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
