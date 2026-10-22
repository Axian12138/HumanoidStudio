"""Open both sampled episodes using the shared Stage 3 timeline and renderer."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

import robosimstudio as rss
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.visualization.kinematic import KinematicStudio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--port", type=int, default=8510)
    args = ap.parse_args()
    run = args.run.resolve()
    with np.load(run / "motion.npz") as z:
        names = tuple(z["robot_joint_names"])
        bodies = tuple(z["robot_body_names"])
    motion = ReferenceMotion.load(
        run / "motion.npz", joint_names=names, body_names=bodies
    )
    robot = rss.EmbodimentSpec(
        name="nx1",
        asset=rss.AssetSpec(path=str(run / "nx1_free.xml")),
        preserve_free_bodies=True,
        enable_self_collisions=False,
        arms=(
            rss.ArmSpec(
                name="whole_body",
                joint_names=names,
                ee_body_name="right_wrist_yaw_link",
                home_q=tuple(motion.joint_pos[0]),
            ),
        ),
    )
    scene = rss.Scene(
        cell="void",
        name="NX1 original teleop replay — Stage 3 kinematics",
        physics=rss.MUJOCO_500HZ,
    )
    scene.add(rss.robot(robot))
    segments = json.loads((run / "segments.json").read_text())
    tracks = []
    for i, label in enumerate(("01_1014", "02_1014")):
        folder = run / label
        if not (folder / "object.glb").exists():
            continue
        with np.load(folder / "object_track.npz") as z:
            pos = z["position_w"]
            quat = z["quaternion_xyzw"]
        scene.add(
            rss.body(
                name="object_" + label,
                asset=str(folder / "object.glb"),
                at=tuple(pos[0]),
                mass=0.1,
            )
        )
        n = motion.frames
        s = segments[i]["start"]
        e = segments[i]["stop"]
        fullpos = np.repeat(pos[:1], n, axis=0)
        fullquat = np.repeat(quat[:1], n, axis=0)
        fullpos[s:e] = pos
        fullquat[s:e] = quat
        fullpos[e:] = pos[-1]
        fullquat[e:] = quat[-1]
        np.savez_compressed(
            folder / "combined_track.npz",
            position_w=fullpos,
            quaternion_xyzw=fullquat,
            fps=50,
            source_time_s=np.arange(n) / 50,
        )
        tracks.append(
            {"name": "object_" + label, "path": label + "/combined_track.npz"}
        )
    manifest = run / "tracks.json"
    manifest.write_text(json.dumps({"objects": tracks, "joints": []}))
    sim = scene.build()
    viewer = KinematicStudio(
        sim,
        motion,
        start=0,
        stop=motion.frames,
        port=args.port,
        human_motion=run / "human.npz",
        tracks_manifest=manifest if tracks else None,
        segments=segments,
        follow_root=True,
        camera_eye_offset=(1.8, -2.4, 1.2),
        camera_look_offset=(0, 0, 0),
    )
    for label in ("01_1014", "02_1014"):
        if (run / label / "room.glb").exists():
            mesh = trimesh.load(run / label / "room.glb", force="mesh")
            viewer.server.scene.add_mesh_trimesh(
                "/room/" + label, mesh, cast_shadow=False, receive_shadow=False
            )
    viewer.server.scene.add_label(
        "/review_note",
        "Orange markers: recorded fruit positions; original fruit meshes unavailable. Room 02 displayed +12 m in X.",
        position=tuple(motion.body_pos_w[0, 0] + [0, 0, 1.2]),
    )
    viewer.run()


if __name__ == "__main__":
    main()
