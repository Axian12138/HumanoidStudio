"""Review one G1 candidate in its original room and episode clock."""

import argparse
import json
from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.visualization.kinematic import KinematicStudio


class SourceClockStudio(KinematicStudio):
    """Keep the shared player, adding the recording's source-frame mapping."""

    def __init__(self, *args, timing, **kwargs):
        with np.load(timing) as data:
            self._source_frames = data["source_frame"].copy()
        self._source_label = None
        super().__init__(*args, **kwargs)
        self._source_label = self.server.scene.add_label(
            "/source_clock", "", position=tuple(self.motion.body_pos_w[0, 0] + [0, 0, 0.9])
        )
        self._set_frame(self.cursor)

    def _set_frame(self, frame):
        super()._set_frame(frame)
        if self._source_label is not None:
            self._source_label.text = f"G1 frame {frame}; source frame {self._source_frames[frame]:.2f}; wall time {self._source_offset_s + frame / 50:.2f}s"
            self._source_label.position = tuple(self.motion.body_pos_w[frame, 0] + [0, 0, 0.9])

    def _public_state(self):
        state = super()._public_state()
        state["original_source_frame"] = float(self._source_frames[self.cursor])
        state["clock"] = "resampled recording wall clock"
        return state


def main():
    import trimesh

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--source-room", type=Path, required=True)
    ap.add_argument("--robot", type=Path, required=True)
    ap.add_argument("--motion-order", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8520)
    args = ap.parse_args()
    run = args.candidate.resolve()
    order = json.loads(args.motion_order.read_text())
    names = tuple(order["robot_joint_names"])
    bodies = tuple(order["robot_body_names"])
    motion = ReferenceMotion.load(run / "body_prior.npz", joint_names=names, body_names=bodies)
    robot = rss.EmbodimentSpec(
        name="g1",
        asset=rss.AssetSpec(path=str(args.robot.resolve())),
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
        cell="void", name="G1 body prior | original human | wall clock | kinematic candidate", physics=rss.MUJOCO_500HZ
    )
    scene.add(rss.robot(robot))
    with np.load(run / "sidecars/object_0.npz") as z:
        pos = z["position_w"][0]
    scene.add(rss.sphere(name="fruit_position_marker", radius=0.025, at=tuple(pos), mass=0.01, color=(1.0, 0.4, 0.0)))
    viewer = SourceClockStudio(
        scene.build(),
        motion,
        start=0,
        stop=motion.frames,
        port=args.port,
        human_motion=run / "sidecars/human.npz",
        timing=run / "sidecars/timing.npz",
        source_offset_s=json.loads((run / "review/provenance.json").read_text()).get("source_offset_s", 0.0),
        object_track=run / "sidecars/object_0.npz",
        object_name="fruit_position_marker",
        follow_root=True,
        camera_look_offset=(0, 0, 0),
    )
    mesh = trimesh.load(args.source_room, force="mesh")
    viewer.server.scene.add_mesh_trimesh("/room/original", mesh, cast_shadow=False, receive_shadow=False)
    viewer.server.scene.add_label(
        "/note",
        "G1 body prior; orange = recorded fruit position ONLY; contact geometry missing; no physics",
        position=tuple(motion.body_pos_w[0, 0] + [0, 0, 1.2]),
    )
    viewer.run()


if __name__ == "__main__":
    main()
