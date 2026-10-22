"""Review a retargeted motion in the RSS scene without tracker or physics steps."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/robosimstudio_humanoid_real2sim_warp")

import newton
import numpy as np

from robosimstudio.humanoid_real2sim.retarget.root_joint import root_joint_index
from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.humanoid_real2sim.visualization.tracks import bind_scene_tracks, load_scene_tracks
from robosimstudio.scene.entity import label_index
from robosimstudio_studio import StudioConfig, StudioSession


class KinematicStudio(StudioSession):
    """The same RSS/Viser scene as repair, with a source-frame scrubber."""

    def __init__(
        self,
        sim,
        motion: ReferenceMotion,
        *,
        start: int,
        stop: int,
        port: int,
        human_motion: Path,
        object_track: Path | None = None,
        tracks_manifest: Path | None = None,
        context_mesh: Path | None = None,
        rigid_track: Path | None = None,
        object_name: str | None = None,
        source_offset_s: float = 0.0,
        segments: list[dict] | None = None,
        camera_eye_offset: tuple[float, float, float] = (1.8, -2.4, 1.2),
        camera_look_offset: tuple[float, float, float] = (-0.2, -0.35, -0.25),
        follow_root: bool = False,
        dark_mode: bool = False,
    ):
        if not 0 <= start < stop <= motion.frames:
            raise ValueError(f"frame window must satisfy 0 <= start < stop <= {motion.frames}")
        self.motion, self.start, self.stop = motion, start, stop
        self.cursor = start
        self.playing = False
        self.playback_rate = 1.0
        self._next_frame_time = time.monotonic()
        self._requests: deque[tuple[str, object]] = deque()
        self._port = port
        self._source_offset_s = source_offset_s
        self._segments = segments or []
        self._track = None
        self._rigid = None
        self._context_mesh = str(context_mesh) if context_mesh is not None else None
        self._object_name = object_name
        scene_tracks = (
            load_scene_tracks(
                tracks_manifest,
                frames=motion.frames,
                fps=motion.fps,
                source_offset_s=source_offset_s,
            )
            if tracks_manifest is not None
            else []
        )
        if object_track is not None and tracks_manifest is not None:
            raise ValueError("use either --object-track or --tracks-manifest")
        with np.load(human_motion, allow_pickle=False) as source:
            human_positions = np.asarray(source["positions_w"], dtype=np.float32)
            human_names = tuple(map(str, source["joint_names"]))
            human_edges = np.asarray(source["edges"], dtype=np.int32)
            human_fps = float(source["fps"])
        if human_positions.shape != (motion.frames, len(human_names), 3):
            raise ValueError("human motion must have one aligned 3D skeleton per retarget frame")
        if (
            human_edges.ndim != 2
            or human_edges.shape[1] != 2
            or np.any(human_edges < 0)
            or np.any(human_edges >= len(human_names))
        ):
            raise ValueError("human motion edges must index its named joints")
        if not np.isclose(human_fps, motion.fps) or not np.isfinite(human_positions).all():
            raise ValueError("human motion FPS or positions do not match the retarget motion")
        self._human_positions = human_positions
        self._human_names = human_names
        self._human_edges = human_edges
        if rigid_track is not None:
            with np.load(rigid_track, allow_pickle=False) as data:
                rigid_positions = np.asarray(data["position_w"], dtype=np.float32)
                rigid_quaternions = np.asarray(data["quaternion_xyzw"], dtype=np.float32)
                wrist_proxies = np.asarray(data["wrist_proxy_w"], dtype=np.float32)
                tracking_valid = np.asarray(data["tracking_valid"], dtype=bool)
                rigid_names = tuple(map(str, data["names"]))
                rigid_fps = float(data["fps"])
                source_range = np.asarray(data["source_range_s"], dtype=float)
            if rigid_names != ("left", "right") or rigid_positions.shape != (motion.frames, 2, 3):
                raise ValueError("rigid track must contain aligned left/right positions")
            if rigid_quaternions.shape != (motion.frames, 2, 4) or wrist_proxies.shape != rigid_positions.shape:
                raise ValueError("rigid track orientations or wrist proxies have the wrong shape")
            if tracking_valid.shape != (motion.frames, 2) or not np.isclose(rigid_fps, motion.fps):
                raise ValueError("rigid track validity or FPS does not match motion")
            if not np.allclose(source_range, [source_offset_s, source_offset_s + motion.frames / motion.fps]):
                raise ValueError("rigid track source time window does not match motion")
            if not all(np.isfinite(item).all() for item in (rigid_positions, rigid_quaternions, wrist_proxies)):
                raise ValueError("rigid track contains non-finite poses")
            self._rigid = (rigid_positions, rigid_quaternions, wrist_proxies, tracking_valid)
        if object_track is not None:
            if not object_name:
                raise ValueError("--object-name is required with --object-track")
            with np.load(object_track, allow_pickle=False) as data:
                position = np.asarray(data["position_w"], dtype=np.float32)
                quat = np.asarray(data["quaternion_xyzw"], dtype=np.float32)
            if position.shape != (motion.frames, 3) or quat.shape != (motion.frames, 4):
                raise ValueError("object track must contain one position and xyzw quaternion per motion frame")
            if not (np.isfinite(position).all() and np.isfinite(quat).all()):
                raise ValueError("object track contains non-finite values")
            self._track = (position, quat)
        super().__init__(
            sim, config=StudioConfig(port=port, panels=("overlay",), speed="1x", backdrop=False, dark_mode=dark_mode)
        )
        self.set_paused(True)
        for folder in self.gui._container_handle_from_uuid.values():
            if getattr(folder, "label", None) in ("Simulation", "How to drive this"):
                folder.visible = False
        self.gui.main_panel.minimize()
        human_scene = self.server.scene
        if context_mesh is not None:
            import trimesh

            table = trimesh.load(context_mesh, force="mesh", process=False)
            if table.is_empty:
                raise ValueError(f"context mesh is empty: {context_mesh}")
            human_scene.add_mesh_simple(
                "/context/table",
                vertices=np.asarray(table.vertices, dtype=np.float32),
                faces=np.asarray(table.faces, dtype=np.uint32),
                color=(177, 128, 82),
                opacity=0.9,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
            )
        self._human_joints = [
            human_scene.add_icosphere(
                f"/human_motion/{name}",
                radius=0.022 if name != "pelvis" else 0.035,
                color=(70, 229, 226),
                opacity=0.32,
                cast_shadow=False,
                position=tuple(human_positions[start, index]),
            )
            for index, name in enumerate(human_names)
        ]
        self._human_lines = human_scene.add_line_segments(
            "/human_motion/skeleton",
            points=human_positions[start, human_edges],
            colors=(70, 229, 226),
            thickness=0.009,
        )
        self._rigid_handles = []
        if self._rigid is not None:
            for side, color in (("left", (255, 174, 54)), ("right", (247, 80, 158))):
                index = 0 if side == "left" else 1
                raw = human_scene.add_icosphere(
                    f"/rigid/{side}/tracker",
                    radius=0.026,
                    color=color,
                    position=tuple(rigid_positions[start, index]),
                    opacity=0.9,
                )
                proxy = human_scene.add_icosphere(
                    f"/rigid/{side}/wrist_proxy",
                    radius=0.018,
                    color=color,
                    position=tuple(wrist_proxies[start, index]),
                    opacity=0.45,
                )
                axes = human_scene.add_line_segments(
                    f"/rigid/{side}/axes",
                    points=np.zeros((3, 2, 3)),
                    colors=np.repeat(np.asarray([[255, 80, 80], [80, 255, 80], [80, 130, 255]])[:, None], 2, axis=1),
                    thickness=0.006,
                )
                offset = human_scene.add_line_segments(
                    f"/rigid/{side}/wrist_comparison",
                    points=np.zeros((1, 2, 3)),
                    colors=color,
                    thickness=0.004,
                )
                self._rigid_handles.append((raw, proxy, axes, offset))
        labels = list(sim.model.joint_label)
        starts = sim.model.joint_q_start.numpy()
        self._root = int(starts[root_joint_index(sim.model, motion.body_names[0])])
        self._joint_coords = np.asarray(
            [starts[label_index(labels, name)] for name in motion.all_joint_names], dtype=np.int32
        )
        self._scene_tracks = bind_scene_tracks(scene_tracks, labels, starts, self._root, self._joint_coords)
        if self._track is not None:
            try:
                self._object_coord = int(starts[label_index(labels, object_name)])
            except KeyError as exc:
                try:
                    self._object_coord = int(starts[label_index(labels, f"{object_name}_free_joint")])
                except KeyError:
                    raise ValueError(f"free joint for object {object_name!r} not found") from exc
        else:
            self._object_coord = None
        self._base_q = sim.pipeline.snapshot.joint_q.astype(np.float32).copy()
        self._zero_qd = np.zeros_like(sim.pipeline.snapshot.joint_qd, dtype=np.float32)
        self._follow_root = follow_root
        self._last_camera_root = motion.body_pos_w[start, 0].astype(float).copy()

        def aim(client):
            focus = motion.body_pos_w[self.cursor, 0]
            eye = tuple(float(v) for v in focus + np.asarray(camera_eye_offset))
            look = tuple(float(v) for v in focus + np.asarray(camera_look_offset))
            client.camera.position, client.camera.look_at = eye, look

        self.server.on_client_connect(aim)
        for client in self.server.get_clients().values():
            aim(client)
        self._set_frame(start)
        self._start_web()

    def _set_frame(self, frame: int) -> None:
        if not self.start <= frame < self.stop:
            raise ValueError("frame outside review window")
        if self._follow_root:
            root_position = self.motion.body_pos_w[frame, 0].astype(float)
            shift = root_position - self._last_camera_root
            if np.any(shift):
                for client in self.server.get_clients().values():
                    # Viser's position setter also translates look_at. Capture
                    # both targets first so the focus is not shifted twice.
                    eye = tuple(np.asarray(client.camera.position) + shift)
                    look = tuple(np.asarray(client.camera.look_at) + shift)
                    client.camera.position = eye
                    client.camera.look_at = look
            self._last_camera_root = root_position
        q = self._base_q.copy()
        root = self._root
        q[root : root + 3] = self.motion.body_pos_w[frame, 0]
        q[root + 3 : root + 7] = self.motion.body_quat_w[frame, 0, [1, 2, 3, 0]]
        q[self._joint_coords] = self.motion.all_joint_pos[frame]
        if self._track is not None:
            coord = self._object_coord
            q[coord : coord + 3] = self._track[0][frame]
            q[coord + 3 : coord + 7] = self._track[1][frame]
        for track, coord in self._scene_tracks:
            q[coord : coord + track.values.shape[1]] = track.values[frame]
        pipeline = self.env.pipeline
        for state in (pipeline.state_0, pipeline.state_1):
            state.joint_q.assign(q)
            state.joint_qd.assign(self._zero_qd)
            newton.eval_fk(self.env.model, state.joint_q, state.joint_qd, state)
        pipeline.mark_dirty()
        for index, node in enumerate(self._human_joints):
            node.position = tuple(self._human_positions[frame, index])
        self._human_lines.points = self._human_positions[frame, self._human_edges]
        if self._rigid is not None:
            from scipy.spatial.transform import Rotation

            positions, quaternions, proxies, valid = self._rigid
            for index, (raw, proxy, axes, offset) in enumerate(self._rigid_handles):
                active = bool(valid[frame, index])
                raw.visible = proxy.visible = axes.visible = offset.visible = active
                if not active:
                    continue
                point = positions[frame, index]
                orientation = Rotation.from_quat(quaternions[frame, index]).as_matrix()
                raw.position = tuple(point)
                proxy.position = tuple(proxies[frame, index])
                axes.points = np.stack((np.repeat(point[None], 3, axis=0), point[None] + orientation.T * 0.09), axis=1)
                wrist_index = self._human_names.index(("left", "right")[index] + "_wrist")
                offset.points = np.asarray([[self._human_positions[frame, wrist_index], proxies[frame, index]]])
        self.cursor = frame
        self._drawn_step = None

    def _public_state(self) -> dict:
        return {
            "cursor": self.cursor,
            "start": self.start,
            "stop": self.stop,
            "fps": self.motion.fps,
            "playing": self.playing,
            "playback_rate": self.playback_rate,
            "source_offset_s": self._source_offset_s,
            "segments": self._segments,
            "human_motion": True,
            "object_track": self._object_name if self._track is not None else None,
            "scene_tracks": [
                {
                    "name": track.name,
                    "kind": track.kind,
                    "path": str(track.path),
                    "position": track.values[self.cursor].tolist(),
                }
                for track, _ in self._scene_tracks
            ],
            "rigid_track": self._rigid is not None,
            "context_mesh": self._context_mesh,
            "rigid_valid": self._rigid[3][self.cursor].tolist() if self._rigid is not None else None,
            "sim_steps": self.env.pipeline.step_count,
        }

    def _start_web(self) -> None:
        studio = self
        page = Path(__file__).with_name("kinematic_web_ui.html")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/state":
                    payload, kind = json.dumps(studio._public_state()).encode(), "application/json"
                elif self.path == "/":
                    payload, kind = page.read_bytes(), "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                if self.path != "/api/command":
                    self.send_error(404)
                    return
                try:
                    data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    command = data["command"]
                    if command not in ("toggle", "seek", "reset", "rate"):
                        raise ValueError("unknown command")
                    value = int(data["value"]) if command == "seek" else None
                    if command == "rate":
                        value = float(data["value"])
                        if value not in (1.0, 0.5, 0.25, 0.1):
                            raise ValueError("unsupported playback rate")
                    if command == "seek" and not studio.start <= value < studio.stop:
                        raise ValueError("frame outside review window")
                    studio._requests.append((command, value))
                    self.send_response(204)
                    self.end_headers()
                except (ValueError, KeyError, TypeError) as exc:
                    self.send_error(400, str(exc))

            def log_message(self, _format, *_args):
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", self._port + 1), Handler)
        Thread(target=self._http.serve_forever, daemon=True).start()
        print(f"[kinematic] open http://localhost:{self._port + 1}/", flush=True)

    def tick(self) -> bool:
        while self._requests:
            command, value = self._requests.popleft()
            if command == "toggle":
                self.playing = not self.playing
                self._next_frame_time = time.monotonic() + 1 / (self.motion.fps * self.playback_rate)
            elif command == "rate":
                self.playback_rate = value
                self._next_frame_time = time.monotonic() + 1 / (self.motion.fps * self.playback_rate)
            elif command == "seek":
                self.playing = False
                self._set_frame(value)
            elif command == "reset":
                self.playing = False
                self._set_frame(self.start)
        now = time.monotonic()
        if self.playing and now >= self._next_frame_time:
            wall_fps = self.motion.fps * self.playback_rate
            frame = min(self.stop - 1, self.cursor + max(1, int((now - self._next_frame_time) * wall_fps) + 1))
            self._set_frame(frame)
            self._next_frame_time = now + 1 / wall_fps
            if frame == self.stop - 1:
                self.playing = False
        # Studio stays paused: tick only services the viewer and never steps physics.
        self.set_paused(True)
        return super().tick()

    def close(self) -> None:
        if hasattr(self, "_http"):
            self._http.shutdown()
            self._http.server_close()
        super().close()
