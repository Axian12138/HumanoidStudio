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

from robosimstudio.humanoid_real2sim.track.timeline import ReferenceMotion
from robosimstudio.scene.entity import label_index
from robosimstudio_studio import StudioConfig, StudioSession


class KinematicStudio(StudioSession):
    """The same RSS/Viser scene as repair, with a source-frame scrubber."""

    def __init__(self, sim, motion: ReferenceMotion, *, start: int, stop: int, port: int,
                 human_motion: Path, object_track: Path | None = None, object_name: str | None = None,
                 source_offset_s: float = 0.0, segments: list[dict] | None = None):
        if not 0 <= start < stop <= motion.frames:
            raise ValueError(f"frame window must satisfy 0 <= start < stop <= {motion.frames}")
        self.motion, self.start, self.stop = motion, start, stop
        self.cursor = start
        self.playing = False
        self._next_frame_time = time.monotonic()
        self._requests: deque[tuple[str, object]] = deque()
        self._port = port
        self._source_offset_s = source_offset_s
        self._segments = segments or []
        self._track = None
        self._object_name = object_name
        with np.load(human_motion, allow_pickle=False) as source:
            human_positions = np.asarray(source["positions_w"], dtype=np.float32)
            human_names = tuple(map(str, source["joint_names"]))
            human_edges = np.asarray(source["edges"], dtype=np.int32)
            human_fps = float(source["fps"])
        if human_positions.shape != (motion.frames, len(human_names), 3):
            raise ValueError("human motion must have one aligned 3D skeleton per retarget frame")
        if human_edges.ndim != 2 or human_edges.shape[1] != 2 or np.any(human_edges < 0) or np.any(human_edges >= len(human_names)):
            raise ValueError("human motion edges must index its named joints")
        if not np.isclose(human_fps, motion.fps) or not np.isfinite(human_positions).all():
            raise ValueError("human motion FPS or positions do not match the retarget motion")
        self._human_positions = human_positions
        self._human_names = human_names
        self._human_edges = human_edges
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
        super().__init__(sim, config=StudioConfig(port=port, panels=("overlay",), speed="1x", backdrop=False))
        self.set_paused(True)
        for folder in self.gui._container_handle_from_uuid.values():
            if getattr(folder, "label", None) in ("Simulation", "How to drive this"):
                folder.visible = False
        self.gui.main_panel.minimize()
        human_scene = self.server.scene
        self._human_joints = [
            human_scene.add_icosphere(
                f"/human_motion/{name}", radius=0.022 if name != "pelvis" else 0.035,
                color=(70, 229, 226), opacity=0.32, cast_shadow=False,
                position=tuple(human_positions[start, index]),
            )
            for index, name in enumerate(human_names)
        ]
        self._human_lines = human_scene.add_line_segments(
            "/human_motion/skeleton",
            points=human_positions[start, human_edges],
            colors=(70, 229, 226), thickness=0.009,
        )
        labels = list(sim.model.joint_label)
        starts = sim.model.joint_q_start.numpy()
        self._root = int(starts[label_index(labels, "floating_base")])
        self._joint_coords = np.asarray(
            [starts[label_index(labels, name)] for name in motion.all_joint_names], dtype=np.int32
        )
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
        focus = motion.body_pos_w[start, 0]
        eye = tuple(float(v) for v in focus + np.array((1.8, -2.4, 1.2)))
        look = tuple(float(v) for v in focus + np.array((-0.2, -0.35, -0.25)))

        def aim(client):
            client.camera.position, client.camera.look_at = eye, look

        self.server.on_client_connect(aim)
        for client in self.server.get_clients().values():
            aim(client)
        self._set_frame(start)
        self._start_web()

    def _set_frame(self, frame: int) -> None:
        if not self.start <= frame < self.stop:
            raise ValueError("frame outside review window")
        q = self._base_q.copy()
        root = self._root
        q[root:root + 3] = self.motion.body_pos_w[frame, 0]
        q[root + 3:root + 7] = self.motion.body_quat_w[frame, 0, [1, 2, 3, 0]]
        q[self._joint_coords] = self.motion.all_joint_pos[frame]
        if self._track is not None:
            coord = self._object_coord
            q[coord:coord + 3] = self._track[0][frame]
            q[coord + 3:coord + 7] = self._track[1][frame]
        pipeline = self.env.pipeline
        for state in (pipeline.state_0, pipeline.state_1):
            state.joint_q.assign(q)
            state.joint_qd.assign(self._zero_qd)
            newton.eval_fk(self.env.model, state.joint_q, state.joint_qd, state)
        pipeline.mark_dirty()
        for index, node in enumerate(self._human_joints):
            node.position = tuple(self._human_positions[frame, index])
        self._human_lines.points = self._human_positions[frame, self._human_edges]
        self.cursor = frame
        self._drawn_step = None

    def _public_state(self) -> dict:
        return {"cursor": self.cursor, "start": self.start, "stop": self.stop,
                "fps": self.motion.fps, "playing": self.playing,
                "source_offset_s": self._source_offset_s, "segments": self._segments,
                "human_motion": True,
                "object_track": self._object_name if self._track is not None else None,
                "sim_steps": self.env.pipeline.step_count}

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
                    if command not in ("toggle", "seek", "reset"):
                        raise ValueError("unknown command")
                    value = int(data["value"]) if command == "seek" else None
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
                self._next_frame_time = time.monotonic() + 1 / self.motion.fps
            elif command == "seek":
                self.playing = False
                self._set_frame(value)
            elif command == "reset":
                self.playing = False
                self._set_frame(self.start)
        now = time.monotonic()
        if self.playing and now >= self._next_frame_time:
            frame = min(self.stop - 1, self.cursor + max(1, int((now - self._next_frame_time) * self.motion.fps) + 1))
            self._set_frame(frame)
            self._next_frame_time = now + 1 / self.motion.fps
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
