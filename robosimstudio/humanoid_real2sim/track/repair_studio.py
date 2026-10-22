"""A small repair controller around the Viser 3D view and RSS physics loop."""

from __future__ import annotations

import json
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import numpy as np

from robosimstudio.humanoid_real2sim.track.timeline import FIVE_POINTS
from robosimstudio_studio import StudioConfig, StudioSession


class RepairStudio(StudioSession):
    def __init__(
        self,
        sim,
        runtime,
        *,
        out: Path,
        port: int,
        stop_frame: int | None = None,
        show_reference: bool = True,
        reference_video: Path | None = None,
        reference_object_track: Path | None = None,
        reference_object_mesh: Path | None = None,
        reference_object_outline: str = "ellipse",
        review_only: bool = False,
    ):
        self.runtime = runtime
        self._stop_frame = stop_frame
        self._show_reference = show_reference
        self._reference_video = reference_video
        self._reference_object_track = reference_object_track
        self._reference_object_mesh = reference_object_mesh
        self._reference_object_outline = reference_object_outline
        self._review_only = review_only
        self._requests: deque[tuple[str, object]] = deque()
        self._out = out
        self._port = port
        self._marks: list[int] = [
            index
            for index, record in enumerate(runtime.records)
            if record.phase == "takeover" and (index == 0 or runtime.records[index - 1].phase != "takeover")
        ]
        self._last_click: dict[str, float] = {}
        self._selected_link = FIVE_POINTS[2]
        self._direction = {name: np.array((0.0, 0.0, 1.0)) for name in FIVE_POINTS}
        self._speed = dict.fromkeys(FIVE_POINTS, 0.30)
        self._angular_speed = dict.fromkeys(FIVE_POINTS, 1.0)
        self._goal_pos: dict[str, np.ndarray] = {}
        self._goal_quat: dict[str, np.ndarray] = {}
        self._staged: dict[str, str] = {}
        self._staged_hands: dict[str, float] = {}
        self._recovery_mode = False
        self._recovery_started = False
        self._recovery_origin_slot = 0
        self._recovery_original_records = []
        self._recovery_frame = 0
        self._recovery_duration_frames = 50
        self._repair_gizmos: dict[str, bool] = {}
        self._syncing_gizmos = False
        self._dirty = True
        super().__init__(
            sim,
            config=StudioConfig(port=port, panels=("overlay",), out=str(out), speed="1x", backdrop=False),
        )
        # StudioSession.adopt resets a supplied Sim. Restore the physical cache
        # after its viewer and panels are built, before placing any edit handles.
        if runtime.records:
            runtime.seek_cached(runtime.cursor)
            self._drawn_step = None
        elif runtime.anchor_source_frame:
            runtime.seek_reference(runtime.anchor_source_frame)
            self._drawn_step = None
        focus = runtime.motion.body_pos_w[runtime.anchor_source_frame, 0]
        eye = tuple(float(v) for v in focus + np.array((1.8, -2.4, 1.2)))
        look = tuple(float(v) for v in focus + np.array((-0.2, -0.35, -0.25)))

        def aim(client):
            client.camera.position, client.camera.look_at = eye, look

        self.server.on_client_connect(aim)
        for client in self.server.get_clients().values():
            aim(client)
        self.set_paused(True)
        self.each_step(runtime.prepare)
        for folder in self.gui._container_handle_from_uuid.values():
            if getattr(folder, "label", None) in ("Simulation", "How to drive this"):
                folder.visible = False
        self.gui.main_panel.minimize()
        self._build_handles()
        self._build_ghost()
        self._start_web()

    def _start_web(self) -> None:
        studio = self
        page = Path(__file__).with_name("web_ui.html")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/reference.mp4" and studio._reference_video is not None:
                    path = studio._reference_video
                    size = path.stat().st_size
                    start, end = 0, size - 1
                    raw_range = self.headers.get("Range", "")
                    if raw_range.startswith("bytes="):
                        bounds = raw_range[6:].split("-", 1)
                        try:
                            start = int(bounds[0]) if bounds[0] else 0
                            end = int(bounds[1]) if bounds[1] else size - 1
                        except ValueError:
                            self.send_error(416)
                            return
                    if not 0 <= start <= end < size:
                        self.send_error(416)
                        return
                    self.send_response(206 if raw_range else 200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(end - start + 1))
                    if raw_range:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    with path.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = stream.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    return
                if self.path == "/api/state":
                    payload = json.dumps(studio._public_state()).encode()
                    kind = "application/json"
                elif self.path == "/":
                    payload = page.read_bytes()
                    kind = "text/html; charset=utf-8"
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
                    command = str(data["command"])
                    if command not in {
                        "toggle",
                        "seek",
                        "reset",
                        "speed",
                        "angular_speed",
                        "hand",
                        "link",
                        "recovery_mode",
                        "recovery_frame",
                        "recovery_duration",
                        "save",
                        "show_reference",
                    }:
                        raise ValueError("unknown command")
                    if studio._review_only and command not in {"toggle", "seek", "reset", "show_reference"}:
                        raise ValueError("Stage 5 review is read-only")
                    studio._ask(command, data.get("value"), pause=command in {"seek", "reset", "recovery_mode"})
                    self.send_response(204)
                    self.end_headers()
                except (ValueError, KeyError, TypeError) as exc:
                    self.send_error(400, str(exc))

            def log_message(self, _format, *_args):
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", self._port + 1), Handler)
        Thread(target=self._http.serve_forever, daemon=True).start()
        print(f"[repair] open http://localhost:{self._port + 1}/", flush=True)

    def _public_state(self) -> dict:
        timeline = self.runtime.timeline
        similarity = None
        if self._recovery_mode:
            achieved = self.runtime._snapshot()["body_pos"]
            timeline = self.runtime.timeline
            if self._recovery_started and timeline.phase == "recovery" and timeline._recovery_body_pos is not None:
                desired = timeline._recovery_body_pos[timeline._recovery_step]
            elif self._recovery_started:
                desired = self.runtime.motion.body_pos_w[timeline.source_frame]
            else:
                desired = self.runtime.motion.body_pos_w[self._recovery_frame]
            distance = np.linalg.norm(achieved - desired, axis=-1)
            similarity = {
                "mean_cm": round(float(distance.mean() * 100), 1),
                "max_cm": round(float(distance.max() * 100), 1),
            }
        return {
            "paused": self.paused,
            "phase": timeline.phase,
            "source": timeline.source_frame,
            "cursor": self.runtime.cursor,
            "cached": len(self.runtime.records),
            "frames": (
                self._stop_frame - self.runtime.anchor_source_frame + 1
                if self._stop_frame is not None
                else max(self.runtime.motion.frames, len(self.runtime.records) + 1)
            ),
            "source_frames": self.runtime.motion.frames,
            "start_frame": self.runtime.anchor_source_frame,
            "stop_frame": self._stop_frame,
            "reference_pid_joint_names": self.runtime.reference_pid_joint_names,
            "show_reference": self._show_reference,
            "reference_video": self._reference_video is not None,
            "reference_object": self._reference_object_track is not None,
            "review_only": self._review_only,
            "recovery_mode": self._recovery_mode,
            "recovery_started": self._recovery_started,
            "recovery_frame": self._recovery_frame,
            "recovery_duration_frames": self._recovery_duration_frames,
            "recovery_origin_slot": self._recovery_origin_slot,
            "recovery_progress": self.runtime.timeline._recovery_step
            if self.runtime.timeline.phase == "recovery"
            else 0,
            "recovery_elapsed_frames": max(0, self.runtime.cursor - self._recovery_origin_slot),
            "recovery_total_frames": self._recovery_duration_frames
            + (
                self.runtime.motion.frames - 1 - self._recovery_frame
                if self._recovery_duration_frames
                else self.runtime.motion.frames - self._recovery_frame
            ),
            "marks": self._marks,
            "selected": self._selected_link,
            "links": [
                {
                    "name": name,
                    "mode": self._staged.get(name, timeline.link_modes[name]),
                    "speed": self._speed[name],
                    "angular_speed": self._angular_speed[name],
                    "goal_pos": self._goal_pos[name].tolist(),
                    "goal_quat_wxyz": self._goal_quat[name].tolist(),
                    "staged": name in self._staged,
                }
                for name in FIVE_POINTS
            ],
            "hands": {
                side: {
                    "amount": self._staged_hands.get(side, self.runtime.hand_override[side]),
                    "staged": side in self._staged_hands,
                }
                for side in ("left", "right")
            },
            "similarity": similarity,
            "status": self.status,
        }

    def _ask(self, command: str, value=None, *, pause: bool = False) -> None:
        if pause:
            self.paused = True
        self._requests.append((command, value))

    def _body_pos(self, name: str) -> np.ndarray:
        i = self.runtime.motion.body_names.index(name)
        return self.runtime.sim.pipeline.snapshot.body_q[self.runtime.body_indices[i], :3].astype(float).copy()

    def _body_quat(self, name: str) -> np.ndarray:
        i = self.runtime.motion.body_names.index(name)
        xyzw = self.runtime.sim.pipeline.snapshot.body_q[self.runtime.body_indices[i], 3:7]
        return np.asarray(xyzw[[3, 0, 1, 2]], dtype=float)

    def _sync_goals_from_timeline(self) -> None:
        timeline = self.runtime.timeline
        for name in FIVE_POINTS:
            if name in timeline.goal_controlled and timeline._goal_pos is not None:
                j = self.runtime.motion.body_names.index(name)
                self._goal_pos[name] = timeline._goal_pos[j].copy()
                self._goal_quat[name] = timeline._goal_quat[j].copy()
                self._gizmos[name].visible = timeline.link_modes[name] == "move"
            else:
                self._goal_pos[name] = self._body_pos(name)
                self._goal_quat[name] = self._body_quat(name)
                self._gizmos[name].visible = False
            self._arrows[name].visible = self._gizmos[name].visible
            self._tips[name].visible = self._gizmos[name].visible

    def _build_handles(self) -> None:
        scene = self.server.scene
        self._dots, self._arrows, self._tips, self._gizmos = {}, {}, {}, {}
        self._anchors = {}
        for name in FIVE_POINTS:
            pos = self._body_pos(name)
            quat = self._body_quat(name)
            self._goal_pos[name], self._goal_quat[name] = pos.copy(), quat.copy()
            tip = pos
            dot = scene.add_icosphere(f"/repair/{name}/dot", radius=0.046, color=(255, 83, 135), position=tuple(pos))
            dot.on_click(lambda _event, link=name: self._dot_clicked(link))
            arrow = scene.add_line_segments(
                f"/repair/{name}/arrow",
                points=np.array([[pos, tip]]),
                colors=(255, 151, 54),
                thickness=0.008,
                visible=False,
            )
            tip_dot = scene.add_icosphere(
                f"/repair/{name}/tip", radius=0.018, color=(255, 151, 54), position=tuple(tip), visible=False
            )
            gizmo = scene.add_transform_controls(
                f"/repair/{name}/direction",
                scale=0.34,
                position=tuple(tip),
                disable_rotations=False,
                disable_sliders=True,
                depth_test=False,
                visible=False,
            )
            gizmo.on_update(lambda event, link=name: self._goal_dragged(link, event))
            self._dots[name], self._arrows[name], self._tips[name], self._gizmos[name] = dot, arrow, tip_dot, gizmo
            self._anchors[name] = pos

    def _dot_clicked(self, name: str) -> None:
        now = time.monotonic()
        double = now - self._last_click.get(name, -100.0) < 0.4
        self._last_click[name] = now
        self._ask("link", (name, double), pause=double)

    def _select_link(self, name: str, double: bool) -> None:
        if name not in FIVE_POINTS:
            raise ValueError("unknown link")
        self._selected_link = name
        if double:
            active = not self._gizmos[name].visible
            self._gizmos[name].visible = active
            self._arrows[name].visible = active
            self._tips[name].visible = active
            self._staged[name] = "move" if active else "hold"
            self._goal_pos[name] = self._body_pos(name)
            self._goal_quat[name] = self._body_quat(name)
            self._gizmos[name].position = tuple(self._goal_pos[name])
            self._gizmos[name].wxyz = tuple(self._goal_quat[name])
            self.set_paused(True)
        self._dirty = True

    def _goal_dragged(self, name: str, event) -> None:
        if self._syncing_gizmos:
            return
        self._goal_pos[name] = np.asarray(event.target.position, dtype=float)
        self._goal_quat[name] = np.asarray(event.target.wxyz, dtype=float)
        self._selected_link = name
        self._staged[name] = "move"
        self.paused = True
        self._dirty = True

    def _build_ghost(self) -> None:
        scene = self.server.scene
        self._ghost_object = None
        self._ghost_object_position = None
        self._ghost_object_quaternion = None
        self._ghost_object_segments = None
        if self._reference_object_track is not None and self._reference_object_mesh is not None:
            import trimesh

            with np.load(self._reference_object_track, allow_pickle=False) as track:
                position = track["position_w"]
                quaternion = track["quaternion_xyzw"]
            if position.shape != (self.runtime.motion.frames, 3) or quaternion.shape != (self.runtime.motion.frames, 4):
                raise ValueError("reference object track must match the robot reference frames")
            self._ghost_object_position = position
            self._ghost_object_quaternion = quaternion
            bounds = trimesh.load(self._reference_object_mesh, force="mesh", process=False).bounds
            if self._reference_object_outline == "box":
                corners = np.asarray(
                    [
                        (x, y, z)
                        for z in (bounds[0, 2], bounds[1, 2])
                        for x, y in (
                            (bounds[0, 0], bounds[0, 1]),
                            (bounds[1, 0], bounds[0, 1]),
                            (bounds[1, 0], bounds[1, 1]),
                            (bounds[0, 0], bounds[1, 1]),
                        )
                    ]
                )
                edges = [(i, (i + 1) % 4) for i in range(4)]
                edges += [(i + 4, (i + 1) % 4 + 4) for i in range(4)]
                edges += [(i, i + 4) for i in range(4)]
                loops = [corners[[a, b]] for a, b in edges]
            else:
                center_xy = (bounds[0, :2] + bounds[1, :2]) / 2
                half_xy = (bounds[1, :2] - bounds[0, :2]) / 2
                height = bounds[1, 2] - bounds[0, 2]
                # A lightweight outline stays legible next to the solid
                # physical object without transparency sorting artifacts.
                angles = np.linspace(0, 2 * np.pi, 25)
                loops = []
                for height_fraction, width_fraction in ((0.05, 0.55), (0.35, 0.72), (0.70, 0.86), (0.98, 0.98)):
                    z = bounds[0, 2] + height_fraction * height
                    points = np.column_stack(
                        (
                            center_xy[0] + width_fraction * half_xy[0] * np.cos(angles),
                            center_xy[1] + width_fraction * half_xy[1] * np.sin(angles),
                            np.full_like(angles, z),
                        )
                    )
                    loops.extend(np.stack((points[:-1], points[1:]), axis=1))
                for angle in np.linspace(0, 2 * np.pi, 9)[:-1]:
                    low = np.r_[
                        center_xy + 0.55 * half_xy * (np.cos(angle), np.sin(angle)), bounds[0, 2] + 0.05 * height
                    ]
                    high = np.r_[
                        center_xy + 0.98 * half_xy * (np.cos(angle), np.sin(angle)), bounds[0, 2] + 0.98 * height
                    ]
                    loops.append(np.stack((low, high)))
            self._ghost_object_segments = np.asarray(loops)
            self._ghost_object = scene.add_line_segments(
                "/reference_object/kinematic_basket",
                points=self._ghost_object_segments + position[0],
                colors=(250, 204, 65),
                thickness=0.005,
                visible=False,
            )
        self._ghost_meshes = []
        self._ghost_model = None
        self._ghost_data = None
        if self.runtime.original_model is not None:
            import mujoco
            import trimesh

            self._ghost_model = mujoco.MjModel.from_xml_path(str(self.runtime.original_model))
            self._ghost_data = mujoco.MjData(self._ghost_model)
            model = self._ghost_model
            for geom in range(model.ngeom):
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] not in (1, 2):
                    continue
                mesh = int(model.geom_dataid[geom])
                va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                vertices = np.asarray(model.mesh_vert[va : va + vn], dtype=np.float32)
                faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int32)
                if len(faces) > 3000:
                    simplified = trimesh.Trimesh(
                        vertices=vertices, faces=faces, process=False
                    ).simplify_quadric_decimation(face_count=3000)
                    vertices = np.asarray(simplified.vertices, dtype=np.float32)
                    faces = np.asarray(simplified.faces, dtype=np.int32)
                handle = scene.add_mesh_simple(
                    f"/reference_robot/geom_{geom:03d}",
                    vertices,
                    faces.astype(np.uint32),
                    color=(80, 224, 235),
                    opacity=0.28,
                    side="double",
                    cast_shadow=False,
                    receive_shadow=False,
                    visible=False,
                )
                self._ghost_meshes.append((geom, handle))
        self._ghost = []
        self._ghost_lines = []
        names = self.runtime.motion.body_names
        frame = self.runtime.motion.body_pos_w[0]
        for i, name in enumerate(names):
            node = scene.add_icosphere(
                f"/reference/{name}",
                radius=0.024 if name != "pelvis" else 0.035,
                color=(57, 229, 228),
                position=tuple(frame[i]),
                visible=False,
            )
            self._ghost.append(node)
        edges = (
            ("pelvis", "left_hip_roll_link"),
            ("left_hip_roll_link", "left_knee_link"),
            ("left_knee_link", "left_ankle_roll_link"),
            ("pelvis", "right_hip_roll_link"),
            ("right_hip_roll_link", "right_knee_link"),
            ("right_knee_link", "right_ankle_roll_link"),
            ("pelvis", "torso_link"),
            ("torso_link", "left_shoulder_roll_link"),
            ("left_shoulder_roll_link", "left_elbow_link"),
            ("left_elbow_link", "left_wrist_yaw_link"),
            ("torso_link", "right_shoulder_roll_link"),
            ("right_shoulder_roll_link", "right_elbow_link"),
            ("right_elbow_link", "right_wrist_yaw_link"),
        )
        for i, (a, b) in enumerate(edges):
            if a not in names or b not in names:
                continue
            node = scene.add_line_segments(
                f"/reference/edge_{i}",
                points=np.array([[frame[names.index(a)], frame[names.index(b)]]]),
                colors=(57, 229, 228),
                thickness=0.004,
                visible=False,
            )
            self._ghost_lines.append((node, names.index(a), names.index(b)))
        self._target_meshes: dict[str, list[tuple[int, object]]] = {name: [] for name in FIVE_POINTS}
        if self._ghost_model is not None:
            import mujoco

            model = self._ghost_model
            for name in FIVE_POINTS:
                root = model.body(name).id
                for geom in range(model.ngeom):
                    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] not in (1, 2):
                        continue
                    body = int(model.geom_bodyid[geom])
                    if name == "pelvis":
                        selected = body == root
                    elif "wrist" in name:
                        selected = body == root or (
                            "hand" in model.body(body).name and self._is_descendant(model, body, root)
                        )
                    else:
                        selected = body == root
                    if not selected:
                        continue
                    mesh = int(model.geom_dataid[geom])
                    va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                    fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                    vertices = np.asarray(model.mesh_vert[va : va + vn], dtype=np.float32)
                    faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=np.uint32)
                    handle = scene.add_mesh_simple(
                        f"/repair/{name}/goal_geom_{geom}",
                        vertices,
                        faces,
                        color=(255, 178, 75),
                        opacity=0.32,
                        side="double",
                        cast_shadow=False,
                        receive_shadow=False,
                        visible=False,
                    )
                    self._target_meshes[name].append((geom, handle))

    @staticmethod
    def _is_descendant(model, body: int, ancestor: int) -> bool:
        while body and body != ancestor:
            body = int(model.body_parentid[body])
        return body == ancestor

    def _update_ghost(self) -> None:
        visible = (self._recovery_mode and not self._recovery_started) or self._show_reference
        frame = (
            self._recovery_frame
            if self._recovery_mode and not self._recovery_started
            else self.runtime.timeline.source_frame
        )
        frame = max(0, min(self.runtime.motion.frames - 1, frame))
        if self._ghost_object is not None:
            self._ghost_object.visible = visible
            if visible:
                from scipy.spatial.transform import Rotation

                self._ghost_object.points = (
                    Rotation.from_quat(self._ghost_object_quaternion[frame])
                    .apply(self._ghost_object_segments.reshape(-1, 3))
                    .reshape(-1, 2, 3)
                    + self._ghost_object_position[frame]
                )
        for i, node in enumerate(self._ghost):
            node.visible = visible
            if visible:
                node.position = tuple(self.runtime.motion.body_pos_w[frame, i])
        for node, a, b in self._ghost_lines:
            node.visible = visible
            if visible:
                positions = self.runtime.motion.body_pos_w[frame]
                node.points = np.array([[positions[a], positions[b]]])
        if self._ghost_model is None or self._ghost_data is None:
            return
        for _, handle in self._ghost_meshes:
            handle.visible = visible
        if not visible:
            return
        import mujoco
        from scipy.spatial.transform import Rotation

        model, data = self._ghost_model, self._ghost_data
        motion = self.runtime.motion
        data.qpos[:3] = motion.body_pos_w[frame, 0]
        data.qpos[3:7] = motion.body_quat_w[frame, 0]
        for name, value in zip(motion.all_joint_names, motion.all_joint_pos[frame], strict=True):
            data.qpos[int(model.joint(name).qposadr[0])] = value
        mujoco.mj_forward(model, data)
        for geom, handle in self._ghost_meshes:
            handle.position = tuple(data.geom_xpos[geom])
            quat = Rotation.from_matrix(data.geom_xmat[geom].reshape(3, 3)).as_quat()
            handle.wxyz = tuple(quat[[3, 0, 1, 2]])

    def _refresh_scene(self) -> None:
        self._syncing_gizmos = True
        try:
            for name in FIVE_POINTS:
                center = self._body_pos(name)
                self._anchors[name] = center
                mode = self._staged.get(name, self.runtime.timeline.link_modes[name])
                self._dots[name].position = tuple(center)
                self._dots[name].visible = not self._recovery_mode
                self._dots[name].color = (
                    (81, 164, 255) if mode == "move" else (255, 165, 63) if mode == "hold" else (255, 83, 135)
                )
                tip = self._goal_pos[name]
                self._arrows[name].points = np.array([[center, tip]])
                self._tips[name].position = tuple(tip)
                if self._gizmos[name].visible:
                    self._gizmos[name].position = tuple(tip)
                    self._gizmos[name].wxyz = tuple(self._goal_quat[name])
            self._update_ghost()
            self._update_target_ghosts()
        finally:
            self._syncing_gizmos = False

    def _update_target_ghosts(self) -> None:
        if self._ghost_model is None or self._ghost_data is None:
            return
        import mujoco
        from scipy.spatial.transform import Rotation

        model, data = self._ghost_model, self._ghost_data
        snapshot = self.runtime.sim.pipeline.snapshot
        q = snapshot.joint_q
        root = self.runtime.root_coord
        data.qpos[:3] = q[root : root + 3]
        data.qpos[3:7] = q[root + 3 : root + 7][[3, 0, 1, 2]]
        labels = list(self.runtime.sim.model.joint_label)
        starts = self.runtime.sim.model.joint_q_start.numpy()
        from robosimstudio.scene.entity import label_index

        for name in self.runtime.motion.all_joint_names:
            data.qpos[int(model.joint(name).qposadr[0])] = q[int(starts[label_index(labels, name)])]
        mujoco.mj_forward(model, data)
        for name in FIVE_POINTS:
            visible = self._gizmos[name].visible and not self._recovery_mode
            root_body = model.body(name).id
            current_pos = data.xpos[root_body]
            current_rot = Rotation.from_quat(data.xquat[root_body][[1, 2, 3, 0]])
            goal_rot = Rotation.from_quat(self._goal_quat[name][[1, 2, 3, 0]])
            delta_rot = goal_rot * current_rot.inv()
            for geom, handle in self._target_meshes[name]:
                handle.visible = visible
                if visible:
                    handle.position = tuple(self._goal_pos[name] + delta_rot.apply(data.geom_xpos[geom] - current_pos))
                    geom_rot = Rotation.from_matrix(data.geom_xmat[geom].reshape(3, 3))
                    quat = (delta_rot * geom_rot).as_quat()
                    handle.wxyz = tuple(quat[[3, 0, 1, 2]])

    def _commit_edit(self) -> bool:
        if self._recovery_mode:
            raise RuntimeError("return to repair before editing a link")
        if not self._staged and not self._staged_hands:
            return False
        timeline = self.runtime.timeline
        if timeline.phase == "recovery":
            raise RuntimeError("wait for recovery to finish")
        changes = {}
        for name, mode in self._staged.items():
            changes[name] = mode
        self._staged.clear()
        hand_changes = {
            side: amount
            for side, amount in self._staged_hands.items()
            if self.runtime.hand_override[side] is None or abs(self.runtime.hand_override[side] - amount) > 1e-5
        }
        self._staged_hands.clear()
        if not changes and not hand_changes:
            return False
        if timeline.phase == "reference" and changes:
            self.runtime.begin_takeover()
        self.runtime.branch_from_here()
        self._marks = [mark for mark in self._marks if mark <= self.runtime.cursor]
        if self.runtime.cursor not in self._marks:
            self._marks.append(self.runtime.cursor)
        for name, mode in changes.items():
            if mode == "move":
                self.runtime.timeline.set_goal(
                    name,
                    self._goal_pos[name],
                    self._goal_quat[name],
                    self._speed[name],
                    self._angular_speed[name],
                )
            else:
                self.runtime.timeline.hold_pose(name, self._body_pos(name), self._body_quat(name))
        for side, amount in hand_changes.items():
            self.runtime.set_hand_amount(side, amount)
        return True

    def _toggle(self) -> None:
        if not self.paused:
            self.set_paused(True)
            return
        if self._stop_frame is not None and self.runtime.timeline.source_frame >= self._stop_frame:
            raise RuntimeError(f"片段已到源帧 {self._stop_frame}; 回退缓存或重置后再播放")
        if self._recovery_mode and not self._recovery_started:
            if self.runtime.timeline.phase != "takeover":
                raise RuntimeError("recovery requires an active repair")
            self.runtime.branch_from_here()
            self.runtime.begin_recovery(self._recovery_frame, self._recovery_duration_frames)
            self._recovery_started = True
            self._staged.clear()
            self._staged_hands.clear()
        elif not self._recovery_mode:
            self._commit_edit()
        self.set_paused(False)

    def _set_recovery_mode(self, enabled: bool) -> None:
        self.set_paused(True)
        if enabled:
            if self.runtime.timeline.phase != "takeover":
                raise RuntimeError("start a link repair before entering recovery")
            if self._staged or self._staged_hands:
                raise RuntimeError("press Play to apply staged edits before entering recovery")
            self._recovery_origin_slot = self.runtime.cursor
            self._recovery_original_records = self.runtime.records.copy()
            timeline = self.runtime.timeline
            self._recovery_frame = min(
                timeline._takeover_frame + timeline.takeover_steps, self.runtime.motion.frames - 1
            )
            self._recovery_duration_frames = 50
            self._repair_gizmos = {name: self._gizmos[name].visible for name in FIVE_POINTS}
            for name in FIVE_POINTS:
                self._gizmos[name].visible = False
                self._arrows[name].visible = False
                self._tips[name].visible = False
            self._recovery_mode = True
            self._recovery_started = False
            self.status = "选择衔接参考帧与插值时长, 然后播放"
        elif self._recovery_mode:
            if self._recovery_started:
                self.runtime.branches.append(self.runtime.records.copy())
                self.runtime.records = self._recovery_original_records.copy()
                self.runtime.seek_cached(self._recovery_origin_slot)
                self.runtime.sim.pipeline.reset_solvers(flags=0)
            self._recovery_mode = False
            self._recovery_started = False
            self._recovery_original_records = []
            for name in FIVE_POINTS:
                visible = self._repair_gizmos.get(name, False)
                self._gizmos[name].visible = visible
                self._arrows[name].visible = visible
                self._tips[name].visible = visible
            self.status = "已回到修复起点并恢复原缓存"
        self._drawn_step = None
        self._dirty = True

    def tick(self) -> bool:
        while self._requests:
            command, value = self._requests.popleft()
            try:
                if command == "toggle":
                    self._toggle()
                elif command == "show_reference":
                    self._show_reference = bool(value)
                    self._dirty = True
                elif command == "seek":
                    if self._recovery_mode:
                        raise RuntimeError("use the recovery timeline to select a reference frame")
                    slot = int(value)
                    self.set_paused(True)
                    self.runtime.seek_cached(slot)
                    self._staged.clear()
                    self._staged_hands.clear()
                    self._sync_goals_from_timeline()
                    self._drawn_step = None
                    self._dirty = True
                elif command == "reset":
                    if self._review_only:
                        self.set_paused(True)
                        self.runtime.seek_cached(0)
                        self._drawn_step = None
                        self._dirty = True
                        continue
                    if self._recovery_mode:
                        self._set_recovery_mode(False)
                    self.set_paused(True)
                    self.runtime.seek_reference(self.runtime.anchor_source_frame)
                    self._staged.clear()
                    self._staged_hands.clear()
                    self._sync_goals_from_timeline()
                    self._marks.clear()
                    self.status = ""
                    self._drawn_step = None
                    self._dirty = True
                elif command == "speed":
                    if self._recovery_mode:
                        raise RuntimeError("return to repair before changing link speed")
                    name, speed = value
                    if name not in FIVE_POINTS or not 0 <= float(speed) <= 0.6:
                        raise ValueError("invalid link speed")
                    speed = float(speed)
                    if abs(speed - self._speed[name]) > 1e-6:
                        self._speed[name] = speed
                        if self._gizmos[name].visible:
                            self._staged[name] = "move"
                            self.set_paused(True)
                elif command == "angular_speed":
                    if self._recovery_mode:
                        raise RuntimeError("return to repair before changing angular speed")
                    name, speed = value
                    if name not in FIVE_POINTS or not 0 <= float(speed) <= 3:
                        raise ValueError("invalid angular speed")
                    speed = float(speed)
                    if abs(speed - self._angular_speed[name]) > 1e-6:
                        self._angular_speed[name] = speed
                        if self._gizmos[name].visible:
                            self._staged[name] = "move"
                            self.set_paused(True)
                elif command == "hand":
                    if self._recovery_mode:
                        raise RuntimeError("return to repair before changing hand closure")
                    side, amount = value
                    if side not in ("left", "right") or not 0 <= float(amount) <= 1:
                        raise ValueError("invalid hand closure")
                    self._staged_hands[side] = float(amount)
                    self.set_paused(True)
                elif command == "link":
                    if self._recovery_mode:
                        raise RuntimeError("return to repair before editing links")
                    self._select_link(str(value[0]), bool(value[1]))
                elif command == "recovery_mode":
                    self._set_recovery_mode(bool(value))
                elif command == "recovery_frame":
                    if not self._recovery_mode or self._recovery_started:
                        raise RuntimeError("enter recovery and pause before selecting its endpoint")
                    self._recovery_frame = max(0, min(self.runtime.motion.frames - 1, int(value)))
                    self._dirty = True
                elif command == "recovery_duration":
                    if not self._recovery_mode or self._recovery_started:
                        raise RuntimeError("enter recovery and pause before changing interpolation duration")
                    frames = int(value)
                    if not 0 <= frames <= 500:
                        raise ValueError("interpolation duration must be 0-10 seconds")
                    self._recovery_duration_frames = frames
                    self._dirty = True
                elif command == "save":
                    path = self.runtime.save(self._out / "motion_repair")
                    self.status = f"saved candidate to {path}"
            except (RuntimeError, ValueError) as exc:
                self.set_paused(True)
                self.status = str(exc)
        cached = not self.paused and self.runtime.cursor < len(self.runtime.records)
        if cached:
            self.runtime.seek_cached(self.runtime.cursor + 1)
            self._drawn_step = None
            self.set_paused(True)
        stepped = super().tick()
        if cached:
            done = (
                self.runtime.cursor == len(self.runtime.records)
                and self.runtime.timeline.phase == "reference"
                and bool(self.runtime.records)
                and self.runtime.records[-1].source_frame >= self.runtime.motion.frames - 1
            )
            self.set_paused(done)
        if stepped:
            try:
                self.runtime.record_after_step()
            except RuntimeError as exc:
                self.set_paused(True)
                self.status = str(exc)
            if (
                self.runtime.timeline.phase == "reference"
                and self.runtime.records[-1].source_frame >= self.runtime.motion.frames - 1
            ):
                self.set_paused(True)
                if self._recovery_mode:
                    self.status = "恢复播放完成, 可保存候选供检查"
            if self._stop_frame is not None and self.runtime.timeline.source_frame >= self._stop_frame:
                self.set_paused(True)
                self.status = f"已到达源帧 {self._stop_frame}, 片段播放暂停"
        if stepped or cached or self._dirty:
            self._refresh_scene()
            self._dirty = False
        return stepped

    def close(self) -> None:
        if hasattr(self, "_http"):
            self._http.shutdown()
            self._http.server_close()
        super().close()
