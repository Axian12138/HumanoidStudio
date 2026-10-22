"""Multi-object review must reject plausible-looking but wrong source windows."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robosimstudio.humanoid_real2sim.visualization.tracks import (
    SceneTrack,
    bind_scene_tracks,
    load_scene_tracks,
)

TIER = "fast"


def write_tracks(tmp_path, *, offset=27.0, quaternion=(0, 0, 0, 1)):
    clock = {"fps": 50.0, "source_time_s": offset + np.arange(3) / 50}
    np.savez(tmp_path / "berry.npz", position_w=np.zeros((3, 3)), quaternion_xyzw=np.tile(quaternion, (3, 1)), **clock)
    np.savez(tmp_path / "drawer.npz", position=np.array([0.3, 0.2, 0.0]), **clock)
    manifest = tmp_path / "tracks.json"
    manifest.write_text(
        json.dumps(
            {
                "objects": [{"name": "berry", "path": "berry.npz"}],
                "joints": [{"name": "drawer", "path": "drawer.npz"}],
            }
        )
    )
    return manifest


class TestSceneTracks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tmp_path = Path(self.directory.name)

    def test_multiple_tracks_have_named_coordinate_bindings(self):
        tmp_path = self.tmp_path
        tracks = load_scene_tracks(write_tracks(tmp_path), frames=3, fps=50, source_offset_s=27)
        bindings = bind_scene_tracks(
            tracks,
            ["floating_base", "elbow", "berry_free_joint", "drawer"],
            np.array([0, 7, 8, 15, 16]),
            0,
            [7],
        )
        q = np.zeros(16)
        for track, coord in bindings:
            q[coord : coord + track.values.shape[1]] = track.values[1]
        assert q[14] == 1
        self.assertAlmostEqual(q[15], 0.2)

    def test_wrong_source_window_is_rejected_even_with_matching_length(self):
        tmp_path = self.tmp_path
        with self.assertRaisesRegex(ValueError, "source timestamps"):
            load_scene_tracks(write_tracks(tmp_path, offset=28), frames=3, fps=50, source_offset_s=27)

    def test_nonunit_quaternion_is_rejected(self):
        tmp_path = self.tmp_path
        with self.assertRaisesRegex(ValueError, "unit norm"):
            load_scene_tracks(write_tracks(tmp_path, quaternion=(0, 0, 0, 2)), frames=3, fps=50, source_offset_s=27)

    def test_scalar_track_cannot_overwrite_robot_joint(self):
        tmp_path = self.tmp_path
        track = SceneTrack("elbow", "scalar", np.zeros((3, 1)), tmp_path)
        with self.assertRaisesRegex(ValueError, "overlaps robot"):
            bind_scene_tracks([track], ["floating_base", "elbow"], [0, 7, 8], 0, [7])

    def test_free_track_cannot_overwrite_scalar_joint(self):
        tmp_path = self.tmp_path
        track = SceneTrack("drawer", "free", np.zeros((3, 7)), tmp_path)
        with self.assertRaisesRegex(ValueError, "coordinate width"):
            bind_scene_tracks([track], ["floating_base", "drawer"], [0, 7, 8], 0, [])

    def test_seek_sets_all_tracks_in_both_states_without_physics(self):
        from viser import CameraHandle

        from robosimstudio.humanoid_real2sim.visualization.kinematic import KinematicStudio

        # Use the real setter: assigning position also translates look_at.
        camera = CameraHandle(SimpleNamespace(_websock_connection=SimpleNamespace(queue_message=lambda _: None)))
        camera._state.update_timestamp = 1.0
        camera._state.look_at = np.array([0.0, 0.0, 1.0])
        camera._state.up_direction = np.array([0.0, 1.0, 0.0])
        camera.position = (2.0, -3.0, 2.0)
        camera.look_at = (0.0, 0.0, 1.0)
        original_eye = camera.position.copy()
        original_look = camera.look_at.copy()
        roots = np.array([[0.0, 0.0, 0.0], [0.8, -0.4, 0.2], [-1.0, 2.0, 0.5]])

        class Buffer:
            def assign(self, value):
                self.value = value.copy()

        states = [SimpleNamespace(joint_q=Buffer(), joint_qd=Buffer()) for _ in range(2)]
        tracks = load_scene_tracks(write_tracks(self.tmp_path), frames=3, fps=50, source_offset_s=27)
        pipeline = SimpleNamespace(state_0=states[0], state_1=states[1], step_count=0, mark_dirty=lambda: None)
        studio = SimpleNamespace(
            start=0,
            stop=3,
            _follow_root=True,
            _last_camera_root=roots[0].copy(),
            server=SimpleNamespace(get_clients=lambda: {0: SimpleNamespace(camera=camera)}),
            _base_q=np.zeros(16),
            _root=0,
            _joint_coords=np.array([7]),
            _track=None,
            _scene_tracks=[(tracks[0], 8), (tracks[1], 15)],
            _zero_qd=np.zeros(14),
            env=SimpleNamespace(pipeline=pipeline, model=object()),
            motion=SimpleNamespace(
                body_pos_w=roots[:, None, :],
                body_quat_w=np.tile([1, 0, 0, 0], (3, 1, 1)),
                all_joint_pos=np.zeros((3, 1)),
            ),
            _human_joints=[],
            _human_positions=np.zeros((3, 2, 3)),
            _human_edges=np.array([[0, 1]]),
            _human_lines=SimpleNamespace(),
            _rigid=None,
        )
        with patch("newton.eval_fk") as fk:
            for frame in (2, 0, 1):
                KinematicStudio._set_frame(studio, frame)
                self.assertEqual(studio.cursor, frame)
                np.testing.assert_allclose(camera.position, original_eye + roots[frame])
                np.testing.assert_allclose(camera.look_at, original_look + roots[frame])
                for state in states:
                    np.testing.assert_array_equal(state.joint_q.value[8:15], tracks[0].values[frame])
                    self.assertAlmostEqual(state.joint_q.value[15], tracks[1].values[frame, 0])
            self.assertEqual(fk.call_count, 6)
        self.assertEqual(pipeline.step_count, 0)

    def test_wall_clock_playback_rate_preserves_source_fps_and_stops_at_endpoint(self):
        from collections import deque
        from unittest.mock import Mock

        from robosimstudio.humanoid_real2sim.visualization.kinematic import KinematicStudio

        for rate, expected in ((1.0, 50), (0.5, 25), (0.25, 12), (0.1, 5)):
            studio = object.__new__(KinematicStudio)
            studio.motion = SimpleNamespace(fps=50)
            studio.start, studio.stop, studio.cursor = 0, 100, 0
            studio.playing, studio.playback_rate = False, 1.0
            studio._requests = deque([("rate", rate), ("toggle", None)])
            studio._set_frame = lambda frame, target=studio: setattr(target, "cursor", frame)
            studio.set_paused = Mock()
            with (
                patch("robosimstudio.humanoid_real2sim.visualization.kinematic.time.monotonic", return_value=0),
                patch("robosimstudio_studio.StudioSession.tick", return_value=True),
            ):
                studio.tick()
            with (
                patch("robosimstudio.humanoid_real2sim.visualization.kinematic.time.monotonic", return_value=1),
                patch("robosimstudio_studio.StudioSession.tick", return_value=True),
            ):
                studio.tick()
            self.assertEqual(studio.cursor, expected)
            self.assertEqual(studio.motion.fps, 50)
            self.assertTrue(studio.playing)
            studio.set_paused.assert_called_with(True)
            with (
                patch("robosimstudio.humanoid_real2sim.visualization.kinematic.time.monotonic", return_value=100),
                patch("robosimstudio_studio.StudioSession.tick", return_value=True),
            ):
                studio.tick()
            self.assertEqual(studio.cursor, 99)
            self.assertFalse(studio.playing)

    def test_rate_change_during_play_and_seek_reset_preserve_rate(self):
        from collections import deque
        from unittest.mock import Mock

        from robosimstudio.humanoid_real2sim.visualization.kinematic import KinematicStudio

        studio = object.__new__(KinematicStudio)
        studio.motion = SimpleNamespace(fps=50)
        studio.start, studio.stop, studio.cursor = 10, 1000, 100
        studio.playing, studio.playback_rate = True, 1.0
        studio._next_frame_time = 0
        studio._requests = deque([("rate", 0.1)])
        studio._set_frame = lambda frame: setattr(studio, "cursor", frame)
        studio.set_paused = Mock()
        with (
            patch("robosimstudio.humanoid_real2sim.visualization.kinematic.time.monotonic", return_value=10),
            patch("robosimstudio_studio.StudioSession.tick", return_value=True),
        ):
            studio.tick()
            self.assertEqual(studio.cursor, 100)
            self.assertTrue(studio.playing)
            self.assertAlmostEqual(studio._next_frame_time, 10.2)
            studio._requests.append(("seek", 250))
            studio.tick()
            self.assertEqual(studio.cursor, 250)
            self.assertFalse(studio.playing)
            self.assertEqual(studio.playback_rate, 0.1)
            studio._requests.extend([("toggle", None), ("reset", None)])
            studio.tick()
            self.assertEqual(studio.cursor, 10)
            self.assertFalse(studio.playing)
            self.assertEqual(studio.playback_rate, 0.1)
        studio.set_paused.assert_called_with(True)
