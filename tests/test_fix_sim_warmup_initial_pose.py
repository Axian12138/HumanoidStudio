"""A free embodiment without a reset writer must survive physics warm-up."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

TIER = "fast"


class Array:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values

    def assign(self, values):
        self.values = np.array(values, dtype=np.float32, copy=True)

    def zero_(self):
        self.values.fill(0)


class TestWarmupInitialPose(unittest.TestCase):
    def test_reset_restores_uncontrolled_free_joint_and_episode_overrides(self):
        from robosimstudio.sim.sim import Sim

        def state():
            return SimpleNamespace(
                joint_q=Array([0, 0, 1, 0, 0, 0, 1]),
                joint_qd=Array(np.zeros(6)),
                body_q=Array(np.zeros((1, 7))),
                body_qd=Array(np.zeros((1, 6))),
            )

        initial, alternate = state(), state()
        sim = SimpleNamespace(
            _start_joint_q=None,
            model=SimpleNamespace(body_count=1),
            pipeline=SimpleNamespace(
                state_0=initial,
                state_1=alternate,
                reset_solvers=lambda **_: None,
                mark_dirty=lambda: None,
            ),
            scene=SimpleNamespace(entities=[]),
            episode=None,
            _park_inactive=lambda: None,
        )
        with (
            patch("newton.eval_fk"),
            patch("warp.copy", side_effect=lambda target, source: target.assign(source.numpy())),
        ):
            Sim.reset_scene(sim)
            # Simulate warm-up advancing an unactuated free body in both buffers.
            initial.joint_q.values[2] = 0.99
            alternate.joint_q.values[2] = 0.98
            Sim.reset_scene(sim)
            np.testing.assert_array_equal(initial.joint_q.numpy(), [0, 0, 1, 0, 0, 0, 1])
            np.testing.assert_array_equal(alternate.joint_q.numpy(), initial.joint_q.numpy())

            # A later episode may still place coordinates after restoring the baseline.
            def place(pipeline, episode):
                pipeline.state_0.joint_q.values[0] = episode

            sim.scene.entities = [SimpleNamespace(reset=place)]
            sim.episode = 2
            Sim.reset_scene(sim)
            self.assertEqual(initial.joint_q.numpy()[0], 2)
            self.assertEqual(alternate.joint_q.numpy()[0], 2)
            self.assertEqual(sim._start_joint_q[0], 0)


if __name__ == "__main__":
    unittest.main()
