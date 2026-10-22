"""Pure sampling contracts; no GPU or task assets required."""

import unittest

import numpy as np

from robosimstudio.humanoid_real2sim.refinement.sampling import Config, optimize, weighted_update

TIER = "fast"


class SamplingTest(unittest.TestCase):
    def test_nonfinite_candidates_and_large_cost_offsets(self):
        actions = np.array([[[1.0]], [[3.0]], [[100.0]]])
        result = weighted_update(actions, np.array([1e9, 1e9, np.inf]), 0.05)
        np.testing.assert_allclose(result, [[2]])
        with self.assertRaises(RuntimeError):
            weighted_update(actions, np.full(3, np.inf), 0.05)

    def test_execution_is_continuous_and_tail_is_not_dropped(self):
        class Integrator:
            lower, upper = np.array([-1.0]), np.array([1.0])
            state = 0.0

            def __init__(self):
                self.executed = []

            def evaluate(self, actions, start):
                before = self.state
                states = before + np.cumsum(actions[:, :, 0], axis=1)
                return ((states - 0.5) ** 2).mean(axis=1)

            def commit(self, actions):
                self.executed.extend(actions[:, 0].tolist())
                self.state += actions[:, 0].sum()

        backend = Integrator()
        cfg = Config(horizon=5, advance=3, samples=16, iterations=3, seed=4)
        result, records = optimize(backend, np.zeros((11, 1)), cfg)
        self.assertEqual(len(backend.executed), 11)
        np.testing.assert_allclose(backend.executed, result[:, 0])
        self.assertEqual(records[-1]["frame"], 11)
        self.assertLess(abs(backend.state - 0.5), 0.5)
        self.assertTrue(np.all(abs(result) <= 1))

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            Config(horizon=4, advance=5)


class ExportTest(unittest.TestCase):
    def test_export_uses_executed_velocity_and_rejects_ambiguous_names(self):
        import tempfile
        from pathlib import Path

        import mujoco

        from robosimstudio.humanoid_real2sim.refinement.export import export_motion, unique_suffix

        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="robot_root">
          <freejoint name="free"/><geom size=".1" mass="1"/>
          <body name="robot_tip" pos="0 0 .2"><joint name="robot_hinge"/>
          <geom size=".05" mass=".1"/></body></body></worldbody></mujoco>""")
        q = np.tile(model.qpos0, (2, 1))
        q[:, 0] = [1, 2]
        v = np.zeros((2, model.nv))
        v[:, 0] = [3, 4]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "motion.npz"
            result = export_motion(model, q, v, ["hinge"], ["root", "tip"], path)
            np.testing.assert_allclose(result.body_pos_w[:, 0, 0], [1, 2])
            np.testing.assert_allclose(result.body_lin_vel_w[:, 0, 0], [3, 4])
            self.assertTrue(path.is_file())
        with self.assertRaises(ValueError):
            unique_suffix(model, "body", "absent")


class InitialStateTest(unittest.TestCase):
    def test_projection_moves_only_selected_joint(self):
        import mujoco

        from robosimstudio.humanoid_real2sim.refinement.initial_state import project_initial

        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
        <geom type="plane" pos="0 0 .5" size="2 2 .1"/>
        <body name="finger" pos="0 0 .48"><joint name="height" type="slide" axis="0 0 1" range="-.2 .2"/>
        <geom type="sphere" size=".02" mass="1"/></body>
        <body name="object" pos="1 0 1"><freejoint/><geom size=".1" mass="1"/></body>
        </worldbody></mujoco>""")
        q = model.qpos0.copy()
        jid = model.joint("height").id
        fixed, report = project_initial(model, q, [jid], max_change=0.1)
        self.assertLess(report["max_penetration_after_m"], 1e-4)
        self.assertGreater(report["max_penetration_before_m"], 0.03)
        unchanged = np.ones(model.nq, dtype=bool)
        unchanged[model.jnt_qposadr[jid]] = False
        np.testing.assert_array_equal(fixed[unchanged], q[unchanged])
        with self.assertRaises(ValueError):
            project_initial(model, q, [int(np.flatnonzero(model.jnt_type == 0)[0])])
