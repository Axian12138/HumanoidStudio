"""GPU prediction must leave CPU execution untouched and replay reproducibly."""

import unittest

import numpy as np

TIER = "medium"


class OfflineDynamicsGPU(unittest.TestCase):
    def test_prediction_isolation_horizon_graph_and_control_replay(self):
        import mujoco

        from robosimstudio.humanoid_real2sim.refinement.warp_backend import WarpBackend

        model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".002"/>
          <worldbody><body name="arm"><joint name="hinge" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 .3" size=".03" mass="1"/>
          </body></worldbody><actuator><position joint="hinge" kp="10" kv="1"
          ctrlrange="-1 1" forcerange="-2 2"/></actuator></mujoco>""")
        q, v = model.qpos0.copy(), np.zeros(model.nv)
        target = np.full((6, model.nq), 0.2)
        backend = WarpBackend(
            model,
            q,
            v,
            target,
            np.ones(model.nq),
            [],
            samples=4,
            horizon=5,
            body_weights=np.ones((model.nbody, 2)),
            contact_geom_kind=np.zeros(model.ngeom, dtype=np.int32),
        )
        actions = np.full((4, 5, 1), 0.2)
        cost = backend.evaluate(actions, 0)
        self.assertTrue(np.isfinite(cost).all())
        np.testing.assert_array_equal(backend.execution.qpos, q)
        # Compare the nested full-horizon graph with five single-step graph launches.
        backend._reset()
        import warp as wp

        for t in range(5):
            backend.index.assign(np.array([t], np.int32))
            wp.capture_launch(backend.graph)
        np.testing.assert_allclose(backend.costs.numpy() / 5, cost, atol=1e-6)
        backend.commit(actions[0])
        recorded = np.array(backend.history_q)
        backend.reset_execution(q, v)
        backend.commit(actions[0, :2])
        backend.commit(actions[0, 2:])
        np.testing.assert_array_equal(backend.history_q, recorded)
