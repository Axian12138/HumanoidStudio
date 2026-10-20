"""Reference targets sent to WholeBody-14 and VR-5."""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.humanoid_real2sim.track.timeline import FIVE_POINTS, ReferenceMotion, RepairTimeline

TIER = "fast"


def _motion(frames: int = 50) -> ReferenceMotion:
    position = np.zeros((frames, len(FIVE_POINTS), 3), dtype=np.float64)
    position[:, 0, 1] = np.arange(frames)  # an untouched reference keeps moving
    quaternion = np.zeros((frames, len(FIVE_POINTS), 4), dtype=np.float64)
    quaternion[..., 0] = 1.0
    return ReferenceMotion(
        joint_pos=np.zeros((frames, 1)),
        body_pos_w=position,
        body_quat_w=quaternion,
        joint_names=("joint",),
        body_names=FIVE_POINTS,
        fps=50,
    )


class TestTakeoverFutureTargets(unittest.TestCase):
    def setUp(self):
        self.motion = _motion()
        self.timeline = RepairTimeline(self.motion, max_edit_m=0.2)
        self.achieved = np.zeros_like(self.motion.body_pos_w[0])
        self.achieved_quat = self.motion.body_quat_w[0].copy()
        self.timeline.take_over(self.achieved, self.achieved_quat)

    def test_direction_is_present_in_the_first_future_reference(self):
        name = "right_wrist_yaw_link"
        body = self.motion.body_names.index(name)
        self.timeline.set_velocity(name, np.array([1.0, 0.0, 0.0]), 0.5)

        position, quaternion = self.timeline.targets((0, 1, 2, 3, 4))

        np.testing.assert_allclose(position[:, body, 0], [0.00, 0.01, 0.02, 0.03, 0.04])
        np.testing.assert_allclose(position[:, body, 1:], 0.0)
        np.testing.assert_array_equal(quaternion[:, body], np.broadcast_to(self.achieved_quat[body], (5, 4)))

    def test_the_future_ramp_advances_with_physical_time(self):
        name = "right_wrist_yaw_link"
        body = self.motion.body_names.index(name)
        self.timeline.set_velocity(name, np.array([1.0, 0.0, 0.0]), 0.5)
        self.timeline.advance()

        position, _ = self.timeline.targets((0, 1, 2, 3, 4))

        np.testing.assert_allclose(position[:, body, 0], [0.01, 0.02, 0.03, 0.04, 0.05])

    def test_hold_is_constant_and_an_untouched_link_keeps_its_reference(self):
        held_name = "left_wrist_yaw_link"
        held = self.motion.body_names.index(held_name)
        untouched = self.motion.body_names.index("pelvis")
        self.timeline.set_link_mode(held_name, "hold")

        position, _ = self.timeline.targets((0, 1, 2, 3, 4))

        np.testing.assert_allclose(position[:, held], np.broadcast_to(self.achieved[held], (5, 3)))
        np.testing.assert_allclose(position[:, untouched, 1], [0, 1, 2, 3, 4])

    def test_every_future_slot_respects_the_total_edit_budget(self):
        timeline = RepairTimeline(self.motion, max_edit_m=0.03)
        timeline.take_over(self.achieved, self.achieved_quat)
        name = "right_wrist_yaw_link"
        body = self.motion.body_names.index(name)
        timeline.set_velocity(name, np.array([1.0, 0.0, 0.0]), 0.5)

        position, _ = timeline.targets((0, 1, 2, 3, 4, 32))

        np.testing.assert_allclose(position[:, body, 0], [0.00, 0.01, 0.02, 0.03, 0.03, 0.03])

    def test_six_d_goal_uses_achieved_pose_and_stops_at_goal(self):
        from scipy.spatial.transform import Rotation

        name = "right_wrist_yaw_link"
        body = self.motion.body_names.index(name)
        goal_quat = Rotation.from_euler("z", 0.4).as_quat()[[3, 0, 1, 2]]
        self.timeline.set_goal(name, np.array([0.04, 0, 0]), goal_quat, 0.5, 2.0)
        pos, quat = self.timeline.targets((0, 1, 2, 3, 4, 32), self.achieved, self.achieved_quat)
        np.testing.assert_allclose(pos[:, body, 0], [0, 0.01, 0.02, 0.03, 0.04, 0.04])
        yaw = Rotation.from_quat(quat[:, body][:, [1, 2, 3, 0]]).as_euler("xyz")[:, 2]
        np.testing.assert_allclose(yaw, [0, 0.04, 0.08, 0.12, 0.16, 0.4], atol=1e-6)
        self.timeline.advance()
        # The physical link can lag completely for one step. The commanded
        # target must still progress instead of resetting to the observation.
        lagged_pos, lagged_quat = self.timeline.targets((0, 1, 2), self.achieved, self.achieved_quat)
        np.testing.assert_allclose(lagged_pos[:, body, 0], [0.01, 0.02, 0.03])
        lagged_yaw = Rotation.from_quat(lagged_quat[:, body][:, [1, 2, 3, 0]]).as_euler("xyz")[:, 2]
        np.testing.assert_allclose(lagged_yaw, [0.04, 0.08, 0.12], atol=1e-6)
        at_goal = self.achieved.copy()
        at_goal[body, 0] = 0.04
        at_goal_quat = self.achieved_quat.copy()
        at_goal_quat[body] = goal_quat
        stopped_pos, stopped_quat = self.timeline.targets((0, 1, 32), at_goal, at_goal_quat)
        np.testing.assert_allclose(stopped_pos[:, body], np.broadcast_to(at_goal[body], (3, 3)))
        np.testing.assert_allclose(stopped_quat[:, body], np.broadcast_to(goal_quat, (3, 4)))


if __name__ == "__main__":
    unittest.main()
