from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "urdf" / "ldjy_right_hand.urdf"


def central_difference(function, qpos, epsilon=1e-6):
    columns = []
    for index in range(len(qpos)):
        positive, negative = qpos.copy(), qpos.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        columns.append((function(positive) - function(negative)) / (2.0 * epsilon))
    return np.column_stack(columns)


class RobotSurfaceJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ldjy_retargeting.robot import RobotWrapper

        cls.robot = RobotWrapper(str(URDF), hand_side="right")
        cls.frame_id = cls.robot.get_link_index("finger1_tip")
        cls.qpos = np.zeros(cls.robot.model.nq)
        cls.qpos[4:8] = [0.18, 0.31, 0.22, 0.13]

    def test_anchor_position_jacobian_matches_central_difference(self):
        local_point = np.array([0.002, -0.001, 0.003])
        _, _, position_jacobian, _ = self.robot.compute_anchor_jacobians(
            self.qpos, self.frame_id, local_point, np.array([0.0, 0.0, 1.0])
        )
        numeric = central_difference(
            lambda q: self.robot.anchor_pose(q, self.frame_id, local_point, [0.0, 0.0, 1.0])[0],
            self.qpos,
        )
        np.testing.assert_allclose(position_jacobian, numeric, atol=2e-5, rtol=2e-3)

    def test_anchor_normal_jacobian_matches_central_difference(self):
        local_normal = np.array([0.0, 0.0, 1.0])
        _, _, _, normal_jacobian = self.robot.compute_anchor_jacobians(
            self.qpos, self.frame_id, np.zeros(3), local_normal
        )
        numeric = central_difference(
            lambda q: self.robot.anchor_pose(q, self.frame_id, np.zeros(3), local_normal)[1], self.qpos
        )
        np.testing.assert_allclose(normal_jacobian, numeric, atol=2e-5, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
