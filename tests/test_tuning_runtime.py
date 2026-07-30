import unittest

import numpy as np

from ldjy_retargeting.tuning.runtime import TuningRuntime


class TuningRuntimeTests(unittest.TestCase):
    def test_apply_config_replaces_retargeter_and_resets_state(self):
        config = {
            "optimizer": {"type": "AdaptiveOptimizerAnalytical"},
            "retarget": {"segment_scaling": {"index": [1.0, 1.0, 1.0]}},
        }
        runtime = TuningRuntime(config, hand_side="right")
        previous = runtime.retargeter
        runtime.retargeter.optimizer.last_qpos = np.ones(20)
        changed = runtime.config
        changed["retarget"]["segment_scaling"]["index"][2] = 1.1

        runtime.apply_config(changed)

        self.assertIsNot(runtime.retargeter, previous)
        self.assertIsNone(runtime.retargeter.optimizer.last_qpos)
        self.assertEqual(runtime.config["retarget"]["segment_scaling"]["index"][2], 1.1)

    def test_tip_offsets_materialize_a_matching_cached_asset(self):
        config = {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}, "retarget": {}}
        runtime = TuningRuntime(config, hand_side="right")
        default_tip = runtime.zero_pose_robot_vector_lengths()[2]
        changed = runtime.config
        changed["tip_offsets"]["finger2"]["axis_mm"] = 5.0

        runtime.apply_config(changed)

        self.assertIn(".cache/tip_tuning", str(runtime.current_urdf_path))
        self.assertTrue(runtime.debug_mjcf_path.is_file())
        self.assertAlmostEqual(
            float(np.linalg.norm(runtime.zero_pose_robot_vector_lengths()[2] - default_tip)),
            0.005,
            places=5,
        )

    def test_tip_preview_materializes_mjcf_without_replacing_optimizer(self):
        runtime = TuningRuntime(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}, "retarget": {}},
            hand_side="right",
        )
        original_retargeter = runtime.retargeter
        preview = runtime.config
        preview["tip_offsets"]["thumb"]["surface_mm"] = 2.0

        mjcf_path = runtime.preview_tip_offsets(preview)

        self.assertTrue(mjcf_path.is_file())
        self.assertIs(runtime.retargeter, original_retargeter)


if __name__ == "__main__":
    unittest.main()
