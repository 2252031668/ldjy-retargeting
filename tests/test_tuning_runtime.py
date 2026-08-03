import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from ldjy_retargeting.tuning.runtime import TuningRuntime


ROOT = Path(__file__).resolve().parents[1]


class TuningRuntimeTests(unittest.TestCase):
    def test_persist_tip_offsets_saves_runtime_yaml_and_asset_source_together(self):
        from ldjy_retargeting.tuning.session import TuningSession
        from ldjy_retargeting.tuning.tip_assets import persist_tip_offsets

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "tuning.yaml"
            shutil.copy2(ROOT / "example/config/adaptive_analytical_video.yaml", config_path)
            session = TuningSession(config_path)
            session.set_value("tip_offsets.thumb.axis_mm", -9.6)
            session.set_value("tip_offsets.finger1.axis_mm", -10.0)
            asset_path = directory / "retarget_tip_offsets.yaml"

            offsets = persist_tip_offsets(session, asset_path)

            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            saved_asset = yaml.safe_load(asset_path.read_text(encoding="utf-8"))
            self.assertFalse(session.is_dirty)
            self.assertEqual(saved_config["tip_offsets"], offsets)
            self.assertEqual(
                saved_asset,
                {"version": 1, "units": "mm", "fingers": offsets},
            )

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
