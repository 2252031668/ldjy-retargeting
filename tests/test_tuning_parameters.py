import copy
import unittest

from ldjy_retargeting.tuning.parameters import (
    normalize_runtime_config,
    parameter_specs,
    set_path,
    validate_runtime_config,
)


class TuningParameterTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "optimizer": {"type": "AdaptiveOptimizerAnalytical"},
            "retarget": {"segment_scaling": {}, "pinch_thresholds": {}},
        }
        normalize_runtime_config(self.config)

    def test_schema_contains_all_fifteen_segment_scales(self):
        paths = {spec.path for spec in parameter_specs()}

        self.assertEqual(
            sum(path.startswith("retarget.segment_scaling.") for path in paths),
            15,
        )
        self.assertNotIn("retarget.scaling", paths)
        self.assertIn("retarget.segment_scaling.index.tip", paths)
        index_tip = next(spec for spec in parameter_specs()
                         if spec.path == "retarget.segment_scaling.index.tip")
        self.assertIn("wrist", index_tip.description_zh)

    def test_segment_scale_path_updates_its_yaml_list_slot(self):
        set_path(self.config, "retarget.segment_scaling.index.tip", 1.12)

        self.assertEqual(self.config["retarget"]["segment_scaling"]["index"], [1.0, 1.03, 1.12])

    def test_validate_rejects_pinch_threshold_order(self):
        invalid = copy.deepcopy(self.config)
        set_path(invalid, "retarget.pinch_thresholds.index.d1", 5.0)
        set_path(invalid, "retarget.pinch_thresholds.index.d2", 2.0)

        with self.assertRaisesRegex(ValueError, "d1.*d2"):
            validate_runtime_config(invalid)

    def test_mano_pad_config_gets_pinch_defaults(self):
        config = {
            "optimizer": {"type": "ManoPadPoseOptimizer"},
            "retarget": {"pad_ik": {}},
        }

        normalize_runtime_config(config)

        self.assertEqual(config["retarget"]["pad_ik"]["pinch_threshold_cm"], 2.0)
        self.assertEqual(config["retarget"]["pad_ik"]["pinch_min_distance_cm"], 0.1)
        self.assertEqual(config["retarget"]["pad_ik"]["pinch_distance_weight"], 4.0)
        self.assertFalse(config["retarget"]["pad_ik"]["thumb_contact_surface_optimization"])
        self.assertEqual(config["retarget"]["pad_ik"]["thumb_contact_surface_weight"], 300.0)
        self.assertEqual(config["retarget"]["pad_ik"]["pinch_position_weight_scale"], 0.25)


if __name__ == "__main__":
    unittest.main()
