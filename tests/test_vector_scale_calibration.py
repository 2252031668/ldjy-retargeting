import unittest
import tempfile
from pathlib import Path

import numpy as np

from ldjy_retargeting.tuning.vector_scale_calibration import (
    CalibrationError,
    compute_scales,
    human_vector_lengths,
)
from ldjy_retargeting.tuning.runtime import TuningRuntime
from ldjy_retargeting.tuning.session import TuningSession


class VectorScaleCalibrationTests(unittest.TestCase):
    def test_human_lengths_follow_thumb_to_pinky_pip_dip_tip_order(self):
        keypoints = np.zeros((21, 3), dtype=np.float64)
        indices = (
            (2, 3, 4),
            (6, 7, 8),
            (10, 11, 12),
            (14, 15, 16),
            (18, 19, 20),
        )
        for finger, triplet in enumerate(indices, start=1):
            for segment, index in enumerate(triplet, start=1):
                keypoints[index] = [finger * 0.01, segment * 0.01, 0.0]

        lengths = human_vector_lengths(keypoints)

        self.assertEqual(lengths.shape, (5, 3))
        self.assertAlmostEqual(lengths[0, 0], np.hypot(0.01, 0.01))
        self.assertAlmostEqual(lengths[4, 2], np.hypot(0.05, 0.03))

    def test_scales_use_median_of_samples(self):
        samples = np.stack(
            [
                np.full((5, 3), 0.10),
                np.full((5, 3), 0.11),
                np.full((5, 3), 0.12),
            ]
        )

        scales = compute_scales(np.full((5, 3), 0.11), samples)

        np.testing.assert_allclose(scales, np.ones((5, 3)))

    def test_scales_reject_entire_result_when_any_value_is_out_of_range(self):
        samples = np.full((3, 5, 3), 0.10)
        robot_lengths = np.full((5, 3), 0.10)
        robot_lengths[3, 1] = 0.16

        with self.assertRaisesRegex(CalibrationError, "outside"):
            compute_scales(robot_lengths, samples)

    def test_human_lengths_reject_zero_or_nonfinite_vectors(self):
        with self.assertRaisesRegex(CalibrationError, "human"):
            human_vector_lengths(np.zeros((21, 3)))

        keypoints = np.zeros((21, 3))
        keypoints[2] = [np.nan, 0.01, 0.0]
        with self.assertRaisesRegex(CalibrationError, "human"):
            human_vector_lengths(keypoints)

    def test_session_applies_all_fifteen_scales_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("retarget:\n  scaling: 1.0\n", encoding="utf-8")
            session = TuningSession(config_path)

            session.set_segment_scalings(np.full((5, 3), 1.12))

            self.assertTrue(session.is_dirty)
            self.assertEqual(
                session.config["retarget"]["segment_scaling"]["thumb"],
                [1.12, 1.12, 1.12],
            )
            self.assertEqual(
                session.config["retarget"]["segment_scaling"]["pinky"],
                [1.12, 1.12, 1.12],
            )

    def test_session_does_not_partially_apply_invalid_scales(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("retarget:\n  scaling: 1.0\n", encoding="utf-8")
            session = TuningSession(config_path)
            before = session.config
            invalid = np.ones((5, 3))
            invalid[3, 1] = 1.6

            with self.assertRaisesRegex(ValueError, "range"):
                session.set_segment_scalings(invalid)

            self.assertEqual(session.config, before)

    def test_runtime_exposes_positive_zero_pose_robot_lengths(self):
        runtime = TuningRuntime(
            {
                "optimizer": {"type": "AdaptiveOptimizerAnalytical"},
                "retarget": {"scaling": 1.0},
            },
            hand_side="right",
        )

        lengths = runtime.zero_pose_robot_vector_lengths()

        self.assertEqual(lengths.shape, (5, 3))
        self.assertTrue(np.all(np.isfinite(lengths)))
        self.assertTrue(np.all(lengths > 0))


if __name__ == "__main__":
    unittest.main()
