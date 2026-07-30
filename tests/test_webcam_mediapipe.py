import sys
import subprocess
import threading
import unittest
from pathlib import Path

import numpy as np
import mediapipe as mp


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"
sys.path.insert(0, str(EXAMPLE_DIR))

class WebcamMediaPipeTest(unittest.TestCase):
    def test_mediapipe_provides_legacy_hands_api(self):
        self.assertTrue(hasattr(mp, "solutions"))
        self.assertTrue(hasattr(mp.solutions, "hands"))

    def test_process_landmarks_centers_wrist_and_normalizes_palm_length(self):
        from input_devices.webcam_mediapipe import WebcamMediaPipe

        device = WebcamMediaPipe.__new__(WebcamMediaPipe)
        device._reference_wrist_to_mid_mcp = 0.09
        device.correct_segments = False

        keypoints = np.zeros((21, 3), dtype=np.float32)
        keypoints[0] = [10.0, 20.0, 30.0]
        keypoints[9] = [10.0, 120.0, 30.0]

        processed = device._process_landmarks(keypoints)

        np.testing.assert_allclose(processed[0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(processed[9]), 0.09)

    def test_simulation_help_does_not_require_visionpro_sdk(self):
        result = subprocess.run(
            [sys.executable, "example/teleop_sim.py", "--help"],
            cwd=EXAMPLE_DIR.parent,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--debug", result.stdout)
        self.assertIn("--robot", result.stdout)
        self.assertIn("openarm", result.stdout)
        self.assertNotIn("--tuning", result.stdout)
        self.assertNotIn("--viz-config", result.stdout)

class PreviewInterfaceTests(unittest.TestCase):
    def test_legacy_input_device_has_no_preview_by_default(self):
        from input_devices.base import InputDeviceBase

        class LegacyDevice(InputDeviceBase):
            def get_fingers_data(self):
                return {
                    "left_fingers": np.zeros((21, 3)),
                    "right_fingers": np.zeros((21, 3)),
                }

        self.assertIsNone(LegacyDevice().get_preview_frame())

    def test_webcam_preview_frame_is_returned_as_a_copy(self):
        from input_devices.webcam_mediapipe import WebcamMediaPipe

        device = WebcamMediaPipe.__new__(WebcamMediaPipe)
        device._lock = threading.Lock()
        device._latest_preview_frame = np.zeros((2, 2, 3), dtype=np.uint8)

        preview = device.get_preview_frame()
        preview[0, 0, 0] = 255

        self.assertEqual(device._latest_preview_frame[0, 0, 0], 0)


if __name__ == "__main__":
    unittest.main()
