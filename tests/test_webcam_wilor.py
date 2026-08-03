from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading
import time
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example"
sys.path.insert(0, str(EXAMPLE_DIR))


def detection(*, is_right: bool, area: float, value: float):
    side = float(np.sqrt(area))
    return SimpleNamespace(
        is_right=is_right,
        bbox_xyxy=np.array([0.0, 0.0, side, side], dtype=np.float32),
        joints_mano=np.full((21, 3), value, dtype=np.float32),
        vertices_mano=np.full((778, 3), value, dtype=np.float32),
        camera_translation=np.array([value, 0.0, 1.0], dtype=np.float32),
    )


class WebcamWiLoRTests(unittest.TestCase):
    def test_selects_the_largest_detection_for_requested_hand_side(self):
        from input_devices.webcam_wilor import WebcamWiLoR

        device = WebcamWiLoR.__new__(WebcamWiLoR)
        device.hand_side = "left"
        device._last_valid_keypoints = None

        keypoints = device._select_keypoints([
            detection(is_right=False, area=100.0, value=1.0),
            detection(is_right=False, area=400.0, value=2.0),
            detection(is_right=True, area=900.0, value=3.0),
        ])

        np.testing.assert_array_equal(keypoints, np.full((21, 3), 2.0, dtype=np.float32))

    def test_keeps_the_last_valid_keypoints_when_requested_hand_is_missing(self):
        from input_devices.webcam_wilor import WebcamWiLoR

        device = WebcamWiLoR.__new__(WebcamWiLoR)
        device.hand_side = "right"
        device._last_valid_keypoints = None
        initial = device._select_keypoints([detection(is_right=True, area=100.0, value=4.0)])
        held = device._select_keypoints([detection(is_right=False, area=500.0, value=9.0)])
        held[0, 0] = -1.0

        np.testing.assert_array_equal(initial, np.full((21, 3), 4.0, dtype=np.float32))
        self.assertEqual(device._last_valid_keypoints[0, 0], 4.0)

    def test_keeps_the_selected_mano_mesh_for_preview_after_a_short_loss(self):
        from input_devices.webcam_wilor import WebcamWiLoR

        device = WebcamWiLoR.__new__(WebcamWiLoR)
        device.hand_side = "right"
        device._last_valid_keypoints = None
        device._last_valid_mano = None

        device._select_keypoints([detection(is_right=True, area=100.0, value=4.0)])
        device._select_keypoints([detection(is_right=False, area=500.0, value=9.0)])

        self.assertIsNotNone(device._last_valid_mano)
        np.testing.assert_array_equal(
            device._last_valid_mano.vertices_mano,
            np.full((778, 3), 4.0, dtype=np.float32),
        )

    def test_overlay_payload_uses_the_selected_mano_mesh_and_raw_camera_frame(self):
        from input_devices.webcam_wilor import WebcamWiLoR

        device = WebcamWiLoR.__new__(WebcamWiLoR)
        device._lock = threading.Lock()
        device._last_valid_mano = detection(is_right=True, area=100.0, value=4.0)
        device._mano_faces = np.array([[0, 1, 2]], dtype=np.int32)
        device._latest_raw_frame = np.full((2, 3, 3), 7, dtype=np.uint8)
        device._frame_sequence = 12

        payload = device.get_mano_overlay_data()
        payload["frame_bgr"][0, 0] = 0

        self.assertEqual(payload["sequence"], 12)
        self.assertTrue(payload["is_right"])
        self.assertEqual(device._latest_raw_frame[0, 0, 0], 7)
        np.testing.assert_array_equal(
            payload["vertices_mano"], np.full((778, 3), 4.0, dtype=np.float32)
        )

    def test_pause_stops_new_wilor_inference_after_the_inflight_frame(self):
        from input_devices.webcam_wilor import WebcamWiLoR

        class Capture:
            def read(self):
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self):
                pass

        class Runner:
            def __init__(self):
                self.count = 0
                self.first_started = threading.Event()
                self.release_first = threading.Event()

            def infer(self, _frame):
                self.count += 1
                if self.count == 1:
                    self.first_started.set()
                    self.release_first.wait(timeout=1.0)
                return []

        device = WebcamWiLoR.__new__(WebcamWiLoR)
        device.hand_side = "right"
        device.show_video = False
        device.cap = Capture()
        device.runner = Runner()
        device._empty = np.zeros((21, 3), dtype=np.float32)
        device._last_valid_keypoints = None
        device._last_valid_mano = None
        device._lock = threading.Lock()
        device._inference_lock = threading.Lock()
        device._stop_event = threading.Event()
        device._resume_event = threading.Event()
        device._resume_event.set()
        device._latest_result = {
            "left_fingers": device._empty.copy(),
            "right_fingers": device._empty.copy(),
        }
        device._latest_preview_frame = None
        device._latest_raw_frame = None
        device._frame_sequence = 0
        device._fps_started_at = time.monotonic()
        device._inference_count = 0
        device._inference_fps = 0.0

        worker = threading.Thread(target=device._capture_loop)
        worker.start()
        try:
            self.assertTrue(device.runner.first_started.wait(timeout=1.0))
            pause_thread = threading.Thread(target=device.set_paused, args=(True,))
            pause_thread.start()
            time.sleep(0.02)
            self.assertTrue(pause_thread.is_alive())
            device.runner.release_first.set()
            pause_thread.join(timeout=1.0)
            self.assertFalse(pause_thread.is_alive())
            time.sleep(0.05)
            self.assertEqual(device.runner.count, 1)

            device.set_paused(False)
            deadline = time.monotonic() + 1.0
            while device.runner.count < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertGreaterEqual(device.runner.count, 2)
        finally:
            device.runner.release_first.set()
            device._stop_event.set()
            device._resume_event.set()
            worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())

    def test_simulation_help_lists_the_wilor_webcam_input(self):
        completed = subprocess.run(
            [sys.executable, "example/teleop_sim.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("webcam_wilor", completed.stdout)

    def test_cleanup_accepts_a_partially_constructed_device(self):
        from input_devices.webcam_wilor import WebcamWiLoR

        WebcamWiLoR.__new__(WebcamWiLoR).cleanup()
