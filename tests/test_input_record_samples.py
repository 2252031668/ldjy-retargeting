"""Tests for live-device record sample publication."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example"))


class InputRecordSampleTests(unittest.TestCase):
    def test_base_queue_drains_in_completion_order(self):
        from input_devices.base import InferenceSample, InputDeviceBase

        class Device(InputDeviceBase):
            def get_fingers_data(self):
                return {}

        device = Device()
        device.publish_inference_sample(InferenceSample(
            timestamp_sec=0.0, frame_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            input_type="webcam", hand_side="right", detected=False, payload={},
        ))
        device.publish_inference_sample(InferenceSample(
            timestamp_sec=0.1, frame_bgr=np.ones((2, 2, 3), dtype=np.uint8),
            input_type="webcam", hand_side="right", detected=True, payload={},
        ))
        samples = device.drain_inference_samples()
        self.assertEqual([sample.timestamp_sec for sample in samples], [0.0, 0.1])
        self.assertEqual(device.drain_inference_samples(), [])

    def test_wilor_selected_detection_becomes_a_record_sample(self):
        from input_devices.webcam_wilor import WebcamWiLoR

        class Detection:
            is_right = True
            bbox_xyxy = np.array((0, 0, 4, 4), dtype=np.float32)
            joints_mano = np.zeros((21, 3), dtype=np.float32)
            vertices_mano = np.zeros((778, 3), dtype=np.float32)
            camera_translation = np.zeros(3, dtype=np.float32)
            global_orient = np.eye(3, dtype=np.float32)
            hand_pose = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))
            betas = np.zeros(10, dtype=np.float32)
            pred_cam = np.zeros(3, dtype=np.float32)

        device = WebcamWiLoR.__new__(WebcamWiLoR)
        device.hand_side = "right"
        device._last_valid_keypoints = None
        device._last_valid_mano = None
        selected = device._select_keypoints([Detection()])
        sample = device._record_sample_from_detection(
            np.zeros((2, 2, 3), dtype=np.uint8), Detection(), selected, 0.25
        )
        self.assertTrue(sample.detected)
        self.assertTrue(sample.payload["detection"].is_right)
        self.assertEqual(sample.input_type, "webcam_wilor")

    def test_wilor_held_pose_is_recorded_as_a_detection_loss(self):
        from input_devices.webcam_wilor import WebcamWiLoR

        device = WebcamWiLoR.__new__(WebcamWiLoR)
        device.hand_side = "right"
        device._last_valid_keypoints = np.ones((21, 3), dtype=np.float32)
        device._last_selected_detection = object()
        device._selected_detection_this_inference = object()
        held = device._select_keypoints([])
        sample = device._record_sample_from_detection(
            np.zeros((2, 2, 3), dtype=np.uint8),
            device._selected_detection_this_inference, held, 0.5,
        )
        self.assertFalse(sample.detected)
        self.assertIsNone(sample.payload["detection"])
        self.assertIsNone(sample.payload["joints_mano"])

    def test_mediapipe_extracts_normalized_detector_landmarks_for_recording(self):
        from input_devices.webcam_mediapipe import WebcamMediaPipe

        class Landmark:
            x, y, z = 0.25, 0.5, -0.1

        class HandLandmarks:
            landmark = [Landmark() for _ in range(21)]

        device = WebcamMediaPipe.__new__(WebcamMediaPipe)
        device.frame_width = 640
        device.frame_height = 480
        device.z_scale = 2.5
        keypoints, image_points, raw = device._extract_hand_landmarks(HandLandmarks())
        self.assertEqual(keypoints.shape, (21, 3))
        self.assertEqual(len(image_points), 21)
        np.testing.assert_allclose(raw[0], [0.25, 0.5, -0.1])
