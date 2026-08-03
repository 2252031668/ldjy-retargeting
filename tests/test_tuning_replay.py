"""Replay uses saved inference outputs and current preprocessing settings."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example"))


class TuningReplayTests(unittest.TestCase):
    def test_cursor_uses_recorded_uneven_timestamps_without_losing_short_ticks(self):
        from ldjy_retargeting.tuning.replay import ReplayCursor

        cursor = ReplayCursor(np.array([0.0, 0.04, 0.16]))
        cursor.play()
        self.assertEqual(cursor.advance(0.02), 0)
        self.assertEqual(cursor.advance(0.03), 1)
        self.assertEqual(cursor.advance(0.12), 2)

    def test_mediapipe_replay_reapplies_current_z_scale(self):
        from ldjy_retargeting.tuning.recording import MediaPipeRecordWriter, RecordSample
        from ldjy_retargeting.tuning.replay import MediaPipeReplay

        raw = np.zeros((21, 3), dtype=np.float32)
        raw[9] = (0.1, 0.1, 0.1)
        raw[8] = (0.2, 0.1, 0.2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = MediaPipeRecordWriter.start(temporary_directory, hand_side="right", width=640, height=480)
            writer.append(RecordSample(
                timestamp_sec=12.0, frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
                detected=True, detector_landmarks=raw,
                processed_landmarks=np.zeros((21, 3), dtype=np.float32),
            ))
            record = writer.finish(config={})
            replay = MediaPipeReplay(record.path)
            flat = replay.input_at(0, {"z_scale": 1.0, "correct_segments": False})
            deep = replay.input_at(0, {"z_scale": 3.0, "correct_segments": False})
            replay.close()
        self.assertFalse(np.allclose(flat, deep))

    def test_wilor_replay_reads_saved_arrays_without_a_runner(self):
        from ldjy_retargeting.tuning.recording import WiLoRRecordSample, WiLoRRecordWriter
        from ldjy_retargeting.tuning.replay import WiLoRReplay

        class Detection:
            bbox_xyxy = np.zeros(4, dtype=np.float32)
            joints_mano = np.ones((21, 3), dtype=np.float32)
            vertices_mano = np.ones((778, 3), dtype=np.float32)
            global_orient = np.eye(3, dtype=np.float32)
            hand_pose = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))
            betas = np.zeros(10, dtype=np.float32)
            pred_cam = np.zeros(3, dtype=np.float32)
            camera_translation = np.zeros(3, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = WiLoRRecordWriter.start(temporary_directory, hand_side="left", width=8, height=6, faces=np.array([[0, 1, 2]]))
            writer.append(WiLoRRecordSample(0.0, np.zeros((6, 8, 3), dtype=np.uint8), True, Detection()))
            record = writer.finish(config={})
            replay = WiLoRReplay(record.path)
            np.testing.assert_allclose(replay.input_at(0), 1.0)
            self.assertEqual(replay.mano_overlay_at(0)["vertices_mano"].shape, (778, 3))
            replay.close()
