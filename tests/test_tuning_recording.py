"""Persistent tuning-record schema tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TuningRecordingTests(unittest.TestCase):
    def test_mediapipe_writer_keeps_video_and_npz_sample_order(self):
        from ldjy_retargeting.tuning.recording import (
            MediaPipeRecordWriter,
            RecordSample,
            load_mediapipe_record,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            writer = MediaPipeRecordWriter.start(
                root, hand_side="right", width=8, height=6
            )
            for timestamp, marker, detected in ((0.0, 1.0, True), (0.043, 2.0, False)):
                landmarks = np.full((21, 3), marker, dtype=np.float32)
                writer.append(RecordSample(
                    timestamp_sec=timestamp,
                    frame_bgr=np.full((6, 8, 3), int(marker), dtype=np.uint8),
                    detected=detected,
                    detector_landmarks=landmarks if detected else None,
                    processed_landmarks=landmarks,
                ))
            info = writer.finish(config={"retarget": {}})
            record = load_mediapipe_record(info.path)

            self.assertEqual(record.timestamp_sec.tolist(), [0.0, 0.043])
            self.assertEqual(float(record.detector_landmarks[0, 0, 0]), 1.0)
            self.assertFalse(bool(record.detected[1]))
            self.assertEqual(record.frame_count, 2)

    def test_list_records_filters_type_and_excludes_incomplete_directories(self):
        from ldjy_retargeting.tuning.recording import (
            MediaPipeRecordWriter,
            RecordSample,
            list_records,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            writer = MediaPipeRecordWriter.start(root, hand_side="left", width=8, height=6)
            writer.append(RecordSample(
                timestamp_sec=0.0,
                frame_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
                detected=False,
                detector_landmarks=None,
                processed_landmarks=np.zeros((21, 3), dtype=np.float32),
            ))
            complete = writer.finish(config={})
            (root / "wilor" / ".unfinished.tmp").mkdir(parents=True)

            records = list_records(root, "webcam")
            self.assertEqual([record.path for record in records], [complete.path])
            self.assertEqual(list_records(root, "webcam_wilor"), [])

    def test_wilor_writer_round_trips_the_selected_mano_result(self):
        from ldjy_retargeting.tuning.recording import (
            WiLoRRecordSample,
            WiLoRRecordWriter,
            load_wilor_record,
        )

        class Detection:
            bbox_xyxy = np.zeros(4, dtype=np.float32)
            joints_mano = np.full((21, 3), 1.0, dtype=np.float32)
            vertices_mano = np.full((778, 3), 2.0, dtype=np.float32)
            global_orient = np.eye(3, dtype=np.float32)
            hand_pose = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))
            betas = np.zeros(10, dtype=np.float32)
            pred_cam = np.ones(3, dtype=np.float32)
            camera_translation = np.ones(3, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = WiLoRRecordWriter.start(
                temporary_directory, hand_side="right", width=8, height=6,
                faces=np.array([[0, 1, 2]], dtype=np.int32),
            )
            writer.append(WiLoRRecordSample(
                timestamp_sec=0.2, frame_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
                detected=True, detection=Detection(),
            ))
            info = writer.finish(config={})
            record = load_wilor_record(info.path)
            self.assertTrue(record.detected[0])
            self.assertEqual(record.arrays["vertices_mano"].shape, (1, 778, 3))
            np.testing.assert_allclose(record.arrays["joints_mano"][0], 1.0)
