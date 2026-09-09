"""Quest HTS input conversion and record-sample contracts."""

from __future__ import annotations

import sys
import socket
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example"))


class QuestHTSTests(unittest.TestCase):
    def test_stop_releases_the_udp_port_for_reapply(self):
        from input_devices.quest_hts import QuestHTS

        reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        reservation.close()
        device = QuestHTS(host="127.0.0.1", port=port)
        try:
            time.sleep(0.02)
            device.stop()
            replacement = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                replacement.bind(("127.0.0.1", port))
            finally:
                replacement.close()
        finally:
            device.stop()

    def test_stop_releases_the_tcp_server_port_for_reapply(self):
        from input_devices.quest_hts import QuestHTS

        reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        reservation.close()
        device = QuestHTS(transport="tcp_server", host="127.0.0.1", port=port)
        try:
            time.sleep(0.02)
            device.stop()
            replacement = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                replacement.bind(("127.0.0.1", port))
            finally:
                replacement.close()
        finally:
            device.stop()

    def test_unity_left_landmarks_convert_to_rfu_without_wrist_pose(self):
        from input_devices.quest_hts import unity_left_landmarks_to_rfu

        source = np.zeros((21, 3), dtype=np.float32)
        source[1] = (2.0, 3.0, 5.0)

        converted = unity_left_landmarks_to_rfu(source)

        np.testing.assert_allclose(converted[1], (2.0, 5.0, 3.0))

    def test_quest_sample_preserves_source_and_converted_landmarks(self):
        from input_devices.quest_hts import QuestHTS

        source = np.ones((21, 3), dtype=np.float32)
        rfu = np.full((21, 3), 2.0, dtype=np.float32)
        sample = QuestHTS._record_sample(
            "right", source, rfu,
            np.array((3.0, 4.0, 5.0)), np.array((0.0, 0.0, 0.0, 1.0)), 0.5,
        )

        self.assertEqual(sample.input_type, "quest_hts")
        self.assertIsNone(sample.frame_bgr)
        np.testing.assert_allclose(sample.payload["landmarks_unity_left"], source)
        np.testing.assert_allclose(sample.payload["landmarks_rfu"], rfu)


if __name__ == "__main__":
    unittest.main()
