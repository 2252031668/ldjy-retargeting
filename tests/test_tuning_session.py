import tempfile
import unittest
from pathlib import Path

import yaml

from ldjy_retargeting.tuning.session import TuningSession


class TuningSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "adaptive_analytical_video.yaml"
        self.config_path.write_text("retarget: {}\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_first_save_creates_immutable_original_backup(self):
        session = TuningSession(self.config_path)
        session.set_value("retarget.segment_scaling.index.tip", 1.2)
        session.save()

        original = self.config_path.with_name(
            "adaptive_analytical_video.yaml.original.yaml"
        )
        first_bytes = original.read_bytes()
        session.set_value("retarget.segment_scaling.index.tip", 1.3)
        session.save()

        self.assertEqual(first_bytes, original.read_bytes())
        self.assertEqual(
            yaml.safe_load(self.config_path.read_text(encoding="utf-8"))["retarget"]["segment_scaling"]["index"][2],
            1.3,
        )

    def test_restore_default_only_changes_memory_until_save(self):
        session = TuningSession(self.config_path)
        session.set_value("retarget.segment_scaling.index.tip", 0.8)
        session.save()

        session.restore_default()

        self.assertTrue(session.is_dirty)
        self.assertEqual(session.config["retarget"]["segment_scaling"]["index"][2], 1.10)
        self.assertEqual(
            yaml.safe_load(self.config_path.read_text(encoding="utf-8"))["retarget"]["segment_scaling"]["index"][2],
            0.8,
        )

    def test_session_does_not_expose_reload_unsaved_changes(self):
        session = TuningSession(self.config_path)

        self.assertFalse(hasattr(session, "reload_disk"))


if __name__ == "__main__":
    unittest.main()
