import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example"
sys.path.insert(0, str(EXAMPLE_DIR))


class TuningGuiCliTests(unittest.TestCase):
    def test_debug_worker_uses_the_shared_120_hz_control_rate(self):
        from tuning_gui import DEBUG_CONTROL_HZ, MuJoCoDebugWorker

        self.assertEqual(DEBUG_CONTROL_HZ, 120)
        worker = MuJoCoDebugWorker("right", ROOT / "missing.xml")
        worker.set_paused(True)
        self.assertTrue(worker.paused)
        worker.set_paused(False)
        self.assertFalse(worker.paused)

    def test_help_does_not_open_a_camera_or_import_pyside(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "example" / "tuning_gui.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--webcam", result.stdout)
        self.assertIn("--camera-index", result.stdout)


if __name__ == "__main__":
    unittest.main()
