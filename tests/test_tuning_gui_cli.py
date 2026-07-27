import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TuningGuiCliTests(unittest.TestCase):
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
