import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example"
sys.path.insert(0, str(EXAMPLE_DIR))


class TuningGuiCliTests(unittest.TestCase):
    def test_supported_algorithms_are_exposed(self):
        from tuning_gui import algorithm_choices

        choices = {choice.key: choice for choice in algorithm_choices()}

        self.assertEqual(
            set(choices),
            {"adaptive_mediapipe", "adaptive_wilor", "adaptive_quest_hts", "mano_pad_pose_wilor"},
        )
        self.assertEqual(choices["mano_pad_pose_wilor"].input_types, ("webcam_wilor",))
        self.assertEqual(choices["adaptive_quest_hts"].input_types, ("quest_hts",))

    def test_only_wilor_input_exposes_the_mano_overlay_toggle(self):
        from tuning_gui import supports_mano_overlay

        self.assertFalse(supports_mano_overlay("webcam"))
        self.assertTrue(supports_mano_overlay("webcam_wilor"))
        self.assertFalse(supports_mano_overlay("quest_hts"))

    def test_wilor_hides_only_mediapipe_preprocessing_controls(self):
        from tuning_gui import parameter_specs_for_input

        webcam_paths = {spec.path for spec in parameter_specs_for_input("webcam")}
        wilor_paths = {spec.path for spec in parameter_specs_for_input("webcam_wilor")}
        quest_paths = {spec.path for spec in parameter_specs_for_input("quest_hts")}

        self.assertIn("video_input.z_scale", webcam_paths)
        self.assertIn("video_input.correct_segments", webcam_paths)
        self.assertNotIn("video_input.z_scale", wilor_paths)
        self.assertNotIn("video_input.reference_wrist_to_mid_mcp", wilor_paths)
        self.assertNotIn("video_input.correct_segments", wilor_paths)
        self.assertIn("retarget.segment_scaling.index.tip", wilor_paths)
        self.assertIn("retarget.lp_alpha", wilor_paths)
        self.assertIn("tip_offsets.thumb.axis_mm", wilor_paths)
        self.assertEqual(quest_paths, wilor_paths)

    def test_parser_accepts_gui_default_or_one_supported_webcam_device(self):
        from tuning_gui import build_parser, initial_config_path, input_device_type_from_args

        parser = build_parser()
        self.assertEqual(input_device_type_from_args(parser.parse_args([])), "webcam")
        self.assertEqual(input_device_type_from_args(parser.parse_args(["--webcam"])), "webcam")
        self.assertEqual(
            input_device_type_from_args(parser.parse_args(["--webcam-wilor"])),
            "webcam_wilor",
        )
        self.assertEqual(
            input_device_type_from_args(parser.parse_args(["--quest-hts"])),
            "quest_hts",
        )
        self.assertEqual(
            initial_config_path(parser.parse_args(["--quest-hts"])).name,
            "adaptive_analytical_quest_hts.yaml",
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--webcam", "--webcam-wilor"])

    def test_debug_worker_uses_the_shared_120_hz_control_rate(self):
        from tuning_gui import DEBUG_CONTROL_HZ, MuJoCoDebugWorker

        self.assertEqual(DEBUG_CONTROL_HZ, 120)
        worker = MuJoCoDebugWorker("right", ROOT / "missing.xml")
        worker.set_paused(True)
        self.assertTrue(worker.paused)
        worker.set_paused(False)
        self.assertFalse(worker.paused)

    def test_debug_pose_writes_actuator_targets_without_servo_lag(self):
        import mujoco
        from tuning_gui import apply_debug_kinematic_pose

        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body><joint name='a' type='hinge'/><geom type='sphere' size='.01'/><body><joint name='b' type='hinge'/><geom type='sphere' size='.01'/></body></body></worldbody>
            <actuator><position joint='a'/><position joint='b'/></actuator></mujoco>
        """)
        data = mujoco.MjData(model)
        data.qvel[:] = 1.0

        apply_debug_kinematic_pose(model, data, [0.3, -0.2])

        self.assertEqual(data.qpos.tolist(), [0.3, -0.2])
        self.assertEqual(data.qvel.tolist(), [0.0, 0.0])

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
        self.assertIn("--webcam-wilor", result.stdout)
        self.assertIn("--quest-hts", result.stdout)
        self.assertIn("--quest-transport", result.stdout)
        self.assertIn("--camera-index", result.stdout)

    def test_readme_documents_tuning_record_and_replay(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("tuning_records", text)
        self.assertIn("静态调参记录", text)


if __name__ == "__main__":
    unittest.main()
