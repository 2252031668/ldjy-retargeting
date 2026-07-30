from pathlib import Path
import tempfile
import unittest

import numpy as np


class RetargetTipFrameTests(unittest.TestCase):
    def test_unconfigured_offsets_are_five_zero_pairs(self):
        from ldjy_retargeting.retarget_tip_frames import DEFAULT_OFFSETS, normalize_tip_offsets

        self.assertEqual(normalize_tip_offsets(None), DEFAULT_OFFSETS)

    def test_canonical_offsets_allow_exported_nonzero_calibration(self):
        from ldjy_retargeting.retarget_tip_frames import FINGERS, load_tip_offsets

        offsets = load_tip_offsets()

        self.assertEqual(tuple(offsets), FINGERS)
        for values in offsets.values():
            self.assertLessEqual(abs(values["axis_mm"]), 20.0)
            self.assertLessEqual(abs(values["surface_mm"]), 20.0)

    def test_apply_offset_uses_millimetres_along_two_local_axes(self):
        from ldjy_retargeting.retarget_tip_frames import apply_offset

        point = apply_offset(
            base_position=np.array((0.0, 0.0, 0.03)),
            axis_local=np.array((0.0, 0.0, 1.0)),
            surface_local=np.array((1.0, 0.0, 0.0)),
            axis_mm=2.0,
            surface_mm=-3.0,
        )

        np.testing.assert_allclose(point, (-0.003, 0.0, 0.032))

    def test_offsets_require_exactly_five_fingers_and_finite_bounded_values(self):
        from ldjy_retargeting.retarget_tip_frames import DEFAULT_OFFSETS, validate_tip_offsets

        bad = {finger: dict(values) for finger, values in DEFAULT_OFFSETS.items()}
        bad["finger2"]["axis_mm"] = 20.1
        with self.assertRaises(ValueError):
            validate_tip_offsets(bad)

        bad = {finger: dict(values) for finger, values in DEFAULT_OFFSETS.items()}
        bad.pop("thumb")
        with self.assertRaises(ValueError):
            validate_tip_offsets(bad)

    def test_loads_yaml_offsets(self):
        from ldjy_retargeting.retarget_tip_frames import load_tip_offsets

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offsets.yaml"
            path.write_text(
                """version: 1
units: mm
fingers:
  thumb: {axis_mm: 1.0, surface_mm: -2.0}
  finger1: {axis_mm: 0.0, surface_mm: 0.0}
  finger2: {axis_mm: 0.0, surface_mm: 0.0}
  finger3: {axis_mm: 0.0, surface_mm: 0.0}
  finger4: {axis_mm: 0.0, surface_mm: 0.0}
""",
                encoding="utf-8",
            )

            self.assertEqual(load_tip_offsets(path)["thumb"], {"axis_mm": 1.0, "surface_mm": -2.0})

    def test_thumb_surface_axis_uses_its_local_nail_to_pulp_axis(self):
        import mujoco
        from ldjy_retargeting.retarget_tip_frames import task_frame_axes

        model = mujoco.MjModel.from_xml_path(
            "ldjy_retargeting/assets/robots/ldjy_hand/source/step_20_dof_hand.urdf"
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        _, surface = task_frame_axes(
            model,
            data,
            "thumb",
            surface_reference_world=np.array((0.0, 1.0, 0.0)),
        )
        self.assertGreater(abs(float(surface @ np.array((0.0, 0.0, 1.0)))), 0.99)


if __name__ == "__main__":
    unittest.main()
