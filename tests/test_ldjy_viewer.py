import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIEWER_PATH = ROOT / "example" / "ldjy_viewer.py"


def load_viewer():
    spec = importlib.util.spec_from_file_location("ldjy_viewer", VIEWER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LDJYViewerTests(unittest.TestCase):
    def test_defaults_to_right_and_accepts_left(self):
        viewer = load_viewer()

        self.assertEqual(viewer.parse_args([]).side, "right")
        self.assertEqual(viewer.parse_args(["--left"]).side, "left")
        self.assertEqual(viewer.mjcf_path("left").name, "ldjy_left_hand.xml")

    def test_zero_controls_are_clipped_to_actuator_ranges(self):
        viewer = load_viewer()
        model = mujoco.MjModel.from_xml_path(str(viewer.mjcf_path("right")))
        data = mujoco.MjData(model)

        viewer.set_zero_controls(model, data)

        expected = np.clip(0.0, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        np.testing.assert_allclose(data.ctrl, expected)

    def test_draws_five_calibrated_pad_markers_from_mjcf_sites(self):
        viewer = load_viewer()
        sites = "".join(
            f'<site name="right_{finger}_pad_center" pos="{index} 0 0" />'
            for index, finger in enumerate(viewer.FINGERS)
        )
        model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{sites}</worldbody></mujoco>")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = mujoco.MjvScene(model, maxgeom=10)

        viewer.draw_pad_points(scene, model, data, "right")
        viewer.draw_pad_normals(scene, model, data, "right")

        self.assertEqual(scene.ngeom, 10)
        for index in range(5):
            self.assertEqual(scene.geoms[index].type, mujoco.mjtGeom.mjGEOM_SPHERE)
        for index in range(5, 10):
            self.assertEqual(scene.geoms[index].type, mujoco.mjtGeom.mjGEOM_CAPSULE)


if __name__ == "__main__":
    unittest.main()
