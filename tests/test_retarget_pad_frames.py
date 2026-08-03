from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class RetargetPadFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from build_ldjy_urdf import SOURCE_URDF

        cls.model = mujoco.MjModel.from_xml_path(str(SOURCE_URDF))

    def test_source_visual_mesh_generates_one_right_hand_frame_per_finger(self):
        from ldjy_retargeting.retarget_pad_frames import FINGERS, pad_frames_from_source

        frames = pad_frames_from_source(self.model)

        self.assertEqual(tuple(frames), FINGERS)
        for frame in frames.values():
            self.assertEqual(frame.position_m.shape, (3,))
            self.assertEqual(frame.rotation_parent_from_pad.shape, (3, 3))
            self.assertTrue(np.isfinite(frame.position_m).all())
            np.testing.assert_allclose(
                frame.rotation_parent_from_pad.T @ frame.rotation_parent_from_pad,
                np.eye(3),
                atol=1e-9,
            )

    def test_pad_normal_points_to_the_configured_pulp_side(self):
        from ldjy_retargeting.retarget_pad_frames import (
            NAIL_HOLE_COORDINATES,
            pad_frames_from_source,
        )
        from ldjy_retargeting.retarget_tip_frames import task_frame_axes
        from build_ldjy_urdf import CAD_NAIL_TO_PULP

        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        for finger, frame in pad_frames_from_source(self.model).items():
            self.assertEqual(NAIL_HOLE_COORDINATES[finger].shape, (3, 2))
            _, pulp_axis_local = task_frame_axes(
                self.model, data, finger, surface_reference_world=CAD_NAIL_TO_PULP
            )
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{finger}_link4"
            )
            world_from_link4 = data.xmat[body_id].reshape(3, 3)
            pulp_axis = world_from_link4 @ pulp_axis_local
            normal = world_from_link4 @ frame.rotation_parent_from_pad[:, 2]
            self.assertGreater(np.dot(normal, pulp_axis), 0.9)

    def test_pad_center_is_on_the_pulp_side_of_its_three_hole_plane(self):
        from ldjy_retargeting.retarget_pad_frames import (
            NAIL_HOLE_COORDINATES,
            pad_frames_from_source,
        )
        from ldjy_retargeting.retarget_tip_frames import task_frame_axes
        from build_ldjy_urdf import CAD_NAIL_TO_PULP

        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        for finger, frame in pad_frames_from_source(self.model).items():
            distal, pulp = task_frame_axes(
                self.model, data, finger, surface_reference_world=CAD_NAIL_TO_PULP
            )
            lateral = np.cross(pulp, distal)
            lateral /= np.linalg.norm(lateral)
            mesh_id = self.model.geom_dataid[
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{finger}_link4_visual_mesh_collision",
                )
            ]
            vertices = self.model.mesh_vert[
                self.model.mesh_vertadr[mesh_id]:self.model.mesh_vertadr[mesh_id]
                + self.model.mesh_vertnum[mesh_id]
            ]
            hole_center = (
                NAIL_HOLE_COORDINATES[finger][:, 0].mean() * lateral
                + NAIL_HOLE_COORDINATES[finger][:, 1].mean() * distal
                + (vertices @ pulp).min() * pulp
            )
            self.assertGreater(np.dot(frame.position_m - hole_center, pulp), 0.01)


if __name__ == "__main__":
    unittest.main()
