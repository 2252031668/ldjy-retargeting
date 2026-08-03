from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))


class LDJYFrameContractTests(unittest.TestCase):
    def test_right_mano_from_cad_is_a_proper_rotation(self):
        from ldjy_asset_frames import RIGHT_MANO_FROM_CAD

        np.testing.assert_allclose(
            RIGHT_MANO_FROM_CAD @ RIGHT_MANO_FROM_CAD.T,
            np.eye(3),
            atol=1e-12,
        )
        self.assertAlmostEqual(float(np.linalg.det(RIGHT_MANO_FROM_CAD)), 1.0)

    def test_cad_wrist_maps_to_the_mano_origin(self):
        from ldjy_asset_frames import (
            RIGHT_MANO_FROM_CAD,
            WRIST_IN_CAD,
            root_palm_translation,
        )

        np.testing.assert_allclose(
            RIGHT_MANO_FROM_CAD @ WRIST_IN_CAD + root_palm_translation("right"),
            np.zeros(3),
            atol=1e-12,
        )


class LDJYGeneratedURDFTests(unittest.TestCase):
    def test_generated_urdfs_have_side_specific_root_wrist(self):
        import pinocchio as pin

        for side in ("right", "left"):
            model = pin.buildModelFromUrdf(
                str(ASSET / "urdf" / f"ldjy_{side}_hand.urdf")
            )
            self.assertEqual(model.nq, 20)
            self.assertLess(
                model.getFrameId(f"{side}_retarget_wrist", pin.BODY), model.nframes
            )
            self.assertLess(
                model.getFrameId(f"{side}_finger2_tip", pin.BODY), model.nframes
            )
            self.assertLess(
                model.getFrameId(f"{side}_finger2_pad_frame", pin.BODY), model.nframes
            )

    def test_right_zero_pose_task_vectors_use_mano_wrist_axes(self):
        import pinocchio as pin

        model = pin.buildModelFromUrdf(str(ASSET / "urdf" / "ldjy_right_hand.urdf"))
        data = model.createData()
        pin.forwardKinematics(model, data, np.zeros(model.nq))
        pin.updateFramePlacements(model, data)

        wrist = data.oMf[model.getFrameId("right_retarget_wrist", pin.BODY)].translation
        middle_tip = data.oMf[model.getFrameId("right_finger2_tip", pin.BODY)].translation
        index_tip = data.oMf[model.getFrameId("right_finger1_tip", pin.BODY)].translation
        pinky_tip = data.oMf[model.getFrameId("right_finger4_tip", pin.BODY)].translation

        middle_vector = middle_tip - wrist
        self.assertGreater(middle_vector[2], 0.18)
        self.assertLess(abs(middle_vector[0]), 0.01)
        self.assertGreater(index_tip[1] - wrist[1], 0.03)
        self.assertLess(pinky_tip[1] - wrist[1], -0.07)


class LDJYGeneratedMJCFTests(unittest.TestCase):
    def test_nonzero_tip_offsets_match_between_generated_urdf_and_mjcf(self):
        import mujoco
        import pinocchio as pin
        from build_ldjy_mjcf import build_model
        from build_ldjy_urdf import build_urdf

        offsets = {
            "thumb": {"axis_mm": 0.0, "surface_mm": 0.0},
            "finger1": {"axis_mm": 0.0, "surface_mm": 0.0},
            "finger2": {"axis_mm": 5.0, "surface_mm": -3.0},
            "finger3": {"axis_mm": 0.0, "surface_mm": 0.0},
            "finger4": {"axis_mm": 0.0, "surface_mm": 0.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meshes").symlink_to(ASSET / "meshes", target_is_directory=True)
            urdf_dir = root / "urdf"
            mjcf_dir = root / "mjcf"
            for side in ("right", "left"):
                urdf_path = build_urdf(side, offsets=offsets, output_dir=urdf_dir)
                mjcf_path = build_model(
                    side,
                    offsets=offsets,
                    urdf_dir=urdf_dir,
                    output_dir=mjcf_dir,
                )

                urdf_model = pin.buildModelFromUrdf(str(urdf_path))
                urdf_data = urdf_model.createData()
                pin.forwardKinematics(urdf_model, urdf_data, np.zeros(urdf_model.nq))
                pin.updateFramePlacements(urdf_model, urdf_data)
                urdf_tip = urdf_data.oMf[
                    urdf_model.getFrameId(f"{side}_finger2_tip", pin.BODY)
                ].translation

                mjcf_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
                mjcf_data = mujoco.MjData(mjcf_model)
                mujoco.mj_forward(mjcf_model, mjcf_data)
                site_id = mujoco.mj_name2id(
                    mjcf_model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_finger2_link4_tip"
                )
                np.testing.assert_allclose(mjcf_data.site_xpos[site_id], urdf_tip, atol=1e-6)

                default_model = pin.buildModelFromUrdf(
                    str(ASSET / "urdf" / f"ldjy_{side}_hand.urdf")
                )
                default_data = default_model.createData()
                pin.forwardKinematics(default_model, default_data, np.zeros(default_model.nq))
                pin.updateFramePlacements(default_model, default_data)
                default_tip = default_data.oMf[
                    default_model.getFrameId(f"{side}_finger2_tip", pin.BODY)
                ].translation
                self.assertGreater(np.linalg.norm(urdf_tip - default_tip), 1e-4)

    def test_left_positive_flexion_is_the_right_hand_mirror(self):
        import mujoco

        def link_displacement(side: str, joint_name: str, body_name: str) -> np.ndarray:
            model = mujoco.MjModel.from_xml_path(
                str(ASSET / "mjcf" / f"ldjy_{side}_hand.xml")
            )
            data = mujoco.MjData(model)
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint_name}"
            )
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{body_name}"
            )
            mujoco.mj_forward(model, data)
            zero_position = data.xpos[body_id].copy()
            data.qpos[model.jnt_qposadr[joint_id]] = 1e-4
            mujoco.mj_forward(model, data)
            return data.xpos[body_id] - zero_position

        right_displacement = link_displacement(
            "right", "finger2_joint2", "finger2_link4"
        )
        left_displacement = link_displacement(
            "left", "finger2_joint2", "finger2_link4"
        )

        # CAD X mirroring becomes a Y reflection after the common CAD->MANO rotation.
        mano_mirror = np.diag((1.0, -1.0, 1.0))
        np.testing.assert_allclose(
            left_displacement, mano_mirror @ right_displacement, atol=1e-8
        )

    def test_urdf_and_mjcf_task_frames_match_at_zero_pose(self):
        import mujoco
        import pinocchio as pin

        for side in ("right", "left"):
            urdf_model = pin.buildModelFromUrdf(
                str(ASSET / "urdf" / f"ldjy_{side}_hand.urdf")
            )
            urdf_data = urdf_model.createData()
            pin.forwardKinematics(urdf_model, urdf_data, np.zeros(urdf_model.nq))
            pin.updateFramePlacements(urdf_model, urdf_data)

            mjcf_model = mujoco.MjModel.from_xml_path(
                str(ASSET / "mjcf" / f"ldjy_{side}_hand.xml")
            )
            mjcf_data = mujoco.MjData(mjcf_model)
            mujoco.mj_forward(mjcf_model, mjcf_data)

            frame_names = [f"{side}_retarget_wrist"]
            for finger in ("thumb", "finger1", "finger2", "finger3", "finger4"):
                frame_names.extend((f"{side}_{finger}_link3", f"{side}_{finger}_link4"))

            for frame_name in frame_names:
                urdf_point = urdf_data.oMf[
                    urdf_model.getFrameId(frame_name, pin.BODY)
                ].translation
                body_id = mujoco.mj_name2id(
                    mjcf_model, mujoco.mjtObj.mjOBJ_BODY, frame_name
                )
                np.testing.assert_allclose(mjcf_data.xpos[body_id], urdf_point, atol=1e-6)

            for finger in ("thumb", "finger1", "finger2", "finger3", "finger4"):
                urdf_point = urdf_data.oMf[
                    urdf_model.getFrameId(f"{side}_{finger}_tip", pin.BODY)
                ].translation
                site_id = mujoco.mj_name2id(
                    mjcf_model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    f"{side}_{finger}_link4_tip",
                )
                np.testing.assert_allclose(mjcf_data.site_xpos[site_id], urdf_point, atol=1e-6)

                pad_pose = urdf_data.oMf[
                    urdf_model.getFrameId(f"{side}_{finger}_pad_frame", pin.BODY)
                ]
                pad_site_id = mujoco.mj_name2id(
                    mjcf_model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    f"{side}_{finger}_pad_center",
                )
                np.testing.assert_allclose(
                    mjcf_data.site_xpos[pad_site_id], pad_pose.translation, atol=1e-6
                )
                np.testing.assert_allclose(
                    mjcf_data.site_xmat[pad_site_id].reshape(3, 3),
                    pad_pose.rotation,
                    atol=2e-6,
                )


if __name__ == "__main__":
    unittest.main()
