from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "ldjy_retargeting" / "assets" / "robots" / "openarm_hand"
SOURCE_URDF = ASSET / "source" / "openarm_bimanual_20dof_hands.urdf"
URDF = ASSET / "urdf" / "openarm_bimanual_mano.urdf"
MJCF = ASSET / "mjcf" / "openarm_bimanual_mano.xml"

FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")
HAND_MESH_LINKS = (
    "palm",
    *(f"{finger}_link{index}" for finger in FINGERS for index in range(1, 5)),
)
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))


class OpenArmGeneratedAssetTests(unittest.TestCase):
    def test_openarm_tip_offsets_match_standalone_ldjy_task_frames(self):
        import pinocchio as pin
        from build_ldjy_urdf import build_urdf as build_ldjy_urdf
        from build_openarm_hand_urdf import build_urdf as build_openarm_urdf

        offsets = {
            "thumb": {"axis_mm": 0.0, "surface_mm": 0.0},
            "finger1": {"axis_mm": 0.0, "surface_mm": 0.0},
            "finger2": {"axis_mm": 4.0, "surface_mm": -2.0},
            "finger3": {"axis_mm": 0.0, "surface_mm": 0.0},
            "finger4": {"axis_mm": 0.0, "surface_mm": 0.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meshes").symlink_to(
                ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "meshes",
                target_is_directory=True,
            )
            hand_urdf_dir = root / "urdf"
            openarm_urdf = root / "openarm_bimanual_mano.urdf"
            build_openarm_urdf(offsets=offsets, output_path=openarm_urdf)
            openarm_model = pin.buildModelFromUrdf(str(openarm_urdf))
            openarm_data = openarm_model.createData()
            pin.forwardKinematics(openarm_model, openarm_data, np.zeros(openarm_model.nq))
            pin.updateFramePlacements(openarm_model, openarm_data)

            for side in ("left", "right"):
                hand_urdf = build_ldjy_urdf(side, offsets=offsets, output_dir=hand_urdf_dir)
                hand_model = pin.buildModelFromUrdf(str(hand_urdf))
                hand_data = hand_model.createData()
                pin.forwardKinematics(hand_model, hand_data, np.zeros(hand_model.nq))
                pin.updateFramePlacements(hand_model, hand_data)

                hand_wrist = hand_data.oMf[
                    hand_model.getFrameId(f"{side}_retarget_wrist", pin.BODY)
                ].inverse()
                openarm_wrist = openarm_data.oMf[
                    openarm_model.getFrameId(f"{side}_retarget_wrist", pin.BODY)
                ].inverse()
                for finger in FINGERS:
                    hand_tip = hand_wrist * hand_data.oMf[
                        hand_model.getFrameId(f"{side}_{finger}_tip", pin.BODY)
                    ]
                    openarm_tip = openarm_wrist * openarm_data.oMf[
                        openarm_model.getFrameId(f"{side}_{finger}_tip", pin.BODY)
                    ]
                    np.testing.assert_allclose(
                        openarm_tip.translation, hand_tip.translation, atol=1e-6
                    )

    def test_generated_zero_pose_applies_mount_and_j7_calibration(self):
        import pinocchio as pin
        from tools.build_openarm_hand_urdf import (
            J7_HOME_OFFSETS,
            PALM_MOUNT_TRANSLATION_CORRECTIONS,
        )

        source_model = pin.buildModelFromUrdf(str(SOURCE_URDF))
        generated_model = pin.buildModelFromUrdf(str(URDF))
        source_data = source_model.createData()
        generated_data = generated_model.createData()
        source_qpos = np.zeros(source_model.nq)
        for side in ("left", "right"):
            source_qpos[source_model.idx_qs[source_model.getJointId(f"openarm_{side}_joint7")]] = J7_HOME_OFFSETS[side]
        pin.forwardKinematics(source_model, source_data, source_qpos)
        pin.forwardKinematics(
            generated_model, generated_data, np.zeros(generated_model.nq)
        )
        pin.updateFramePlacements(source_model, source_data)
        pin.updateFramePlacements(generated_model, generated_data)

        for side in ("left", "right"):
            source_adapter = source_data.oMf[
                source_model.getFrameId(f"{side}_hand_adapter", pin.BODY)
            ]
            source_palm = source_data.oMf[source_model.getFrameId(f"{side}_palm", pin.BODY)]
            generated_adapter = generated_data.oMf[
                generated_model.getFrameId(f"{side}_hand_adapter", pin.BODY)
            ]
            generated_palm = generated_data.oMf[
                generated_model.getFrameId(f"{side}_palm", pin.BODY)
            ]
            correction = pin.SE3(
                np.eye(3), PALM_MOUNT_TRANSLATION_CORRECTIONS[side]
            )
            expected_mount = correction * (source_adapter.inverse() * source_palm)
            actual_mount = generated_adapter.inverse() * generated_palm
            np.testing.assert_allclose(
                actual_mount.translation, expected_mount.translation, atol=1e-9
            )
            np.testing.assert_allclose(
                actual_mount.rotation, expected_mount.rotation, atol=1e-9
            )

            middle_tip = generated_data.oMf[
                generated_model.getFrameId(f"{side}_finger2_link4", pin.BODY)
            ].translation
            direction = middle_tip - generated_palm.translation
            direction /= np.linalg.norm(direction)
            self.assertAlmostEqual(float(direction[1]), 0.0, places=6)
            self.assertLess(float(direction[2]), -0.99)

            joint = generated_model.getJointId(f"openarm_{side}_joint7")
            source_joint = source_model.getJointId(f"openarm_{side}_joint7")
            np.testing.assert_allclose(
                generated_model.lowerPositionLimit[generated_model.idx_qs[joint]],
                source_model.lowerPositionLimit[source_model.idx_qs[source_joint]] - J7_HOME_OFFSETS[side],
                atol=1e-9,
            )
            np.testing.assert_allclose(
                generated_model.upperPositionLimit[generated_model.idx_qs[joint]],
                source_model.upperPositionLimit[source_model.idx_qs[source_joint]] - J7_HOME_OFFSETS[side],
                atol=1e-9,
            )

    def test_hand_mano_wrist_to_palm_transform_matches_ldjy_hand_assets(self):
        import pinocchio as pin

        openarm_model = pin.buildModelFromUrdf(str(URDF))
        openarm_data = openarm_model.createData()
        pin.forwardKinematics(openarm_model, openarm_data, np.zeros(openarm_model.nq))
        pin.updateFramePlacements(openarm_model, openarm_data)

        ldjy_asset = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
        for side in ("left", "right"):
            hand_model = pin.buildModelFromUrdf(
                str(ldjy_asset / "urdf" / f"ldjy_{side}_hand.urdf")
            )
            hand_data = hand_model.createData()
            pin.forwardKinematics(hand_model, hand_data, np.zeros(hand_model.nq))
            pin.updateFramePlacements(hand_model, hand_data)

            openarm_wrist = openarm_data.oMf[
                openarm_model.getFrameId(f"{side}_retarget_wrist", pin.BODY)
            ]
            openarm_palm = openarm_data.oMf[
                openarm_model.getFrameId(f"{side}_palm", pin.BODY)
            ]
            hand_wrist = hand_data.oMf[
                hand_model.getFrameId(f"{side}_retarget_wrist", pin.BODY)
            ]
            hand_palm = hand_data.oMf[
                hand_model.getFrameId(f"{side}_palm", pin.BODY)
            ]
            openarm_relative = openarm_wrist.inverse() * openarm_palm
            hand_relative = hand_wrist.inverse() * hand_palm
            np.testing.assert_allclose(
                openarm_relative.translation, hand_relative.translation, atol=1e-9
            )
            np.testing.assert_allclose(
                openarm_relative.rotation, hand_relative.rotation, atol=1e-9
            )

    def test_urdf_keeps_a_fixed_root_and_adds_mano_task_frames(self):
        import pinocchio as pin

        model = pin.buildModelFromUrdf(str(URDF))
        self.assertEqual(model.nq, 54)
        self.assertEqual(model.nv, 54)
        for side in ("left", "right"):
            self.assertLess(model.getFrameId(f"{side}_retarget_wrist", pin.BODY), model.nframes)
            for finger in FINGERS:
                self.assertLess(
                    model.getFrameId(f"{side}_{finger}_tip", pin.BODY), model.nframes
                )

    def test_mjcf_has_one_position_actuator_for_each_of_54_joints(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        self.assertEqual(model.nq, 54)
        self.assertEqual(model.nu, 54)
        actuator_joints = set(model.actuator_trnid[:, 0])
        self.assertEqual(actuator_joints, set(range(model.njnt)))
        for actuator_id, joint_id in enumerate(model.actuator_trnid[:, 0]):
            np.testing.assert_allclose(
                model.actuator_ctrlrange[actuator_id], model.jnt_range[joint_id], atol=1e-9
            )
        for side in ("left", "right"):
            self.assertGreater(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_retarget_wrist"),
                0,
            )
            for finger in FINGERS:
                self.assertGreater(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{finger}_tip"),
                    0,
                )

    def test_urdf_and_mjcf_task_frame_positions_match_at_zero_pose(self):
        import mujoco
        import pinocchio as pin

        urdf_model = pin.buildModelFromUrdf(str(URDF))
        urdf_data = urdf_model.createData()
        pin.forwardKinematics(urdf_model, urdf_data, np.zeros(urdf_model.nq))
        pin.updateFramePlacements(urdf_model, urdf_data)

        mjcf_model = mujoco.MjModel.from_xml_path(str(MJCF))
        mjcf_data = mujoco.MjData(mjcf_model)
        mujoco.mj_forward(mjcf_model, mjcf_data)

        for side in ("left", "right"):
            frame_names = [f"{side}_retarget_wrist"]
            frame_names += [f"{side}_{finger}_tip" for finger in FINGERS]
            for frame_name in frame_names:
                urdf_position = urdf_data.oMf[
                    urdf_model.getFrameId(frame_name, pin.BODY)
                ].translation
                body_id = mujoco.mj_name2id(
                    mjcf_model, mujoco.mjtObj.mjOBJ_BODY, frame_name
                )
                np.testing.assert_allclose(
                    mjcf_data.xpos[body_id], urdf_position, atol=1e-6
                )

    def test_mjcf_preserves_openarm_red_and_gold_appearance(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        colors = model.geom_rgba[:, :3]
        expected = (
            np.array((0.24, 0.008, 0.015)),
            np.array((0.58, 0.025, 0.035)),
            np.array((0.88, 0.50, 0.055)),
        )
        for color in expected:
            self.assertTrue(np.any(np.all(np.isclose(colors, color, atol=1e-4), axis=1)))

    def test_mjcf_keeps_distinct_left_and_right_hand_mesh_assets(self):
        import mujoco

        root = ET.parse(MJCF).getroot()
        mesh_files = {
            mesh.attrib["name"]: mesh.attrib["file"]
            for mesh in root.findall("./asset/mesh")
        }
        model = mujoco.MjModel.from_xml_path(str(MJCF))
        for link in HAND_MESH_LINKS:
            self.assertEqual(
                mesh_files[f"left_{link}"],
                f"../meshes/hand_mirrored/{link}.stl",
            )
            self.assertEqual(
                mesh_files[f"right_{link}"], f"../meshes/hand/{link}.stl"
            )

            mesh_ids = []
            for side in ("left", "right"):
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{link}"
                )
                geom_id = model.body_geomadr[body_id]
                mesh_ids.append(model.geom_dataid[geom_id])
            self.assertNotEqual(*mesh_ids)

    def test_visual_surface_meshes_are_used_for_openarm_collision(self):
        import mujoco

        urdf_root = ET.parse(URDF).getroot()
        collision_files = [
            mesh.attrib["filename"]
            for mesh in urdf_root.findall(".//collision/geometry/mesh")
        ]
        self.assertTrue(collision_files)
        self.assertFalse(
            any("openarm/arm/collision" in mesh_file for mesh_file in collision_files)
        )
        self.assertTrue(
            any("openarm_visual_clean/arm/visual" in mesh_file for mesh_file in collision_files)
        )

        mjcf_root = ET.parse(MJCF).getroot()
        mesh_files = [mesh.attrib["file"] for mesh in mjcf_root.findall("./asset/mesh")]
        self.assertTrue(
            any("openarm_visual_clean/arm/visual" in mesh_file for mesh_file in mesh_files)
        )
        self.assertFalse(
            any("openarm/arm/collision" in mesh_file for mesh_file in mesh_files)
        )

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        robot_meshes = np.flatnonzero(
            model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
        )
        self.assertTrue(np.all(model.geom_contype[robot_meshes] > 0))

    def test_mjcf_has_bright_scene_lighting(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        self.assertGreaterEqual(model.nlight, 3)
        self.assertGreaterEqual(float(model.vis.headlight.ambient[0]), 0.4)
        self.assertGreaterEqual(float(model.vis.headlight.diffuse[0]), 0.75)

    def test_mjcf_has_a_checker_floor_and_stable_zero_control_dynamics(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.assertGreaterEqual(floor_id, 0)
        self.assertEqual(model.geom_type[floor_id], mujoco.mjtGeom.mjGEOM_PLANE)
        floor_material = model.geom_matid[floor_id]
        self.assertGreaterEqual(floor_material, 0)
        self.assertTrue(np.any(model.mat_texid[floor_material] >= 0))

        data = mujoco.MjData(model)
        peak_velocity = 0.0
        for _ in range(5_000):
            mujoco.mj_step(model, data)
            peak_velocity = max(peak_velocity, float(np.abs(data.qvel).max()))
        self.assertLess(peak_velocity, 1.0)
        self.assertLess(float(np.abs(data.qvel).max()), 0.05)

    def test_hand_mcp_step_reaches_ninety_percent_within_350ms_without_overshoot(self):
        import mujoco
        from ldjy_retargeting import Retargeter
        from ldjy_retargeting.openarm_control import OpenArmTeleopControl

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        data = mujoco.MjData(model)
        control = OpenArmTeleopControl(model, "left")
        control.set_initial_pose(data)
        retargeter = Retargeter.from_config(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}, "left"
        )
        names = retargeter.optimizer.robot.dof_joint_names
        command = np.clip(
            0.0,
            retargeter.optimizer.robot.joint_limits[:, 0],
            retargeter.optimizer.robot.joint_limits[:, 1],
        )
        data.ctrl[:] = control.targets(names, command)
        for _ in range(1_000):
            mujoco.mj_step(model, data)

        target = 0.8
        command[names.index("left_finger1_joint2")] = target
        data.ctrl[:] = control.targets(names, command)
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger1_joint2"
        )
        qpos_address = model.jnt_qposadr[joint_id]
        samples = []
        for _ in range(175):
            mujoco.mj_step(model, data)
            samples.append(data.qpos[qpos_address])

        self.assertGreaterEqual(max(samples), target * 0.9)
        self.assertLessEqual(max(samples), target * 1.02)
        for _ in range(2_325):
            mujoco.mj_step(model, data)
        self.assertLess(abs(data.qvel[model.jnt_dofadr[joint_id]]), 1e-3)

    def test_left_arm_tracks_representative_joint_targets_without_residual_shake(self):
        import mujoco

        targets = (
            np.array((-0.4, -0.4, 0.3, 0.6, 0.3, 0.2, 0.2)),
            np.array((0.4, -1.0, -0.4, 1.0, -0.4, 0.3, -0.3)),
            np.array((-0.8, -0.8, 0.4, 0.4, 0.2, -0.2, 0.3)),
        )
        for target in targets:
            model = mujoco.MjModel.from_xml_path(str(MJCF))
            data = mujoco.MjData(model)
            joints = [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_left_joint{i}"
                )
                for i in range(1, 8)
            ]
            actuators = [
                next(
                    actuator_id
                    for actuator_id in range(model.nu)
                    if model.actuator_trnid[actuator_id, 0] == joint_id
                )
                for joint_id in joints
            ]
            data.ctrl[actuators] = target
            for _ in range(500):
                mujoco.mj_step(model, data)
            qpos = np.array([data.qpos[model.jnt_qposadr[joint]] for joint in joints])
            self.assertLess(float(np.abs(qpos - target).max()), 0.08)
            for _ in range(4_500):
                mujoco.mj_step(model, data)
            qpos = np.array([data.qpos[model.jnt_qposadr[joint]] for joint in joints])
            qvel = np.array([data.qvel[model.jnt_dofadr[joint]] for joint in joints])
            self.assertLess(float(np.abs(qpos - target).max()), 0.035)
            self.assertLess(float(np.abs(qvel).max()), 0.1)

    def test_arm_joint_tuning_is_shared_by_left_and_right_arms(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        for index in range(1, 8):
            values = []
            for side in ("left", "right"):
                joint = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{index}"
                )
                actuator = next(
                    actuator_id
                    for actuator_id in range(model.nu)
                    if model.actuator_trnid[actuator_id, 0] == joint
                )
                dof = model.jnt_dofadr[joint]
                values.append(
                    (
                        model.actuator_gainprm[actuator, 0],
                        -model.actuator_biasprm[actuator, 2],
                        model.dof_damping[dof],
                        model.dof_armature[dof],
                    )
                )
            np.testing.assert_allclose(values[0], values[1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
