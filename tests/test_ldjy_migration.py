from pathlib import Path
import sys
import pickle
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"
sys.path.insert(0, str(EXAMPLE_DIR))


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"


class LDJYAssetTests(unittest.TestCase):
    def test_ldjy_assets_are_vendored(self):
        self.assertTrue((ASSET / "source" / "step_20_dof_hand.urdf").is_file())
        self.assertTrue((ASSET / "urdf" / "ldjy_left_hand.urdf").is_file())
        self.assertTrue((ASSET / "urdf" / "ldjy_right_hand.urdf").is_file())
        self.assertFalse((ASSET / "urdf" / "ldjy_hand.urdf").exists())
        self.assertTrue((ASSET / "mjcf" / "ldjy_left_hand.xml").is_file())
        self.assertTrue((ASSET / "mjcf" / "ldjy_right_hand.xml").is_file())


class LDJYDefaultKinematicsTests(unittest.TestCase):
    def test_tip_position_targets_reuse_the_fifteen_vector_tip_scales(self):
        """Pinch TipPos and FullHand TIP must use one per-finger target length."""
        from ldjy_retargeting import Retargeter

        optimizer = Retargeter.from_config({
            "optimizer": {"type": "AdaptiveOptimizerAnalytical"},
            "retarget": {
                "segment_scaling": {
                    "thumb": [0.8, 0.9, 1.0],
                    "index": [0.9, 1.0, 1.1],
                    "middle": [1.0, 1.1, 1.2],
                    "ring": [1.1, 1.2, 1.3],
                    "pinky": [1.2, 1.3, 1.4],
                },
            },
        }).optimizer
        keypoints = np.zeros((21, 3), dtype=np.float64)
        for finger, tip_index in enumerate(optimizer.MP_TIP_INDICES):
            keypoints[tip_index] = [0.01 * (finger + 1), 0.02, -0.03]

        tip_pos_targets = optimizer._compute_tip_vectors(
            keypoints, optimizer.segment_scaling[:, 2]
        )
        full_hand_targets = optimizer._compute_full_hand_vectors(
            keypoints, optimizer.segment_scaling
        )

        np.testing.assert_allclose(tip_pos_targets, full_hand_targets[10:15])

    def test_default_optimizer_uses_generated_side_specific_urdf(self):
        from ldjy_retargeting import Retargeter

        for side in ("right", "left"):
            optimizer = Retargeter.from_config(
                {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}, side
            ).optimizer
            self.assertEqual(optimizer.robot.dof_joint_names[0], f"{side}_finger1_joint1")
            self.assertLess(
                optimizer.robot.get_link_index("retarget_wrist"),
                optimizer.robot.model.nframes,
            )

    def test_runtime_keypoint_preparation_keeps_mano_task_coordinates(self):
        from ldjy_retargeting import Retargeter
        from ldjy_retargeting.mediapipe import apply_mediapipe_transformations

        retargeter = Retargeter.from_config(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}
        )
        raw_keypoints = np.zeros((21, 3), dtype=np.float64)
        raw_keypoints[5] = [0.02, 0.0, 0.0]
        raw_keypoints[9] = [0.0, 0.09, 0.0]
        raw_keypoints[17] = [-0.02, 0.0, 0.0]
        raw_keypoints[8] = [0.03, 0.12, 0.01]

        np.testing.assert_allclose(
            retargeter._prepare_keypoints(raw_keypoints),
            apply_mediapipe_transformations(raw_keypoints, "right"),
        )

    def test_default_optimizer_uses_ldjy_semantic_frames(self):
        from ldjy_retargeting import Retargeter

        retargeter = Retargeter.from_config(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}
        )
        optimizer = retargeter.optimizer
        self.assertEqual(optimizer.origin_link_name, "retarget_wrist")
        self.assertEqual(
            optimizer.task_link_names,
            ["thumb_tip", "finger1_tip", "finger2_tip", "finger3_tip", "finger4_tip"],
        )
        self.assertEqual(
            optimizer.link3_names,
            [
                "thumb_link3",
                "finger1_link3",
                "finger2_link3",
                "finger3_link3",
                "finger4_link3",
            ],
        )
        self.assertEqual(
            optimizer.robot.dof_joint_names[:4],
            [
                "right_finger1_joint1",
                "right_finger1_joint2",
                "right_finger1_joint3",
                "right_finger1_joint4",
            ],
        )


class LDJYJointMappingTests(unittest.TestCase):
    def test_mjcf_side_prefix_is_ignored_when_reordering_qpos(self):
        from utils.config_paths import qpos_reorder_perm

        source = [
            *(f"finger1_joint{i}" for i in range(1, 5)),
            *(f"finger2_joint{i}" for i in range(1, 5)),
            *(f"finger3_joint{i}" for i in range(1, 5)),
            *(f"thumb_joint{i}" for i in range(1, 5)),
            *(f"finger4_joint{i}" for i in range(1, 5)),
        ]
        target = [f"right_{name}" for name in source]

        np.testing.assert_array_equal(qpos_reorder_perm(source, target), np.arange(20))

    def test_bundled_mjcf_joint_order_aligns_with_the_urdf(self):
        import mujoco
        from ldjy_retargeting import Retargeter
        from ldjy_retargeting.joint_mapping import qpos_reorder_perm

        retargeter = Retargeter.from_config(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}
        )
        model = mujoco.MjModel.from_xml_path(str(ASSET / "mjcf" / "ldjy_right_hand.xml"))
        mjcf_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
        ]

        # URDF orders thumb before finger4, while MuJoCo orders finger4 before
        # thumb. The permutation is therefore intentionally non-identity.
        np.testing.assert_array_equal(
            qpos_reorder_perm(retargeter.optimizer.robot.dof_joint_names, mjcf_names),
            np.array([*range(12), *range(16, 20), *range(12, 16)]),
        )


class LDJYDebugSupportTests(unittest.TestCase):
    def test_debug_overlay_accepts_the_active_hand_for_a_bimanual_model(self):
        import mujoco
        from ldjy_retargeting import Retargeter
        from ldjy_retargeting.viz.debug_overlay import DebugOverlay

        openarm_mjcf = (
            ROOT
            / "ldjy_retargeting"
            / "assets"
            / "robots"
            / "openarm_hand"
            / "mjcf"
            / "openarm_bimanual_mano.xml"
        )
        model = mujoco.MjModel.from_xml_path(str(openarm_mjcf))
        overlay = DebugOverlay(model, "left")
        self.assertEqual(overlay.hand_side, "left")
        data = mujoco.MjData(model)
        scene = mujoco.MjvScene(model, maxgeom=256)
        optimizer = Retargeter.from_config(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}, "left"
        ).optimizer
        overlay.draw(scene, data, optimizer, np.zeros((21, 3)), np.zeros(5))
        self.assertGreater(scene.ngeom, 0)

    def test_debug_mesh_transparency_only_changes_visible_meshes(self):
        import mujoco
        from teleop_sim import DEBUG_MESH_ALPHA, set_debug_mesh_transparency

        model = mujoco.MjModel.from_xml_path(str(ASSET / "mjcf" / "ldjy_right_hand.xml"))
        visible_mesh_id = next(
            i for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_rgba[i, 3] > 0
        )
        hidden_mesh_id = next(
            i for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_rgba[i, 3] == 0
        )

        set_debug_mesh_transparency(model)

        self.assertAlmostEqual(model.geom_rgba[visible_mesh_id, 3], DEBUG_MESH_ALPHA)
        self.assertEqual(model.geom_rgba[hidden_mesh_id, 3], 0.0)

    def test_first_optimizer_guess_is_the_zero_pose_clipped_to_limits(self):
        from ldjy_retargeting import Retargeter

        optimizer = Retargeter.from_config(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}
        ).optimizer

        np.testing.assert_allclose(
            optimizer._get_init_qpos(None),
            np.clip(0.0, optimizer.robot.joint_limits[:, 0], optimizer.robot.joint_limits[:, 1]),
        )

    def test_debug_mode_labels_expose_the_blend_weight(self):
        from ldjy_retargeting.viz.debug_overlay import mode_label

        self.assertEqual(mode_label(0.0), "FullHandVec")
        self.assertEqual(mode_label(0.35), "Blend(alpha=0.35)")
        self.assertEqual(mode_label(0.7), "TipDirVec(alpha=0.70)")

    def test_debug_overlay_does_not_draw_a_command_pose(self):
        import mujoco
        from ldjy_retargeting import Retargeter
        from ldjy_retargeting.viz.debug_overlay import DebugOverlay

        model = mujoco.MjModel.from_xml_path(str(ASSET / "mjcf" / "ldjy_right_hand.xml"))
        data = mujoco.MjData(model)
        overlay = DebugOverlay(model)
        optimizer = Retargeter.from_config(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}
        ).optimizer
        scene = mujoco.MjvScene(model, maxgeom=256)

        overlay.draw(scene, data, optimizer, np.zeros((21, 3)), np.zeros(5))

        purple = np.array([0.8, 0.25, 1.0, 0.95])
        self.assertFalse(
            any(np.allclose(scene.geoms[i].rgba, purple) for i in range(scene.ngeom))
        )

    def test_debug_overlay_can_hide_skeleton_and_target_rays_independently(self):
        import mujoco
        from ldjy_retargeting import Retargeter
        from ldjy_retargeting.viz.debug_overlay import DebugOverlay

        model = mujoco.MjModel.from_xml_path(str(ASSET / "mjcf" / "ldjy_right_hand.xml"))
        data = mujoco.MjData(model)
        optimizer = Retargeter.from_config(
            {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}
        ).optimizer
        scene = mujoco.MjvScene(model, maxgeom=256)
        overlay = DebugOverlay(model, show_skeleton=False, show_rays=False)

        overlay.draw(scene, data, optimizer, np.zeros((21, 3)), np.zeros(5))

        self.assertEqual(scene.ngeom, 0)

    def test_pinch_dominant_fingers_use_red_rays_without_full_hand_rays(self):
        from ldjy_retargeting.viz.debug_overlay import _active_target_segments

        class Optimizer:
            segment_scaling = np.ones((5, 3))
            MP_TIP_INDICES = [4, 8, 12, 16, 20]
            MP_DIP_INDICES = [3, 7, 11, 15, 19]

            @staticmethod
            def _compute_full_hand_vectors(keypoints, scaling):
                return np.ones((15, 3))

            @staticmethod
            def _compute_tip_vectors(keypoints, scaling):
                return np.ones((5, 3))

            @staticmethod
            def _compute_tip_dirs(keypoints):
                return np.tile(np.array((0.0, 0.0, 1.0)), (5, 1))

        segments = _active_target_segments(
            Optimizer(), np.zeros((21, 3)), np.array((0.7, 0.7, 0.0, 0.0, 0.0))
        )

        self.assertEqual(sum(segment.kind == "full" for segment in segments), 9)
        self.assertEqual(sum(segment.kind == "pinch" for segment in segments), 4)
        self.assertTrue(all("TH" not in segment.label for segment in segments if segment.kind == "full"))
        self.assertTrue(all("F1" not in segment.label for segment in segments if segment.kind == "full"))

    def test_pinch_direction_arrow_ends_at_its_tip_position_target(self):
        from ldjy_retargeting.viz.debug_overlay import _active_target_segments

        class Optimizer:
            segment_scaling = np.ones((5, 3))
            MP_TIP_INDICES = [4, 8, 12, 16, 20]
            MP_DIP_INDICES = [3, 7, 11, 15, 19]

            @staticmethod
            def _compute_full_hand_vectors(keypoints, scaling):
                return np.zeros((15, 3))

            @staticmethod
            def _compute_tip_vectors(keypoints, scaling):
                return np.tile(np.array((10.0, 20.0, 30.0)), (5, 1))

            @staticmethod
            def _compute_tip_dirs(keypoints):
                return np.tile(np.array((0.0, 0.0, 1.0)), (5, 1))

        segments = _active_target_segments(
            Optimizer(), np.zeros((21, 3)), np.array((0.7, 0.0, 0.0, 0.0, 0.0))
        )
        tip_pos = next(segment for segment in segments if segment.label == "TipPos TH")
        tip_dir = next(segment for segment in segments if segment.label == "TipDir TH")

        np.testing.assert_allclose(tip_dir.end, tip_pos.end)
        self.assertLess(tip_dir.start[2], tip_dir.end[2])


class LDJYRetargetingSmokeTests(unittest.TestCase):
    def test_recorded_keypoints_produce_a_finite_ldjy_pose_within_limits(self):
        from ldjy_retargeting import Retargeter

        with open(ROOT / "example" / "data" / "avp1.pkl", "rb") as f:
            frames = pickle.load(f)
        keypoints = next(
            frame["left_fingers"] for frame in frames
            if not np.allclose(frame["left_fingers"], 0)
        )
        retargeter = Retargeter.from_yaml(
            str(ROOT / "example" / "config" / "adaptive_analytical_avp.yaml"),
            hand_side="left",
        )
        qpos = retargeter.retarget(keypoints, apply_filter=False)
        limits = retargeter.optimizer.robot.joint_limits

        self.assertEqual(qpos.shape, (20,))
        self.assertTrue(np.all(np.isfinite(qpos)))
        self.assertTrue(np.all(qpos >= limits[:, 0]))
        self.assertTrue(np.all(qpos <= limits[:, 1]))


class LDJYPackagingTests(unittest.TestCase):
    def test_project_metadata_contains_only_ldjy_package(self):
        with open(ROOT / "pyproject.toml", "rb") as f:
            project = tomllib.load(f)

        self.assertEqual(project["project"]["name"], "ldjy-retargeting")
        self.assertEqual(
            project["tool"]["setuptools"]["packages"]["find"]["include"],
            ["ldjy_retargeting*"],
        )

    def test_docs_describe_realtime_tuning_gui_and_baseline_backup(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        startup_modes = (ROOT / "docs" / "startup-modes.md").read_text(encoding="utf-8")

        self.assertIn("tuning_gui.py --webcam", readme)
        self.assertIn("original.yaml", startup_modes)


if __name__ == "__main__":
    unittest.main()
