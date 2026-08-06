from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ldjy_retargeting.mano_ldjy_overlay import (
    OverlayScene,
    average_surface_normal,
    apply_registration,
    fit_static_registration,
    solve_exact_hand_pose_constraints,
)


ROOT = Path(__file__).resolve().parents[1]
LDJY_MJCF = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "mjcf" / "ldjy_right_hand.xml"


def test_overlay_scene_keeps_ldjy_controls_and_updates_mano_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
        ],
        dtype=float,
    )
    scene = OverlayScene(
        LDJY_MJCF,
        vertices,
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
    )
    try:
        assert scene.model.nu == 20
        mesh_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_MESH, "mano_overlay_mesh")
        before = scene.model.mesh_vert.copy()
        scene.update_mano_mesh(vertices + np.array([0.02, 0.0, 0.0]))
        assert not np.array_equal(scene.model.mesh_vert, before)
        assert mesh_id >= 0
    finally:
        scene.close()


def test_registration_is_independent_static_scale_rotation_and_translation():
    registered = apply_registration(
        np.array([[1.0, 0.0, 0.0]]),
        rotation=np.array([0.0, 0.0, np.pi / 2]),
        translation=np.array([0.1, 0.2, 0.3]),
        scale=2.0,
    )

    np.testing.assert_allclose(registered, [[0.1, 2.2, 0.3]], atol=1e-12)


def test_surface_normal_uses_mesh_face_winding_not_arbitrary_pad_point_order():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])

    normal = average_surface_normal(vertices, faces, [0, 2, 1])

    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0])


def test_static_fit_uses_pad_positions_and_normals_to_recover_shape_and_registration():
    base_points = np.array(
        [[-0.03, 0.0, 0.01], [-0.01, 0.01, 0.03], [0.0, 0.0, 0.04], [0.02, -0.01, 0.03], [0.04, 0.0, 0.01]]
    )
    point_deformation = np.array(
        [[-0.01, 0.0, 0.0], [-0.005, 0.0, 0.002], [0.0, 0.0, 0.003], [0.005, 0.0, 0.002], [0.01, 0.0, 0.0]]
    )
    base_normals = np.array(
        [[0.0, -1.0, 0.0], [0.1, -0.99, 0.1], [0.0, -0.98, 0.2], [-0.1, -0.99, 0.1], [0.0, -1.0, 0.0]]
    )
    normal_deformation = np.array(
        [[0.2, 0.0, 0.0], [0.1, 0.0, 0.1], [0.0, 0.0, 0.15], [-0.1, 0.0, 0.1], [-0.2, 0.0, 0.0]]
    )
    base_directions = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 0.99], [0.0, 0.1, 0.99], [-0.1, 0.0, 0.99], [0.0, -0.1, 0.99]]
    )
    direction_deformation = np.array(
        [[0.1, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [-0.05, 0.0, 0.0], [0.0, -0.05, 0.0]]
    )

    def sample(betas):
        normals = base_normals + betas[0] * normal_deformation
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return base_points + betas[0] * point_deformation, normals

    def sample_directions(betas):
        directions = base_directions + betas[0] * direction_deformation
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)

    expected_beta = np.array([0.7])
    expected_rotation = np.array([0.25, -0.35, 0.2])
    expected_translation = np.array([0.04, -0.02, 0.08])
    expected_scale = 1.2
    points, normals = sample(expected_beta)
    target_points = apply_registration(points, expected_rotation, expected_translation, expected_scale)
    target_normals = normals @ __import__("scipy").spatial.transform.Rotation.from_rotvec(expected_rotation).as_matrix().T
    target_directions = sample_directions(expected_beta) @ __import__("scipy").spatial.transform.Rotation.from_rotvec(expected_rotation).as_matrix().T

    betas, _, rotation, translation, scale, position_rms = fit_static_registration(
        sample,
        target_points,
        target_normals,
        beta_count=1,
        direction_sample=sample_directions,
        target_directions=target_directions,
        max_nfev=200,
    )

    fitted_points, fitted_normals = sample(betas)
    np.testing.assert_allclose(
        apply_registration(fitted_points, rotation, translation, scale), target_points, atol=2e-5
    )
    np.testing.assert_allclose(
        fitted_normals @ __import__("scipy").spatial.transform.Rotation.from_rotvec(rotation).as_matrix().T,
        target_normals,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        sample_directions(betas) @ __import__("scipy").spatial.transform.Rotation.from_rotvec(rotation).as_matrix().T,
        target_directions,
        atol=2e-4,
    )
    assert position_rms < 2e-5


def test_static_fit_keeps_unselected_variables_fixed_and_skips_unselected_constraints():
    source = np.array(
        [[-0.02, 0.0, 0.0], [-0.01, 0.01, 0.01], [0.0, 0.0, 0.02], [0.01, -0.01, 0.01], [0.02, 0.0, 0.0]]
    )
    deformation = np.array(
        [[-0.002, 0.0, 0.0], [-0.001, 0.0, 0.0], [0.0, 0.0, 0.001], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]]
    )
    normals = np.tile([[0.0, -1.0, 0.0]], (5, 1))

    def sample(betas):
        return source + betas[0] * deformation, normals

    fixed_beta = np.array([0.6])
    fixed_rotation = np.array([0.2, -0.1, 0.3])
    target = apply_registration(
        source + fixed_beta[0] * deformation,
        fixed_rotation,
        np.array([0.03, -0.02, 0.04]),
        1.1,
    )
    betas, _, rotation, _, _, position_rms = fit_static_registration(
        sample,
        target,
        normals,
        beta_count=1,
        initial_betas=fixed_beta,
        initial_rotation=fixed_rotation,
        initial_translation=np.zeros(3),
        initial_scale=1.0,
        fit_betas=False,
        fit_rotation=False,
        use_normals=False,
    )

    np.testing.assert_allclose(betas, fixed_beta)
    np.testing.assert_allclose(rotation, fixed_rotation)
    assert position_rms < 1e-6


def test_static_fit_rotates_the_optional_palm_normal_without_position_constraint():
    source_points = np.zeros((5, 3))
    source_normals = np.tile([[0.0, 0.0, 1.0]], (5, 1))
    source_palm_normal = np.array([0.0, 0.0, 1.0])
    target_palm_normal = Rotation.from_rotvec([0.4, -0.3, 0.0]).apply(source_palm_normal)

    def sample(_betas):
        return source_points, source_normals

    _, _, rotation, _, _, _ = fit_static_registration(
        sample,
        source_points,
        source_normals,
        beta_count=1,
        palm_normal_sample=lambda _betas: source_palm_normal,
        target_palm_normal=target_palm_normal,
        use_positions=False,
        use_normals=False,
        use_directions=False,
        use_palm_normal=True,
        fit_betas=False,
        fit_translation=False,
        fit_scale=False,
    )

    np.testing.assert_allclose(
        Rotation.from_rotvec(rotation).apply(source_palm_normal), target_palm_normal, atol=1e-7
    )


def test_static_fit_can_softly_align_corresponding_finger_lines():
    source_points = np.zeros((5, 3))
    source_normals = np.tile([[0.0, 0.0, 1.0]], (5, 1))
    source_starts = np.array(
        [[-0.03, 0.00, 0.01], [-0.01, 0.01, 0.02], [0.00, 0.00, 0.03], [0.02, -0.01, 0.02], [0.04, 0.00, 0.01]]
    )
    source_directions = np.array(
        [[0.0, 0.0, 1.0], [0.2, 0.0, 0.98], [0.0, 0.3, 0.95], [-0.2, 0.1, 0.97], [0.1, -0.3, 0.95]]
    )
    source_directions /= np.linalg.norm(source_directions, axis=1, keepdims=True)
    expected_rotation = np.array([0.2, -0.3, 0.15])
    expected_translation = np.array([0.04, -0.02, 0.08])
    matrix = Rotation.from_rotvec(expected_rotation).as_matrix()
    target_directions = source_directions @ matrix.T
    # These starts are intentionally shifted along their own lines.
    target_starts = apply_registration(source_starts, expected_rotation, expected_translation, 1.0)
    target_starts += target_directions * np.array([[0.01], [-0.015], [0.02], [0.005], [-0.01]])

    def sample(_betas):
        return source_points, source_normals

    _, _, rotation, translation, _, _ = fit_static_registration(
        sample,
        source_points,
        source_normals,
        beta_count=1,
        direction_sample=lambda _betas: source_directions,
        target_directions=target_directions,
        direction_start_sample=lambda _betas: source_starts,
        target_direction_starts=target_starts,
        use_positions=False,
        use_normals=False,
        use_directions=True,
        use_direction_lines=True,
        fit_betas=False,
        fit_scale=False,
        initial_rotation=np.zeros(3),
        initial_translation=np.zeros(3),
        initial_scale=1.0,
    )

    registered_starts = apply_registration(source_starts, rotation, translation, 1.0)
    offsets = registered_starts - target_starts
    perpendicular = offsets - np.sum(offsets * target_directions, axis=1, keepdims=True) * target_directions
    np.testing.assert_allclose(perpendicular, 0.0, atol=2e-5)


def test_static_fit_limits_automatic_betas_to_three():
    base_points = np.zeros((5, 3))
    deformation = np.tile([[0.01, 0.0, 0.0]], (5, 1))
    normals = np.tile([[0.0, 0.0, 1.0]], (5, 1))

    def sample(betas):
        return base_points + betas[0] * deformation, normals

    betas, _, _, _, _, _ = fit_static_registration(
        sample,
        base_points + 6.0 * deformation,
        normals,
        beta_count=1,
        use_normals=False,
        use_directions=False,
        use_palm_normal=False,
        fit_rotation=False,
        fit_translation=False,
        fit_scale=False,
    )

    np.testing.assert_allclose(betas, [3.0], atol=1e-7)


def test_static_fit_uses_only_selected_pad_positions():
    source_points = np.zeros((5, 3))
    normals = np.tile([[0.0, 0.0, 1.0]], (5, 1))
    target_points = np.array(
        [[0.02, -0.01, 0.03], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4], [-0.4, 0.0, 0.0]]
    )

    _, _, _, translation, _, _ = fit_static_registration(
        lambda _betas: (source_points, normals),
        target_points,
        normals,
        beta_count=1,
        use_normals=False,
        use_directions=False,
        use_palm_normal=False,
        position_mask=np.array([True, False, False, False, False]),
        fit_betas=False,
        fit_rotation=False,
        fit_scale=False,
    )

    np.testing.assert_allclose(translation, target_points[0], atol=1e-7)


def test_static_fit_optimizes_only_selected_axis_angle_hand_pose_joints():
    source_points = np.zeros((5, 3))
    source_normals = np.tile([[0.0, 0.0, 1.0]], (5, 1))
    target_points = source_points.copy()
    target_points[0, 0] = 0.01

    def sample(_betas):
        return source_points, source_normals

    def pose_sample(_betas, hand_pose):
        points = source_points.copy()
        points[0, 0] += hand_pose[0, 0]
        return points, source_normals

    selected_joints = np.zeros(15, dtype=bool)
    selected_joints[0] = True
    _, hand_pose, _, _, _, _ = fit_static_registration(
        sample,
        target_points,
        source_normals,
        beta_count=1,
        pose_sample=pose_sample,
        initial_hand_pose=np.zeros((15, 3)),
        fit_hand_joints=selected_joints,
        use_normals=False,
        use_directions=False,
        use_palm_normal=False,
        use_hand_pose_prior=False,
        fit_betas=False,
        fit_rotation=False,
        fit_translation=False,
        fit_scale=False,
    )

    np.testing.assert_allclose(hand_pose[0, 0], 0.01, atol=1e-7)
    np.testing.assert_allclose(hand_pose[1:], 0.0, atol=1e-12)


def test_static_fit_can_softly_enforce_straight_fingers_with_selected_hand_pose():
    points = np.zeros((5, 3))
    normals = np.tile([[0.0, 0.0, 1.0]], (5, 1))
    initial_hand_pose = np.zeros((15, 3))
    initial_hand_pose[0, 0] = 0.05
    selected_joints = np.zeros(15, dtype=bool)
    selected_joints[0] = True

    def sample(_betas):
        return points, normals

    def joint_sample(_betas, hand_pose):
        joints = np.zeros((21, 3))
        for chain_index, chain in enumerate(((2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))):
            for offset, index in enumerate(chain):
                joints[index] = [chain_index * 0.1, 0.0, float(offset)]
        joints[3, 1] = hand_pose[0, 0]
        return joints

    _, hand_pose, _, _, _, _ = fit_static_registration(
        sample,
        points,
        normals,
        beta_count=1,
        pose_sample=lambda betas, pose: sample(betas),
        joint_pose_sample=joint_sample,
        initial_hand_pose=initial_hand_pose,
        fit_hand_joints=selected_joints,
        use_positions=False,
        use_normals=False,
        use_directions=False,
        use_palm_normal=False,
        use_straight_fingers=True,
        fit_betas=False,
        fit_rotation=False,
        fit_translation=False,
        fit_scale=False,
    )

    np.testing.assert_allclose(hand_pose[0, 0], 0.0, atol=1e-7)


def test_exact_straightening_adjusts_each_distal_joint_in_order():
    initial_hand_pose = np.zeros((15, 3))
    initial_hand_pose[0, 1] = 0.2
    initial_hand_pose[1, 1] = 0.35
    initial_hand_pose[2, 1] = -0.2
    selected_joints = np.zeros(15, dtype=bool)
    selected_joints[[0, 1, 2]] = True

    def joint_sample(_betas, hand_pose):
        joints = np.zeros((21, 3))
        for chain_index, chain in enumerate(((2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))):
            for offset, index in enumerate(chain):
                joints[index] = [chain_index * 0.1, 0.0, float(offset)]
        mcp_angle = hand_pose[0, 1]
        pip_angle = mcp_angle + hand_pose[1, 1]
        dip_angle = pip_angle + hand_pose[2, 1]
        joints[6] = [0.1, np.sin(mcp_angle), np.cos(mcp_angle)]
        joints[7] = joints[6] + [0.0, np.sin(pip_angle), np.cos(pip_angle)]
        joints[8] = joints[7] + [0.0, np.sin(dip_angle), np.cos(dip_angle)]
        return joints

    hand_pose = solve_exact_hand_pose_constraints(
        joint_sample,
        np.zeros(1),
        initial_hand_pose,
        selected_joints,
        hand_pose_reference=np.zeros((15, 3)),
        use_straight_fingers=True,
    )

    joints = joint_sample(np.zeros(1), hand_pose)
    line_direction = joints[6] - joints[5]
    np.testing.assert_allclose(np.cross(joints[7] - joints[5], line_direction), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.cross(joints[8] - joints[5], line_direction), 0.0, atol=1e-7)
    np.testing.assert_allclose(hand_pose[0], initial_hand_pose[0], atol=1e-12)
    np.testing.assert_allclose(hand_pose[1:3], 0.0, atol=1e-7)
    np.testing.assert_allclose(hand_pose[3:], 0.0, atol=1e-12)


def test_exact_straightening_escapes_axis_angle_stationary_points():
    initial_hand_pose = np.zeros((15, 3))
    initial_hand_pose[1, 0] = np.pi - 1e-5
    selected_joints = np.zeros(15, dtype=bool)
    selected_joints[1] = True

    def joint_sample(_betas, hand_pose):
        joints = np.zeros((21, 3))
        for chain_index, chain in enumerate(((2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))):
            for offset, index in enumerate(chain):
                joints[index] = [chain_index * 0.1, 0.0, float(offset)]
        joints[7, 1] = np.cos(hand_pose[1, 0])
        return joints

    hand_pose = solve_exact_hand_pose_constraints(
        joint_sample,
        np.zeros(1),
        initial_hand_pose,
        selected_joints,
        hand_pose_reference=initial_hand_pose,
        use_straight_fingers=True,
    )

    assert abs(np.cos(hand_pose[1, 0])) < 1e-7


def test_exact_straightening_skips_unselected_fingers():
    initial_hand_pose = np.zeros((15, 3))
    initial_hand_pose[1, 1] = 0.2
    selected_joints = np.zeros(15, dtype=bool)
    selected_joints[1] = True

    def joint_sample(_betas, hand_pose):
        joints = np.zeros((21, 3))
        for chain_index, chain in enumerate(((2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))):
            for offset, index in enumerate(chain):
                joints[index] = [chain_index * 0.1, 0.0, float(offset)]
        joints[7, 1] = hand_pose[1, 1]
        joints[11, 1] = 0.1
        return joints

    hand_pose = solve_exact_hand_pose_constraints(
        joint_sample,
        np.zeros(1),
        initial_hand_pose,
        selected_joints,
        hand_pose_reference=initial_hand_pose,
        use_straight_fingers=True,
        straight_finger_mask=np.array([False, True, False, False, False]),
    )

    np.testing.assert_allclose(hand_pose[1, 1], 0.0, atol=1e-7)


def test_exact_finger_plane_constraint_allows_finger_bases_off_the_palm_plane():
    initial_hand_pose = np.zeros((15, 3))
    selected_joints = np.zeros(15, dtype=bool)
    selected_joints[0] = True

    def joint_sample(_betas, _hand_pose):
        joints = np.zeros((21, 3))
        joints[0] = [0.0, 0.0, 0.0]
        joints[5] = [1.0, 0.0, 0.0]
        joints[17] = [0.0, 0.0, 1.0]
        for start, end, base in ((5, 8, joints[5]), (9, 12, [0.0, 0.02, 0.2]), (13, 16, [0.3, -0.03, 0.4]), (17, 20, joints[17])):
            for offset, index in enumerate(range(start, end + 1)):
                joints[index] = np.asarray(base) + [0.0, 0.0, offset * 0.1]
        return joints

    hand_pose = solve_exact_hand_pose_constraints(
        joint_sample,
        np.zeros(1),
        initial_hand_pose,
        selected_joints,
        hand_pose_reference=initial_hand_pose,
        use_palm_plane=True,
    )

    np.testing.assert_allclose(hand_pose, initial_hand_pose)
