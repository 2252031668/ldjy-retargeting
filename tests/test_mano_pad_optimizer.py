import numpy as np
import pytest

from ldjy_retargeting.opt.mano_pad_pose import ManoPadPoseOptimizer


def _optimizer():
    return ManoPadPoseOptimizer({
        "optimizer": {"hand_side": "right"},
        "retarget": {
            "pad_ik": {
                "position_weight": 1.0,
                "normal_weight": 0.2,
                "smooth_weight": 0.1,
                "maxeval": 20,
            }
        },
    })


def test_pad_ik_keeps_finite_bounded_warm_start():
    optimizer = _optimizer()
    q_prev = np.zeros(optimizer.num_joints)
    optimizer.robot.compute_forward_kinematics(q_prev)
    targets = optimizer.targets_from_qpos(q_prev)

    solved = optimizer.solve(targets, q_prev)

    assert solved.shape == (optimizer.num_joints,)
    assert np.isfinite(solved).all()
    limits = optimizer.robot.joint_limits
    assert np.all(solved >= limits[:, 0] - 1e-10)
    assert np.all(solved <= limits[:, 1] + 1e-10)
    np.testing.assert_allclose(solved, q_prev, atol=1e-5)


def test_disabled_finger_remains_at_previous_command():
    optimizer = _optimizer()
    optimizer.finger_configs["finger2"]["enabled"] = False
    q_prev = np.linspace(-0.1, 0.1, optimizer.num_joints)
    optimizer.robot.compute_forward_kinematics(q_prev)
    targets = optimizer.targets_from_qpos(q_prev)

    solved = optimizer.solve(targets, q_prev)

    indices = optimizer.finger_qpos_indices[2]
    clipped = np.clip(q_prev, optimizer.robot.joint_limits[:, 0], optimizer.robot.joint_limits[:, 1])
    np.testing.assert_array_equal(solved[indices], clipped[indices])


def test_pinch_activation_uses_thumb_to_finger_target_distances():
    optimizer = _optimizer()
    targets = optimizer.targets_from_qpos(np.zeros(optimizer.num_joints))
    targets.pinch_distances_m[0] = 0.015

    active, distances, target_distances = optimizer.pinch_state(targets)

    assert active.tolist() == [False, True, False, False, False]
    assert distances[1] == pytest.approx(0.015)
    assert target_distances[1] == pytest.approx(0.015)

    targets.pinch_distances_m[0] = 0.0005
    _, _, target_distances = optimizer.pinch_state(targets)
    assert target_distances[1] == pytest.approx(0.001)


def test_pinch_position_weight_releases_at_thirty_percent_of_threshold():
    optimizer = _optimizer()
    optimizer.pinch_threshold_m = 0.03

    scales = optimizer.pinch_position_scales(np.array([0.03, 0.0195, 0.009, 0.0]))

    np.testing.assert_allclose(scales, [0.25, 0.125, 0.0, 0.0])


def test_pinch_contact_weight_increases_as_the_pair_closes():
    optimizer = _optimizer()
    optimizer.pinch_threshold_m = 0.03
    optimizer.thumb_contact_surface_weight = 300.0

    weights = optimizer.pinch_contact_weights(np.array([0.03, 0.0195, 0.009, 0.0]))

    np.testing.assert_allclose(weights, [0.0, 105.0, 210.0, 300.0])


def test_pinch_contact_surface_uses_the_nearest_active_finger():
    optimizer = _optimizer()

    pair = optimizer.nearest_pinch_finger(
        np.array([False, True, True, False, False]),
        np.array([0.0, 0.015, 0.010, 0.0, 0.0]),
    )

    assert pair == 2


def test_pinch_joint_solve_moves_thumb_and_reduces_active_pair_distance_error():
    optimizer = _optimizer()
    q0 = np.zeros(optimizer.num_joints)
    targets = optimizer.targets_from_qpos(q0)
    original_distance = np.linalg.norm(targets.pad_positions[1] - targets.pad_positions[0])
    targets.pad_positions[1] = targets.pad_positions[0] + [0.005, 0.0, 0.0]
    targets.pinch_distances_m[0] = 0.005

    solved = optimizer.solve(targets, q0)
    actual = optimizer.targets_from_qpos(solved)
    actual_distance = np.linalg.norm(actual.pad_positions[1] - actual.pad_positions[0])
    thumb = optimizer.finger_qpos_indices[0]

    assert np.linalg.norm(solved[thumb] - q0[thumb]) > 1e-5
    assert abs(actual_distance - 0.005) < abs(original_distance - 0.005)


def test_thumb_contact_surface_targets_normals_along_the_solved_pinch_line():
    optimizer = _optimizer()
    optimizer.thumb_contact_surface_optimization = True
    q0 = np.zeros(optimizer.num_joints)
    targets = optimizer.targets_from_qpos(q0)
    targets.pad_positions[1] = targets.pad_positions[0] + [0.005, 0.0, 0.0]
    targets.pinch_distances_m[0] = 0.005
    targets.pad_normals[[0, 1]] = [0.0, 1.0, 0.0]

    optimizer.solve(targets, q0)

    actual = optimizer.last_diagnostics["pad_actual_m"]
    direction = actual[1] - actual[0]
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(optimizer.last_diagnostics["pad_target_normals"][0], direction)
    np.testing.assert_allclose(optimizer.last_diagnostics["pad_target_normals"][1], -direction)


def test_pinch_joint_solve_supports_multiple_active_fingers():
    optimizer = _optimizer()
    q0 = np.zeros(optimizer.num_joints)
    targets = optimizer.targets_from_qpos(q0)
    targets.pad_positions[1] = targets.pad_positions[0] + [0.015, 0.0, 0.0]
    targets.pad_positions[2] = targets.pad_positions[0] + [0.018, 0.0, 0.0]
    targets.pinch_distances_m[:2] = [0.015, 0.018]

    solved = optimizer.solve(targets, q0)

    assert optimizer.last_diagnostics["pinch_active"].tolist() == [False, True, True, False, False]
    assert np.isfinite(solved).all()
