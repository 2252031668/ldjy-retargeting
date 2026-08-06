import numpy as np

from ldjy_retargeting.mano_conventions import (
    MANO_HAND_POSE_FINGER_ORDER,
    MANO_ACTIVE_SKELETON_EDGES,
    MEDIAPIPE_FINGER_DIRECTION_JOINTS,
    MEDIAPIPE_21_SKELETON_EDGES,
    mano_native_to_mediapipe_keypoints,
)


def test_native_mano_keypoints_are_reordered_to_mediapipe_order():
    native_joints = np.column_stack((np.arange(16), np.zeros(16), np.zeros(16)))
    fingertip_points = np.column_stack((np.arange(100, 105), np.zeros(5), np.zeros(5)))

    keypoints = mano_native_to_mediapipe_keypoints(native_joints, fingertip_points)

    np.testing.assert_array_equal(
        keypoints[:, 0],
        [0, 13, 14, 15, 100, 1, 2, 3, 101, 4, 5, 6, 102, 10, 11, 12, 103, 7, 8, 9, 104],
    )


def test_hand_pose_keeps_mano_native_finger_order():
    assert MANO_HAND_POSE_FINGER_ORDER == ("Index", "Middle", "Pinky", "Ring", "Thumb")


def test_finger_direction_pairs_follow_mediapipe_order():
    assert MEDIAPIPE_FINGER_DIRECTION_JOINTS == ((2, 3), (5, 6), (9, 10), (13, 14), (17, 18))


def test_mano_skeleton_edges_follow_public_mediapipe_keypoints():
    assert MEDIAPIPE_21_SKELETON_EDGES == (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    )
    assert MANO_ACTIVE_SKELETON_EDGES == tuple(
        edge for edge in MEDIAPIPE_21_SKELETON_EDGES if edge[1] not in (4, 8, 12, 16, 20)
    )
