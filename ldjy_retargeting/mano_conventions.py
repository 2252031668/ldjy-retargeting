"""MANO keypoint conventions shared by viewers and retargeting tools."""

import numpy as np


# MANO's 15 local rotations follow its native kinematic-tree order.
MANO_HAND_POSE_FINGER_ORDER = ("Index", "Middle", "Pinky", "Ring", "Thumb")
MEDIAPIPE_FINGER_DIRECTION_JOINTS = ((2, 3), (5, 6), (9, 10), (13, 14), (17, 18))
MEDIAPIPE_21_SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
MANO_ACTIVE_SKELETON_EDGES = tuple(
    edge for edge in MEDIAPIPE_21_SKELETON_EDGES if edge[1] not in (4, 8, 12, 16, 20)
)
MANO_ACTIVE_KEYPOINT_INDICES = (0,) + tuple(end for _, end in MANO_ACTIVE_SKELETON_EDGES)

# smplx MANOLayer's 16 joints, reordered without fingertip vertices.
_NATIVE_TO_MEDIAPIPE_BASE = np.array(
    (0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9), dtype=np.intp
)
_MEDIAPIPE_BASE_SLOTS = np.array(
    (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19), dtype=np.intp
)
_MEDIAPIPE_TIP_SLOTS = np.array((4, 8, 12, 16, 20), dtype=np.intp)


def mano_native_to_mediapipe_keypoints(native_joints, fingertip_points):
    """Return the public 21 MANO keypoints in MediaPipe/WiLoR order."""
    native_joints = np.asarray(native_joints)
    fingertip_points = np.asarray(fingertip_points)
    if native_joints.shape != (16, 3):
        raise ValueError(f"native_joints must have shape (16, 3), got {native_joints.shape}")
    if fingertip_points.shape != (5, 3):
        raise ValueError(f"fingertip_points must have shape (5, 3), got {fingertip_points.shape}")

    keypoints = np.empty((21, 3), dtype=np.result_type(native_joints, fingertip_points))
    keypoints[_MEDIAPIPE_BASE_SLOTS] = native_joints[_NATIVE_TO_MEDIAPIPE_BASE]
    keypoints[_MEDIAPIPE_TIP_SLOTS] = fingertip_points
    return keypoints
