"""OpenCV overlays used by the live WiLoR tuning preview."""

from __future__ import annotations

import cv2
import numpy as np


MANO_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
RIGHT_COLOR = (60, 220, 60)
LEFT_COLOR = (80, 180, 255)


def full_image_focal_length(width: int, height: int) -> float:
    """Match WiLoR's full-image focal scaling."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return 5000.0 / 256.0 * max(width, height)


def draw_mano_mesh_overlay(
    image_bgr: np.ndarray,
    *,
    vertices_mano: np.ndarray,
    camera_translation: np.ndarray,
    faces: np.ndarray,
    is_right: bool,
    focal_length: float | None = None,
    opacity: float = 0.62,
) -> np.ndarray:
    """Project WiLoR's MANO mesh with OpenCV, without an EGL renderer."""
    image = np.asarray(image_bgr, dtype=np.uint8)
    vertices = np.asarray(vertices_mano, dtype=np.float32)
    translation = np.asarray(camera_translation, dtype=np.float32)
    mesh_faces = np.asarray(faces, dtype=np.int32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must have shape (H, W, 3)")
    if vertices.shape != (778, 3) or translation.shape != (3,):
        raise ValueError("WiLoR MANO mesh data has an invalid shape")
    if mesh_faces.ndim != 2 or mesh_faces.shape[1] != 3:
        raise ValueError("WiLoR MANO faces must have shape (F, 3)")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be within [0, 1]")

    height, width = image.shape[:2]
    focal = full_image_focal_length(width, height) if focal_length is None else float(focal_length)
    camera_vertices = vertices + translation
    depth = camera_vertices[:, 2]
    projected = np.full((len(vertices), 2), np.nan, dtype=np.float32)
    valid_vertices = np.isfinite(camera_vertices).all(axis=1) & (depth > 1e-6)
    projected[valid_vertices, 0] = camera_vertices[valid_vertices, 0] / depth[valid_vertices] * focal + width / 2.0
    projected[valid_vertices, 1] = camera_vertices[valid_vertices, 1] / depth[valid_vertices] * focal + height / 2.0
    valid_faces = (mesh_faces >= 0).all(axis=1) & (mesh_faces < len(vertices)).all(axis=1)
    valid_faces &= valid_vertices[mesh_faces].all(axis=1)
    valid_faces &= np.isfinite(projected[mesh_faces]).all(axis=(1, 2))
    if not np.any(valid_faces):
        return image.copy()

    canvas, mask = image.copy(), np.zeros((height, width), dtype=np.uint8)
    color = RIGHT_COLOR if is_right else LEFT_COLOR
    indices = np.flatnonzero(valid_faces)
    for face_index in indices[np.argsort(depth[mesh_faces[indices]].mean(axis=1))[::-1]]:
        polygon = np.rint(projected[mesh_faces[face_index]]).astype(np.int32)
        cv2.fillConvexPoly(canvas, polygon, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)
    result = image.copy()
    covered = mask > 0
    result[covered] = np.rint(
        (1.0 - opacity) * image[covered].astype(np.float32)
        + opacity * canvas[covered].astype(np.float32)
    ).astype(np.uint8)
    return result


def draw_mano_skeleton_overlay(
    image_bgr: np.ndarray,
    *,
    joints_mano: np.ndarray,
    camera_translation: np.ndarray,
    is_right: bool,
    focal_length: float | None = None,
) -> np.ndarray:
    """Project WiLoR's 21 MANO joints onto its source camera image."""
    image = np.asarray(image_bgr, dtype=np.uint8).copy()
    joints = np.asarray(joints_mano, dtype=np.float32)
    translation = np.asarray(camera_translation, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must have shape (H, W, 3)")
    if joints.shape != (21, 3) or translation.shape != (3,):
        raise ValueError("WiLoR MANO joint data has an invalid shape")
    height, width = image.shape[:2]
    focal = full_image_focal_length(width, height) if focal_length is None else float(focal_length)
    points_3d = joints + translation
    if not np.isfinite(points_3d).all() or np.any(points_3d[:, 2] <= 1e-6):
        return image
    points = np.empty((21, 2), dtype=np.int32)
    points[:, 0] = np.rint(points_3d[:, 0] / points_3d[:, 2] * focal + width / 2.0)
    points[:, 1] = np.rint(points_3d[:, 1] / points_3d[:, 2] * focal + height / 2.0)
    color = RIGHT_COLOR if is_right else LEFT_COLOR
    for start, end in MANO_BONES:
        cv2.line(image, tuple(points[start]), tuple(points[end]), color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)
    return image
