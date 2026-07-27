"""Length calibration for the 15 wrist-origin retargeting vectors."""

from __future__ import annotations

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
SEGMENT_NAMES = ("pip", "dip", "tip")
KEYPOINT_INDICES = np.array(
    ((2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)),
    dtype=np.int64,
)
_MIN_LENGTH = 1e-8


class CalibrationError(ValueError):
    """Raised when a zero-pose length calibration cannot be applied."""


def _validate_lengths(lengths: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(lengths, dtype=np.float64)
    if values.shape != (5, 3):
        raise CalibrationError(f"{name} lengths must have shape (5, 3), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values <= _MIN_LENGTH):
        raise CalibrationError(f"{name} lengths must be finite and greater than {_MIN_LENGTH}")
    return values


def human_vector_lengths(keypoints: np.ndarray) -> np.ndarray:
    """Return wrist-to-PIP/DIP/TIP lengths in thumb-to-pinky order."""
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (21, 3):
        raise CalibrationError(f"expected keypoints shape (21, 3), got {points.shape}")
    vectors = points[KEYPOINT_INDICES] - points[0]
    return _validate_lengths(np.linalg.norm(vectors, axis=2), "human")


def robot_vector_lengths(
    origin: np.ndarray,
    pip: np.ndarray,
    dip: np.ndarray,
    tip: np.ndarray,
) -> np.ndarray:
    """Return robot wrist-to-PIP/DIP/TIP lengths in `(5, 3)` layout."""
    origin_point = np.asarray(origin, dtype=np.float64)
    if origin_point.shape != (3,):
        raise CalibrationError(f"robot origin must have shape (3,), got {origin_point.shape}")
    vectors = np.stack(
        (
            np.asarray(pip, dtype=np.float64) - origin_point,
            np.asarray(dip, dtype=np.float64) - origin_point,
            np.asarray(tip, dtype=np.float64) - origin_point,
        ),
        axis=1,
    )
    return _validate_lengths(np.linalg.norm(vectors, axis=2), "robot")


def compute_scales(
    robot_lengths: np.ndarray,
    human_samples: np.ndarray,
    lower: float = 0.5,
    upper: float = 1.5,
) -> np.ndarray:
    """Match median human lengths to zero-pose robot lengths without clipping."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or lower > upper:
        raise CalibrationError("invalid scale bounds")
    samples = np.asarray(human_samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1:] != (5, 3) or samples.shape[0] == 0:
        raise CalibrationError("human samples must have shape (frames, 5, 3)")
    human_median = _validate_lengths(np.median(samples, axis=0), "human median")
    scales = _validate_lengths(robot_lengths, "robot") / human_median
    if np.any(scales < lower) or np.any(scales > upper):
        raise CalibrationError(f"suggested scale outside [{lower}, {upper}]")
    return scales


__all__ = [
    "CalibrationError",
    "FINGER_NAMES",
    "KEYPOINT_INDICES",
    "SEGMENT_NAMES",
    "compute_scales",
    "human_vector_lengths",
    "robot_vector_lengths",
]
