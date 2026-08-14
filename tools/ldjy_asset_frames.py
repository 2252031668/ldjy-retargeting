"""Static coordinate contract for generated LDJY hand assets.

Column-vector convention:
    p_mano = R_mano_from_cad @ (p_cad - wrist_in_cad)

MediaPipe input is already canonicalized into the MANO wrist frame before it
reaches the retargeting optimizer. Generated URDF and MJCF assets therefore
express their task frames in this same coordinate system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


# CAD right-hand coordinates -> MANO right-hand wrist coordinates. This is a
# +90 degree rotation around CAD Z when represented as a column-vector matrix.
RIGHT_MANO_FROM_CAD = np.array(
    (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)

# Original CAD palm frame location of the task-space wrist. The point is
# behind the physical palm and makes the four non-thumb finger rays coplanar
# at the zero pose.
WRIST_IN_CAD = np.array((0.0, -0.015, -0.03), dtype=np.float64)
DEFAULT_ROOT_PALM_TRANSLATION = -(RIGHT_MANO_FROM_CAD @ WRIST_IN_CAD)

ROOT = Path(__file__).resolve().parents[1]
RETARGET_WRIST_CALIBRATION_PATH = (
    ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "retarget_wrist_calibration.yaml"
)


def _calibrated_root_frames() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load the accumulated MANO-wrist point and root-to-palm translation for each side."""
    default_points = {side: np.zeros(3) for side in ("right", "left")}
    default_translations = {side: DEFAULT_ROOT_PALM_TRANSLATION.copy() for side in default_points}
    if not RETARGET_WRIST_CALIBRATION_PATH.exists():
        return default_points, default_translations
    payload = yaml.safe_load(RETARGET_WRIST_CALIBRATION_PATH.read_text(encoding="utf-8")) or {}
    points, translations = {}, {}
    for side in default_points:
        entry = payload.get(side, {})
        point = np.asarray(
            entry.get("mano_joint0_in_original_root", entry.get("mano_joint0_in_previous_root", default_points[side])),
            dtype=float,
        )
        translation = np.asarray(entry.get("root_to_palm_translation", default_translations[side]), dtype=float)
        if point.shape != (3,) or translation.shape != (3,):
            raise ValueError(f"invalid {side} retarget wrist calibration")
        points[side] = point
        translations[side] = translation
    return points, translations


RETARGET_WRIST_POINT_IN_ORIGINAL_ROOT, _ROOT_PALM_TRANSLATIONS = _calibrated_root_frames()


def root_palm_translation(side: str) -> np.ndarray:
    """Return the generated root-wrist -> CAD-palm translation for ``side``.

    The generated CAD chain is mirrored before this transform is applied for
    the left hand, so both side-specific CAD frames use this same contract.
    """
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported side for the frame contract: {side}")
    return _ROOT_PALM_TRANSLATIONS[side].copy()
