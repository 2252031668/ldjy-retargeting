"""Static coordinate contract for generated LDJY hand assets.

Column-vector convention:
    p_mano = R_mano_from_cad @ (p_cad - wrist_in_cad)

MediaPipe input is already canonicalized into the MANO wrist frame before it
reaches the retargeting optimizer. Generated URDF and MJCF assets therefore
express their task frames in this same coordinate system.
"""

from __future__ import annotations

import numpy as np


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


def root_palm_translation(side: str) -> np.ndarray:
    """Return the generated root-wrist -> CAD-palm translation for ``side``.

    The generated CAD chain is mirrored before this transform is applied for
    the left hand, so both side-specific CAD frames use this same contract.
    """
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported side for the frame contract: {side}")
    return -(RIGHT_MANO_FROM_CAD @ WRIST_IN_CAD)
