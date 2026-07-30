"""Measured assembly and natural-zero calibration for generated OpenArm assets."""

from __future__ import annotations

import numpy as np


# Translation in each hand-adapter frame that aligns the visual-mesh mating-face
# centroids. The source asset is retained unchanged; this applies only to the
# generated OpenArm asset.
PALM_MOUNT_TRANSLATION_CORRECTIONS = {
    "left": np.array((-0.0004, 0.00072, 0.00002), dtype=np.float64),
    "right": np.array((-0.0004, -0.00072, 0.00002), dtype=np.float64),
}

# Source J7 angles that make the middle-finger direction vertical and down at
# the all-other-joints-zero arm pose. Generated q7=0 represents these poses.
J7_HOME_OFFSETS = {
    "left": 0.22071467042253595,
    "right": -0.2207146703886431,
}
