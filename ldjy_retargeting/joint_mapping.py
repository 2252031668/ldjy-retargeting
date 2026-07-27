"""Joint-name helpers shared by LDJY simulation and visualization consumers."""

from __future__ import annotations

import numpy as np


def normalize_joint_name(name: str) -> str:
    """Remove the MJCF-only side prefix from an LDJY joint name."""
    for prefix in ("left_", "right_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def qpos_reorder_perm(src_joint_names, dst_joint_names):
    """Return indices that reorder optimizer qpos into a consumer's joint order.

    The LDJY URDF uses unprefixed joint names, while the left and right MJCFs
    add ``left_`` or ``right_``. ``None`` means the lists cannot be aligned
    one-to-one.
    """
    if not dst_joint_names:
        return None

    normalized_source = [normalize_joint_name(name) for name in src_joint_names]
    normalized_target = [normalize_joint_name(name) for name in dst_joint_names]
    if len(normalized_source) != len(normalized_target):
        return None
    if len(set(normalized_source)) != len(normalized_source):
        return None

    source_index = {name: i for i, name in enumerate(normalized_source)}
    try:
        return np.array([source_index[name] for name in normalized_target], dtype=int)
    except KeyError:
        return None
