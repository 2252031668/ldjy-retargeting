"""Shared task-tip configuration and local coordinate helpers.

The generated ``*_tip`` frames are virtual retargeting points.  Their
longitudinal coordinate follows PIP -> DIP, while the surface coordinate lies
in the finger thickness direction and is perpendicular to the longitudinal
axis.  All public offsets are expressed in millimetres.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import yaml


FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")
OFFSET_LIMIT_MM = 20.0
DEFAULT_OFFSET_FILE = (
    Path(__file__).resolve().parent
    / "assets"
    / "robots"
    / "ldjy_hand"
    / "retarget_tip_offsets.yaml"
)
DEFAULT_OFFSETS = {
    finger: {"axis_mm": 0.0, "surface_mm": 0.0} for finger in FINGERS
}


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"{name} must be a non-zero finite vector")
    return vector / norm


def validate_tip_offsets(offsets: Mapping[str, Mapping[str, float]]) -> None:
    """Validate the exact five-finger, two-axis task-tip configuration."""
    if set(offsets) != set(FINGERS):
        raise ValueError(f"tip offsets must contain exactly: {', '.join(FINGERS)}")
    for finger in FINGERS:
        values = offsets[finger]
        if set(values) != {"axis_mm", "surface_mm"}:
            raise ValueError(f"{finger} must contain axis_mm and surface_mm")
        for name, value in values.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{finger}.{name} must be numeric") from exc
            if not np.isfinite(numeric) or abs(numeric) > OFFSET_LIMIT_MM:
                raise ValueError(
                    f"{finger}.{name} must be finite and within "
                    f"[-{OFFSET_LIMIT_MM:g}, {OFFSET_LIMIT_MM:g}] mm"
                )


def normalize_tip_offsets(offsets: Mapping[str, Mapping[str, float]] | None) -> dict[str, dict[str, float]]:
    """Return a validated, independent copy of configured or zero offsets."""
    normalized = deepcopy(DEFAULT_OFFSETS if offsets is None else offsets)
    validate_tip_offsets(normalized)
    return {
        finger: {
            "axis_mm": float(normalized[finger]["axis_mm"]),
            "surface_mm": float(normalized[finger]["surface_mm"]),
        }
        for finger in FINGERS
    }


def load_tip_offsets(path: Path = DEFAULT_OFFSET_FILE) -> dict[str, dict[str, float]]:
    """Load the canonical task-tip configuration from YAML."""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping) or document.get("version") != 1:
        raise ValueError("tip offset file must declare version: 1")
    if document.get("units") != "mm":
        raise ValueError("tip offset file must declare units: mm")
    fingers = document.get("fingers")
    if not isinstance(fingers, Mapping):
        raise ValueError("tip offset file must contain a fingers mapping")
    return normalize_tip_offsets(fingers)


def apply_offset(
    base_position: np.ndarray,
    axis_local: np.ndarray,
    surface_local: np.ndarray,
    axis_mm: float,
    surface_mm: float,
) -> np.ndarray:
    """Return a link-local task point after its two millimetre offsets."""
    axis = _unit(axis_local, name="axis_local")
    surface = np.asarray(surface_local, dtype=np.float64)
    surface -= axis * np.dot(surface, axis)
    surface = _unit(surface, name="surface_local perpendicular to axis_local")
    return (
        np.asarray(base_position, dtype=np.float64)
        + axis * (float(axis_mm) / 1000.0)
        + surface * (float(surface_mm) / 1000.0)
    )


def task_frame_axes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    finger: str,
    *,
    side: str = "",
    surface_reference_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PIP->DIP and nail/pulp axes in the link-4 local frame.

    ``surface_reference_world`` represents the chosen zero-pose nail-to-pulp
    direction for the input model.  It is projected perpendicular to the
    longitudinal axis so the two GUI controls remain independent.
    """
    if finger not in FINGERS:
        raise ValueError(f"unsupported finger: {finger}")
    prefix = f"{side}_" if side else ""
    joint3 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{finger}_joint3"
    )
    joint4 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{finger}_joint4"
    )
    if min(joint3, joint4) < 0:
        raise ValueError(f"missing joints for {prefix}{finger}")
    axis_world = _unit(data.xanchor[joint4] - data.xanchor[joint3], name="PIP->DIP")
    body_id = model.jnt_bodyid[joint4]
    world_from_link4 = data.xmat[body_id].reshape(3, 3)
    axis_local = world_from_link4.T @ axis_world
    axis_local = _unit(axis_local, name="local PIP->DIP")
    # The thumb's distal-link X axis spans its width.  Projecting the palm
    # normal there selects that lateral direction, not nail -> pulp.  Its
    # local Z axis is the authored nail/pulp thickness reference.
    if finger == "thumb":
        surface_local = np.array((0.0, 0.0, 1.0))
    else:
        surface_local = world_from_link4.T @ np.asarray(
            surface_reference_world, dtype=np.float64
        )
    surface_local = surface_local - axis_local * np.dot(surface_local, axis_local)
    return axis_local, _unit(surface_local, name="local nail-to-pulp")
