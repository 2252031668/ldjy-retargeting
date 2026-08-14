"""Shared data preparation for side-by-side WiLoR MANO comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .mano_ldjy_overlay import average_surface_normal, mirror_registration_for_left


WILOR_LEFT_FROM_RIGHT = np.diag((-1.0, 1.0, 1.0))
PAD_ORDER = ("Thumb", "Index", "Middle", "Ring", "Pinky")
PAD_VERTEX_IDS = {
    "Thumb": 763,
    "Index": 355,
    "Middle": 438,
    "Ring": 573,
    "Pinky": 690,
}
PAD_3PT_VERTEX_IDS = {
    "Index": (328, 343, 350),
    "Middle": (438, 455, 439),
}


@dataclass(frozen=True)
class RobotMANOReference:
    beta_robot: np.ndarray
    mano_scale: float
    registration_rotation: np.ndarray
    registration_scale: float

    @property
    def mesh_scale(self) -> float:
        return self.mano_scale * self.registration_scale


@dataclass(frozen=True)
class ComparisonHand:
    vertices: np.ndarray
    joints: np.ndarray
    pads: np.ndarray
    pad_normals: np.ndarray


def load_robot_mano_reference(path: str | Path, hand_side: str) -> RobotMANOReference:
    """Load robot shape and the static MANO-to-LDJY frame calibration."""
    hand_side = hand_side.lower()
    if hand_side not in {"left", "right"}:
        raise ValueError(f"unsupported hand side: {hand_side}")
    source = Path(path).resolve()
    left_source = source.with_name("mano_ldjy_left_reference.yaml")
    mirrored = hand_side == "left" and not left_source.exists()
    if mirrored:
        source = source
    elif hand_side == "left":
        source = left_source
    if not source.exists():
        raise FileNotFoundError(f"MANO-LDJY reference not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    mano = payload.get("mano", {})
    registration = payload.get("mano_to_ldjy_registration", {})
    betas = np.asarray(mano.get("betas"), dtype=np.float64)
    rotation = np.asarray(registration.get("rotation_axis_angle"), dtype=np.float64)
    scale = float(mano.get("scale", 1.0))
    registration_scale = float(registration.get("scale", 1.0))
    if betas.shape != (10,) or not np.isfinite(betas).all():
        raise ValueError("reference betas must have shape (10)")
    if rotation.shape != (3,) or not np.isfinite(rotation).all():
        raise ValueError("reference rotation must have shape (3,)")
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(registration_scale) or registration_scale <= 0:
        raise ValueError("reference scales must be positive and finite")
    if mirrored:
        rotation, _, registration_scale = mirror_registration_for_left(
            rotation, np.zeros(3), registration_scale
        )
    return RobotMANOReference(
        beta_robot=betas,
        mano_scale=scale,
        registration_rotation=rotation,
        registration_scale=registration_scale,
    )


def _axis_angle_pose(value: Any) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape == (15, 3, 3):
        matrices = pose
        axis_angle = Rotation.from_matrix(matrices).as_rotvec()
    elif pose.shape == (15, 3):
        matrices = Rotation.from_rotvec(pose).as_matrix()
        axis_angle = pose
    else:
        raise ValueError("hand_pose must have shape (15, 3, 3) or (15, 3)")
    if not np.isfinite(axis_angle).all():
        raise ValueError("hand_pose must be finite")
    return axis_angle, matrices


def _global_axis_angle(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (3, 3):
        value = Rotation.from_matrix(value).as_rotvec()
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("global_orient must have shape (3, 3) or (3,)")
    return value


def _build_hand(
    mano_model,
    betas: np.ndarray,
    hand_pose: np.ndarray,
    global_orient: np.ndarray,
    reference: RobotMANOReference,
    hand_side: str,
    display_offset: np.ndarray,
) -> ComparisonHand:
    vertices, joints, _ = mano_model.compute(
        betas,
        hand_pose,
        global_orient,
        np.zeros(3, dtype=np.float64),
        reference.mano_scale,
    )
    vertices = np.asarray(vertices, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    if hand_side == "left":
        vertices = vertices @ WILOR_LEFT_FROM_RIGHT
        joints = joints @ WILOR_LEFT_FROM_RIGHT
    return _place_hand(
        vertices,
        joints,
        mano_model.faces,
        reference,
        display_offset,
        flip_normals=hand_side == "left",
    )


def _place_hand(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    reference: RobotMANOReference,
    display_offset: np.ndarray,
    *,
    flip_normals: bool = False,
) -> ComparisonHand:
    vertices = np.asarray(vertices, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    if vertices.shape != (778, 3) or joints.shape != (21, 3):
        raise ValueError("MANO vertices/joints must have shapes (778, 3) and (21, 3)")
    root = joints[0].copy()
    rotation = Rotation.from_rotvec(reference.registration_rotation).as_matrix()
    vertices = (vertices - root) @ rotation.T * reference.registration_scale
    joints = (joints - root) @ rotation.T * reference.registration_scale
    offset = np.asarray(display_offset, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("display_offset must be a finite length-3 vector")
    vertices += offset
    joints += offset

    pads, normals = [], []
    for name in PAD_ORDER:
        point_ids = PAD_3PT_VERTEX_IDS.get(name, (PAD_VERTEX_IDS[name],))
        pads.append(vertices[list(point_ids)].mean(axis=0))
        normal = average_surface_normal(vertices, faces, point_ids)
        # A left WiLoR hand is an X reflection; unchanged triangle winding reverses its normal.
        normals.append(-normal if flip_normals else normal)
    return ComparisonHand(vertices, joints, np.asarray(pads), np.asarray(normals))


def build_comparison_hands(
    mano_model,
    reference_path: str | Path,
    hand_side: str,
    parameters: dict[str, Any],
    human_offset: np.ndarray,
    robot_offset: np.ndarray,
    *,
    video_vertices: np.ndarray | None = None,
    video_joints: np.ndarray | None = None,
) -> tuple[ComparisonHand, ComparisonHand]:
    """Build video-shape and robot-shape MANO hands from one WiLoR frame."""
    reference = load_robot_mano_reference(reference_path, hand_side)
    hand_pose, _ = _axis_angle_pose(parameters["hand_pose"])
    global_orient = _global_axis_angle(parameters.get("global_orient", np.zeros(3)))
    video_betas = np.asarray(parameters["betas"], dtype=np.float64)
    if video_betas.shape != (10,) or not np.isfinite(video_betas).all():
        raise ValueError("video betas must have shape (10)")
    if video_vertices is None or video_joints is None:
        human = _build_hand(
            mano_model, video_betas, hand_pose, global_orient,
            reference, hand_side, human_offset,
        )
    else:
        human = _place_hand(
            video_vertices,
            video_joints,
            mano_model.faces,
            reference,
            human_offset,
            flip_normals=hand_side == "left",
        )
    robot = _build_hand(
        mano_model, reference.beta_robot, hand_pose, global_orient,
        reference, hand_side, robot_offset,
    )
    return human, robot


__all__ = [
    "ComparisonHand",
    "WILOR_LEFT_FROM_RIGHT",
    "RobotMANOReference",
    "build_comparison_hands",
    "load_robot_mano_reference",
]
