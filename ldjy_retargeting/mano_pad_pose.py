"""MANO pad targets for the absolute WiLoR hand-pose retargeting mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
import trimesh

from .mano_ldjy_overlay import average_surface_normal, mirror_registration_for_left


MANO_LEFT_FROM_RIGHT = np.diag((1.0, -1.0, 1.0))
DISTAL_MANO_JOINTS = np.array((15, 3, 6, 12, 9), dtype=np.intp)


def distal_triangle_indices(faces: np.ndarray, lbs_weights: np.ndarray) -> tuple[np.ndarray, ...]:
    """Assign each MANO triangle to one distal segment by dominant skinning joint."""
    faces = np.asarray(faces, dtype=np.intp)
    weights = np.asarray(lbs_weights, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3 or weights.ndim != 2:
        raise ValueError("MANO faces must be (n, 3) and skinning weights must be (n, joints)")
    if np.any(faces < 0) or np.any(faces >= len(weights)):
        raise ValueError("MANO faces reference an invalid vertex")
    dominant_joints = np.argmax(weights, axis=1)
    return tuple(
        np.flatnonzero(np.count_nonzero(dominant_joints[faces] == joint, axis=1) >= 2)
        for joint in DISTAL_MANO_JOINTS
    )


def _segment_segment_distance_squared(first_start, first_end, second_start, second_end) -> np.ndarray:
    """Return all pairwise squared distances between two broadcastable segment arrays."""
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = first_start - second_start
    first_length = np.einsum("...i,...i->...", first_direction, first_direction)
    cross_length = np.einsum("...i,...i->...", first_direction, second_direction)
    second_length = np.einsum("...i,...i->...", second_direction, second_direction)
    first_offset = np.einsum("...i,...i->...", first_direction, offset)
    second_offset = np.einsum("...i,...i->...", second_direction, offset)
    determinant = first_length * second_length - cross_length * cross_length
    epsilon = 1e-12

    first_numerator = cross_length * second_offset - second_length * first_offset
    second_numerator = first_length * second_offset - cross_length * first_offset
    first_denominator = determinant.copy()
    second_denominator = determinant.copy()
    parallel = determinant <= epsilon
    first_numerator = np.where(parallel, 0.0, first_numerator)
    first_denominator = np.where(parallel, 1.0, first_denominator)
    second_numerator = np.where(parallel, second_offset, second_numerator)
    second_denominator = np.where(parallel, second_length, second_denominator)

    below = first_numerator < 0.0
    above = first_numerator > first_denominator
    first_numerator = np.where(below, 0.0, np.where(above, first_denominator, first_numerator))
    second_numerator = np.where(below, second_offset, np.where(above, second_offset + cross_length, second_numerator))
    second_denominator = np.where(below | above, second_length, second_denominator)

    below = second_numerator < 0.0
    above = second_numerator > second_denominator
    second_numerator = np.where(below, 0.0, np.where(above, second_denominator, second_numerator))
    first_numerator = np.where(
        below,
        np.where(-first_offset < 0.0, 0.0, np.where(-first_offset > first_length, first_denominator, -first_offset)),
        first_numerator,
    )
    first_denominator = np.where(below, np.where((-first_offset >= 0.0) & (-first_offset <= first_length), first_length, first_denominator), first_denominator)
    edge_offset = -first_offset + cross_length
    first_numerator = np.where(
        above,
        np.where(edge_offset < 0.0, 0.0, np.where(edge_offset > first_length, first_denominator, edge_offset)),
        first_numerator,
    )
    first_denominator = np.where(above, np.where((edge_offset >= 0.0) & (edge_offset <= first_length), first_length, first_denominator), first_denominator)

    first_parameter = np.divide(first_numerator, first_denominator, out=np.zeros_like(first_numerator), where=np.abs(first_numerator) > epsilon)
    second_parameter = np.divide(second_numerator, second_denominator, out=np.zeros_like(second_numerator), where=np.abs(second_numerator) > epsilon)
    separation = offset + first_parameter[..., None] * first_direction - second_parameter[..., None] * second_direction
    return np.einsum("...i,...i->...", separation, separation)


def triangle_mesh_distance(first_triangles: np.ndarray, second_triangles: np.ndarray) -> float:
    """Return the exact minimum distance between two small triangle surface sets."""
    first = np.asarray(first_triangles, dtype=np.float64)
    second = np.asarray(second_triangles, dtype=np.float64)
    if first.ndim != 3 or first.shape[1:] != (3, 3) or second.ndim != 3 or second.shape[1:] != (3, 3):
        raise ValueError("triangle meshes must have shape (n, 3, 3)")
    if not len(first) or not len(second):
        raise ValueError("triangle meshes must be nonempty")

    # Vertex distance is a cheap finite upper bound. AABB pairs farther than it
    # cannot contain the closest surface pair, so skip their exact tests.
    vertex_delta = first[:, None, :, None, :] - second[None, :, None, :, :]
    distance_squared = float(np.min(np.sum(vertex_delta * vertex_delta, axis=-1)))
    first_min, first_max = first.min(axis=1), first.max(axis=1)
    second_min, second_max = second.min(axis=1), second.max(axis=1)
    gap = np.maximum(
        np.maximum(first_min[:, None] - second_max[None], second_min[None] - first_max[:, None]),
        0.0,
    )
    candidates = np.flatnonzero(np.sum(gap * gap, axis=-1) <= distance_squared)
    first_indices, second_indices = np.unravel_index(candidates, (len(first), len(second)))
    first = first[first_indices]
    second = second[second_indices]
    for vertex in range(3):
        points = first[:, vertex]
        triangles = second
        closest = trimesh.triangles.closest_point(triangles, points)
        distance_squared = min(distance_squared, float(np.min(np.sum((points - closest) ** 2, axis=1))))
        points = second[:, vertex]
        triangles = first
        closest = trimesh.triangles.closest_point(triangles, points)
        distance_squared = min(distance_squared, float(np.min(np.sum((points - closest) ** 2, axis=1))))
    for first_edge in range(3):
        for second_edge in range(3):
            squared = _segment_segment_distance_squared(
                first[:, first_edge], first[:, (first_edge + 1) % 3],
                second[:, second_edge], second[:, (second_edge + 1) % 3],
            )
            distance_squared = min(distance_squared, float(np.min(squared)))
    return float(np.sqrt(max(distance_squared, 0.0)))


@dataclass(frozen=True)
class ManoPoseInput:
    hand_pose: np.ndarray
    global_orient: np.ndarray
    translation: np.ndarray
    betas: np.ndarray


@dataclass(frozen=True)
class ManoPadTarget:
    pad_positions: np.ndarray
    pad_normals: np.ndarray
    pinch_distances_m: np.ndarray
    mano_joints: np.ndarray
    mano_vertices: np.ndarray


def _finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _project_so3(matrices: np.ndarray) -> np.ndarray:
    projected = np.empty_like(matrices, dtype=np.float64)
    for index, matrix in enumerate(matrices):
        left, _, right = np.linalg.svd(matrix)
        if np.linalg.det(left @ right) < 0.0:
            left[:, -1] *= -1.0
        projected[index] = left @ right
    return projected


class ManoPadTargetBuilder:
    """Convert absolute WiLoR local pose parameters to registered pad targets."""

    def __init__(
        self,
        mano_model,
        reference_path: str | Path,
        hand_side: str,
        *,
        pad_vertex_ids: Mapping[str, int],
        pad_3pt_vertex_ids: Mapping[str, Sequence[int]],
        pad_order: Sequence[str],
    ) -> None:
        self.mano_model = mano_model
        self.reference_path = Path(reference_path).resolve()
        self.hand_side = hand_side.lower()
        if self.hand_side not in {"left", "right"}:
            raise ValueError(f"unsupported hand side: {hand_side}")
        self.pad_vertex_ids = dict(pad_vertex_ids)
        self.pad_3pt_vertex_ids = {name: tuple(values) for name, values in pad_3pt_vertex_ids.items()}
        self.pad_order = tuple(pad_order)
        if len(self.pad_order) != 5:
            raise ValueError("pad_order must contain five fingers")
        weights = getattr(self.mano_model, "lbs_weights", None)
        self.distal_face_indices = (
            distal_triangle_indices(self.mano_model.faces, weights)
            if weights is not None else None
        )
        self._load_reference()

    def _load_reference(self) -> None:
        source_path = self.reference_path
        mirrored_from_right = False
        if self.hand_side == "left":
            left_path = source_path.with_name("mano_ldjy_left_reference.yaml")
            if left_path.exists():
                source_path = left_path
            else:
                mirrored_from_right = True
        if not source_path.exists():
            raise FileNotFoundError(f"MANO-LDJY reference not found: {source_path}")
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        mano = payload.get("mano", {})
        registration = payload.get("mano_to_ldjy_registration", {})
        self.beta_robot = _finite_array(mano.get("betas"), (10,), "reference betas")
        self.mano_scale = float(mano.get("scale", 1.0))
        rotation = _finite_array(
            registration.get("rotation_axis_angle"), (3,), "registration rotation"
        )
        translation = _finite_array(
            registration.get("translation"), (3,), "registration translation"
        )
        scale = float(registration.get("scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("registration scale must be positive and finite")
        if mirrored_from_right:
            rotation, translation, scale = mirror_registration_for_left(
                rotation, translation, scale
            )
        self.registration_rotation = rotation
        self.registration_translation = translation
        self.registration_scale = scale

    def build(self, pose_input: ManoPoseInput) -> ManoPadTarget:
        hand_pose = _finite_array(pose_input.hand_pose, (15, 3, 3), "hand_pose")
        _finite_array(pose_input.global_orient, (3, 3), "global_orient")
        _finite_array(pose_input.translation, (3,), "translation")
        _finite_array(pose_input.betas, (10,), "betas")
        hand_pose = _project_so3(hand_pose)
        hand_pose_axis_angle = Rotation.from_matrix(hand_pose).as_rotvec()
        vertices, joints, _ = self.mano_model.compute(
            self.beta_robot,
            hand_pose_axis_angle,
            np.zeros(3),
            np.zeros(3),
            self.mano_scale,
        )
        video_vertices, video_joints, _ = self.mano_model.compute(
            pose_input.betas,
            hand_pose_axis_angle,
            np.zeros(3),
            np.zeros(3),
            self.mano_scale,
        )
        vertices = np.asarray(vertices, dtype=np.float64)
        joints = np.asarray(joints, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("MANO vertices must be finite with shape (n, 3)")
        if joints.shape != (21, 3) or not np.isfinite(joints).all():
            raise ValueError("MANO joints must be finite with shape (21, 3)")

        pad_points = []
        pad_normals = []
        faces = np.asarray(self.mano_model.faces, dtype=np.int64)
        for name in self.pad_order:
            point_ids = self.pad_3pt_vertex_ids.get(name)
            if point_ids is None:
                point_ids = (self.pad_vertex_ids[name],)
            pad_points.append(vertices[list(point_ids)].mean(axis=0))
            pad_normals.append(average_surface_normal(vertices, faces, point_ids))
        pad_points = np.asarray(pad_points)
        pad_normals = np.asarray(pad_normals)
        video_pad_points = []
        for name in self.pad_order:
            point_ids = self.pad_3pt_vertex_ids.get(name)
            if point_ids is None:
                point_ids = (self.pad_vertex_ids[name],)
            video_pad_points.append(np.asarray(video_vertices, dtype=np.float64)[list(point_ids)].mean(axis=0))
        video_pad_points = np.asarray(video_pad_points)
        video_joints = np.asarray(video_joints, dtype=np.float64)

        mirror = MANO_LEFT_FROM_RIGHT if self.hand_side == "left" else np.eye(3)
        vertices = vertices @ mirror
        joints = joints @ mirror
        pad_points = pad_points @ mirror
        pad_normals = pad_normals @ mirror
        video_pad_points = video_pad_points @ mirror
        video_joints = video_joints @ mirror

        registration = Rotation.from_rotvec(self.registration_rotation).as_matrix()
        root = joints[0]
        pad_positions = (
            (pad_points - root) @ registration.T * self.registration_scale
        )
        pad_normals = pad_normals @ registration.T
        pad_normals /= np.linalg.norm(pad_normals, axis=1, keepdims=True)
        robot_hand_scale = np.linalg.norm(joints[9] - joints[0])
        video_hand_scale = np.linalg.norm(video_joints[9] - video_joints[0])
        if video_hand_scale <= 1e-12:
            raise ValueError("video MANO hand scale must be nonzero")
        if self.distal_face_indices is None:
            pinch_distances = np.linalg.norm(video_pad_points[1:] - video_pad_points[0], axis=1)
        else:
            video_triangles = [np.asarray(video_vertices)[np.asarray(faces)[indices]] for indices in self.distal_face_indices]
            pinch_distances = np.array([
                triangle_mesh_distance(video_triangles[0], triangles)
                for triangles in video_triangles[1:]
            ])
        pinch_distances *= robot_hand_scale / video_hand_scale * self.registration_scale
        registered_vertices = vertices @ registration.T * self.registration_scale
        registered_vertices += self.registration_translation
        registered_joints = joints @ registration.T * self.registration_scale
        registered_joints += self.registration_translation
        return ManoPadTarget(
            pad_positions, pad_normals, pinch_distances, registered_joints, registered_vertices
        )


__all__ = [
    "ManoPadTarget", "ManoPadTargetBuilder", "ManoPoseInput",
    "distal_triangle_indices", "triangle_mesh_distance",
]
