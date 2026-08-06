"""Minimal MuJoCo scene used to inspect MANO and LDJY registration."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
from typing import Callable

import mujoco
import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


def apply_registration(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Apply the static MANO-to-LDJY registration, independently of pose."""
    return float(scale) * np.asarray(points) @ Rotation.from_rotvec(rotation).as_matrix().T + translation


def average_surface_normal(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_ids: np.ndarray | list[int],
) -> np.ndarray:
    """Average mesh-face normals adjacent to the supplied surface vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
    face_vertices = vertices[faces]
    face_normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.where(lengths > 1e-12, lengths, 1.0)
    adjacent = np.isin(faces, vertex_ids).any(axis=1)
    if not np.any(adjacent):
        raise ValueError("surface vertices have no adjacent faces")
    normal = face_normals[adjacent].mean(axis=0)
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        raise ValueError("surface normal must be non-zero")
    return normal / length


def fit_static_registration(
    sample: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    target_points: np.ndarray,
    target_normals: np.ndarray,
    *,
    beta_count: int = 10,
    pose_sample: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    direction_sample: Callable[[np.ndarray], np.ndarray] | None = None,
    direction_pose_sample: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    target_directions: np.ndarray | None = None,
    direction_start_sample: Callable[[np.ndarray], np.ndarray] | None = None,
    direction_start_pose_sample: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    target_direction_starts: np.ndarray | None = None,
    palm_normal_sample: Callable[[np.ndarray], np.ndarray] | None = None,
    palm_normal_pose_sample: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    target_palm_normal: np.ndarray | None = None,
    joint_pose_sample: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    use_positions: bool = True,
    position_weight: float = 1.0,
    position_mask: np.ndarray | None = None,
    use_normals: bool = True,
    normal_weight: float = 0.5,
    normal_mask: np.ndarray | None = None,
    use_directions: bool = True,
    direction_weight: float = 0.5,
    direction_mask: np.ndarray | None = None,
    use_direction_lines: bool = False,
    direction_line_mask: np.ndarray | None = None,
    use_palm_normal: bool = True,
    palm_normal_weight: float = 0.5,
    use_straight_fingers: bool = False,
    straight_fingers_weight: float = 0.5,
    straight_finger_mask: np.ndarray | None = None,
    use_palm_plane: bool = False,
    palm_plane_weight: float = 0.5,
    use_hand_pose_prior: bool = False,
    hand_pose_prior_weight: float = 0.1,
    fit_betas: bool = True,
    fit_hand_joints: np.ndarray | None = None,
    fit_rotation: bool = True,
    fit_translation: bool = True,
    fit_scale: bool = True,
    initial_betas: np.ndarray | None = None,
    initial_hand_pose: np.ndarray | None = None,
    hand_pose_reference: np.ndarray | None = None,
    initial_rotation: np.ndarray | None = None,
    initial_translation: np.ndarray | None = None,
    initial_scale: float | None = None,
    beta_bound: float = 3.0,
    max_nfev: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit shape, selected MANO axis-angle joints, and a static similarity transform."""
    target_points = np.asarray(target_points, dtype=np.float64)
    if target_points.shape != (5, 3):
        raise ValueError("static registration requires five pad points")
    if beta_count <= 0:
        raise ValueError("beta_count must be positive")
    if beta_bound <= 0.0:
        raise ValueError("beta_bound must be positive")
    position_mask = _five_mask(position_mask, "position_mask")
    normal_mask = _five_mask(normal_mask, "normal_mask")
    direction_mask = _five_mask(direction_mask, "direction_mask")
    direction_line_mask = _five_mask(direction_line_mask, "direction_line_mask")
    straight_finger_mask = _five_mask(straight_finger_mask, "straight_finger_mask")
    if use_normals:
        target_normals = _normalize_rows(np.asarray(target_normals, dtype=np.float64))
        if target_normals.shape != (5, 3):
            raise ValueError("static registration requires five pad normals")
    direction_source = direction_pose_sample or direction_sample
    if (direction_source is None) != (target_directions is None):
        raise ValueError("a direction sample and target_directions must be provided together")
    if use_directions and direction_source is not None:
        target_directions = _normalize_rows(np.asarray(target_directions, dtype=np.float64))
        if target_directions.shape != (5, 3):
            raise ValueError("static registration requires five finger directions")
    use_positions = use_positions and np.any(position_mask)
    use_normals = use_normals and np.any(normal_mask)
    use_directions = use_directions and direction_source is not None and np.any(direction_mask)
    direction_start_source = direction_start_pose_sample or direction_start_sample
    if use_direction_lines and (direction_start_source is None or target_direction_starts is None):
        raise ValueError("direction starts are required for finger-line alignment")
    if use_direction_lines:
        target_direction_starts = np.asarray(target_direction_starts, dtype=np.float64)
        if target_direction_starts.shape != (5, 3):
            raise ValueError("finger-line alignment requires five target direction starts")
    direction_line_mask &= direction_mask
    use_direction_lines = use_direction_lines and use_directions and np.any(direction_line_mask)
    palm_normal_source = palm_normal_pose_sample or palm_normal_sample
    if (palm_normal_source is None) != (target_palm_normal is None):
        raise ValueError("a palm-normal sample and target_palm_normal must be provided together")
    if use_palm_normal and palm_normal_source is not None:
        target_palm_normal = _normalize_rows(
            np.asarray(target_palm_normal, dtype=np.float64).reshape(1, 3)
        )[0]
    use_palm_normal = use_palm_normal and palm_normal_source is not None
    use_straight_fingers = use_straight_fingers and np.any(straight_finger_mask)
    if (use_straight_fingers or use_palm_plane) and joint_pose_sample is None:
        raise ValueError("joint_pose_sample is required for MANO geometry priors")

    hand_pose = _initial_hand_pose(initial_hand_pose)
    hand_pose_reference = _initial_hand_pose(
        hand_pose if hand_pose_reference is None else hand_pose_reference
    )
    fit_hand_joints = _hand_joint_mask(fit_hand_joints)
    if np.any(fit_hand_joints) and pose_sample is None:
        raise ValueError("pose_sample is required when optimizing hand_pose")
    enabled_weights = (
        (use_positions, position_weight),
        (use_normals, normal_weight),
        (use_directions, direction_weight),
        (use_palm_normal, palm_normal_weight),
        (use_straight_fingers, straight_fingers_weight),
        (use_palm_plane, palm_plane_weight),
        (use_hand_pose_prior and np.any(fit_hand_joints), hand_pose_prior_weight),
    )
    if not any(enabled for enabled, _ in enabled_weights):
        raise ValueError("select at least one fit constraint")
    if any(weight <= 0.0 for enabled, weight in enabled_weights if enabled):
        raise ValueError("enabled fit weights must be positive")

    betas = _initial_vector(initial_betas, beta_count, 0.0)
    rotation = _initial_vector(initial_rotation, 3, 0.0)
    translation = _initial_vector(initial_translation, 3, 0.0)
    scale = 1.0 if initial_scale is None else float(initial_scale)
    def source_sample(current_betas, current_hand_pose):
        if pose_sample is not None:
            return pose_sample(current_betas, current_hand_pose)
        return sample(current_betas)

    def source_directions(current_betas, current_hand_pose):
        if direction_pose_sample is not None:
            return direction_pose_sample(current_betas, current_hand_pose)
        return direction_sample(current_betas)

    def source_direction_starts(current_betas, current_hand_pose):
        if direction_start_pose_sample is not None:
            return direction_start_pose_sample(current_betas, current_hand_pose)
        return direction_start_sample(current_betas)

    def source_palm_normal(current_betas, current_hand_pose):
        if palm_normal_pose_sample is not None:
            return palm_normal_pose_sample(current_betas, current_hand_pose)
        return palm_normal_sample(current_betas)

    source_points, _ = source_sample(betas, hand_pose)
    if use_positions:
        estimated_rotation, estimated_translation, estimated_scale = _similarity_initialization(
            source_points, target_points
        )
        if fit_rotation:
            rotation = estimated_rotation
        if fit_translation:
            translation = estimated_translation
        if fit_scale:
            scale = estimated_scale
    pose_start = beta_count
    pose_end = pose_start + 45
    rotation_start = pose_end
    translation_start = rotation_start + 3
    initial = np.concatenate((betas, hand_pose.ravel(), rotation, translation, [scale]))
    lower = np.concatenate((
        np.full(beta_count, -beta_bound), np.full(45, -np.pi), np.full(3, -np.pi), np.full(3, -0.2), [0.5]
    ))
    upper = np.concatenate((
        np.full(beta_count, beta_bound), np.full(45, np.pi), np.full(3, np.pi), np.full(3, 0.2), [2.0]
    ))
    active = np.concatenate((
        np.full(beta_count, fit_betas),
        np.repeat(fit_hand_joints, 3),
        np.full(3, fit_rotation),
        np.full(3, fit_translation),
        [fit_scale],
    ))
    active_indices = np.flatnonzero(active)

    def residual(active_parameters):
        parameters = initial.copy()
        parameters[active_indices] = active_parameters
        betas = parameters[:beta_count]
        hand_pose = parameters[pose_start:pose_end].reshape(15, 3)
        source_points, source_normals = source_sample(betas, hand_pose)
        matrix = Rotation.from_rotvec(parameters[rotation_start:translation_start]).as_matrix()
        registered_points = parameters[-1] * np.asarray(source_points) @ matrix.T + parameters[translation_start:translation_start + 3]
        residuals = []
        if use_positions:
            residuals.append((position_weight * (registered_points[position_mask] - target_points[position_mask]) / 0.005).ravel())
        if use_normals:
            registered_normals = _normalize_rows(np.asarray(source_normals)) @ matrix.T
            residuals.append((normal_weight * (registered_normals[normal_mask] - target_normals[normal_mask])).ravel())
        if use_directions:
            directions = _normalize_rows(np.asarray(source_directions(betas, hand_pose)))
            residuals.append((direction_weight * (directions @ matrix.T - target_directions)[direction_mask]).ravel())
        if use_direction_lines:
            registered_starts = apply_registration(
                source_direction_starts(betas, hand_pose),
                parameters[rotation_start:translation_start],
                parameters[translation_start:translation_start + 3],
                parameters[-1],
            )
            offsets = registered_starts[direction_line_mask] - target_direction_starts[direction_line_mask]
            line_directions = target_directions[direction_line_mask]
            perpendicular_offsets = offsets - np.sum(
                offsets * line_directions, axis=1, keepdims=True
            ) * line_directions
            residuals.append((direction_weight * perpendicular_offsets / 0.005).ravel())
        if use_palm_normal:
            palm_normal = _normalize_rows(
                np.asarray(source_palm_normal(betas, hand_pose), dtype=np.float64).reshape(1, 3)
            )
            residuals.append(
                (palm_normal_weight * (palm_normal @ matrix.T - target_palm_normal)).ravel()
            )
        if use_straight_fingers or use_palm_plane:
            registered_joints = apply_registration(
                joint_pose_sample(betas, hand_pose),
                parameters[rotation_start:translation_start],
                parameters[translation_start:translation_start + 3],
                parameters[-1],
            )
            if use_straight_fingers:
                residuals.append(
                    straight_fingers_weight * _straight_finger_residual(registered_joints, straight_finger_mask)
                )
            if use_palm_plane:
                residuals.append(palm_plane_weight * _finger_plane_residual(registered_joints))
        if use_hand_pose_prior and np.any(fit_hand_joints):
            residuals.append(
                hand_pose_prior_weight * (hand_pose[fit_hand_joints] - hand_pose_reference[fit_hand_joints]).ravel()
            )
        return np.concatenate(residuals)

    if len(active_indices):
        # MANO inference is float32.  A zero beta needs a visible perturbation
        # or scipy's numerical derivative is quantized away.
        x0 = initial[active_indices].copy()
        beta_active = active_indices < beta_count
        x0 = np.clip(x0, lower[active_indices] + 1e-9, upper[active_indices] - 1e-9)
        x0[beta_active & (np.abs(x0) < 0.05)] = 0.1
        diff_step = np.where(beta_active, 0.1, 1e-5)
        result = least_squares(
            residual,
            x0,
            bounds=(lower[active_indices], upper[active_indices]),
            max_nfev=max_nfev,
            diff_step=diff_step,
        )
        initial[active_indices] = result.x
    betas = initial[:beta_count]
    hand_pose = initial[pose_start:pose_end].reshape(15, 3)
    rotation = initial[rotation_start:translation_start]
    translation = initial[translation_start:translation_start + 3]
    scale = float(initial[-1])
    source_points, _ = source_sample(betas, hand_pose)
    position_rms = float(np.sqrt(np.mean((apply_registration(source_points, rotation, translation, scale) - target_points) ** 2)))
    return betas, hand_pose, rotation, translation, scale, position_rms


def solve_exact_hand_pose_constraints(
    joint_pose_sample: Callable[[np.ndarray, np.ndarray], np.ndarray],
    betas: np.ndarray,
    initial_hand_pose: np.ndarray,
    fit_hand_joints: np.ndarray,
    *,
    hand_pose_reference: np.ndarray,
    use_straight_fingers: bool = False,
    straight_finger_mask: np.ndarray | None = None,
    use_palm_plane: bool = False,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Apply strict palm-plane and sequential no-twist finger straightening constraints."""
    initial_hand_pose = _initial_hand_pose(initial_hand_pose)
    hand_pose_reference = _initial_hand_pose(hand_pose_reference)
    fit_hand_joints = _hand_joint_mask(fit_hand_joints)
    straight_finger_mask = _five_mask(straight_finger_mask, "straight_finger_mask")
    use_straight_fingers = use_straight_fingers and np.any(straight_finger_mask)
    if not (use_straight_fingers or use_palm_plane):
        return initial_hand_pose
    pose = initial_hand_pose.copy()
    if use_palm_plane:
        pose = _align_finger_bases_to_palm_plane(
            joint_pose_sample, betas, pose, hand_pose_reference, fit_hand_joints, tolerance
        )
    if use_straight_fingers:
        pose = _straighten_fingers_sequentially(
            joint_pose_sample, betas, pose, hand_pose_reference, fit_hand_joints, straight_finger_mask, tolerance
        )

    joints = joint_pose_sample(np.asarray(betas, dtype=np.float64), pose)
    residuals = []
    if use_straight_fingers:
        residuals.append(_straight_finger_residual(joints, straight_finger_mask))
    if use_palm_plane:
        residuals.append(_finger_plane_residual(joints))
    maximum_error = float(np.max(np.abs(np.concatenate(residuals))))
    if maximum_error > tolerance:
        raise ValueError(f"exact hand-pose constraints are infeasible (max residual {maximum_error:.3g})")
    return pose


# Each step changes only the joint that moves its target landmark.  This keeps
# the already-aligned proximal segment fixed instead of redistributing bend over
# the whole finger.
_SEQUENTIAL_STRAIGHTENING_STEPS = (
    ((1,), 1, 5, 6, 7, "Index PIP"),
    ((2,), 1, 5, 6, 8, "Index DIP"),
    ((4,), 2, 9, 10, 11, "Middle PIP"),
    ((5,), 2, 9, 10, 12, "Middle DIP"),
    ((7,), 4, 17, 18, 19, "Pinky PIP"),
    ((8,), 4, 17, 18, 20, "Pinky DIP"),
    ((10,), 3, 13, 14, 15, "Ring PIP"),
    ((11,), 3, 13, 14, 16, "Ring DIP"),
    ((13, 14), 0, 2, 3, 4, "Thumb MCP/IP"),
)
_PALM_PLANE_BASE_STEPS = (
    (0, 5, 6, "Index MCP"),
    (3, 9, 10, "Middle MCP"),
    (6, 17, 18, "Pinky MCP"),
    (9, 13, 14, "Ring MCP"),
)


def _straighten_fingers_sequentially(
    joint_pose_sample: Callable[[np.ndarray, np.ndarray], np.ndarray],
    betas: np.ndarray,
    pose: np.ndarray,
    reference: np.ndarray,
    fit_hand_joints: np.ndarray,
    straight_finger_mask: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    # MANO fingertip points are mesh vertices. Pose blend shapes create weak
    # cross-finger coupling, so repeat the ordered local corrections until the
    # visible 21-point chains themselves are straight.
    for _ in range(8):
        for joint_indices, finger_index, line_start, line_end, target, label in _SEQUENTIAL_STRAIGHTENING_STEPS:
            if not straight_finger_mask[finger_index]:
                continue
            def residual(joints, start=line_start, end=line_end, point=target):
                return _point_line_residual(joints[point], joints[start], joints[end])

            pose = _solve_joint_equality(
                joint_pose_sample, betas, pose, reference, fit_hand_joints, joint_indices, residual, label, tolerance
            )
        joints = joint_pose_sample(np.asarray(betas, dtype=np.float64), pose)
        if float(np.max(np.abs(_straight_finger_residual(joints, straight_finger_mask)))) <= tolerance:
            return pose
    return pose


def _align_finger_bases_to_palm_plane(
    joint_pose_sample: Callable[[np.ndarray, np.ndarray], np.ndarray],
    betas: np.ndarray,
    pose: np.ndarray,
    reference: np.ndarray,
    fit_hand_joints: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    for joint_index, start, end, label in _PALM_PLANE_BASE_STEPS:
        def residual(joints, point_start=start, point_end=end):
            normal = np.cross(joints[5] - joints[0], joints[17] - joints[0])
            length = np.linalg.norm(normal)
            if length <= 1e-12:
                raise ValueError("palm-plane anchors must not be collinear")
            return np.array([np.dot(joints[point_end] - joints[point_start], normal / length) / 0.005])

        pose = _solve_joint_equality(
            joint_pose_sample, betas, pose, reference, fit_hand_joints, (joint_index,), residual, label, tolerance
        )
    return pose


def _solve_joint_equality(
    joint_pose_sample: Callable[[np.ndarray, np.ndarray], np.ndarray],
    betas: np.ndarray,
    pose: np.ndarray,
    reference: np.ndarray,
    fit_hand_joints: np.ndarray,
    joint_indices: tuple[int, ...],
    residual: Callable[[np.ndarray], np.ndarray],
    label: str,
    tolerance: float,
) -> np.ndarray:
    joint_indices = tuple(joint_indices)

    def values(candidate):
        candidate_pose = pose.copy()
        candidate_pose[list(joint_indices)] = candidate.reshape(len(joint_indices), 3)
        joints = joint_pose_sample(np.asarray(betas, dtype=np.float64), candidate_pose)
        return np.asarray(residual(joints), dtype=np.float64)

    current = pose[list(joint_indices)].ravel()
    current_error = float(np.max(np.abs(values(current))))
    if current_error <= tolerance:
        return pose
    missing = [index for index in joint_indices if not fit_hand_joints[index]]
    if missing:
        raise ValueError(f"cannot satisfy {label}: select hand-pose joint(s) {', '.join(map(str, missing))}")

    reference_rotations = Rotation.from_rotvec(reference[list(joint_indices)])

    def rotation_change(candidate):
        rotations = Rotation.from_rotvec(candidate.reshape(len(joint_indices), 3))
        relative = rotations * reference_rotations.inv()
        changes = relative.as_rotvec()
        return 0.5 * np.sum(changes * changes)

    starts = [current, reference[list(joint_indices)].ravel(), np.zeros_like(current)]
    if len(joint_indices) == 1:
        for axis in range(3):
            for angle in (-np.pi / 2, np.pi / 2):
                start = np.zeros_like(current)
                start[axis] = angle
                starts.append(start)
    feasible_candidates = []
    best_error = np.inf
    for start in starts:
        feasible = least_squares(
            values,
            np.clip(start, -np.pi, np.pi),
            bounds=(-np.pi, np.pi),
            diff_step=0.01,
            max_nfev=100,
        )
        feasible_error = float(np.max(np.abs(values(feasible.x))))
        best_error = min(best_error, feasible_error)
        if feasible_error <= tolerance:
            feasible_candidates.append(feasible.x)
    if not feasible_candidates:
        raise ValueError(f"cannot satisfy {label} (max residual {best_error:.3g})")
    feasible_values = min(feasible_candidates, key=rotation_change)
    constraint_rows = _independent_constraint_rows(values, feasible_values)
    if len(constraint_rows) == 0:
        return pose

    result = minimize(
        rotation_change,
        feasible_values,
        method="SLSQP",
        bounds=[(-np.pi + 1e-9, np.pi - 1e-9)] * len(current),
        constraints=[{"type": "eq", "fun": lambda candidate: values(candidate)[constraint_rows]}],
        options={"ftol": tolerance, "maxiter": 300, "eps": 1e-4},
    )
    maximum_error = float(np.max(np.abs(values(result.x))))
    if not result.success or maximum_error > tolerance:
        result_values = feasible_values
    else:
        result_values = result.x
    result_pose = pose.copy()
    result_pose[list(joint_indices)] = result_values.reshape(len(joint_indices), 3)
    return result_pose


def _point_line_residual(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> np.ndarray:
    direction = np.asarray(line_end, dtype=np.float64) - np.asarray(line_start, dtype=np.float64)
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError("finger baseline endpoints must be distinct")
    direction /= length
    tangent = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tangent, direction)) > 0.9:
        tangent = np.array([0.0, 1.0, 0.0])
    normal_a = np.cross(direction, tangent)
    normal_a /= np.linalg.norm(normal_a)
    normal_b = np.cross(direction, normal_a)
    offset = np.asarray(point, dtype=np.float64) - line_start
    return np.array([np.dot(offset, normal_a), np.dot(offset, normal_b)]) / 0.005


def _independent_constraint_rows(constraint_values: Callable[[np.ndarray], np.ndarray], values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    baseline = constraint_values(values)
    jacobian = np.empty((len(baseline), len(values)))
    for index in range(len(values)):
        shifted = values.copy()
        shifted[index] += 1e-3
        jacobian[:, index] = (constraint_values(shifted) - baseline) / 1e-3
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] <= 1e-10:
        return np.empty(0, dtype=np.intp)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-7))
    _, _, pivots = qr(jacobian.T, pivoting=True, mode="economic")
    return np.sort(pivots[:rank])


def _initial_vector(values: np.ndarray | None, size: int, default: float) -> np.ndarray:
    if values is None:
        return np.full(size, default, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (size,):
        raise ValueError(f"expected initial value shape {(size,)}, got {values.shape}")
    return values.copy()


def _initial_hand_pose(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.zeros((15, 3), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (15, 3):
        raise ValueError(f"expected hand_pose shape (15, 3), got {values.shape}")
    return values.copy()


def _hand_joint_mask(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.zeros(15, dtype=bool)
    values = np.asarray(values, dtype=bool)
    if values.shape != (15,):
        raise ValueError(f"expected hand-joint mask shape (15,), got {values.shape}")
    return values.copy()


def _five_mask(values: np.ndarray | None, name: str) -> np.ndarray:
    if values is None:
        return np.ones(5, dtype=bool)
    values = np.asarray(values, dtype=bool)
    if values.shape != (5,):
        raise ValueError(f"expected {name} shape (5,), got {values.shape}")
    return values.copy()


_STRAIGHT_FINGER_CHAINS = (
    (2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


def _straight_finger_residual(joints: np.ndarray, finger_mask: np.ndarray | None = None) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    finger_mask = _five_mask(finger_mask, "finger_mask")
    residuals = []
    for enabled, chain in zip(finger_mask, _STRAIGHT_FINGER_CHAINS):
        if not enabled:
            continue
        start = joints[chain[0]]
        direction = joints[chain[-1]] - start
        length = np.linalg.norm(direction)
        if length <= 1e-12:
            raise ValueError("finger endpoints must be distinct")
        for index in chain[1:-1]:
            residuals.append(np.cross(joints[index] - start, direction) / length / 0.005)
    return np.concatenate(residuals)


def _finger_plane_residual(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    origin = joints[0]
    normal = np.cross(joints[5] - origin, joints[17] - origin)
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        raise ValueError("palm-plane anchors must not be collinear")
    normal /= length
    residuals = []
    for start, end in ((5, 8), (9, 12), (13, 16), (17, 20)):
        direction = joints[end] - joints[start]
        direction_length = np.linalg.norm(direction)
        if direction_length <= 1e-12:
            raise ValueError("finger endpoints must be distinct")
        residuals.append(np.dot(direction / direction_length, normal))
    return np.asarray(residuals)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("vectors must be non-zero")
    return vectors / lengths


def _similarity_initialization(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    left, _, right = np.linalg.svd(centered_source.T @ centered_target)
    matrix = right.T @ left.T
    if np.linalg.det(matrix) < 0:
        right[-1] *= -1
        matrix = right.T @ left.T
    denominator = np.sum(centered_source ** 2)
    scale = 1.0 if denominator <= 1e-12 else np.sum(centered_target * (centered_source @ matrix.T)) / denominator
    translation = target_center - scale * source_center @ matrix.T
    return Rotation.from_matrix(matrix).as_rotvec(), translation, float(scale)


class OverlayScene:
    """An LDJY model with one non-colliding, dynamically updated MANO mesh."""

    def __init__(
        self,
        ldjy_mjcf_path: str | Path,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        self._ldjy_mjcf_path = Path(ldjy_mjcf_path).resolve()
        self._faces = np.asarray(faces, dtype=np.int32)
        vertices = np.asarray(vertices, dtype=np.float64)
        self._validate_mesh(vertices, self._faces)

        # Keep the generated OBJ next to its temporary MJCF: source asset paths
        # in the LDJY MJCF stay valid while MuJoCo compiles the combined scene.
        self._tmpdir = Path(tempfile.mkdtemp(prefix="mano_ldjy_overlay_"))
        self._obj_path = self._tmpdir / "mano_overlay.obj"
        self._write_obj(vertices, self._faces)

        fd, temporary_xml = tempfile.mkstemp(
            prefix="mano_ldjy_overlay_", suffix=".xml", dir=self._ldjy_mjcf_path.parent
        )
        os.close(fd)
        self._xml_path = Path(temporary_xml)
        self._write_combined_mjcf()
        try:
            self.model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        except Exception:
            self.close()
            raise
        finally:
            self._xml_path.unlink(missing_ok=True)
        self.data = mujoco.MjData(self.model)

        self.mesh_id = self._object_id(mujoco.mjtObj.mjOBJ_MESH, "mano_overlay_mesh")
        self.geom_id = self._object_id(mujoco.mjtObj.mjOBJ_GEOM, "mano_overlay_geom")
        self._vertadr = self.model.mesh_vertadr[self.mesh_id]
        self._vertnum = self.model.mesh_vertnum[self.mesh_id]
        self._mesh_pos = self.model.mesh_pos[self.mesh_id].copy()
        self._mesh_scale_inv = np.reciprocal(
            self.model.mesh_scale[self.mesh_id],
            where=np.abs(self.model.mesh_scale[self.mesh_id]) > 1e-10,
        )
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, self.model.mesh_quat[self.mesh_id])
        self._mesh_R = matrix.reshape(3, 3)
        self.update_mano_mesh(vertices)

    def close(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def update_mano_mesh(self, vertices: np.ndarray) -> None:
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.shape != (self._vertnum, 3):
            raise ValueError(f"expected MANO vertices with shape {(self._vertnum, 3)}, got {vertices.shape}")
        stored_vertices = (vertices - self._mesh_pos) * self._mesh_scale_inv @ self._mesh_R
        end = self._vertadr + self._vertnum
        self.model.mesh_vert[self._vertadr:end] = stored_vertices.astype(np.float32)

        face_vertices = stored_vertices[self._faces]
        normals = np.cross(
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 0],
        )
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(lengths > 1e-12, lengths, 1.0)
        vertex_normals = np.zeros_like(stored_vertices)
        np.add.at(vertex_normals, self._faces.reshape(-1), np.repeat(normals, 3, axis=0))
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals /= np.where(lengths > 1e-12, lengths, 1.0)
        self.model.mesh_normal[self._vertadr:end] = vertex_normals.astype(np.float32)
        mujoco.mj_forward(self.model, self.data)

    def set_ldjy_alpha(self, alpha: float) -> None:
        """Make the robot visual meshes translucent without touching collisions."""
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name and name.endswith("_visual_mesh_collision"):
                self.model.geom_rgba[geom_id, 3] = alpha

    def _object_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"combined scene is missing {name!r}")
        return object_id

    @staticmethod
    def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("MANO vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("MANO faces must have shape (M, 3)")
        if len(vertices) == 0 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("MANO faces must reference the supplied vertices")

    def _write_obj(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        with self._obj_path.open("w", encoding="ascii") as handle:
            for vertex in vertices:
                handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
            for face in faces:
                handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    def _write_combined_mjcf(self) -> None:
        tree = ET.parse(self._ldjy_mjcf_path)
        root = tree.getroot()
        asset = root.find("asset")
        worldbody = root.find("worldbody")
        if asset is None or worldbody is None:
            raise ValueError("LDJY MJCF must contain asset and worldbody elements")
        ET.SubElement(asset, "mesh", name="mano_overlay_mesh", file=str(self._obj_path))
        body = ET.SubElement(worldbody, "body", name="mano_overlay")
        ET.SubElement(
            body,
            "geom",
            name="mano_overlay_geom",
            type="mesh",
            mesh="mano_overlay_mesh",
            rgba="0.25 0.75 1 0.45",
            contype="0",
            conaffinity="0",
            mass="0",
        )
        tree.write(self._xml_path, encoding="utf-8", xml_declaration=True)
