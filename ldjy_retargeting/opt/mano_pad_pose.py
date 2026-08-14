"""Independent bounded IK for absolute MANO finger-pad targets."""

from __future__ import annotations

from typing import Optional
import time

import numpy as np
from scipy.optimize import minimize

from .base import BaseOptimizer
from ..mano_pad_pose import ManoPadTarget


FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")
PAD_NORMAL = np.array((0.0, 0.0, 1.0), dtype=np.float64)
PINCH_POSITION_RELEASE_RATIO = 0.30


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    if norm <= 1e-12:
        raise ValueError("normal must be nonzero")
    return value / norm


class ManoPadPoseOptimizer(BaseOptimizer):
    """Solve independent pad IK, or joint thumb-finger IK during a pinch."""

    def __init__(self, config: dict):
        super().__init__(config)
        retarget = config.get("retarget", {})
        settings = retarget.get("pad_ik", {})
        self.position_scale_m = max(float(settings.get("position_scale_m", 0.01)), 1e-6)
        self.normal_scale = max(float(settings.get("normal_scale", 1.0)), 1e-6)
        self.pinch_threshold_m = max(float(settings.get("pinch_threshold_cm", 2.0)), 0.0) / 100.0
        self.pinch_min_distance_m = max(float(settings.get("pinch_min_distance_cm", 0.1)), 0.0) / 100.0
        self.pinch_distance_weight = max(float(settings.get("pinch_distance_weight", 4.0)), 0.0)
        self.thumb_contact_surface_optimization = bool(
            settings.get("thumb_contact_surface_optimization", False)
        )
        self.thumb_contact_surface_weight = max(
            float(settings.get("thumb_contact_surface_weight", 300.0)), 0.0
        )
        self.pinch_position_weight_scale = np.clip(
            float(settings.get("pinch_position_weight_scale", 0.25)), 0.0, 1.0
        )
        self.maxeval = max(int(settings.get("maxeval", 30)), 1)
        self.ftol_rel = max(float(settings.get("ftol_rel", 1e-5)), 0.0)
        self.finger_configs = {
            finger: {
                "enabled": True,
                "use_positions": True,
                "position_weight": 1.0,
                "use_normals": True,
                "normal_weight": 0.2,
                "smooth_weight": 0.05,
                "lp_alpha": 0.2,
                **(settings.get(finger) or {}),
            }
            for finger in FINGERS
        }
        self.pad_link_names = [f"{finger}_pad" for finger in FINGERS]
        self.pad_link_indices = []
        for name in self.pad_link_names:
            try:
                self.pad_link_indices.append(self.robot.get_link_index(name))
            except RuntimeError:
                # Older cached tuning assets used the explicit *_pad_frame name.
                self.pad_link_indices.append(self.robot.get_link_index(f"{name}_frame"))
        self.finger_qpos_indices = [
            np.array(
                [self.robot.get_actuated_qpos_index(f"{finger}_link{link}") for link in range(1, 5)],
                dtype=np.int64,
            )
            for finger in FINGERS
        ]
        self.last_qpos: Optional[np.ndarray] = None
        self.last_diagnostics: dict = {}

    def pinch_state(self, targets: ManoPadTarget) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return active thumb-finger pinch pairs and their MANO target distances."""
        distances = np.asarray(targets.pinch_distances_m, dtype=np.float64)
        active = distances < self.pinch_threshold_m
        target_distances = np.maximum(distances, self.pinch_min_distance_m)
        return (
            np.r_[False, active],
            np.r_[0.0, distances],
            np.r_[0.0, target_distances],
        )

    def pinch_position_scales(self, distances: np.ndarray) -> np.ndarray:
        """Fade absolute pad targets out from the pinch threshold to 30% of it."""
        ratio = np.asarray(distances, dtype=np.float64) / self.pinch_threshold_m
        return self.pinch_position_weight_scale * np.clip(
            (ratio - PINCH_POSITION_RELEASE_RATIO) / (1.0 - PINCH_POSITION_RELEASE_RATIO),
            0.0, 1.0,
        )

    def pinch_contact_weights(self, distances: np.ndarray) -> np.ndarray:
        """Increase the contact-face constraint smoothly as a pinch closes."""
        alpha = np.clip(
            1.0 - np.asarray(distances, dtype=np.float64) / self.pinch_threshold_m,
            0.0,
            1.0,
        )
        return self.thumb_contact_surface_weight * alpha

    @staticmethod
    def nearest_pinch_finger(pinch_active: np.ndarray, distances: np.ndarray) -> int | None:
        """Return the closest active non-thumb finger, if one exists."""
        candidates = np.flatnonzero(np.asarray(pinch_active)[1:]) + 1
        if not len(candidates):
            return None
        return int(candidates[np.argmin(np.asarray(distances)[candidates])])

    def targets_from_qpos(self, qpos: np.ndarray) -> ManoPadTarget:
        """Build a target object from the current robot pose for offline checks."""
        qpos = self._validate_qpos(qpos)
        positions, normals = [], []
        self.robot.compute_forward_kinematics(qpos)
        for link_id in self.pad_link_indices:
            position, normal, _, _ = self.robot.compute_anchor_jacobians(
                qpos, link_id, np.zeros(3), PAD_NORMAL
            )
            positions.append(position)
            normals.append(_unit(normal))
        positions = np.asarray(positions)
        return ManoPadTarget(
            pad_positions=positions,
            pad_normals=np.asarray(normals),
            pinch_distances_m=np.linalg.norm(positions[1:] - positions[0], axis=1),
            mano_joints=np.zeros((21, 3)),
            mano_vertices=np.zeros((778, 3)),
        )

    def solve(
        self,
        targets: ManoPadTarget,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        targets = self._validate_targets(targets)
        previous = self._validate_qpos(
            self.last_qpos if last_qpos is None and self.last_qpos is not None
            else (np.zeros(self.num_joints) if last_qpos is None else last_qpos)
        )
        result = previous.copy()
        started_at = time.perf_counter()
        diagnostics = {"pad_targets_m": targets.pad_positions.copy(),
                       "pad_target_normals": targets.pad_normals.copy(),
                       "pad_actual_m": np.zeros((5, 3)),
                       "pad_actual_normals": np.zeros((5, 3)),
                       "pad_errors_m": np.zeros((5, 3)),
                       "pad_normal_errors": np.zeros((5, 3)),
                       "pad_costs": np.zeros(5)}
        pinch_active, pinch_distances, pinch_targets = self.pinch_state(targets)
        enabled = np.asarray(
            [bool(self.finger_configs[finger].get("enabled", True)) for finger in FINGERS]
        )
        pinch_active &= enabled
        if not enabled[0]:
            pinch_active[:] = False
        diagnostics.update({
            "pinch_active": pinch_active.copy(),
            "pinch_distances_m": pinch_distances.copy(),
            "pinch_target_distances_m": pinch_targets.copy(),
        })
        contact_surface_finger = (
            self.nearest_pinch_finger(pinch_active, pinch_distances)
            if self.thumb_contact_surface_optimization else None
        )
        diagnostics["thumb_contact_surface_finger"] = contact_surface_finger
        diagnostics["thumb_contact_surface_weight"] = (
            self.pinch_contact_weights(pinch_distances)[contact_surface_finger]
            if contact_surface_finger is not None else 0.0
        )

        if np.any(pinch_active[1:]):
            active_fingers = np.r_[0, np.flatnonzero(pinch_active[1:]) + 1]
            indices = np.concatenate([self.finger_qpos_indices[index] for index in active_fingers])
            local_indices = {
                finger_index: np.arange(block * 4, (block + 1) * 4)
                for block, finger_index in enumerate(active_fingers)
            }
            x0 = previous[indices].copy()
            lower = self.robot.joint_limits[indices, 0]
            upper = self.robot.joint_limits[indices, 1]
            alpha = np.clip(1.0 - pinch_distances / self.pinch_threshold_m, 0.0, 1.0)
            position_scales = self.pinch_position_scales(pinch_distances)
            thumb_position_scale = np.min(position_scales[pinch_active])
            contact_weight = diagnostics["thumb_contact_surface_weight"]
            contact_local_index = (
                int(np.flatnonzero(active_fingers == contact_surface_finger)[0])
                if contact_surface_finger is not None else None
            )

            def pinch_objective(local_q, want_gradient=False):
                candidate = result.copy()
                candidate[indices] = local_q
                positions, normals, position_jacobians, normal_jacobians = [], [], [], []
                residuals, jacobians = [], []
                for finger_index in active_fingers:
                    position, normal, position_jac, normal_jac = self.robot.compute_anchor_jacobians(
                        candidate, self.pad_link_indices[finger_index], np.zeros(3), PAD_NORMAL
                    )
                    settings = self.finger_configs[FINGERS[finger_index]]
                    position_weight = max(float(settings.get("position_weight", 1.0)), 0.0)
                    position_weight *= thumb_position_scale if finger_index == 0 else position_scales[finger_index]
                    if bool(settings.get("use_positions", True)) and position_weight:
                        residuals.append(
                            np.sqrt(position_weight) * (position - targets.pad_positions[finger_index])
                            / self.position_scale_m
                        )
                        jacobians.append(
                            np.sqrt(position_weight) * position_jac[:, indices] / self.position_scale_m
                        )
                    normal = _unit(normal)
                    smooth_weight = max(float(settings.get("smooth_weight", 0.05)), 0.0)
                    if smooth_weight:
                        finger_local_indices = local_indices[finger_index]
                        smooth_jacobian = np.zeros((4, len(indices)))
                        smooth_jacobian[np.arange(4), finger_local_indices] = np.sqrt(smooth_weight)
                        residuals.append(np.sqrt(smooth_weight) * (local_q[finger_local_indices] - x0[finger_local_indices]))
                        jacobians.append(smooth_jacobian)
                    positions.append(position)
                    normals.append(normal)
                    position_jacobians.append(position_jac[:, indices])
                    normal_jacobians.append(normal_jac[:, indices])
                contact_direction = None
                contact_direction_jacobian = None
                if contact_local_index is not None:
                    delta = positions[contact_local_index] - positions[0]
                    distance = np.linalg.norm(delta)
                    if distance > 1e-12:
                        contact_direction = delta / distance
                        contact_direction_jacobian = (
                            (np.eye(3) - np.outer(contact_direction, contact_direction)) / distance
                            @ (position_jacobians[contact_local_index] - position_jacobians[0])
                        )
                for local_finger_index, finger_index in enumerate(active_fingers):
                    settings = self.finger_configs[FINGERS[finger_index]]
                    normal_weight = max(float(settings.get("normal_weight", 0.2)), 0.0)
                    is_contact_pair = contact_direction is not None and finger_index in (0, contact_surface_finger)
                    if is_contact_pair:
                        normal_weight = contact_weight
                    if not (bool(settings.get("use_normals", True)) and normal_weight):
                        continue
                    if is_contact_pair and finger_index == 0:
                        target_normal = contact_direction
                        target_jacobian = contact_direction_jacobian
                    elif is_contact_pair:
                        target_normal = -contact_direction
                        target_jacobian = -contact_direction_jacobian
                    else:
                        target_normal = targets.pad_normals[finger_index]
                        target_jacobian = 0.0
                    residuals.append(
                        np.sqrt(normal_weight) * (normals[local_finger_index] - target_normal)
                        / self.normal_scale
                    )
                    jacobians.append(
                        np.sqrt(normal_weight) * (
                            normal_jacobians[local_finger_index] - target_jacobian
                        ) / self.normal_scale
                    )
                for local_finger_index, finger_index in enumerate(active_fingers[1:], start=1):
                    delta = positions[local_finger_index] - positions[0]
                    distance = np.linalg.norm(delta)
                    if distance <= 1e-12:
                        continue
                    weight = self.pinch_distance_weight * alpha[finger_index]
                    if not weight:
                        continue
                    residuals.append(np.array([
                        np.sqrt(weight) * (distance - pinch_targets[finger_index]) / self.position_scale_m
                    ]))
                    jacobians.append((
                        np.sqrt(weight) * (delta / distance) @ (
                            position_jacobians[local_finger_index] - position_jacobians[0]
                        ) / self.position_scale_m
                    )[None, :])
                residual = np.concatenate(residuals) if residuals else np.zeros(0)
                value = 0.5 * float(residual @ residual)
                if not want_gradient:
                    return value
                jacobian = np.vstack(jacobians)
                return value, jacobian.T @ residual

            outcome = minimize(
                lambda values: pinch_objective(values, True),
                np.clip(x0, lower, upper),
                jac=True,
                bounds=list(zip(lower, upper)),
                method="L-BFGS-B",
                options={"maxiter": self.maxeval, "ftol": self.ftol_rel},
            )
            result[indices] = np.clip(outcome.x, lower, upper)
            diagnostics["pad_costs"][active_fingers] = pinch_objective(result[indices]) / len(active_fingers)

        for finger_index, finger in enumerate(FINGERS):
            settings = self.finger_configs[finger]
            indices = self.finger_qpos_indices[finger_index]
            if (finger_index == 0 and np.any(pinch_active[1:])) or pinch_active[finger_index]:
                continue
            if not settings.get("enabled", True):
                continue
            use_position = bool(settings.get("use_positions", True))
            use_normal = bool(settings.get("use_normals", True))
            if not (use_position or use_normal):
                continue
            x0 = previous[indices].copy()
            lower = self.robot.joint_limits[indices, 0]
            upper = self.robot.joint_limits[indices, 1]

            def objective(local_q, want_gradient=False):
                candidate = result.copy()
                candidate[indices] = local_q
                position, normal, position_jac, normal_jac = self.robot.compute_anchor_jacobians(
                    candidate, self.pad_link_indices[finger_index], np.zeros(3), PAD_NORMAL
                )
                normal = _unit(normal)
                residuals = []
                jacobians = []
                if use_position:
                    weight = max(float(settings.get("position_weight", 1.0)), 0.0)
                    scale = self.position_scale_m
                    residuals.append(np.sqrt(weight) * (position - targets.pad_positions[finger_index]) / scale)
                    jacobians.append(np.sqrt(weight) * position_jac[:, indices] / scale)
                if use_normal:
                    weight = max(float(settings.get("normal_weight", 0.2)), 0.0)
                    residuals.append(np.sqrt(weight) * (normal - targets.pad_normals[finger_index]) / self.normal_scale)
                    jacobians.append(np.sqrt(weight) * normal_jac[:, indices] / self.normal_scale)
                smooth_weight = max(float(settings.get("smooth_weight", 0.05)), 0.0)
                if smooth_weight:
                    residuals.append(np.sqrt(smooth_weight) * (local_q - x0))
                    jacobians.append(np.sqrt(smooth_weight) * np.eye(len(indices)))
                residual = np.concatenate(residuals) if residuals else np.zeros(0)
                value = 0.5 * float(residual @ residual)
                if not want_gradient:
                    return value
                jacobian = np.vstack(jacobians)
                return value, jacobian.T @ residual

            outcome = minimize(
                lambda values: objective(values, True),
                np.clip(x0, lower, upper),
                jac=True,
                bounds=list(zip(lower, upper)),
                method="L-BFGS-B",
                options={"maxiter": self.maxeval, "ftol": self.ftol_rel},
            )
            result[indices] = np.clip(outcome.x, lower, upper)
            diagnostics["pad_costs"][finger_index] = float(objective(result[indices]))

        for finger_index, link_id in enumerate(self.pad_link_indices):
            position, normal, _, _ = self.robot.compute_anchor_jacobians(
                result, link_id, np.zeros(3), PAD_NORMAL
            )
            diagnostics["pad_actual_m"][finger_index] = position
            diagnostics["pad_actual_normals"][finger_index] = _unit(normal)
            diagnostics["pad_errors_m"][finger_index] = position - targets.pad_positions[finger_index]
            diagnostics["pad_normal_errors"][finger_index] = (
                diagnostics["pad_actual_normals"][finger_index] - targets.pad_normals[finger_index]
            )
        if contact_surface_finger is not None:
            delta = diagnostics["pad_actual_m"][contact_surface_finger] - diagnostics["pad_actual_m"][0]
            distance = np.linalg.norm(delta)
            if distance > 1e-12:
                direction = delta / distance
                diagnostics["pad_target_normals"][0] = direction
                diagnostics["pad_target_normals"][contact_surface_finger] = -direction
                diagnostics["pad_normal_errors"][0] = diagnostics["pad_actual_normals"][0] - direction
                diagnostics["pad_normal_errors"][contact_surface_finger] = (
                    diagnostics["pad_actual_normals"][contact_surface_finger] + direction
                )

        self.last_qpos = result.copy()
        diagnostics["solve_time_ms"] = (time.perf_counter() - started_at) * 1000.0
        self.last_diagnostics = diagnostics
        return result

    def compute_cost(self, qpos: np.ndarray, targets: ManoPadTarget) -> float:
        """Return the sum of the five independent objective values."""
        previous = self.last_qpos
        self.last_qpos = self._validate_qpos(qpos).copy()
        try:
            solved = self.solve(targets, qpos)
            return float(np.sum(self.last_diagnostics["pad_costs"]))
        finally:
            self.last_qpos = previous

    def _validate_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.num_joints,) or not np.isfinite(qpos).all():
            raise ValueError(f"qpos must be finite with shape ({self.num_joints},)")
        return np.clip(qpos, self.robot.joint_limits[:, 0], self.robot.joint_limits[:, 1])

    @staticmethod
    def _validate_targets(targets: ManoPadTarget) -> ManoPadTarget:
        positions = np.asarray(targets.pad_positions, dtype=np.float64)
        normals = np.asarray(targets.pad_normals, dtype=np.float64)
        pinch_distances = np.asarray(targets.pinch_distances_m, dtype=np.float64)
        if positions.shape != (5, 3) or normals.shape != (5, 3):
            raise ValueError("MANO pad targets must have shape (5, 3)")
        if pinch_distances.shape != (4,):
            raise ValueError("MANO pinch distances must have shape (4,)")
        if not np.isfinite(positions).all() or not np.isfinite(normals).all() or np.any(pinch_distances < 0.0):
            raise ValueError("MANO pad targets must be finite")
        normals = np.asarray([_unit(normal) for normal in normals])
        return ManoPadTarget(
            pad_positions=positions,
            pad_normals=normals,
            pinch_distances_m=pinch_distances,
            mano_joints=targets.mano_joints,
            mano_vertices=targets.mano_vertices,
        )
