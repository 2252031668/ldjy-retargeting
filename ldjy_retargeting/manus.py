"""Lossless MANUS frames, neutral calibration, and full-skeleton retargeting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .opt.base import LPFilter
from .robot import RobotWrapper


CALIBRATION_SCHEMA = "ldjy-retargeting.manus-calibration.v1"
ERGONOMICS_CALIBRATION_SCHEMA = "ldjy-retargeting.manus-ergonomics-calibration.v1"
CHAIN_HAND = 13
FINGER_CHAINS = {5: "thumb", 6: "finger1", 7: "finger2", 8: "finger3", 9: "finger4"}
JOINT_NAMES = {1: "link1", 2: "link2", 3: "link3", 4: "link4", 5: "tip"}
ERGONOMICS_FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")
DIRECT_ERGONOMICS_FINGERS = ("finger1", "finger2", "finger3")


def ergonomics_indices(side: str) -> dict[str, np.ndarray]:
    """Return MANUS' documented 40-value Ergonomics layout for one hand."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    base = 0 if side == "left" else 20
    return {finger: np.arange(base + 4 * index, base + 4 * index + 4)
            for index, finger in enumerate(ERGONOMICS_FINGERS)}


def urdf_geometry_fingerprint(urdf_path: str | Path) -> str:
    """Bind a saved Hybrid calibration to the exact robot model file."""
    return hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest()


def _array(value: Any, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result.copy()


@dataclass(frozen=True)
class ManusTopology:
    """MANUS node metadata in exactly the same order as the transform arrays."""

    node_ids: np.ndarray
    parent_ids: np.ndarray
    chain_types: np.ndarray
    sides: np.ndarray
    finger_joint_types: np.ndarray

    def __post_init__(self) -> None:
        count = np.asarray(self.node_ids).size
        for name in ("node_ids", "parent_ids", "chain_types", "sides", "finger_joint_types"):
            object.__setattr__(self, name, _array(getattr(self, name), np.int64, (count,), name))
        if len(set(self.node_ids.tolist())) != count:
            raise ValueError("MANUS node IDs must be unique")

    @property
    def node_count(self) -> int:
        return int(self.node_ids.size)

    def matches(self, other: "ManusTopology") -> bool:
        if self.node_count != other.node_count or set(self.node_ids) != set(other.node_ids):
            return False
        left = np.argsort(self.node_ids)
        right = np.argsort(other.node_ids)
        return all(np.array_equal(getattr(self, name)[left], getattr(other, name)[right]) for name in (
            "node_ids", "parent_ids", "chain_types", "sides", "finger_joint_types"
        ))

    def semantic_keys(self) -> list[tuple[int, int]]:
        return list(zip(self.chain_types.tolist(), self.finger_joint_types.tolist()))


@dataclass(frozen=True)
class ManusFrame:
    """One fully copied Raw Skeleton sample with its latest Ergonomics sample."""

    received_timestamp_sec: float
    sdk_timestamp: int
    glove_id: int
    side: str
    positions: np.ndarray
    rotations: np.ndarray  # SDK quaternion order: w, x, y, z
    scales: np.ndarray
    topology: ManusTopology
    ergonomics: np.ndarray
    ergonomics_timestamp: int = 0
    ergonomics_valid: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.received_timestamp_sec) or self.sdk_timestamp < 0:
            raise ValueError("MANUS timestamps must be finite/non-negative")
        if self.side not in {"left", "right"}:
            raise ValueError("MANUS frame side must be left or right")
        n = self.topology.node_count
        object.__setattr__(self, "positions", _array(self.positions, np.float64, (n, 3), "positions"))
        rotations = _array(self.rotations, np.float64, (n, 4), "rotations")
        norms = np.linalg.norm(rotations, axis=1)
        if not np.isfinite(rotations).all() or np.any(norms <= 1e-12):
            raise ValueError("MANUS rotations must be finite non-zero quaternions")
        object.__setattr__(self, "rotations", rotations / norms[:, None])
        object.__setattr__(self, "scales", _array(self.scales, np.float64, (n, 3), "scales"))
        ergonomics = np.asarray(self.ergonomics, dtype=np.float64)
        if ergonomics.ndim != 1:
            raise ValueError("ergonomics must be a vector")
        object.__setattr__(self, "ergonomics", ergonomics.copy())
        if self.ergonomics_valid and not np.isfinite(ergonomics).all():
            raise ValueError("valid MANUS Ergonomics data must be finite")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.scales).all():
            raise ValueError("MANUS transforms must be finite")


def quaternion_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def average_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Average quaternions after resolving the q/-q representation ambiguity."""
    q = np.asarray(quaternions, dtype=np.float64).copy()
    if q.ndim != 2 or q.shape[1] != 4 or not len(q):
        raise ValueError("quaternions must have shape (samples, 4)")
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norms <= 1e-12):
        raise ValueError("quaternions must be finite and non-zero")
    q /= norms
    q[np.sum(q * q[0], axis=1) < 0] *= -1
    result = q.mean(axis=0)
    return result / np.linalg.norm(result)


def wrist_local(frame: ManusFrame) -> tuple[np.ndarray, np.ndarray, int]:
    roots = np.flatnonzero(frame.topology.chain_types == CHAIN_HAND)
    if len(roots) != 1:
        raise ValueError(f"expected exactly one MANUS Hand node, got {len(roots)}")
    root = int(roots[0])
    matrices = quaternion_matrices_wxyz(frame.rotations)
    root_rotation = matrices[root]
    positions = (frame.positions - frame.positions[root]) @ root_rotation
    rotations = np.einsum("ij,njk->nik", root_rotation.T, matrices)
    return positions, rotations, root


def _rotation_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return xyzw[[3, 0, 1, 2]]


def semantic_frame_name(chain: int, joint: int) -> str | None:
    if chain == CHAIN_HAND:
        return "retarget_wrist"
    finger = FINGER_CHAINS.get(chain)
    suffix = JOINT_NAMES.get(joint)
    return None if finger is None or suffix is None else f"{finger}_{suffix}"


@dataclass(frozen=True)
class ManusCalibration:
    glove_id: int
    side: str
    topology: ManusTopology
    reference_positions: np.ndarray
    reference_rotations: np.ndarray
    alignment: np.ndarray
    rotation_offsets: np.ndarray
    sdk_version: str = "unknown"

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"} or self.glove_id < 0:
            raise ValueError("invalid MANUS calibration glove/side")
        n = self.topology.node_count
        object.__setattr__(self, "reference_positions", _array(
            self.reference_positions, np.float64, (n, 3), "reference_positions"
        ))
        object.__setattr__(self, "reference_rotations", _array(
            self.reference_rotations, np.float64, (n, 4), "reference_rotations"
        ))
        object.__setattr__(self, "alignment", _array(self.alignment, np.float64, (3, 3), "alignment"))
        object.__setattr__(self, "rotation_offsets", _array(
            self.rotation_offsets, np.float64, (n, 3, 3), "rotation_offsets"
        ))
        if not all(np.isfinite(value).all() for value in (
            self.reference_positions, self.reference_rotations, self.alignment, self.rotation_offsets
        )):
            raise ValueError("MANUS calibration arrays must be finite")
        if not np.allclose(self.alignment.T @ self.alignment, np.eye(3), atol=1e-5) or np.linalg.det(self.alignment) < 0:
            raise ValueError("MANUS calibration alignment must be a proper rotation")

    def matches(self, frame: ManusFrame) -> bool:
        return self.glove_id == frame.glove_id and self.side == frame.side and self.topology.matches(frame.topology)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as f:
                temporary = Path(f.name)
            np.savez_compressed(temporary, **self.as_arrays())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "schema": np.array(CALIBRATION_SCHEMA), "sdk_version": np.array(self.sdk_version),
            "glove_id": np.array(self.glove_id, dtype=np.uint32), "side": np.array(self.side),
            "node_ids": self.topology.node_ids, "parent_ids": self.topology.parent_ids,
            "chain_types": self.topology.chain_types, "sides": self.topology.sides,
            "finger_joint_types": self.topology.finger_joint_types,
            "reference_positions": self.reference_positions,
            "reference_rotations": self.reference_rotations,
            "alignment": self.alignment, "rotation_offsets": self.rotation_offsets,
        }

    @classmethod
    def load(cls, path: str | Path) -> "ManusCalibration":
        with np.load(path, allow_pickle=False) as a:
            if str(a["schema"]) != CALIBRATION_SCHEMA:
                raise ValueError("unsupported MANUS calibration schema")
            topology = ManusTopology(*(a[name] for name in (
                "node_ids", "parent_ids", "chain_types", "sides", "finger_joint_types"
            )))
            return cls(int(a["glove_id"]), str(a["side"]), topology,
                       a["reference_positions"], a["reference_rotations"], a["alignment"],
                       a["rotation_offsets"], str(a["sdk_version"]))


@dataclass(frozen=True)
class ManusErgonomicsCalibration:
    """One glove/side calibration for the Ergonomics Hybrid path."""

    glove_id: int
    side: str
    urdf_fingerprint: str
    open_degrees: np.ndarray
    direct_gain: np.ndarray
    direct_offset: np.ndarray
    special_gain: np.ndarray
    special_offset: np.ndarray
    pinch_distances_m: np.ndarray
    sdk_version: str = "unknown"

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"} or self.glove_id < 0 or not self.urdf_fingerprint:
            raise ValueError("invalid MANUS Ergonomics calibration identity")
        for name, shape in (("open_degrees", (20,)), ("direct_gain", (12,)),
                            ("direct_offset", (12,)), ("special_gain", (8,)),
                            ("special_offset", (8,)), ("pinch_distances_m", (4,))):
            object.__setattr__(self, name, _array(getattr(self, name), np.float64, shape, name))
        if np.any(self.pinch_distances_m <= 0):
            raise ValueError("pinch distances must be positive")

    def matches(self, frame: ManusFrame, urdf_fingerprint: str) -> bool:
        return self.glove_id == frame.glove_id and self.side == frame.side and self.urdf_fingerprint == urdf_fingerprint

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {"schema": np.array(ERGONOMICS_CALIBRATION_SCHEMA), "glove_id": np.array(self.glove_id, dtype=np.uint32),
                "side": np.array(self.side), "urdf_fingerprint": np.array(self.urdf_fingerprint),
                "open_degrees": self.open_degrees, "direct_gain": self.direct_gain,
                "direct_offset": self.direct_offset, "special_gain": self.special_gain,
                "special_offset": self.special_offset, "pinch_distances_m": self.pinch_distances_m,
                "sdk_version": np.array(self.sdk_version)}

    def save(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as f:
            temporary = Path(f.name)
        try:
            np.savez_compressed(temporary, **self.as_arrays()); os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ManusErgonomicsCalibration":
        with np.load(path, allow_pickle=False) as a:
            if str(a["schema"]) != ERGONOMICS_CALIBRATION_SCHEMA:
                raise ValueError("unsupported MANUS Ergonomics calibration schema")
            return cls(int(a["glove_id"]), str(a["side"]), str(a["urdf_fingerprint"]),
                       a["open_degrees"], a["direct_gain"], a["direct_offset"],
                       a["special_gain"], a["special_offset"], a["pinch_distances_m"], str(a["sdk_version"]))


class ManusFullSkeletonRetargeter:
    """Bounded full-skeleton least-squares solver, independent of the 21-point stack."""

    def __init__(self, config: dict[str, Any]):
        optimizer = config["optimizer"]
        self.hand_side = optimizer["hand_side"]
        self.robot = RobotWrapper(optimizer["urdf_path"], self.hand_side)
        self.num_joints = self.robot.model.nq
        self.limits = self.robot.joint_limits.astype(np.float64)
        cfg = config.get("retarget", {})
        self.position_scale = float(cfg.get("position_scale_m", .01))
        self.direction_scale = float(cfg.get("direction_scale", .1))
        self.rotation_scale = np.deg2rad(float(cfg.get("rotation_scale_deg", 10.0)))
        self.position_weight = float(cfg.get("position_weight", 1.0))
        self.direction_weight = float(cfg.get("direction_weight", .5))
        self.rotation_weight = float(cfg.get("rotation_weight", .5))
        self.temporal_scale = float(cfg.get("temporal_scale_rad", .1))
        self.temporal_weight = float(cfg.get("temporal_weight", .05))
        self.zero_weight = float(cfg.get("zero_weight", .001))
        self.max_nfev = int(cfg.get("max_nfev", 30))
        self.filter = LPFilter(float(cfg.get("lp_alpha", .15)))
        self.last_qpos: np.ndarray | None = None
        self.last_diagnostics: dict[str, Any] = {}
        self.calibration: ManusCalibration | None = None
        self._frame_indices: list[int] = []
        self._robot_frame_ids: list[int] = []
        self._parent_mapped_indices: list[int] = []
        self._robot_zero_positions = np.empty((0, 3))
        self._robot_zero_rotations = np.empty((0, 3, 3))

    def reset(self) -> None:
        self.last_qpos = None
        self.filter.reset()

    def _mapping(self, topology: ManusTopology) -> tuple[list[int], list[int]]:
        seen: set[tuple[int, int]] = set()
        frame_indices = []
        for i, key in enumerate(topology.semantic_keys()):
            name = semantic_frame_name(*key)
            if name is None:
                continue
            if key in seen:
                raise ValueError(f"duplicate MANUS semantic node chain/joint {key}")
            seen.add(key)
            frame_indices.append(i)
        if not any(topology.chain_types[i] == CHAIN_HAND for i in frame_indices):
            raise ValueError("MANUS topology has no unique Hand node")
        id_to_index = {int(node_id): i for i, node_id in enumerate(topology.node_ids)}

        def depth(index: int) -> int:
            seen_parents = set()
            result, parent_id = 0, int(topology.parent_ids[index])
            while parent_id in id_to_index:
                parent_index = id_to_index[parent_id]
                # MANUS uses parentId == nodeId for a root in some raw topologies.
                if parent_index == index:
                    break
                if parent_id in seen_parents:
                    raise ValueError("MANUS topology contains a parent cycle")
                seen_parents.add(parent_id)
                result += 1
                index = parent_index
                parent_id = int(topology.parent_ids[index])
            return result

        frame_indices.sort(key=depth)
        robot_ids = [self.robot.get_link_index(semantic_frame_name(
            int(topology.chain_types[i]), int(topology.finger_joint_types[i])
        )) for i in frame_indices]
        return frame_indices, robot_ids

    def _configure_topology(self, topology: ManusTopology) -> None:
        self._frame_indices, self._robot_frame_ids = self._mapping(topology)
        id_to_source = {int(node_id): i for i, node_id in enumerate(topology.node_ids)}
        mapped_lookup = {source: mapped for mapped, source in enumerate(self._frame_indices)}
        parents: list[int] = []
        for source in self._frame_indices:
            parent = id_to_source.get(int(topology.parent_ids[source]))
            if parent == source:
                parent = None
            visited = {source}
            while parent is not None and parent not in mapped_lookup:
                if parent in visited:
                    raise ValueError("MANUS topology contains a parent cycle")
                visited.add(parent)
                parent = id_to_source.get(int(topology.parent_ids[parent]))
                if parent in visited:
                    raise ValueError("MANUS topology contains a parent cycle")
            parents.append(mapped_lookup.get(parent, -1))
        self._parent_mapped_indices = parents
        self.robot.compute_forward_kinematics(np.zeros(self.num_joints))
        poses = [self.robot.get_link_pose(i) for i in self._robot_frame_ids]
        root_mapped = next(i for i, source in enumerate(self._frame_indices)
                           if topology.chain_types[source] == CHAIN_HAND)
        root = poses[root_mapped]
        self._robot_zero_positions = np.stack([root[:3, :3].T @ (p[:3, 3] - root[:3, 3]) for p in poses])
        self._robot_zero_rotations = np.stack([root[:3, :3].T @ p[:3, :3] for p in poses])

    def calibrate(self, frames: Iterable[ManusFrame], *, sdk_version: str = "unknown") -> ManusCalibration:
        frames = list(frames)
        if len(frames) < 50:
            raise ValueError("MANUS neutral calibration needs at least 50 frames")
        if len({frame.sdk_timestamp for frame in frames}) < 50:
            raise ValueError("MANUS neutral calibration needs 50 unique SDK frames")
        first = frames[0]
        if any(f.glove_id != first.glove_id or f.side != first.side or not f.topology.matches(first.topology) for f in frames):
            raise ValueError("MANUS neutral frames must have identical glove, side and topology")
        local = []
        for frame in frames:
            positions_for_frame, rotations_for_frame, _ = wrist_local(frame)
            by_id = {int(node_id): i for i, node_id in enumerate(frame.topology.node_ids)}
            order = [by_id[int(node_id)] for node_id in first.topology.node_ids]
            local.append((positions_for_frame[order], rotations_for_frame[order]))
        positions = np.stack([item[0] for item in local])
        rotations = np.stack([item[1] for item in local])
        reference_positions = np.median(positions, axis=0)
        position_deviation = np.median(np.linalg.norm(positions - reference_positions, axis=2), axis=0)
        reference_quaternions = np.stack([
            average_quaternions_wxyz(np.stack([_rotation_to_wxyz(r[node]) for r in rotations]))
            for node in range(first.topology.node_count)
        ])
        reference_matrices = quaternion_matrices_wxyz(reference_quaternions)
        angles = np.stack([
            Rotation.from_matrix(reference_matrices[node].T @ rotations[sample, node]).magnitude()
            for sample in range(len(frames)) for node in range(first.topology.node_count)
        ]).reshape(len(frames), first.topology.node_count)
        rotation_deviation = np.median(angles, axis=0)
        if position_deviation.max(initial=0.0) > .003 or rotation_deviation.max(initial=0.0) > np.deg2rad(5):
            raise ValueError(
                f"MANUS neutral pose unstable: {position_deviation.max()*1000:.1f} mm, "
                f"{np.rad2deg(rotation_deviation.max()):.1f} deg"
            )
        self._configure_topology(first.topology)
        source = reference_positions[self._frame_indices]
        valid = np.linalg.norm(source, axis=1) > 1e-6
        source_unit = source[valid] / np.linalg.norm(source[valid], axis=1, keepdims=True)
        target = self._robot_zero_positions[valid]
        target_unit = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
        u, _, vt = np.linalg.svd(source_unit.T @ target_unit)
        alignment = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
        offsets = np.tile(np.eye(3), (first.topology.node_count, 1, 1))
        for mapped, source_index in enumerate(self._frame_indices):
            transformed = alignment @ reference_matrices[source_index] @ alignment.T
            offsets[source_index] = transformed.T @ self._robot_zero_rotations[mapped]
        calibration = ManusCalibration(
            first.glove_id, first.side, first.topology, reference_positions,
            reference_quaternions, alignment, offsets, sdk_version,
        )
        self.calibration = calibration
        return calibration

    def set_calibration(self, calibration: ManusCalibration) -> None:
        if calibration.side != self.hand_side:
            raise ValueError("MANUS calibration hand side does not match runtime")
        self._configure_topology(calibration.topology)
        self.calibration = calibration
        self.reset()

    def _targets(self, frame: ManusFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calibration = self.calibration
        if calibration is None:
            raise RuntimeError("尚未采集 MANUS 中立姿态")
        if not calibration.matches(frame):
            raise ValueError("MANUS frame does not match the loaded glove/topology calibration")
        positions, rotations, _ = wrist_local(frame)
        frame_by_id = {int(node_id): i for i, node_id in enumerate(frame.topology.node_ids)}
        order = [frame_by_id[int(node_id)] for node_id in calibration.topology.node_ids]
        positions, rotations = positions[order], rotations[order]
        source = positions[self._frame_indices]
        aligned_rotations = np.stack([
            calibration.alignment @ rotations[i] @ calibration.alignment.T @ calibration.rotation_offsets[i]
            for i in self._frame_indices
        ])
        targets = np.zeros_like(source)
        directions = np.zeros_like(source)
        for mapped, source_index in enumerate(self._frame_indices):
            parent = self._parent_mapped_indices[mapped]
            if parent < 0:
                continue
            parent_source = self._frame_indices[parent]
            direction = calibration.alignment @ (positions[source_index] - positions[parent_source])
            norm = np.linalg.norm(direction)
            if norm <= 1e-9:
                reference = calibration.reference_positions[source_index] - calibration.reference_positions[parent_source]
                direction = calibration.alignment @ reference
                norm = np.linalg.norm(direction)
            direction /= max(norm, 1e-12)
            length = np.linalg.norm(self._robot_zero_positions[mapped] - self._robot_zero_positions[parent])
            directions[mapped] = direction
            targets[mapped] = targets[parent] + length * direction
        return targets, directions, aligned_rotations

    def _root_mapped_index(self) -> int:
        if self.calibration is None:
            raise RuntimeError("尚未采集 MANUS 中立姿态")
        return next(i for i, source in enumerate(self._frame_indices)
                    if self.calibration.topology.chain_types[source] == CHAIN_HAND)

    def _mapped_node_ids(self) -> np.ndarray:
        if self.calibration is None:
            return np.empty(0, dtype=np.int64)
        return self.calibration.topology.node_ids[self._frame_indices].copy()

    def _mapped_semantics(self) -> list[str | None]:
        if self.calibration is None:
            return []
        return [semantic_frame_name(self.calibration.topology.chain_types[i],
                                    self.calibration.topology.finger_joint_types[i])
                for i in self._frame_indices]

    def _extra_residual(
        self, qpos: np.ndarray, *, with_jacobian: bool, root_rotation: np.ndarray,
        root_translation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
        del qpos, root_rotation, root_translation
        return np.empty(0), (np.empty((0, self.num_joints)) if with_jacobian else None), {}

    def _robot_poses(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pin.forwardKinematics(self.robot.model, self.robot.data, qpos)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        placements = [self.robot.data.oMf[i] for i in self._robot_frame_ids]
        root_index = self._root_mapped_index()
        root = placements[root_index]
        positions = np.stack([root.rotation.T @ (p.translation - root.translation) for p in placements])
        rotations = np.stack([root.rotation.T @ p.rotation for p in placements])
        return positions, rotations

    @staticmethod
    def _so3_left_jacobian_inverse(vector: np.ndarray) -> np.ndarray:
        angle = np.linalg.norm(vector)
        skew = pin.skew(vector)
        if angle < 1e-6:
            return np.eye(3) - .5 * skew + skew @ skew / 12.0
        coefficient = 1.0 / angle**2 - 1.0 / (2.0 * angle * np.tan(angle / 2.0))
        return np.eye(3) - .5 * skew + coefficient * skew @ skew

    def _robot_poses_and_jacobians(self, qpos: np.ndarray):
        pin.forwardKinematics(self.robot.model, self.robot.data, qpos)
        pin.computeJointJacobians(self.robot.model, self.robot.data, qpos)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        placements = [self.robot.data.oMf[i] for i in self._robot_frame_ids]
        root_index = self._root_mapped_index()
        root = placements[root_index]
        positions, rotations, position_jacobians, angular_jacobians = [], [], [], []
        for frame_id, placement in zip(self._robot_frame_ids, placements):
            spatial = pin.getFrameJacobian(
                self.robot.model, self.robot.data, frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            positions.append(root.rotation.T @ (placement.translation - root.translation))
            rotations.append(root.rotation.T @ placement.rotation)
            position_jacobians.append(root.rotation.T @ spatial[:3])
            angular_jacobians.append(root.rotation.T @ spatial[3:])
        return tuple(map(np.stack, (positions, rotations, position_jacobians, angular_jacobians)))

    def solve(self, frame: ManusFrame) -> tuple[np.ndarray, dict[str, Any]]:
        start = time.perf_counter()
        targets, target_directions, target_rotations = self._targets(frame)
        previous = np.zeros(self.num_joints) if self.last_qpos is None else self.last_qpos.copy()
        lower, upper = self.limits[:, 0], self.limits[:, 1]
        initial = np.clip(previous, lower, upper)
        # scipy's relative finite-difference step is ineffective at exact zero.
        initial[np.abs(initial) < 1e-8] = np.minimum(np.maximum(1e-4, lower[np.abs(initial) < 1e-8]), upper[np.abs(initial) < 1e-8])

        extra_diagnostics: dict[str, np.ndarray] = {}

        def components(qpos: np.ndarray, with_jacobian: bool = False):
            if with_jacobian:
                positions, rotations, position_jacobians, angular_jacobians = self._robot_poses_and_jacobians(qpos)
            else:
                positions, rotations = self._robot_poses(qpos)
            position_error = positions - targets
            direction_error = np.zeros_like(positions)
            rotation_error = np.zeros_like(positions)
            direction_jacobians = np.zeros((len(positions), 3, self.num_joints))
            rotation_jacobians = np.zeros((len(positions), 3, self.num_joints))
            for i, parent in enumerate(self._parent_mapped_indices):
                if parent >= 0:
                    segment = positions[i] - positions[parent]
                    length = max(np.linalg.norm(segment), 1e-12)
                    unit = segment / length
                    direction_error[i] = unit - target_directions[i]
                    if with_jacobian:
                        direction_jacobians[i] = (
                            (np.eye(3) - np.outer(unit, unit)) / length
                            @ (position_jacobians[i] - position_jacobians[parent])
                        )
                rotation_error[i] = Rotation.from_matrix(target_rotations[i].T @ rotations[i]).as_rotvec()
                if with_jacobian:
                    rotation_jacobians[i] = (
                        self._so3_left_jacobian_inverse(rotation_error[i])
                        @ target_rotations[i].T @ angular_jacobians[i]
                    )
            root_index = self._root_mapped_index()
            if with_jacobian:
                root_pose = self.robot.data.oMf[self._robot_frame_ids[root_index]]
                extra, extra_jacobian, diagnostics = self._extra_residual(
                    qpos, with_jacobian=True, root_rotation=root_pose.rotation,
                    root_translation=root_pose.translation,
                )
            else:
                root_pose = self.robot.data.oMf[self._robot_frame_ids[root_index]]
                extra, _, diagnostics = self._extra_residual(
                    qpos, with_jacobian=False, root_rotation=root_pose.rotation,
                    root_translation=root_pose.translation,
                )
            extra_diagnostics.clear()
            extra_diagnostics.update(diagnostics)
            residual = np.concatenate((
                (self.position_weight / self.position_scale * position_error).ravel(),
                (self.direction_weight / self.direction_scale * direction_error).ravel(),
                (self.rotation_weight / self.rotation_scale * rotation_error).ravel(),
                self.temporal_weight / self.temporal_scale * (qpos - previous),
                self.zero_weight * qpos,
                extra,
            ))
            values = (residual, positions, rotations, position_error, rotation_error)
            if not with_jacobian:
                return values
            jacobian = np.vstack((
                self.position_weight / self.position_scale * position_jacobians.reshape(-1, self.num_joints),
                self.direction_weight / self.direction_scale * direction_jacobians.reshape(-1, self.num_joints),
                self.rotation_weight / self.rotation_scale * rotation_jacobians.reshape(-1, self.num_joints),
                self.temporal_weight / self.temporal_scale * np.eye(self.num_joints),
                self.zero_weight * np.eye(self.num_joints),
                extra_jacobian,
            ))
            return values + (jacobian,)

        failure = None
        try:
            # Pinocchio's frame-Jacobian cache must be populated before SciPy's
            # first residual/Jacobian pair; otherwise its first trust-region
            # step can observe stale frame placements on this model.
            components(initial, True)
            result = least_squares(lambda q: components(q)[0], initial,
                                   jac=lambda q: components(q, True)[-1], bounds=(lower, upper),
                                   loss="huber", f_scale=1.0, max_nfev=self.max_nfev,
                                   x_scale="jac")
            if not result.success or not np.isfinite(result.x).all():
                failure = result.message
                solution = previous
            else:
                solution = result.x
        except Exception as exc:
            failure = str(exc)
            solution = previous
            result = None
        filtered = np.clip(self.filter.next(solution), lower, upper)
        if failure is None:
            self.last_qpos = filtered.copy()
        residual, actual_positions, actual_rotations, position_error, rotation_error = components(filtered)
        elapsed_ms = (time.perf_counter() - start) * 1000
        squared = residual * residual
        huber_cost = float(np.sum(np.where(squared <= 1.0, squared, 2.0 * np.sqrt(squared) - 1.0)))
        direction_errors = np.zeros(len(actual_positions))
        for index, parent in enumerate(self._parent_mapped_indices):
            if parent >= 0:
                actual_direction = actual_positions[index] - actual_positions[parent]
                actual_direction /= max(np.linalg.norm(actual_direction), 1e-12)
                direction_errors[index] = np.linalg.norm(actual_direction - target_directions[index])
        diagnostics = {
            "manus_node_ids": self._mapped_node_ids(),
            "manus_semantics": self._mapped_semantics(),
            "manus_target_positions_m": targets.copy(), "manus_target_rotations": target_rotations.copy(),
            "manus_actual_positions_m": actual_positions.copy(), "manus_actual_rotations": actual_rotations.copy(),
            "manus_parent_indices": np.asarray(self._parent_mapped_indices, dtype=np.int64),
            "manus_position_errors_m": np.linalg.norm(position_error, axis=1),
            "manus_direction_errors": direction_errors,
            "manus_rotation_errors_rad": np.linalg.norm(rotation_error, axis=1),
            "manus_raw_positions_m": frame.positions.copy(), "manus_raw_rotations_wxyz": frame.rotations.copy(),
            "manus_raw_scales": frame.scales.copy(), "manus_raw_node_ids": frame.topology.node_ids.copy(),
            "manus_raw_parent_ids": frame.topology.parent_ids.copy(),
            "manus_ergonomics": frame.ergonomics.copy(),
            "manus_ergonomics_timestamp": frame.ergonomics_timestamp,
            "manus_ergonomics_valid": frame.ergonomics_valid,
            "cost": huber_cost, "solve_ms": elapsed_ms,
            "nfev": 0 if result is None else int(result.nfev), "failure": failure,
            "qpos": filtered.copy(), "pinch_alphas": np.zeros(5),
        }
        diagnostics.update(extra_diagnostics)
        self.last_diagnostics = diagnostics
        return filtered, diagnostics


class ManusErgonomicsHybridRetargeter:
    """Official Ergonomics for central fingers plus Raw-Skeleton IK for thumb/pinky."""

    direct_indices = np.arange(12)
    special_indices = np.arange(12, 20)

    def __init__(self, config: dict[str, Any]):
        optimizer, cfg = config["optimizer"], config.get("retarget", {})
        self.hand_side = optimizer["hand_side"]
        self.robot = RobotWrapper(optimizer["urdf_path"], self.hand_side)
        self.limits = self.robot.joint_limits.astype(np.float64)
        self.num_joints = self.robot.model.nq
        self.urdf_fingerprint = urdf_geometry_fingerprint(optimizer["urdf_path"])
        self.position_scale = float(cfg.get("position_scale_m", .01))
        self.tip_weight = float(cfg.get("tip_position_weight", 3.0))
        self.pad_weight = float(cfg.get("pad_position_weight", 3.0))
        self.direction_weight = float(cfg.get("direction_weight", .5))
        self.prior_weight = float(cfg.get("ergonomics_prior_weight", .1))
        self.temporal_weight = float(cfg.get("temporal_weight", .05))
        self.temporal_scale = float(cfg.get("temporal_scale_rad", .1))
        self.zero_weight = float(cfg.get("zero_weight", .001))
        self.max_nfev = int(cfg.get("max_nfev", 30))
        self.filter = LPFilter(float(cfg.get("lp_alpha", .15)))
        self.calibration: ManusErgonomicsCalibration | None = None
        self.last_qpos: np.ndarray | None = None
        self.last_diagnostics: dict[str, Any] = {}
        self._ids = {name: self.robot.get_link_index(name) for name in (
            "retarget_wrist", "thumb_link4", "thumb_tip", "thumb_pad", "finger4_link3", "finger4_link4",
            "finger4_tip", "finger4_pad", "finger1_tip", "finger2_tip", "finger3_tip",
            "finger1_pad", "finger2_pad", "finger3_pad")}
        self._zero_positions = self._positions(np.zeros(self.num_joints))

    def reset(self) -> None:
        self.last_qpos = None; self.filter.reset()

    def set_calibration(self, calibration: ManusErgonomicsCalibration) -> None:
        if calibration.side != self.hand_side or calibration.urdf_fingerprint != self.urdf_fingerprint:
            raise ValueError("MANUS Ergonomics calibration does not match current hand/URDF")
        self.calibration = calibration; self.reset()

    @staticmethod
    def _stable_pose(frames: Iterable[ManusFrame]) -> ManusFrame:
        values = list(frames)
        if len(values) < 50 or len({f.sdk_timestamp for f in values}) < 50:
            raise ValueError("each Ergonomics calibration pose needs 50 unique frames")
        first = values[0]
        if any(not f.ergonomics_valid or f.glove_id != first.glove_id or f.side != first.side for f in values):
            raise ValueError("Ergonomics calibration samples must be valid and from one glove/side")
        return values[len(values) // 2]

    def calibrate(self, poses: dict[str, Iterable[ManusFrame]], *, sdk_version: str = "unknown") -> ManusErgonomicsCalibration:
        required = ("open", "spread", "fist", "pinch_finger1", "pinch_finger2", "pinch_finger3", "pinch_finger4")
        if set(required) - set(poses):
            raise ValueError("missing Ergonomics calibration poses")
        samples = {name: self._stable_pose(poses[name]) for name in required}
        first = samples["open"]
        if any(f.glove_id != first.glove_id or f.side != first.side for f in samples.values()):
            raise ValueError("Ergonomics calibration poses must use one glove/side")
        idx = ergonomics_indices(self.hand_side)
        def angles(frame: ManusFrame, fingers: tuple[str, ...]) -> np.ndarray:
            return np.concatenate([frame.ergonomics[idx[f]] for f in fingers])
        open_all = angles(first, ERGONOMICS_FINGERS)
        direct_open, direct_spread, direct_fist = (angles(samples[name], DIRECT_ERGONOMICS_FINGERS)
                                                   for name in ("open", "spread", "fist"))
        target = np.zeros(12)
        for finger in range(3):
            base = 4 * finger
            target[base] = .75 * (self.limits[base, 1] if direct_spread[base] >= direct_open[base] else self.limits[base, 0])
            target[base + 1:base + 4] = .8 * self.limits[base + 1:base + 4, 1]
        reference = direct_spread.copy()
        reference.reshape(3, 4)[:, 1:] = direct_fist.reshape(3, 4)[:, 1:]
        delta = np.deg2rad(reference - direct_open)
        if np.any(np.abs(delta) < np.deg2rad(3)):
            raise ValueError("spread/fist movement is too small; please repeat the pose")
        gain = target / delta; offset = np.zeros(12)
        special_open = np.concatenate([first.ergonomics[idx["thumb"]], first.ergonomics[idx["finger4"]]])
        special_ref = np.concatenate([samples["fist"].ergonomics[idx["thumb"]], samples["fist"].ergonomics[idx["finger4"]]])
        special_target = .55 * self.limits[12:20, 1]
        special_delta = np.deg2rad(special_ref - special_open)
        special_gain = special_target / np.where(np.abs(special_delta) >= np.deg2rad(3), special_delta, np.deg2rad(3))
        distances = []
        for finger in ("finger1", "finger2", "finger3", "finger4"):
            raw = self._raw_targets(samples[f"pinch_{finger}"])
            distances.append(np.linalg.norm(raw["thumb_tip"] - raw[f"{finger}_tip"]))
        result = ManusErgonomicsCalibration(first.glove_id, self.hand_side, self.urdf_fingerprint,
            open_all, gain, offset, special_gain, np.zeros(8), np.maximum(distances, .002), sdk_version)
        self.set_calibration(result)
        return result

    def _positions(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        pin.forwardKinematics(self.robot.model, self.robot.data, qpos); pin.updateFramePlacements(self.robot.model, self.robot.data)
        root = self.robot.data.oMf[self._ids["retarget_wrist"]]
        return {name: root.rotation.T @ (self.robot.data.oMf[index].translation - root.translation)
                for name, index in self._ids.items()}

    def _raw_targets(self, frame: ManusFrame) -> dict[str, np.ndarray]:
        local, _, root = wrist_local(frame)
        semantic = {semantic_frame_name(int(chain), int(joint)): i for i, (chain, joint) in
                    enumerate(zip(frame.topology.chain_types, frame.topology.finger_joint_types))}
        needed = ("thumb_tip", "thumb_link4", "finger4_tip", "finger4_link3", "finger4_link4",
                  "finger1_link1", "finger2_link1", "finger3_link1", "finger1_tip", "finger2_tip", "finger3_tip")
        if any(name not in semantic for name in needed):
            raise ValueError("Raw Skeleton lacks the nodes needed by Ergonomics Hybrid")
        source_forward = local[semantic["finger2_link1"]]
        source_lateral = local[semantic["finger3_link1"]] - local[semantic["finger1_link1"]]
        source_forward /= max(np.linalg.norm(source_forward), 1e-12); source_lateral /= max(np.linalg.norm(source_lateral), 1e-12)
        source_normal = np.cross(source_lateral, source_forward); source_normal /= max(np.linalg.norm(source_normal), 1e-12)
        source_lateral = np.cross(source_forward, source_normal)
        robot_forward = self._zero_positions["finger2_tip"]
        robot_lateral = self._zero_positions["finger3_tip"] - self._zero_positions["finger1_tip"]
        robot_forward /= max(np.linalg.norm(robot_forward), 1e-12); robot_lateral /= max(np.linalg.norm(robot_lateral), 1e-12)
        robot_normal = np.cross(robot_lateral, robot_forward); robot_normal /= max(np.linalg.norm(robot_normal), 1e-12)
        robot_lateral = np.cross(robot_forward, robot_normal)
        transform = np.stack((robot_lateral, robot_forward, robot_normal), axis=1) @ np.stack((source_lateral, source_forward, source_normal), axis=1).T
        pairs = [("finger1", "finger1_tip"), ("finger2", "finger2_tip"), ("finger3", "finger3_tip")]
        scale = np.median([np.linalg.norm(self._zero_positions[target])
                           / max(np.linalg.norm(local[semantic[target]] - local[semantic[f"{name}_link1"]]), 1e-9)
                           for name, target in pairs])
        return {name: scale * transform @ local[index] for name, index in semantic.items() if name in needed}

    def solve(self, frame: ManusFrame) -> tuple[np.ndarray, dict[str, Any]]:
        start = time.perf_counter()
        calibration = self.calibration
        if calibration is None:
            raise RuntimeError("尚未采集 MANUS Ergonomics Hybrid 标定")
        if not calibration.matches(frame, self.urdf_fingerprint):
            raise ValueError("MANUS Ergonomics Hybrid 标定与当前手套/URDF 不匹配")
        if not frame.ergonomics_valid or len(frame.ergonomics) != 40:
            raise ValueError("MANUS Ergonomics 无效或不是 40 维")
        indices = ergonomics_indices(self.hand_side)
        direct = np.concatenate([frame.ergonomics[indices[name]] for name in DIRECT_ERGONOMICS_FINGERS])
        special = np.concatenate([frame.ergonomics[indices["thumb"]], frame.ergonomics[indices["finger4"]]])
        base = np.zeros(self.num_joints)
        base[:12] = np.clip(calibration.direct_offset + calibration.direct_gain * np.deg2rad(
            direct - calibration.open_degrees[4:16]), self.limits[:12, 0], self.limits[:12, 1])
        prior = np.clip(calibration.special_offset + calibration.special_gain * np.deg2rad(
            special - np.concatenate((calibration.open_degrees[:4], calibration.open_degrees[16:20]))),
            self.limits[12:20, 0], self.limits[12:20, 1])
        raw = self._raw_targets(frame)
        previous = base.copy() if self.last_qpos is None else self.last_qpos.copy()
        initial = np.clip(previous[12:20], self.limits[12:20, 0], self.limits[12:20, 1])

        def residual(x: np.ndarray) -> np.ndarray:
            q = base.copy(); q[12:20] = x
            actual = self._positions(q)
            values = [self.tip_weight / self.position_scale * (actual["thumb_tip"] - raw["thumb_tip"]),
                      self.tip_weight / self.position_scale * (actual["finger4_tip"] - raw["finger4_tip"]),
                      self.pad_weight / self.position_scale * (actual["thumb_pad"] - raw["thumb_tip"]),
                      self.pad_weight / self.position_scale * (actual["finger4_pad"] - raw["finger4_tip"])]
            for actual_a, actual_b, raw_a, raw_b in (("thumb_tip", "thumb_link4", "thumb_tip", "thumb_link4"),
                                                       ("finger4_tip", "finger4_link4", "finger4_tip", "finger4_link4"),
                                                       ("finger4_link4", "finger4_link3", "finger4_link4", "finger4_link3")):
                a = actual[actual_a] - actual[actual_b]; b = raw[raw_a] - raw[raw_b]
                values.append(self.direction_weight * (a / max(np.linalg.norm(a), 1e-12) - b / max(np.linalg.norm(b), 1e-12)))
            for i, finger in enumerate(("finger1", "finger2", "finger3", "finger4")):
                target = min(np.linalg.norm(raw["thumb_tip"] - raw[f"{finger}_tip"]), calibration.pinch_distances_m[i])
                if target < .04:
                    values.append(2.0 * (np.linalg.norm(actual["thumb_pad"] - actual[f"{finger}_pad"] if finger != "finger4" else actual["finger4_pad"]) - target) / self.position_scale)
            values.extend((self.prior_weight * (x - prior), self.temporal_weight / self.temporal_scale * (x - previous[12:20]), self.zero_weight * x))
            return np.concatenate([np.atleast_1d(value).ravel() for value in values])

        failure = None
        try:
            result = least_squares(residual, initial, bounds=(self.limits[12:20, 0], self.limits[12:20, 1]),
                                   loss="huber", f_scale=1.0, max_nfev=self.max_nfev)
            if not result.success or not np.isfinite(result.x).all():
                failure, solution = result.message, previous[12:20]
            else:
                solution = result.x
        except Exception as exc:
            result, failure, solution = None, str(exc), previous[12:20]
        qpos = base.copy(); qpos[12:20] = solution
        filtered = np.clip(self.filter.next(qpos), self.limits[:, 0], self.limits[:, 1])
        if failure is None:
            self.last_qpos = filtered.copy()
        actual = self._positions(filtered)
        diagnostics = {"qpos": filtered.copy(), "failure": failure, "solve_ms": (time.perf_counter() - start) * 1000,
                       "nfev": 0 if result is None else int(result.nfev), "manus_ergonomics": frame.ergonomics.copy(),
                       "ergonomics_direct_degrees": direct, "ergonomics_direct_qpos": base[:12].copy(),
                       "ergonomics_special_prior": prior, "pad_targets_m": np.stack((raw["thumb_tip"], raw["finger4_tip"])),
                       "pad_actual_m": np.stack((actual["thumb_pad"], actual["finger4_pad"])),
                       "manus_hybrid_raw_targets_m": np.stack((raw["thumb_tip"], raw["finger4_tip"])),
                       "manus_hybrid_pinch_distances_m": calibration.pinch_distances_m.copy()}
        self.last_diagnostics = diagnostics
        return filtered, diagnostics




__all__ = [
    "CALIBRATION_SCHEMA", "ERGONOMICS_CALIBRATION_SCHEMA", "ManusCalibration",
    "ManusErgonomicsCalibration", "ManusErgonomicsHybridRetargeter", "ManusFrame",
    "ManusFullSkeletonRetargeter", "ManusTopology", "average_quaternions_wxyz",
    "ergonomics_indices", "semantic_frame_name", "urdf_geometry_fingerprint", "wrist_local",
]
