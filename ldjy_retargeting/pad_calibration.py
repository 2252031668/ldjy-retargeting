"""Surface-constrained LDJY finger-pad calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import yaml


FINGERS = ("finger1", "finger2", "finger3", "thumb", "finger4")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAD_POINTS_PATH = (
    ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "retarget_pad_points.yaml"
)


@dataclass(frozen=True)
class SurfacePoint:
    """A point on a mesh face, represented by its barycentric coordinates."""

    face: int
    barycentric: np.ndarray


class TriangleSurface:
    """Walk a point over a triangle mesh without leaving its surface."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (n, 3)")
        if not len(self.faces):
            raise ValueError("surface has no faces")
        self.triangles = self.vertices[self.faces]
        edge01 = self.triangles[:, 1] - self.triangles[:, 0]
        edge02 = self.triangles[:, 2] - self.triangles[:, 0]
        raw_normals = np.cross(edge01, edge02)
        lengths = np.linalg.norm(raw_normals, axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("surface contains degenerate faces")
        self.normals = raw_normals / lengths[:, None]
        self._adjacent = self._build_adjacency()

    def _build_adjacency(self) -> np.ndarray:
        adjacent = np.full((len(self.faces), 3), -1, dtype=np.int64)
        edges: dict[tuple[int, int], tuple[int, int]] = {}
        for face_id, face in enumerate(self.faces):
            for opposite in range(3):
                edge = tuple(sorted((face[(opposite + 1) % 3], face[(opposite + 2) % 3])))
                previous = edges.get(edge)
                if previous is None:
                    edges[edge] = (face_id, opposite)
                else:
                    other_face, other_opposite = previous
                    adjacent[face_id, opposite] = other_face
                    adjacent[other_face, other_opposite] = face_id
        return adjacent

    def position(self, point: SurfacePoint) -> np.ndarray:
        return point.barycentric @ self.triangles[point.face]

    def _barycentric(self, face_id: int, position: np.ndarray) -> np.ndarray:
        triangle = self.triangles[face_id]
        edge01 = triangle[1] - triangle[0]
        edge02 = triangle[2] - triangle[0]
        relative = position - triangle[0]
        dot00 = edge01 @ edge01
        dot01 = edge01 @ edge02
        dot11 = edge02 @ edge02
        dot20 = relative @ edge01
        dot21 = relative @ edge02
        denominator = dot00 * dot11 - dot01 * dot01
        second = (dot11 * dot20 - dot01 * dot21) / denominator
        third = (dot00 * dot21 - dot01 * dot20) / denominator
        return np.array((1.0 - second - third, second, third))

    def project(self, position: np.ndarray) -> SurfacePoint:
        """Return the closest point on the surface to ``position``."""
        point = np.asarray(position, dtype=float)
        triangles = self.triangles
        base = triangles[:, 0]
        edge01 = triangles[:, 1] - base
        edge02 = triangles[:, 2] - base
        normals = self.normals
        plane_points = point - ((point - base) * normals).sum(axis=1)[:, None] * normals

        dot00 = (edge01 * edge01).sum(axis=1)
        dot01 = (edge01 * edge02).sum(axis=1)
        dot11 = (edge02 * edge02).sum(axis=1)
        relative = plane_points - base
        dot20 = (relative * edge01).sum(axis=1)
        dot21 = (relative * edge02).sum(axis=1)
        denominator = dot00 * dot11 - dot01 * dot01
        second = (dot11 * dot20 - dot01 * dot21) / denominator
        third = (dot00 * dot21 - dot01 * dot20) / denominator
        barycentric = np.column_stack((1.0 - second - third, second, third))
        inside = np.all(barycentric >= 0.0, axis=1)

        def nearest_segment(start: np.ndarray, end: np.ndarray) -> np.ndarray:
            direction = end - start
            fraction = ((point - start) * direction).sum(axis=1) / (direction * direction).sum(axis=1)
            return start + np.clip(fraction, 0.0, 1.0)[:, None] * direction

        edges = (
            nearest_segment(triangles[:, 0], triangles[:, 1]),
            nearest_segment(triangles[:, 1], triangles[:, 2]),
            nearest_segment(triangles[:, 2], triangles[:, 0]),
        )
        edge_distances = np.column_stack([((candidate - point) ** 2).sum(axis=1) for candidate in edges])
        edge_choice = edge_distances.argmin(axis=1)
        edge_points = np.stack(edges)[edge_choice, np.arange(len(triangles))]
        candidates = np.where(inside[:, None], plane_points, edge_points)
        face_id = int(((candidates - point) ** 2).sum(axis=1).argmin())
        return SurfacePoint(face_id, self._barycentric(face_id, candidates[face_id]))

    def move(self, point: SurfacePoint, direction: np.ndarray, distance: float) -> SurfacePoint:
        """Move along adjacent triangles, stopping at an open mesh boundary."""
        if distance < 0.0:
            raise ValueError("distance must be non-negative")
        current = SurfacePoint(point.face, np.asarray(point.barycentric, dtype=float).copy())
        remaining = float(distance)
        requested = np.asarray(direction, dtype=float)
        for _ in range(len(self.faces) + 1):
            normal = self.normals[current.face]
            tangent = requested - normal * (requested @ normal)
            tangent_length = np.linalg.norm(tangent)
            if remaining <= 1e-12 or tangent_length <= 1e-12:
                return current
            delta = tangent / tangent_length * remaining
            start = self.position(current)
            target = start + delta
            target_barycentric = self._barycentric(current.face, target)
            if np.all(target_barycentric >= -1e-10):
                return SurfacePoint(current.face, np.maximum(target_barycentric, 0.0) / target_barycentric.sum())

            crossed = np.where(target_barycentric < -1e-10)[0]
            fractions = current.barycentric[crossed] / (
                current.barycentric[crossed] - target_barycentric[crossed]
            )
            crossed_opposite = int(crossed[np.argmin(fractions)])
            fraction = float(fractions.min())
            boundary = start + delta * fraction
            neighbour = self._adjacent[current.face, crossed_opposite]
            if neighbour < 0:
                return SurfacePoint(current.face, self._barycentric(current.face, boundary))
            remaining *= 1.0 - fraction
            current = SurfacePoint(int(neighbour), self._barycentric(int(neighbour), boundary))
        return current

    @staticmethod
    def _normal_transport(old_normal: np.ndarray, new_normal: np.ndarray) -> np.ndarray:
        """Return the smallest rotation taking one face normal to the next."""
        axis = np.cross(old_normal, new_normal)
        sine = float(np.linalg.norm(axis))
        cosine = float(np.clip(old_normal @ new_normal, -1.0, 1.0))
        if sine <= 1e-12:
            return np.eye(3)
        axis /= sine
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)

    def move_with_transport(
        self, point: SurfacePoint, direction: np.ndarray, distance: float
    ) -> tuple[SurfacePoint, np.ndarray]:
        """Move across sharp edges while parallel-transporting a tangent direction."""
        if distance < 0.0:
            raise ValueError("distance must be non-negative")
        current = SurfacePoint(point.face, np.asarray(point.barycentric, dtype=float).copy())
        tangent = np.asarray(direction, dtype=float)
        tangent -= self.normals[current.face] * (tangent @ self.normals[current.face])
        length = np.linalg.norm(tangent)
        transport = np.eye(3)
        if length <= 1e-12:
            return current, transport
        tangent /= length
        remaining = float(distance)
        for _ in range(len(self.faces) + 1):
            if remaining <= 1e-12:
                return current, transport
            start = self.position(current)
            target = start + tangent * remaining
            target_barycentric = self._barycentric(current.face, target)
            if np.all(target_barycentric >= -1e-10):
                return (
                    SurfacePoint(
                        current.face,
                        np.maximum(target_barycentric, 0.0) / target_barycentric.sum(),
                    ),
                    transport,
                )
            crossed = np.where(target_barycentric < -1e-10)[0]
            fractions = current.barycentric[crossed] / (
                current.barycentric[crossed] - target_barycentric[crossed]
            )
            crossed_opposite = int(crossed[np.argmin(fractions)])
            fraction = float(fractions.min())
            boundary = start + tangent * remaining * fraction
            neighbour = self._adjacent[current.face, crossed_opposite]
            if neighbour < 0:
                return SurfacePoint(current.face, self._barycentric(current.face, boundary)), transport
            step_transport = self._normal_transport(
                self.normals[current.face], self.normals[neighbour]
            )
            tangent = step_transport @ tangent
            transport = step_transport @ transport
            remaining *= 1.0 - fraction
            current = SurfacePoint(int(neighbour), self._barycentric(int(neighbour), boundary))
        return current, transport


def link4_visual_surface(
    model: mujoco.MjModel, data: mujoco.MjData, finger: str, side: str = ""
) -> TriangleSurface:
    """Return a distal visual mesh in its link4 local coordinates."""
    prefix = f"{side}_" if side else ""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{finger}_link4")
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{finger}_link4_visual_mesh_collision"
    )
    if min(body_id, geom_id) < 0:
        raise ValueError(f"missing visual link4 mesh for {prefix}{finger}")
    mesh_id = model.geom_dataid[geom_id]
    vertices = model.mesh_vert[
        model.mesh_vertadr[mesh_id]: model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]
    ]
    faces = model.mesh_face[
        model.mesh_faceadr[mesh_id]: model.mesh_faceadr[mesh_id] + model.mesh_facenum[mesh_id]
    ].reshape(-1, 3)
    rotation = data.xmat[body_id].reshape(3, 3)
    world_vertices = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    local_vertices = (world_vertices - data.xpos[body_id]) @ rotation
    return TriangleSurface(local_vertices, faces)


def load_pad_points(path: Path = DEFAULT_PAD_POINTS_PATH) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != set(FINGERS):
        raise ValueError(f"{path} must define exactly: {', '.join(FINGERS)}")
    points = {finger: np.asarray(payload[finger], dtype=float) for finger in FINGERS}
    if any(point.shape != (3,) or not np.isfinite(point).all() for point in points.values()):
        raise ValueError(f"{path} must contain three finite coordinates per finger")
    return points


def save_pad_points(points: dict[str, np.ndarray], path: Path = DEFAULT_PAD_POINTS_PATH) -> Path:
    if set(points) != set(FINGERS):
        raise ValueError(f"points must define exactly: {', '.join(FINGERS)}")
    payload = {finger: [float(value) for value in np.asarray(points[finger])] for finger in FINGERS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
