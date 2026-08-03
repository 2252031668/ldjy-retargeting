"""Derive LDJY finger-pad frames from the visual CAD surface at zero pose."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .retarget_tip_frames import FINGERS, task_frame_axes


CAD_NAIL_TO_PULP = np.array((0.0, 1.0, 0.0))
# Coordinates are (lateral, distal) in the local basis
# ``cross(nail_to_pulp, PIP_to_DIP), PIP_to_DIP``.  They are the centers of
# the three through-holes on the CAD nail plate, measured once from the source
# mesh.  Keeping these explicit makes a source-asset change reviewable while
# the normal and opposite-side point still come from the actual mesh.
NAIL_HOLE_COORDINATES = {
    "thumb": np.array(((-0.00549, 0.02837), (0.00439, 0.02833), (-0.00061, 0.03617))),
    "finger1": np.array(((-0.00569, 0.01750), (0.00431, 0.01750), (-0.00057, 0.02554))),
    "finger2": np.array(((-0.00569, 0.01750), (0.00431, 0.01750), (-0.00057, 0.02554))),
    "finger3": np.array(((-0.00569, 0.01750), (0.00431, 0.01750), (-0.00057, 0.02554))),
    "finger4": np.array(((-0.00422, 0.02775), (0.00582, 0.02767), (0.00082, 0.03563))),
}
RAY_ORIGIN_MARGIN_M = 0.002
PAD_VISUAL_RADIUS_M = np.array((0.007, 0.006))


@dataclass(frozen=True)
class RobotPadFrame:
    """A right-handed pad frame expressed in its link4 parent coordinates."""

    position_m: np.ndarray
    rotation_parent_from_pad: np.ndarray
    radius_m: np.ndarray


def _mesh_in_link4_coordinates(model: mujoco.MjModel, finger: str) -> tuple[np.ndarray, np.ndarray]:
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{finger}_link4_visual_mesh_collision"
    )
    if geom_id < 0:
        raise ValueError(f"missing visual mesh for {finger}_link4")
    mesh_id = model.geom_dataid[geom_id]
    vertex_start, vertex_count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    face_start, face_count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    vertices = model.mesh_vert[vertex_start:vertex_start + vertex_count]
    faces = model.mesh_face[face_start:face_start + face_count]
    rotation = Rotation.from_quat(model.geom_quat[geom_id][[1, 2, 3, 0]]).as_matrix()
    return vertices @ rotation.T + model.geom_pos[geom_id], faces


def _furthest_mesh_hit(
    vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Return the far exterior hit for one ray using vectorized Moller-Trumbore."""
    triangles = vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    usable = np.abs(determinant) > 1e-10
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=usable)
    offset = origin - triangles[:, 0]
    bary_u = inverse * np.einsum("ij,ij->i", offset, h)
    q = np.cross(offset, edge1)
    bary_v = inverse * (q @ direction)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    hit = usable & (bary_u >= -1e-8) & (bary_v >= -1e-8)
    hit &= bary_u + bary_v <= 1.0 + 1e-8
    hit &= distance > 1e-7
    if not np.any(hit):
        raise ValueError("nail-hole ray does not intersect the opposite CAD surface")
    return origin + float(distance[hit].max()) * direction


def _pad_from_nail_holes(
    vertices: np.ndarray, faces: np.ndarray, distal: np.ndarray, pulp: np.ndarray, holes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a pad center and frame from the three nail holes and a mesh ray."""
    lateral = np.cross(pulp, distal)
    lateral /= np.linalg.norm(lateral)
    nail_coordinate = float((vertices @ pulp).min())
    hole_points = (
        holes[:, :1] * lateral
        + holes[:, 1:] * distal
        + nail_coordinate * pulp
    )
    nail_normal = np.cross(hole_points[1] - hole_points[0], hole_points[2] - hole_points[0])
    nail_normal /= np.linalg.norm(nail_normal)
    if np.dot(nail_normal, pulp) < 0.0:
        nail_normal *= -1.0
    nail_center = hole_points.mean(axis=0)
    center = _furthest_mesh_hit(
        vertices,
        faces,
        nail_center - RAY_ORIGIN_MARGIN_M * nail_normal,
        nail_normal,
    )
    tangent = distal - nail_normal * np.dot(distal, nail_normal)
    tangent /= np.linalg.norm(tangent)
    rotation = np.column_stack((tangent, np.cross(nail_normal, tangent), nail_normal))
    return center, rotation, PAD_VISUAL_RADIUS_M.copy()


def pad_frames_from_hand_model(
    model: mujoco.MjModel,
    *,
    side: str = "",
    surface_reference_world: np.ndarray = CAD_NAIL_TO_PULP,
) -> dict[str, RobotPadFrame]:
    """Return five zero-pose link4-local pad frames from a compiled hand model."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result: dict[str, RobotPadFrame] = {}
    for finger in FINGERS:
        axis, pulp = task_frame_axes(
            model,
            data,
            finger,
            side=side,
            surface_reference_world=surface_reference_world,
        )
        center, rotation, radius = _pad_from_nail_holes(
            *_mesh_in_link4_coordinates(model, finger),
            axis,
            pulp,
            NAIL_HOLE_COORDINATES[finger],
        )
        result[finger] = RobotPadFrame(center, rotation, radius)
    return result


def pad_frames_from_source(model: mujoco.MjModel) -> dict[str, RobotPadFrame]:
    """Return five zero-pose frames from the original unprefixed right CAD hand."""
    return pad_frames_from_hand_model(model)
