"""MuJoCo overlays and diagnostics for adaptive LDJY retargeting."""

from __future__ import annotations

import mujoco
import numpy as np

from ..opt.base import M_TO_CM


FINGERS = (
    ("thumb", "TH"),
    ("finger1", "F1"),
    ("finger2", "F2"),
    ("finger3", "F3"),
    ("finger4", "F4"),
)
ACTUAL_JOINT_RGBA = np.array((1.0, 0.85, 0.1, 1.0))
ACTUAL_LINK_RGBA = np.array((0.1, 0.85, 0.9, 0.95))
TARGET_RGBA = np.array((0.25, 1.0, 0.25, 0.95))
IDENTITY_MAT = np.eye(3).ravel()


def mode_label(alpha: float) -> str:
    """Return the dominant adaptive loss mode without hiding blend weights."""
    if alpha <= 0.05:
        return "FullHandVec"
    if alpha >= 0.65:
        return f"TipDirVec(alpha={alpha:.2f})"
    return f"Blend(alpha={alpha:.2f})"


def _object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Model is missing {name}")
    return object_id


def _hand_prefix(model: mujoco.MjModel) -> str:
    for prefix in ("right", "left"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_palm") >= 0:
            return prefix
    raise ValueError("Model is missing an LDJY palm body")


def _add_sphere(scene, position, label, rgba, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((radius, 0.0, 0.0)),
        np.asarray(position, dtype=np.float64),
        IDENTITY_MAT,
        rgba,
    )
    geom.label = label
    scene.ngeom += 1


def _add_link(scene, start, end, label, rgba, width):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        IDENTITY_MAT,
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.label = label
    scene.ngeom += 1


def _draw_hand(model, data, scene, joint_rgba, link_rgba, label_prefix, radius, width):
    prefix = _hand_prefix(model)
    wrist_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_retarget_wrist")
    wrist = data.xpos[wrist_id].copy()
    _add_sphere(scene, wrist, f"{label_prefix} wrist", joint_rgba, radius)

    for finger, short_name in FINGERS:
        parent = wrist
        for number in range(1, 5):
            joint_id = _object_id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{finger}_joint{number}"
            )
            position = data.xanchor[joint_id].copy()
            _add_link(scene, parent, position, f"{label_prefix} {short_name}", link_rgba, width)
            _add_sphere(scene, position, f"{label_prefix} {short_name} J{number}", joint_rgba, radius)
            parent = position

        site_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_{finger}_link4_tip"
        )
        tip = data.site_xpos[site_id].copy()
        _add_link(scene, parent, tip, f"{label_prefix} {short_name}", link_rgba, width)
        _add_sphere(scene, tip, f"{label_prefix} {short_name} tip", joint_rgba, radius)


def _active_target_segments(optimizer, mediapipe_keypoints, pinch_alphas):
    """Return current FullHand and TipDir target vectors in wrist-local meters."""
    if not all(
        hasattr(optimizer, method)
        for method in ("_compute_full_hand_vectors", "_compute_tip_vectors", "_compute_tip_dirs")
    ):
        return []

    full_hand = optimizer._compute_full_hand_vectors(
        mediapipe_keypoints, optimizer.segment_scaling
    ) / M_TO_CM
    tip_vectors = optimizer._compute_tip_vectors(
        mediapipe_keypoints, optimizer.scaling
    ) / M_TO_CM
    tip_dirs = optimizer._compute_tip_dirs(mediapipe_keypoints)
    dip_to_tip_length = np.linalg.norm(
        mediapipe_keypoints[optimizer.MP_TIP_INDICES]
        - mediapipe_keypoints[optimizer.MP_DIP_INDICES],
        axis=1,
    )

    segments = []
    origin = np.zeros(3, dtype=np.float64)
    for finger_index, alpha in enumerate(pinch_alphas):
        if 1.0 - alpha > 0.01:
            for offset, name in ((0, "PIP"), (5, "DIP"), (10, "TIP")):
                segments.append((origin, full_hand[offset + finger_index], f"Full {name}"))
        if alpha > 0.01:
            segments.append((origin, tip_vectors[finger_index], "TipPos"))
            dip_target = full_hand[5 + finger_index]
            segments.append(
                (
                    dip_target,
                    dip_target + tip_dirs[finger_index] * dip_to_tip_length[finger_index],
                    "TipDir",
                )
            )
    return segments


class DebugOverlay:
    """Draw physical pose and active adaptive target vectors."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model

    def draw(self, scene, actual_data, optimizer, mediapipe_keypoints, pinch_alphas) -> None:
        scene.ngeom = 0
        _draw_hand(
            self.model,
            actual_data,
            scene,
            ACTUAL_JOINT_RGBA,
            ACTUAL_LINK_RGBA,
            "actual",
            radius=0.0035,
            width=0.0015,
        )
        prefix = _hand_prefix(self.model)
        wrist_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_retarget_wrist"
        )
        wrist_pos = actual_data.xpos[wrist_id].copy()
        wrist_rot = actual_data.xmat[wrist_id].reshape(3, 3).copy()
        for start, end, label in _active_target_segments(
            optimizer, mediapipe_keypoints, pinch_alphas
        ):
            world_start = start @ wrist_rot.T + wrist_pos
            world_end = end @ wrist_rot.T + wrist_pos
            _add_link(scene, world_start, world_end, label, TARGET_RGBA, width=0.00055)
            _add_sphere(scene, world_end, label, TARGET_RGBA, radius=0.0018)


def format_joint_diagnostics(model, data, actuator_targets, joint_mode_labels, cost):
    """Build concise terminal diagnostics for command, actual, and adaptive mode."""
    lines = [f"DEBUG cost={cost:.4f} modes=" + ", ".join(joint_mode_labels)]
    for actuator_id, target in enumerate(actuator_targets):
        joint_id = model.actuator_trnid[actuator_id, 0]
        qpos_address = model.jnt_qposadr[joint_id]
        actual = data.qpos[qpos_address]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        lower, upper = model.actuator_ctrlrange[actuator_id]
        lines.append(
            f"  {name}: cmd={target:+.3f} actual={actual:+.3f} "
            f"err={actual - target:+.3f} range=[{lower:+.3f}, {upper:+.3f}]"
        )
    return "\n".join(lines)
