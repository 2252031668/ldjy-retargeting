#!/usr/bin/env python3
"""Inspect a semi-transparent MANO hand over the right LDJY hand.

The MuJoCo Control panel owns all 20 LDJY controls.  The Qt window controls
MANO parameters and a static MANO-to-LDJY registration.

Usage:
    .venv/bin/python example/mano_ldjy_overlay_viewer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.transform import Rotation

from ldjy_retargeting.mano_conventions import (
    MANO_ACTIVE_KEYPOINT_INDICES,
    MANO_ACTIVE_SKELETON_EDGES,
    MEDIAPIPE_21_SKELETON_EDGES,
    MEDIAPIPE_FINGER_DIRECTION_JOINTS,
)
from ldjy_retargeting.mano_ldjy_overlay import (
    OverlayScene,
    average_surface_normal,
    apply_registration,
    fit_static_registration,
    solve_exact_hand_pose_constraints,
)

from mano_viewer import (
    FINGER_JOINT_NAMES,
    FINGER_ORDER,
    MANO_MODEL_PATH,
    PAD_3PT_VERTEX_IDS,
    PAD_VERTEX_IDS,
    TIP_ORDER,
    MANOModel,
    SliderWithSpinbox,
)


ROOT = Path(__file__).resolve().parents[1]
LDJY_MJCF = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "mjcf" / "ldjy_right_hand.xml"
REFERENCE_PATH = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "mano_ldjy_reference.yaml"
LDJY_FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")
FINGER_LABELS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
LDJY_DIRECTION_JOINTS = (
    ("thumb", 2, 3),
    ("finger1", 2, 3),
    ("finger2", 2, 3),
    ("finger3", 2, 3),
    ("finger4", 3, 4),
)
MANO_DIRECTION_JOINTS = MEDIAPIPE_FINGER_DIRECTION_JOINTS
LDJY_PALM_NORMAL_JOINTS = (("thumb", 2), ("finger1", 2), ("finger3", 2))
MANO_PALM_NORMAL_KEYPOINTS = (0, 5, 17)
PALM_NORMAL_LENGTH = 0.04
IDENTITY_MAT = np.eye(3).ravel()
FIT_DEFAULTS = {
    "use_positions": True,
    "position_weight": 1.0,
    "position_mask": [True] * 5,
    "use_normals": True,
    "normal_weight": 0.5,
    "normal_mask": [True] * 5,
    "use_directions": True,
    "direction_weight": 0.5,
    "direction_mask": [True] * 5,
    "use_direction_lines": False,
    "direction_line_mask": [True] * 5,
    "use_palm_normal": True,
    "palm_normal_weight": 0.5,
    "use_straight_fingers": False,
    "straight_fingers_weight": 0.5,
    "straight_fingers_exact": False,
    "straight_finger_mask": [True] * 5,
    "use_palm_plane": False,
    "palm_plane_weight": 0.5,
    "palm_plane_exact": False,
    "use_hand_pose_prior": True,
    "hand_pose_prior_weight": 0.1,
    "hand_pose_prior_reference": "current",
    "fit_hand_joints": [False] * 15,
    "fit_betas": True,
    "beta_bound": 3.0,
    "fit_rotation": True,
    "fit_translation": True,
    "fit_scale": True,
}


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


def _add_line(scene, start, end, label, rgba, width=0.00035):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), IDENTITY_MAT, rgba
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


def _plane_normal(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    return np.cross(second - first, third - first)


class FitSettingsDialog(QDialog):
    """Modal settings panel for one explicit static-registration run."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Static Fit Settings")
        layout = QVBoxLayout(self)

        constraints = QGroupBox("Constraints")
        constraint_layout = QVBoxLayout(constraints)
        self.constraint_controls = {}
        self.direction_lines = None
        self.direction_line_controls = []
        for key, label, weight_key in (
            ("use_positions", "Pad positions", "position_weight"),
            ("use_normals", "Pad normals", "normal_weight"),
            ("use_directions", "Finger directions", "direction_weight"),
            ("use_palm_normal", "Palm normal", "palm_normal_weight"),
        ):
            row = QHBoxLayout()
            enabled = QCheckBox(label)
            enabled.setChecked(settings.get(key, FIT_DEFAULTS[key]))
            weight = QDoubleSpinBox()
            weight.setRange(0.001, 100.0)
            weight.setDecimals(3)
            weight.setSingleStep(0.1)
            weight.setValue(settings.get(weight_key, FIT_DEFAULTS[weight_key]))
            mask_key = {
                "use_positions": "position_mask",
                "use_normals": "normal_mask",
                "use_directions": "direction_mask",
            }.get(key)
            selected_fingers = settings.get(mask_key, FIT_DEFAULTS[mask_key]) if mask_key else []
            if mask_key and len(selected_fingers) != 5:
                selected_fingers = FIT_DEFAULTS[mask_key]
            finger_controls = []
            for finger, selected in zip(FINGER_LABELS, selected_fingers):
                checkbox = QCheckBox(finger)
                checkbox.setChecked(bool(selected))
                finger_controls.append(checkbox)

            def sync_constraint(_checked=False, enabled=enabled, weight=weight, controls=finger_controls):
                weight.setEnabled(enabled.isChecked())
                for control in controls:
                    control.setEnabled(enabled.isChecked())

            enabled.toggled.connect(sync_constraint)
            sync_constraint()
            row.addWidget(enabled)
            row.addWidget(QLabel("weight"))
            row.addWidget(weight)
            for control in finger_controls:
                row.addWidget(control)
            if key == "use_directions":
                self.direction_lines = QCheckBox("same line")
                self.direction_lines.setChecked(
                    settings.get("use_direction_lines", FIT_DEFAULTS["use_direction_lines"])
                )
                row.addWidget(self.direction_lines)
                selected_lines = settings.get("direction_line_mask", FIT_DEFAULTS["direction_line_mask"])
                if len(selected_lines) != 5:
                    selected_lines = FIT_DEFAULTS["direction_line_mask"]
                for finger, selected in zip(FINGER_LABELS, selected_lines):
                    checkbox = QCheckBox(finger)
                    checkbox.setChecked(bool(selected))
                    self.direction_line_controls.append(checkbox)
                    row.addWidget(checkbox)

                def sync_direction_lines(_checked=False, enabled=enabled):
                    active = enabled.isChecked() and self.direction_lines.isChecked()
                    self.direction_lines.setEnabled(enabled.isChecked())
                    for control in self.direction_line_controls:
                        control.setEnabled(active)

                enabled.toggled.connect(sync_direction_lines)
                self.direction_lines.toggled.connect(sync_direction_lines)
                sync_direction_lines()
            constraint_layout.addLayout(row)
            self.constraint_controls[key] = (enabled, weight, weight_key, finger_controls, mask_key)
        layout.addWidget(constraints)

        geometry = QGroupBox("MANO geometry priors")
        geometry_layout = QVBoxLayout(geometry)
        self.geometry_controls = {}
        self.straight_finger_controls = []
        for key, label, weight_key, exact_key in (
            ("use_straight_fingers", "Straight fingers", "straight_fingers_weight", "straight_fingers_exact"),
            ("use_palm_plane", "Finger directions in palm plane", "palm_plane_weight", "palm_plane_exact"),
        ):
            row = QHBoxLayout()
            enabled = QCheckBox(label)
            enabled.setChecked(settings.get(key, FIT_DEFAULTS[key]))
            weight = QDoubleSpinBox()
            weight.setRange(0.001, 100.0)
            weight.setDecimals(3)
            weight.setSingleStep(0.1)
            weight.setValue(settings.get(weight_key, FIT_DEFAULTS[weight_key]))
            exact = QCheckBox("linear constraint")
            exact.setChecked(settings.get(exact_key, FIT_DEFAULTS[exact_key]))
            finger_controls = []
            if key == "use_straight_fingers":
                selected_fingers = settings.get("straight_finger_mask", FIT_DEFAULTS["straight_finger_mask"])
                if len(selected_fingers) != 5:
                    selected_fingers = FIT_DEFAULTS["straight_finger_mask"]
                for finger, selected in zip(FINGER_LABELS, selected_fingers):
                    checkbox = QCheckBox(finger)
                    checkbox.setChecked(bool(selected))
                    finger_controls.append(checkbox)
                    self.straight_finger_controls.append(checkbox)

            def sync_weight(_checked=False, enabled=enabled, exact=exact, weight=weight, controls=finger_controls):
                weight.setEnabled(enabled.isChecked() and not exact.isChecked())
                for control in controls:
                    control.setEnabled(enabled.isChecked())

            enabled.toggled.connect(sync_weight)
            exact.toggled.connect(sync_weight)
            sync_weight()
            row.addWidget(enabled)
            row.addWidget(QLabel("weight"))
            row.addWidget(weight)
            row.addWidget(exact)
            for control in finger_controls:
                row.addWidget(control)
            geometry_layout.addLayout(row)
            self.geometry_controls[key] = (enabled, weight, weight_key, exact, exact_key)

        prior_row = QHBoxLayout()
        self.hand_pose_prior = QCheckBox("Hand pose prior")
        self.hand_pose_prior.setChecked(settings.get("use_hand_pose_prior", FIT_DEFAULTS["use_hand_pose_prior"]))
        self.hand_pose_prior_weight = QDoubleSpinBox()
        self.hand_pose_prior_weight.setRange(0.001, 100.0)
        self.hand_pose_prior_weight.setDecimals(3)
        self.hand_pose_prior_weight.setSingleStep(0.01)
        self.hand_pose_prior_weight.setValue(
            settings.get("hand_pose_prior_weight", FIT_DEFAULTS["hand_pose_prior_weight"])
        )
        self.hand_pose_prior_reference = QComboBox()
        self.hand_pose_prior_reference.addItem("Current GUI pose", "current")
        self.hand_pose_prior_reference.addItem("Zero pose", "zero")
        reference_index = self.hand_pose_prior_reference.findData(
            settings.get("hand_pose_prior_reference", FIT_DEFAULTS["hand_pose_prior_reference"])
        )
        self.hand_pose_prior_reference.setCurrentIndex(max(reference_index, 0))
        self.hand_pose_prior.toggled.connect(self.hand_pose_prior_weight.setEnabled)
        self.hand_pose_prior.toggled.connect(self.hand_pose_prior_reference.setEnabled)
        prior_row.addWidget(self.hand_pose_prior)
        prior_row.addWidget(QLabel("weight"))
        prior_row.addWidget(self.hand_pose_prior_weight)
        prior_row.addWidget(self.hand_pose_prior_reference)
        geometry_layout.addLayout(prior_row)
        layout.addWidget(geometry)

        variables = QGroupBox("Variables to optimize")
        variable_layout = QVBoxLayout(variables)
        self.variable_controls = {}
        for key, label in (
            ("fit_betas", "betas (10)"),
            ("fit_rotation", "static rotation"),
            ("fit_translation", "static translation"),
            ("fit_scale", "static scale"),
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(settings.get(key, FIT_DEFAULTS[key]))
            variable_layout.addWidget(checkbox)
            self.variable_controls[key] = checkbox
        self.beta_bound = SliderWithSpinbox(
            "auto beta bound", 0.5, 10.0,
            settings.get("beta_bound", FIT_DEFAULTS["beta_bound"]), 0.1,
        )
        self.beta_bound.setEnabled(self.variable_controls["fit_betas"].isChecked())
        self.variable_controls["fit_betas"].toggled.connect(self.beta_bound.setEnabled)
        variable_layout.addWidget(self.beta_bound)
        layout.addWidget(variables)

        hand_pose_group = QGroupBox("Hand pose joints to optimize (axis-angle)")
        hand_pose_layout = QVBoxLayout(hand_pose_group)
        selected_joints = settings.get("fit_hand_joints", FIT_DEFAULTS["fit_hand_joints"])
        if len(selected_joints) != 15:
            selected_joints = FIT_DEFAULTS["fit_hand_joints"]
        self.hand_pose_controls = []
        for finger_index, finger in enumerate(FINGER_ORDER):
            finger_group = QGroupBox(finger)
            finger_layout = QHBoxLayout(finger_group)
            for joint_index, joint_name in enumerate(FINGER_JOINT_NAMES[finger]):
                checkbox = QCheckBox(joint_name)
                checkbox.setChecked(bool(selected_joints[finger_index * 3 + joint_index]))
                finger_layout.addWidget(checkbox)
                self.hand_pose_controls.append(checkbox)
            hand_pose_layout.addWidget(finger_group)
        layout.addWidget(hand_pose_group)

        actions = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        fit = QPushButton("Fit")
        fit.clicked.connect(self.accept)
        actions.addStretch()
        actions.addWidget(cancel)
        actions.addWidget(fit)
        layout.addLayout(actions)

    def settings(self):
        settings = {}
        for key, (enabled, weight, weight_key, finger_controls, mask_key) in self.constraint_controls.items():
            settings[key] = enabled.isChecked()
            settings[weight_key] = weight.value()
            if mask_key:
                settings[mask_key] = [checkbox.isChecked() for checkbox in finger_controls]
        settings["use_direction_lines"] = self.direction_lines.isChecked()
        settings["direction_line_mask"] = [checkbox.isChecked() for checkbox in self.direction_line_controls]
        for key, (enabled, weight, weight_key, exact, exact_key) in self.geometry_controls.items():
            settings[key] = enabled.isChecked()
            settings[weight_key] = weight.value()
            settings[exact_key] = exact.isChecked()
        settings["use_hand_pose_prior"] = self.hand_pose_prior.isChecked()
        settings["hand_pose_prior_weight"] = self.hand_pose_prior_weight.value()
        settings["hand_pose_prior_reference"] = self.hand_pose_prior_reference.currentData()
        settings["fit_hand_joints"] = [checkbox.isChecked() for checkbox in self.hand_pose_controls]
        settings["straight_finger_mask"] = [checkbox.isChecked() for checkbox in self.straight_finger_controls]
        for key, checkbox in self.variable_controls.items():
            settings[key] = checkbox.isChecked()
        settings["beta_bound"] = self.beta_bound.value()
        return settings


class OverlayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MANO / LDJY Registration Debug")
        self.setMinimumWidth(440)
        if not MANO_MODEL_PATH.exists():
            raise FileNotFoundError(f"MANO model not found: {MANO_MODEL_PATH}")
        if not LDJY_MJCF.exists():
            raise FileNotFoundError(f"LDJY MJCF not found: {LDJY_MJCF}")

        self.mano = MANOModel(str(MANO_MODEL_PATH))
        self._expanded_to_orig = self.mano.faces.reshape(-1).astype(np.int64)
        self._expanded_faces = np.arange(len(self._expanded_to_orig), dtype=np.int32).reshape(-1, 3)
        self.scene = OverlayScene(
            LDJY_MJCF,
            self.mano.default_vertices[self._expanded_to_orig],
            self._expanded_faces,
        )
        self.scene.set_ldjy_alpha(0.35)
        self.fit_settings = dict(FIT_DEFAULTS)
        self._last_mano_state = None
        self._last_mano_output = None
        self.viewer = mujoco.viewer.launch_passive(self.scene.model, self.scene.data)
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 0.5
        self.viewer.cam.lookat[:] = [0.0, 0.0, 0.05]

        self._build_ui()
        self._load_reference()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)
        self.status.setText("Use MuJoCo Control for LDJY; use this window for MANO and static registration.")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        actions = QHBoxLayout()
        reset = QPushButton("Reset MANO")
        reset.clicked.connect(self._reset_mano)
        auto_fit = QPushButton("Fit settings")
        auto_fit.clicked.connect(self._show_fit_settings)
        save = QPushButton("Save reference")
        save.clicked.connect(self._save_reference)
        for button in (reset, auto_fit, save):
            actions.addWidget(button)
        root.addLayout(actions)

        visibility = QVBoxLayout()
        ldjy_visibility = QHBoxLayout()
        self.show_ldjy_skeleton = QCheckBox("LDJY skeleton")
        self.show_ldjy_tips = QCheckBox("LDJY tip sites")
        self.show_ldjy_pads = QCheckBox("LDJY pad points")
        self.show_ldjy_normals = QCheckBox("LDJY pad normals")
        for checkbox in (
            self.show_ldjy_skeleton,
            self.show_ldjy_tips,
            self.show_ldjy_pads,
            self.show_ldjy_normals,
        ):
            checkbox.setChecked(True)
            ldjy_visibility.addWidget(checkbox)
        visibility.addLayout(ldjy_visibility)
        mano_visibility = QHBoxLayout()
        self.show_mano_keypoints = QCheckBox("MANO keypoints + skeleton")
        self.show_mano_active_skeleton = QCheckBox("MANO active skeleton")
        self.show_mano_keypoints.setChecked(True)
        self.show_mano_active_skeleton.setChecked(False)
        mano_visibility.addWidget(self.show_mano_keypoints)
        mano_visibility.addWidget(self.show_mano_active_skeleton)
        visibility.addLayout(mano_visibility)
        palm_visibility = QHBoxLayout()
        self.show_ldjy_palm_normal = QCheckBox("LDJY palm normal")
        self.show_mano_palm_normal = QCheckBox("MANO palm normal")
        self.show_ldjy_palm_normal.setChecked(False)
        self.show_mano_palm_normal.setChecked(False)
        palm_visibility.addWidget(self.show_ldjy_palm_normal)
        palm_visibility.addWidget(self.show_mano_palm_normal)
        visibility.addLayout(palm_visibility)
        root.addLayout(visibility)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        content = QWidget()
        layout = QVBoxLayout(content)
        self.beta_sliders = self._add_vector_group(layout, "Betas (shape)", "beta", 10, -10.0, 10.0, 0.01)
        self.global_orient_sliders = self._add_vector_group(
            layout, "MANO global orient (axis-angle rad)", "axis", 3, -np.pi, np.pi, 0.01
        )

        pose_group = QGroupBox("MANO hand pose (15 joints x 3 axis-angle rad)")
        pose_layout = QVBoxLayout(pose_group)
        self.hand_pose_sliders = []
        for finger in FINGER_ORDER:
            finger_group = QGroupBox(finger)
            finger_layout = QVBoxLayout(finger_group)
            finger_sliders = []
            for joint_name in FINGER_JOINT_NAMES[finger]:
                axes = []
                for axis in "XYZ":
                    slider = SliderWithSpinbox(f"{joint_name} {axis}", -np.pi, np.pi, 0.0, 0.01)
                    finger_layout.addWidget(slider)
                    axes.append(slider)
                finger_sliders.append(axes)
            pose_layout.addWidget(finger_group)
            self.hand_pose_sliders.append(finger_sliders)
        layout.addWidget(pose_group)

        self.translation_sliders = self._add_vector_group(
            layout, "MANO translation (m)", "pos", 3, -0.2, 0.2, 0.001
        )
        scale_group = QGroupBox("MANO runtime scale")
        scale_layout = QVBoxLayout(scale_group)
        self.mano_scale = SliderWithSpinbox("scale", 0.1, 3.0, 1.0, 0.01)
        scale_layout.addWidget(self.mano_scale)
        layout.addWidget(scale_group)

        registration = QGroupBox("Static MANO -> LDJY registration")
        registration_layout = QVBoxLayout(registration)
        self.registration_rotation = []
        self.registration_translation = []
        for axis in "XYZ":
            slider = SliderWithSpinbox(f"R {axis}", -np.pi, np.pi, 0.0, 0.01)
            registration_layout.addWidget(slider)
            self.registration_rotation.append(slider)
        for axis in "XYZ":
            slider = SliderWithSpinbox(f"t {axis} (m)", -0.2, 0.2, 0.0, 0.001)
            registration_layout.addWidget(slider)
            self.registration_translation.append(slider)
        self.registration_scale = SliderWithSpinbox("s", 0.5, 2.0, 1.0, 0.001)
        registration_layout.addWidget(self.registration_scale)
        layout.addWidget(registration)

        layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    @staticmethod
    def _add_vector_group(layout, title, prefix, count, minimum, maximum, step):
        group = QGroupBox(title)
        form = QVBoxLayout(group)
        sliders = []
        labels = "XYZ" if count == 3 else [f"{index:02d}" for index in range(count)]
        for label in labels:
            slider = SliderWithSpinbox(f"{prefix}_{label}", minimum, maximum, 0.0, step)
            form.addWidget(slider)
            sliders.append(slider)
        layout.addWidget(group)
        return sliders

    def _parameters(self):
        betas = np.array([slider.value() for slider in self.beta_sliders])
        global_orient = np.array([slider.value() for slider in self.global_orient_sliders])
        translation = np.array([slider.value() for slider in self.translation_sliders])
        hand_pose = np.zeros((15, 3))
        for finger_index in range(5):
            for joint_index in range(3):
                for axis_index in range(3):
                    hand_pose[finger_index * 3 + joint_index, axis_index] = self.hand_pose_sliders[finger_index][joint_index][axis_index].value()
        return betas, hand_pose, global_orient, translation, self.mano_scale.value()

    def _registration(self):
        return (
            np.array([slider.value() for slider in self.registration_rotation]),
            np.array([slider.value() for slider in self.registration_translation]),
            self.registration_scale.value(),
        )

    def _reset_mano(self):
        for slider in self.beta_sliders:
            slider.set_value(0.0)
        self._set_mano_neutral_runtime()
        self.status.setText("MANO runtime parameters reset; static registration and LDJY controls were kept.")

    def _set_mano_neutral_runtime(self):
        for slider in self.global_orient_sliders + self.translation_sliders:
            slider.set_value(0.0)
        for finger in self.hand_pose_sliders:
            for joint in finger:
                for slider in joint:
                    slider.set_value(0.0)
        self.mano_scale.set_value(1.0)

    def _show_fit_settings(self):
        dialog = FitSettingsDialog(self.fit_settings, self)
        try:
            if dialog.exec():
                self.fit_settings = dialog.settings()
                self._auto_register()
        finally:
            dialog.deleteLater()

    def _auto_register(self):
        """Fit MANO parameters and static registration to the visible LDJY pose."""
        initial_betas, initial_hand_pose, _, _, _ = self._parameters()
        initial_rotation, initial_translation, initial_scale = self._registration()
        mujoco.mj_forward(self.scene.model, self.scene.data)
        target_points, target_normals = self._ldjy_pad_samples()
        target_direction_starts, target_directions = self._ldjy_finger_lines()
        target_palm_normal = self._ldjy_palm_normal()
        zero_vector = np.zeros(3)
        cached_betas = None
        cached_hand_pose = None
        cached_directions = None
        cached_direction_starts = None
        cached_palm_normal = None
        cached_joints = None
        cached_pads = None
        cached_normals = None

        def sample(betas, hand_pose):
            nonlocal cached_betas, cached_hand_pose, cached_directions, cached_direction_starts, cached_palm_normal
            nonlocal cached_joints, cached_pads, cached_normals
            if cached_betas is not None and np.array_equal(betas, cached_betas) and np.array_equal(hand_pose, cached_hand_pose):
                return cached_pads, cached_normals
            vertices, joints, pads = self.mano.compute(betas, hand_pose, zero_vector, zero_vector, 1.0)
            cached_betas = np.asarray(betas).copy()
            cached_hand_pose = np.asarray(hand_pose).copy()
            cached_directions = self._mano_finger_directions(joints)
            cached_direction_starts = self._mano_finger_direction_starts(joints)
            cached_palm_normal = self._mano_palm_normal(joints)
            cached_joints = joints
            cached_pads = pads
            cached_normals = self._mano_pad_normals(vertices)
            return cached_pads, cached_normals

        def sample_directions(betas, hand_pose):
            sample(betas, hand_pose)
            return cached_directions

        def sample_direction_starts(betas, hand_pose):
            sample(betas, hand_pose)
            return cached_direction_starts

        def sample_palm_normal(betas, hand_pose):
            sample(betas, hand_pose)
            return cached_palm_normal

        def sample_joints(betas, hand_pose):
            sample(betas, hand_pose)
            return cached_joints

        fit_hand_joints = np.asarray(self.fit_settings["fit_hand_joints"], dtype=bool)
        position_mask = np.asarray(self.fit_settings["position_mask"], dtype=bool)
        normal_mask = np.asarray(self.fit_settings["normal_mask"], dtype=bool)
        direction_mask = np.asarray(self.fit_settings["direction_mask"], dtype=bool)
        direction_line_mask = np.asarray(self.fit_settings["direction_line_mask"], dtype=bool)
        straight_finger_mask = np.asarray(self.fit_settings["straight_finger_mask"], dtype=bool)
        pose_reference = (
            initial_hand_pose
            if self.fit_settings["hand_pose_prior_reference"] == "current"
            else np.zeros((15, 3))
        )
        exact_straight = (
            self.fit_settings["use_straight_fingers"]
            and self.fit_settings["straight_fingers_exact"]
            and np.any(straight_finger_mask)
        )
        exact_palm_plane = self.fit_settings["use_palm_plane"] and self.fit_settings["palm_plane_exact"]
        least_squares_hand_joints = np.zeros(15, dtype=bool) if (exact_straight or exact_palm_plane) else fit_hand_joints
        has_static_fit_constraint = any((
            self.fit_settings["use_positions"] and np.any(position_mask),
            self.fit_settings["use_normals"] and np.any(normal_mask),
            self.fit_settings["use_directions"] and np.any(direction_mask),
            self.fit_settings["use_palm_normal"],
            self.fit_settings["use_straight_fingers"] and not exact_straight and np.any(straight_finger_mask),
            self.fit_settings["use_palm_plane"] and not exact_palm_plane,
            self.fit_settings["use_hand_pose_prior"] and np.any(least_squares_hand_joints),
        ))

        try:
            if has_static_fit_constraint:
                betas, hand_pose, rotation, translation, scale, position_rms = fit_static_registration(
                    lambda betas: sample(betas, initial_hand_pose),
                    target_points,
                    target_normals,
                    pose_sample=sample,
                    direction_pose_sample=sample_directions,
                    target_directions=target_directions,
                    direction_start_pose_sample=sample_direction_starts,
                    target_direction_starts=target_direction_starts,
                    palm_normal_pose_sample=sample_palm_normal,
                    target_palm_normal=target_palm_normal,
                    joint_pose_sample=sample_joints,
                    use_positions=self.fit_settings["use_positions"],
                    position_weight=self.fit_settings["position_weight"],
                    position_mask=position_mask,
                    use_normals=self.fit_settings["use_normals"],
                    normal_weight=self.fit_settings["normal_weight"],
                    normal_mask=normal_mask,
                    use_directions=self.fit_settings["use_directions"],
                    direction_weight=self.fit_settings["direction_weight"],
                    direction_mask=direction_mask,
                    use_direction_lines=self.fit_settings["use_direction_lines"],
                    direction_line_mask=direction_line_mask,
                    use_palm_normal=self.fit_settings["use_palm_normal"],
                    palm_normal_weight=self.fit_settings["palm_normal_weight"],
                    use_straight_fingers=self.fit_settings["use_straight_fingers"] and not exact_straight,
                    straight_fingers_weight=self.fit_settings["straight_fingers_weight"],
                    straight_finger_mask=straight_finger_mask,
                    use_palm_plane=self.fit_settings["use_palm_plane"] and not exact_palm_plane,
                    palm_plane_weight=self.fit_settings["palm_plane_weight"],
                    use_hand_pose_prior=self.fit_settings["use_hand_pose_prior"],
                    hand_pose_prior_weight=self.fit_settings["hand_pose_prior_weight"],
                    fit_betas=self.fit_settings["fit_betas"],
                    fit_hand_joints=least_squares_hand_joints,
                    fit_rotation=self.fit_settings["fit_rotation"],
                    fit_translation=self.fit_settings["fit_translation"],
                    fit_scale=self.fit_settings["fit_scale"],
                    initial_betas=initial_betas,
                    initial_hand_pose=initial_hand_pose,
                    hand_pose_reference=pose_reference,
                    initial_rotation=initial_rotation,
                    initial_translation=initial_translation,
                    initial_scale=initial_scale,
                    beta_bound=self.fit_settings["beta_bound"],
                )
            else:
                betas, hand_pose = initial_betas, initial_hand_pose
                rotation, translation, scale = initial_rotation, initial_translation, initial_scale
                position_rms = None
            if exact_straight or exact_palm_plane:
                hand_pose = solve_exact_hand_pose_constraints(
                    sample_joints,
                    betas,
                    initial_hand_pose,
                    fit_hand_joints,
                    hand_pose_reference=pose_reference,
                    use_straight_fingers=exact_straight,
                    straight_finger_mask=straight_finger_mask,
                    use_palm_plane=exact_palm_plane,
                )
        except Exception as error:
            self.status.setText(f"Auto fit failed: {error}")
            return
        for slider, value in zip(self.beta_sliders, betas):
            slider.set_value(value)
        for joint_index, joint in enumerate(hand_pose):
            for axis_index, value in enumerate(joint):
                self.hand_pose_sliders[joint_index // 3][joint_index % 3][axis_index].set_value(value)
        for slider, value in zip(self.registration_rotation, rotation):
            slider.set_value(value)
        for slider, value in zip(self.registration_translation, translation):
            slider.set_value(value)
        self.registration_scale.set_value(scale)
        mode = " with exact hand-pose constraints" if (exact_straight or exact_palm_plane) else ""
        if position_rms is None:
            self.status.setText(f"Auto fit complete{mode}: static registration unchanged.")
        else:
            self.status.setText(f"Auto fit complete{mode}: pad position RMS {position_rms * 1000:.2f} mm.")

    def _save_reference(self):
        betas, hand_pose, global_orient, translation, scale = self._parameters()
        rotation, static_translation, static_scale = self._registration()
        payload = {
            "mano": {
                "betas": betas.tolist(),
                "hand_pose": hand_pose.tolist(),
                "global_orient": global_orient.tolist(),
                "translation": translation.tolist(),
                "scale": float(scale),
            },
            "mano_to_ldjy_registration": {
                "rotation_axis_angle": rotation.tolist(),
                "translation": static_translation.tolist(),
                "scale": float(static_scale),
            },
            "fit_settings": self.fit_settings,
        }
        REFERENCE_PATH.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.status.setText(f"Saved reference: {REFERENCE_PATH}")

    def _load_reference(self):
        if not REFERENCE_PATH.exists():
            return
        payload = yaml.safe_load(REFERENCE_PATH.read_text(encoding="utf-8")) or {}
        mano = payload.get("mano", {})
        for slider, value in zip(self.beta_sliders, mano.get("betas", [])):
            slider.set_value(value)
        for slider, value in zip(self.global_orient_sliders, mano.get("global_orient", [])):
            slider.set_value(value)
        for finger_index, joints in enumerate(mano.get("hand_pose", [])):
            if finger_index >= 15:
                break
            for axis_index, value in enumerate(joints):
                if axis_index < 3:
                    self.hand_pose_sliders[finger_index // 3][finger_index % 3][axis_index].set_value(value)
        for slider, value in zip(self.translation_sliders, mano.get("translation", [])):
            slider.set_value(value)
        if "scale" in mano:
            self.mano_scale.set_value(mano["scale"])

        registration = payload.get("mano_to_ldjy_registration", {})
        for slider, value in zip(self.registration_rotation, registration.get("rotation_axis_angle", [])):
            slider.set_value(value)
        for slider, value in zip(self.registration_translation, registration.get("translation", [])):
            slider.set_value(value)
        if "scale" in registration:
            self.registration_scale.set_value(registration["scale"])
        self.fit_settings.update(payload.get("fit_settings", {}))

    def _ldjy_pad_samples(self):
        points = []
        normals = []
        for finger in LDJY_FINGERS:
            site_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"right_{finger}_pad_center"
            )
            if site_id < 0:
                raise ValueError(f"LDJY model is missing right_{finger}_pad_center")
            points.append(self.scene.data.site_xpos[site_id].copy())
            normals.append(self.scene.data.site_xmat[site_id].reshape(3, 3)[:, 2].copy())
        return np.asarray(points), np.asarray(normals)

    def _mano_pad_normals(self, vertices):
        normals = []
        for finger in TIP_ORDER:
            vertex_ids = PAD_3PT_VERTEX_IDS.get(finger, [PAD_VERTEX_IDS[finger]])
            normals.append(average_surface_normal(vertices, self.mano.faces, vertex_ids))
        return np.asarray(normals)

    @staticmethod
    def _mano_finger_directions(joints):
        return np.asarray([joints[end] - joints[start] for start, end in MANO_DIRECTION_JOINTS])

    @staticmethod
    def _mano_finger_direction_starts(joints):
        return np.asarray([joints[start] for start, _ in MANO_DIRECTION_JOINTS])

    def _ldjy_finger_directions(self):
        return self._ldjy_finger_lines()[1]

    def _ldjy_finger_lines(self):
        starts = []
        directions = []
        for finger, start_number, end_number in LDJY_DIRECTION_JOINTS:
            start_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"right_{finger}_joint{start_number}"
            )
            end_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"right_{finger}_joint{end_number}"
            )
            start = self.scene.data.xanchor[start_id].copy()
            starts.append(start)
            directions.append(self.scene.data.xanchor[end_id] - start)
        return np.asarray(starts), np.asarray(directions)

    @staticmethod
    def _mano_palm_normal(joints):
        return _plane_normal(*(joints[index] for index in MANO_PALM_NORMAL_KEYPOINTS))

    def _ldjy_palm_points(self):
        points = []
        for finger, number in LDJY_PALM_NORMAL_JOINTS:
            joint_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"right_{finger}_joint{number}"
            )
            if joint_id < 0:
                raise ValueError(f"LDJY model is missing right_{finger}_joint{number}")
            points.append(self.scene.data.xanchor[joint_id].copy())
        return np.asarray(points)

    def _ldjy_palm_normal(self):
        return _plane_normal(*self._ldjy_palm_points())

    def _draw_ldjy_skeleton(self, visual):
        palm_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
        palm = self.scene.data.xpos[palm_id]
        _add_sphere(visual, palm, "LDJY palm", (0.1, 0.9, 0.85, 1.0), 0.0025)
        for finger in LDJY_FINGERS:
            previous = palm
            for number in range(1, 5):
                joint_id = mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"right_{finger}_joint{number}"
                )
                point = self.scene.data.xanchor[joint_id]
                _add_line(visual, previous, point, f"{finger} link", (0.1, 0.85, 0.85, 0.9), 0.00055)
                _add_sphere(visual, point, f"{finger} J{number}", (0.1, 0.9, 0.85, 1.0), 0.0018)
                previous = point
            site_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"right_{finger}_link4_tip"
            )
            _add_line(visual, previous, self.scene.data.site_xpos[site_id], f"{finger} distal", (0.1, 0.85, 0.85, 0.9), 0.00055)

    def _draw_overlays(self, mano_vertices, mano_joints, mano_pads, rotation, translation, scale):
        visual = self.viewer.user_scn
        visual.ngeom = 0
        if self.show_ldjy_skeleton.isChecked():
            self._draw_ldjy_skeleton(visual)
        if self.show_ldjy_tips.isChecked():
            for finger in LDJY_FINGERS:
                site_id = mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"right_{finger}_link4_tip"
                )
                if site_id >= 0:
                    _add_sphere(visual, self.scene.data.site_xpos[site_id], f"{finger} tip", (0.1, 0.7, 1.0, 1.0), 0.002)
        if self.show_ldjy_pads.isChecked() or self.show_ldjy_normals.isChecked():
            for finger in LDJY_FINGERS:
                site_id = mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"right_{finger}_pad_center"
                )
                if site_id < 0:
                    continue
                point = self.scene.data.site_xpos[site_id]
                if self.show_ldjy_pads.isChecked():
                    _add_sphere(visual, point, f"{finger} pad", (1.0, 0.15, 0.05, 1.0), 0.002)
                if self.show_ldjy_normals.isChecked():
                    normal = self.scene.data.site_xmat[site_id].reshape(3, 3)[:, 2]
                    _add_line(visual, point, point + normal * 0.01, f"{finger} pad normal", (1.0, 0.85, 0.1, 1.0))
        if self.show_ldjy_palm_normal.isChecked():
            palm_points = self._ldjy_palm_points()
            normal = self._ldjy_palm_normal()
            normal /= np.linalg.norm(normal)
            center = palm_points.mean(axis=0)
            _add_line(
                visual, center, center + normal * PALM_NORMAL_LENGTH, "LDJY palm normal",
                (0.1, 0.95, 0.9, 1.0), 0.0008,
            )

        rotation_matrix = Rotation.from_rotvec(rotation).as_matrix()
        registered_joints = apply_registration(mano_joints, rotation, translation, scale)
        if self.show_mano_keypoints.isChecked():
            for index, point in enumerate(registered_joints):
                _add_sphere(visual, point, f"MANO joint {index}", (0.35, 1.0, 0.3, 1.0), 0.0017)
            for start, end in MEDIAPIPE_21_SKELETON_EDGES:
                _add_line(
                    visual, registered_joints[start], registered_joints[end], "MANO keypoint bone",
                    (0.35, 1.0, 0.3, 0.7), 0.00035,
                )
        if self.show_mano_active_skeleton.isChecked():
            for index in MANO_ACTIVE_KEYPOINT_INDICES:
                _add_sphere(visual, registered_joints[index], "MANO active joint", (0.2, 0.8, 1.0, 1.0), 0.0014)
            for start, end in MANO_ACTIVE_SKELETON_EDGES:
                _add_line(
                    visual, registered_joints[start], registered_joints[end], "MANO active bone",
                    (0.2, 0.8, 1.0, 0.9), 0.00055,
                )
        if self.show_mano_palm_normal.isChecked():
            palm_points = registered_joints[list(MANO_PALM_NORMAL_KEYPOINTS)]
            normal = _plane_normal(*palm_points)
            normal /= np.linalg.norm(normal)
            center = palm_points.mean(axis=0)
            _add_line(
                visual, center, center + normal * PALM_NORMAL_LENGTH, "MANO palm normal",
                (1.0, 0.25, 0.8, 1.0), 0.0008,
            )
        registered_normals = self._mano_pad_normals(mano_vertices) @ rotation_matrix.T
        for index, finger in enumerate(TIP_ORDER):
            point = apply_registration(mano_pads[index:index + 1], rotation, translation, scale)[0]
            normal = registered_normals[index]
            _add_sphere(visual, point, f"MANO {finger} pad", (1.0, 0.5, 0.0, 1.0), 0.002)
            _add_line(visual, point, point + normal * 0.01, f"MANO {finger} normal", (0.95, 0.95, 0.35, 1.0))

    def _refresh(self):
        if not self.viewer.is_running():
            self._timer.stop()
            self.close()
            return
        parameters = self._parameters()
        rotation, translation, scale = self._registration()
        state = tuple(np.concatenate((
            parameters[0], parameters[1].ravel(), parameters[2], parameters[3], [parameters[4]],
            rotation, translation, [scale],
        )))
        if state != self._last_mano_state:
            vertices, joints, pads = self.mano.compute(*parameters)
            registered = apply_registration(vertices, rotation, translation, scale)
            self.scene.update_mano_mesh(registered[self._expanded_to_orig])
            self.viewer.update_mesh(self.scene.mesh_id)
            self._last_mano_state = state
            self._last_mano_output = (vertices, joints, pads)
        else:
            vertices, joints, pads = self._last_mano_output
        for _ in range(10):
            mujoco.mj_step(self.scene.model, self.scene.data)
        self._draw_overlays(vertices, joints, pads, rotation, translation, scale)
        self.viewer.sync()

    def closeEvent(self, event):
        self._timer.stop()
        self.viewer.close()
        self.scene.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = OverlayWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
