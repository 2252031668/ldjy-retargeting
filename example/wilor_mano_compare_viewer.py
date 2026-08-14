#!/usr/bin/env python3
"""Replay WiLoR records and compare video MANO with robot-shape MANO.

Usage:
    uv run --extra wilor --extra gui --extra tuning \
        python example/wilor_mano_compare_viewer.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ldjy_retargeting.mano_compare import (
    ComparisonHand,
    PAD_ORDER,
    build_comparison_hands,
    load_robot_mano_reference,
)
from ldjy_retargeting.tuning.recording import list_records
from ldjy_retargeting.tuning.replay import WiLoRReplay

try:
    from mano_viewer import MANO_MODEL_PATH, MANOModel
except ModuleNotFoundError:
    from example.mano_viewer import MANO_MODEL_PATH, MANOModel


ROOT = Path(__file__).resolve().parents[1]
RECORDS_ROOT = ROOT / "outputs" / "tuning_records"
REFERENCE_PATH = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "mano_ldjy_reference.yaml"
DISPLAY_OFFSETS = (np.array((-0.10, 0.0, 0.0)), np.array((0.10, 0.0, 0.0)))
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
IDENTITY = np.eye(3, dtype=np.float64).ravel()
HUMAN_RGBA = np.array((0.08, 0.55, 1.0, 0.42), dtype=np.float64)
ROBOT_RGBA = np.array((1.0, 0.38, 0.08, 0.42), dtype=np.float64)
HUMAN_POINT_RGBA = np.array((0.15, 1.0, 0.3, 1.0), dtype=np.float64)
ROBOT_POINT_RGBA = np.array((1.0, 0.15, 0.1, 1.0), dtype=np.float64)
HUMAN_NORMAL_RGBA = np.array((0.1, 1.0, 0.75, 1.0), dtype=np.float64)
ROBOT_NORMAL_RGBA = np.array((1.0, 0.8, 0.1, 1.0), dtype=np.float64)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, default=RECORDS_ROOT)
    parser.add_argument("--record", type=Path, default=None, help="open this WiLoR record directly")
    return parser.parse_args(args)


def _add_sphere(scene, position, label, rgba, radius=0.0022):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((radius, 0.0, 0.0)), np.asarray(position), IDENTITY, rgba,
    )
    geom.label = label
    scene.ngeom += 1


def _add_link(scene, start, end, label, rgba, width=0.00045):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3), np.zeros(3), IDENTITY, rgba,
    )
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
        np.asarray(start), np.asarray(end),
    )
    geom.label = label
    scene.ngeom += 1


class ComparisonScene:
    """One MuJoCo viewer containing two mutable MANO meshes."""

    def __init__(self, mano_model: MANOModel, human: ComparisonHand, robot: ComparisonHand):
        self.mano_model = mano_model
        self.faces = np.asarray(mano_model.faces, dtype=np.int32)
        self._expanded_to_orig = self.faces.reshape(-1).astype(np.int64)
        self._expanded_faces = np.arange(len(self._expanded_to_orig), dtype=np.int32).reshape(-1, 3)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mano_compare_")
        self.model = self._create_model(human, robot)
        self.data = mujoco.MjData(self.model)
        self._mesh_transforms = [self._mesh_transform(index) for index in range(self.model.nmesh)]
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 0.45
        self.viewer.cam.lookat[:] = [0.0, 0.08, 0.0]
        self.show_keypoints = True
        self.show_skeleton = True
        self.show_human_pads = False
        self.show_human_normals = False
        self.show_robot_pads = True
        self.show_robot_normals = True
        self.update(human, robot)

    def _write_obj(self, path: Path, vertices: np.ndarray) -> None:
        expanded = np.asarray(vertices)[self._expanded_to_orig]
        with path.open("w", encoding="ascii") as handle:
            handle.write("o mano_mesh\n")
            for vertex in expanded:
                handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for face in self._expanded_faces:
                handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    def _create_model(self, human: ComparisonHand, robot: ComparisonHand):
        human_path = Path(self._tmpdir.name) / "human.obj"
        robot_path = Path(self._tmpdir.name) / "robot.obj"
        self._write_obj(human_path, human.vertices)
        self._write_obj(robot_path, robot.vertices)
        mjcf = f"""
        <mujoco>
          <worldbody>
            <geom name="human_mesh_geom" type="mesh" mesh="human_mesh"
                  rgba="{HUMAN_RGBA[0]} {HUMAN_RGBA[1]} {HUMAN_RGBA[2]} {HUMAN_RGBA[3]}"
                  contype="0" conaffinity="0" mass="0.001" priority="1" />
            <geom name="robot_mesh_geom" type="mesh" mesh="robot_mesh"
                  rgba="{ROBOT_RGBA[0]} {ROBOT_RGBA[1]} {ROBOT_RGBA[2]} {ROBOT_RGBA[3]}"
                  contype="0" conaffinity="0" mass="0.001" priority="1" />
          </worldbody>
          <asset>
            <mesh name="human_mesh" file="{human_path}" />
            <mesh name="robot_mesh" file="{robot_path}" />
          </asset>
        </mujoco>
        """
        return mujoco.MjModel.from_xml_string(mjcf)

    @staticmethod
    def _mesh_slice(model, mesh_id):
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        return start, start + count

    def _mesh_transform(self, mesh_id):
        quat = self.model.mesh_quat[mesh_id]
        matrix = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(matrix, quat)
        scale = self.model.mesh_scale[mesh_id]
        return (
            self.model.mesh_pos[mesh_id].copy(),
            matrix.reshape(3, 3),
            np.where(np.abs(scale) > 1e-12, 1.0 / scale, 0.0),
        )

    def _to_mesh_storage(self, mesh_id: int, vertices: np.ndarray) -> np.ndarray:
        position, rotation, inverse_scale = self._mesh_transforms[mesh_id]
        return (np.asarray(vertices, dtype=np.float64) - position) * inverse_scale @ rotation

    def _set_mesh(self, mesh_id: int, vertices: np.ndarray) -> None:
        start, end = self._mesh_slice(self.model, mesh_id)
        expanded = np.asarray(vertices)[self._expanded_to_orig]
        stored = self._to_mesh_storage(mesh_id, expanded)
        self.model.mesh_vert[start:end] = stored.astype(np.float32)
        v0 = stored[self._expanded_faces[:, 0]]
        v1 = stored[self._expanded_faces[:, 1]]
        v2 = stored[self._expanded_faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(lengths > 1e-12, lengths, 1.0)
        normal_start = int(self.model.mesh_normaladr[mesh_id])
        self.model.mesh_normal[normal_start:normal_start + len(expanded)] = np.repeat(
            normals, 3, axis=0
        ).astype(np.float32)

    def _draw_hand(self, scene, hand: ComparisonHand, prefix: str, point_rgba, normal_rgba, show_pads, show_normals):
        if self.show_skeleton:
            for start, end in HAND_CONNECTIONS:
                _add_link(scene, hand.joints[start], hand.joints[end], "", point_rgba, 0.00055)
        if self.show_keypoints:
            for point in hand.joints:
                _add_sphere(scene, point, "", point_rgba, 0.0017)
        if show_pads:
            for point in hand.pads:
                _add_sphere(scene, point, "", point_rgba, 0.0026)
        if show_normals:
            for point, normal in zip(hand.pads, hand.pad_normals):
                _add_link(scene, point, point + normal * 0.025, "", normal_rgba, 0.0005)

    def update(self, human: ComparisonHand, robot: ComparisonHand) -> None:
        self._set_mesh(0, human.vertices)
        self._set_mesh(1, robot.vertices)
        self.viewer.update_mesh(0)
        self.viewer.update_mesh(1)
        with self.viewer.lock():
            self._draw_scene(human, robot)
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def _draw_scene(self, human, robot):
        scene = self.viewer.user_scn
        scene.ngeom = 0
        self._draw_hand(
            scene, human, "Video", HUMAN_POINT_RGBA, HUMAN_NORMAL_RGBA,
            self.show_human_pads, self.show_human_normals,
        )
        self._draw_hand(
            scene, robot, "Robot beta", ROBOT_POINT_RGBA, ROBOT_NORMAL_RGBA,
            self.show_robot_pads, self.show_robot_normals,
        )

    def set_visibility(self, **options) -> None:
        for name, value in options.items():
            setattr(self, name, bool(value))

    def close(self) -> None:
        self.viewer.close()
        self._tmpdir.cleanup()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, records_root: Path, direct_record: Path | None = None):
        super().__init__()
        self.setWindowTitle("WiLoR MANO 形状与姿态对比")
        self.resize(1320, 820)
        self.records_root = Path(records_root)
        self.mano_model = MANOModel(str(MANO_MODEL_PATH))
        self.replay: WiLoRReplay | None = None
        self.scene: ComparisonScene | None = None
        self._current_hands: tuple[ComparisonHand, ComparisonHand] | None = None
        self._references = {}
        self._build_ui()
        self._load_records(direct_record)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("WiLoR 记录"))
        self.record_combo = QtWidgets.QComboBox()
        self.record_combo.setMinimumWidth(300)
        self.record_combo.currentIndexChanged.connect(self._record_changed)
        top.addWidget(self.record_combo)
        self.info_label = QtWidgets.QLabel()
        top.addWidget(self.info_label, stretch=1)
        layout.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.video_label = QtWidgets.QLabel("选择记录")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(560, 420)
        self.video_label.setStyleSheet("background:#101820;color:#c8d4df")
        split.addWidget(self.video_label)
        controls = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(controls)
        control_layout.addWidget(QtWidgets.QLabel("MuJoCo 中蓝色为视频形状 MANO，橙色为 beta_robot MANO"))
        self.keypoints = self._check("21 点", True, self._visibility_changed)
        self.skeleton = self._check("骨架", True, self._visibility_changed)
        self.human_pads = self._check("视频 MANO 指腹点", False, self._visibility_changed)
        self.human_normals = self._check("视频 MANO 指腹法线", False, self._visibility_changed)
        self.robot_pads = self._check("机器人 MANO 指腹点", True, self._visibility_changed)
        self.robot_normals = self._check("机器人 MANO 指腹法线", True, self._visibility_changed)
        for widget in (self.keypoints, self.skeleton, self.human_pads, self.human_normals, self.robot_pads, self.robot_normals):
            control_layout.addWidget(widget)
        control_layout.addStretch()
        split.addWidget(controls)
        split.setSizes([720, 340])
        layout.addWidget(split, stretch=1)

        playback = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("播放")
        self.play_button.clicked.connect(self._toggle_play)
        playback.addWidget(self.play_button)
        for text, delta in (("|<", -10**9), ("<", -1), (">", 1)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _, d=delta: self._step(d))
            playback.addWidget(button)
        self.speed_combo = QtWidgets.QComboBox()
        for speed in (0.25, 0.5, 1.0, 2.0):
            self.speed_combo.addItem(f"{speed:g}x", speed)
        self.speed_combo.setCurrentIndex(2)
        playback.addWidget(self.speed_combo)
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline.sliderMoved.connect(self._seek)
        playback.addWidget(self.timeline, stretch=1)
        self.frame_label = QtWidgets.QLabel("0 / 0")
        playback.addWidget(self.frame_label)
        layout.addLayout(playback)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    @staticmethod
    def _check(text, checked, callback):
        box = QtWidgets.QCheckBox(text)
        box.setChecked(checked)
        box.toggled.connect(callback)
        return box

    def _load_records(self, direct_record):
        records = list_records(self.records_root, "webcam_wilor")
        if direct_record is not None:
            direct_record = Path(direct_record).resolve()
            records = [info for info in records if info.path.resolve() == direct_record] or records
        for info in records:
            self.record_combo.addItem(f"{info.name} ({info.hand_side})", info)
        if not records:
            self.status.setText(f"没有找到 WiLoR 记录: {self.records_root / 'wilor'}")
            return
        self._record_changed(0)

    def _record_changed(self, index):
        if index < 0 or self.record_combo.itemData(index) is None:
            return
        info = self.record_combo.itemData(index)
        try:
            if self.replay is not None:
                self.replay.close()
            self.replay = WiLoRReplay(info.path)
            self.timeline.setRange(0, max(0, info.frame_count - 1))
            self.timeline.setValue(0)
            self._render(0)
            self.info_label.setText(f"{info.hand_side} | {info.frame_count} 帧")
        except Exception as exc:
            self.status.setText(f"记录加载失败: {exc}")

    def _render(self, index, *, seek=True):
        if self.replay is None:
            return
        if seek:
            index = self.replay.cursor.seek(index)
        else:
            index = int(np.clip(index, 0, self.replay.record.frame_count - 1))
        frame = self.replay.preview_at(index)
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            ))
        params = self.replay.mano_parameters_at(index)
        overlay = self.replay.mano_overlay_at(index)
        if params is None or overlay is None:
            self.status.setText(f"第 {index} 帧没有有效 MANO")
            return
        side = self.replay.hand_side
        human, robot = build_comparison_hands(
            self.mano_model, REFERENCE_PATH, side, params,
            DISPLAY_OFFSETS[0], DISPLAY_OFFSETS[1],
            video_vertices=overlay["vertices_mano"],
            video_joints=overlay["joints_mano"],
        )
        self._current_hands = human, robot
        if self.scene is None:
            self.scene = ComparisonScene(self.mano_model, human, robot)
        else:
            self._set_scene_visibility()
            self.scene.update(human, robot)
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        timestamp = float(self.replay.record.timestamp_sec[index])
        self.frame_label.setText(f"{index + 1} / {self.replay.record.frame_count}  {timestamp:.3f}s")
        reference = self._references.setdefault(
            side, load_robot_mano_reference(REFERENCE_PATH, side)
        )
        beta_delta = np.linalg.norm(np.asarray(params["betas"], dtype=np.float64) - reference.beta_robot)
        mesh_rms_mm = 1000 * np.sqrt(np.mean(
            ((human.vertices - DISPLAY_OFFSETS[0]) - (robot.vertices - DISPLAY_OFFSETS[1])) ** 2
        ))
        self.status.setText(
            f"当前手: {side} | blue=记录网格，orange=beta_robot | "
            f"beta L2 差={beta_delta:.3f}，对齐后网格 RMS={mesh_rms_mm:.2f} mm"
        )

    def _visibility_changed(self):
        if self.scene is not None:
            self._set_scene_visibility()
            if self._current_hands is not None:
                self.scene.update(*self._current_hands)

    def _set_scene_visibility(self):
        self.scene.set_visibility(
            show_keypoints=self.keypoints.isChecked(),
            show_skeleton=self.skeleton.isChecked(),
            show_human_pads=self.human_pads.isChecked(),
            show_human_normals=self.human_normals.isChecked(),
            show_robot_pads=self.robot_pads.isChecked(),
            show_robot_normals=self.robot_normals.isChecked(),
        )

    def _toggle_play(self):
        if self.replay is None:
            return
        if self.replay.cursor.playing:
            self.replay.cursor.pause()
            self.play_button.setText("播放")
        else:
            self.replay.cursor.play()
            self.play_button.setText("暂停")

    def _seek(self, value):
        self._render(value)

    def _step(self, delta):
        if self.replay is None:
            return
        if abs(delta) > 100:
            self._render(0)
        else:
            self._render(self.replay.current_index + delta)

    def _tick(self):
        if self.replay is None or not self.replay.cursor.playing:
            return
        elapsed = 0.033 * float(self.speed_combo.currentData())
        index = self.replay.cursor.advance(elapsed)
        self._render(index, seek=False)
        if not self.replay.cursor.playing:
            self.play_button.setText("播放")

    def closeEvent(self, event):
        self.timer.stop()
        if self.replay is not None:
            self.replay.close()
        if self.scene is not None:
            self.scene.close()
        super().closeEvent(event)


def main(args=None):
    options = parse_args(args)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(options.records_root, options.record)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
