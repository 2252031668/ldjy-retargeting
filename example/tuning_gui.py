"""Desktop GUI for real-time LDJY retargeting parameter tuning."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LDJY 实时重定向调参 GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/adaptive_analytical_video.yaml",
                        help="调参 YAML，相对路径以 example 目录为基准")
    parser.add_argument("--webcam", action="store_true",
                        help="使用 USB 摄像头和 MediaPipe 输入")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="OpenCV USB 摄像头索引")
    parser.add_argument("--hand", choices=("left", "right"), default="right",
                        help="要重定向的手侧")
    return parser


def _resolve_config_path(config: str) -> Path:
    path = Path(config)
    return path if path.is_absolute() else EXAMPLE_DIR / path


def _visible_mesh_alpha(model, alpha: float = 0.3) -> None:
    import mujoco

    for geom_id in range(model.ngeom):
        if (
            model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_rgba[geom_id, 3] > 0
        ):
            model.geom_rgba[geom_id, 3] = alpha


class MuJoCoDebugWorker(threading.Thread):
    """Own the passive MuJoCo viewer and draw the current debug scene."""

    def __init__(self, hand_side: str):
        super().__init__(name="ldjy-mujoco-debug", daemon=True)
        self.hand_side = hand_side
        self._state_lock = threading.Lock()
        self._frame: tuple[np.ndarray, dict[str, Any], Any] | None = None
        self._stop_event = threading.Event()
        self._viewer = None
        self.error: Exception | None = None

    def submit(self, qpos: np.ndarray, diagnostics: dict[str, Any], optimizer: Any) -> None:
        frame = (
            np.asarray(qpos, dtype=np.float64).copy(),
            {
                "mediapipe_kp": np.asarray(diagnostics["mediapipe_kp"], dtype=np.float64).copy(),
                "pinch_alphas": np.asarray(
                    diagnostics.get("pinch_alphas", np.zeros(5)), dtype=np.float64
                ).copy(),
            },
            optimizer,
        )
        with self._state_lock:
            self._frame = frame

    def stop(self) -> None:
        self._stop_event.set()
        viewer = self._viewer
        if viewer is not None:
            viewer.close()

    def run(self) -> None:
        import mujoco
        import mujoco.viewer

        from ldjy_retargeting.joint_mapping import qpos_reorder_perm
        from ldjy_retargeting.viz.debug_overlay import DebugOverlay

        try:
            mjcf_path = (
                PROJECT_ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
                / "mjcf" / f"ldjy_{self.hand_side}_hand.xml"
            )
            model = mujoco.MjModel.from_xml_path(str(mjcf_path))
            _visible_mesh_alpha(model)
            data = mujoco.MjData(model)
            overlay = DebugOverlay(model)
            viewer = mujoco.viewer.launch_passive(model, data)
            self._viewer = viewer
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -20
            viewer.cam.distance = 0.5
            viewer.cam.lookat[:] = [0, 0, 0.05]

            actuator_joint_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
                for i in range(model.nu)
            ]
            qpos_perm: np.ndarray | None = None
            for actuator_id in range(model.nu):
                if model.actuator_ctrllimited[actuator_id]:
                    lower, upper = model.actuator_ctrlrange[actuator_id]
                    data.ctrl[actuator_id] = np.clip(0.0, lower, upper)
            for _ in range(100):
                mujoco.mj_step(model, data)

            while viewer.is_running() and not self._stop_event.is_set():
                with self._state_lock:
                    frame = self._frame
                if frame is not None:
                    qpos, diagnostics, optimizer = frame
                    if qpos_perm is None:
                        qpos_perm = qpos_reorder_perm(
                            optimizer.robot.dof_joint_names, actuator_joint_names
                        )
                        if qpos_perm is None:
                            raise ValueError("LDJY URDF 与 MJCF 关节名无法重排")
                    actuator_targets = qpos[qpos_perm]
                    with viewer.lock():
                        data.ctrl[:] = actuator_targets
                        mujoco.mj_step(model, data)
                        overlay.draw(
                            viewer.user_scn,
                            data,
                            optimizer,
                            diagnostics["mediapipe_kp"],
                            diagnostics["pinch_alphas"],
                        )
                viewer.sync()
                time.sleep(0.01)
        except Exception as exc:  # surfaced in the GUI status strip
            self.error = exc
        finally:
            self._viewer = None


def _run_gui(args: argparse.Namespace) -> int:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "未安装 GUI 依赖。请执行: uv sync --extra gui --extra tuning"
        ) from exc

    from input_devices.webcam_mediapipe import WebcamMediaPipe
    from ldjy_retargeting.tuning.parameters import get_path, parameter_specs
    from ldjy_retargeting.tuning.runtime import TuningRuntime
    from ldjy_retargeting.tuning.session import TuningSession
    from ldjy_retargeting.tuning.vector_scale_calibration import (
        CalibrationError,
        compute_scales,
        human_vector_lengths,
    )

    config_path = _resolve_config_path(args.config).resolve()

    class TuningMainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("LDJY 实时重定向调参")
            self.resize(1440, 900)
            self.session = TuningSession(config_path)
            self.runtime = TuningRuntime(
                self.session.config, args.hand, yaml_dir=config_path.parent
            )
            self.device = WebcamMediaPipe(
                hand_side=args.hand,
                camera_index=args.camera_index,
                video_config=self.session.config.get("video_input", {}),
                show_video=False,
            )
            self.debug_worker = MuJoCoDebugWorker(args.hand)
            self.debug_worker.start()
            self._widgets: dict[str, tuple[Any, Any, Any]] = {}
            self._last_tick = time.monotonic()
            self._fps = 0.0
            self._calibration_samples: list[np.ndarray] = []
            self._calibration_started_at: float | None = None
            self._calibration_target_frames = 45
            self._calibration_timeout_seconds = 3.0

            self._build_ui()
            self._apply_timer = QtCore.QTimer(self)
            self._apply_timer.setSingleShot(True)
            self._apply_timer.setInterval(100)
            self._apply_timer.timeout.connect(self._apply_live_config)
            self._frame_timer = QtCore.QTimer(self)
            self._frame_timer.setInterval(33)
            self._frame_timer.timeout.connect(self._tick)
            self._frame_timer.start()

        def _build_ui(self) -> None:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            self.setCentralWidget(splitter)

            preview_panel = QtWidgets.QWidget()
            preview_layout = QtWidgets.QVBoxLayout(preview_panel)
            title = QtWidgets.QLabel("OpenCV / MediaPipe 检测画面")
            title.setStyleSheet("font-weight: 600; font-size: 16px;")
            preview_layout.addWidget(title)
            self.preview_label = QtWidgets.QLabel("等待 USB 摄像头画面...")
            self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setMinimumSize(640, 480)
            self.preview_label.setStyleSheet("background: #101820; color: #b8c7d9; border: 1px solid #405060;")
            preview_layout.addWidget(self.preview_label, stretch=1)
            self.camera_status = QtWidgets.QLabel()
            preview_layout.addWidget(self.camera_status)
            splitter.addWidget(preview_panel)

            parameter_panel = QtWidgets.QWidget()
            parameter_layout = QtWidgets.QVBoxLayout(parameter_panel)
            heading = QtWidgets.QLabel("运行时参数")
            heading.setStyleSheet("font-weight: 600; font-size: 16px;")
            parameter_layout.addWidget(heading)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            toolbox = QtWidgets.QToolBox()
            groups: dict[str, list[Any]] = defaultdict(list)
            for spec in parameter_specs():
                groups[spec.group].append(spec)
            for group_name, specs in groups.items():
                toolbox.addItem(self._create_group(specs), group_name)
            scroll.setWidget(toolbox)
            parameter_layout.addWidget(scroll, stretch=1)

            actions = QtWidgets.QHBoxLayout()
            save_button = QtWidgets.QPushButton("保存 YAML")
            save_button.clicked.connect(self._save)
            default_button = QtWidgets.QPushButton("恢复默认")
            default_button.clicked.connect(self._restore_default)
            actions.addWidget(save_button)
            actions.addWidget(default_button)
            parameter_layout.addLayout(actions)
            self.dirty_label = QtWidgets.QLabel()
            parameter_layout.addWidget(self.dirty_label)
            self.status_label = QtWidgets.QLabel("MuJoCo debug 窗口正在启动...")
            self.status_label.setWordWrap(True)
            parameter_layout.addWidget(self.status_label)
            splitter.addWidget(parameter_panel)
            splitter.setSizes([800, 640])
            self._update_dirty_label()

        def _create_group(self, specs: list[Any]) -> QtWidgets.QWidget:
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(widget)
            layout.setContentsMargins(10, 10, 10, 10)
            if specs[0].group == "高级机械约束":
                warning = QtWidgets.QLabel("高级约束默认关闭；应在有 LDJY 实测依据时再启用。")
                warning.setStyleSheet("color: #b07020;")
                warning.setWordWrap(True)
                layout.addWidget(warning)
            if specs[0].group == "15 条目标向量":
                layout.addWidget(self._create_calibration_controls())
            for spec in specs:
                layout.addWidget(self._create_control(spec))
            layout.addStretch(1)
            return widget

        def _create_calibration_controls(self) -> QtWidgets.QGroupBox:
            group = QtWidgets.QGroupBox("自动零位标定")
            layout = QtWidgets.QVBoxLayout(group)
            instructions = QtWidgets.QLabel(
                "掌心朝相机，手腕稳定，五指自然张开并尽量伸直。"
                "采集 45 帧后，将 MANO 射线长度匹配到 LDJY 零位。"
            )
            instructions.setWordWrap(True)
            layout.addWidget(instructions)
            self.calibration_button = QtWidgets.QPushButton("开始采集（45 帧）")
            self.calibration_button.clicked.connect(self._start_calibration)
            layout.addWidget(self.calibration_button)
            self.calibration_status = QtWidgets.QLabel("尚未标定")
            self.calibration_status.setWordWrap(True)
            layout.addWidget(self.calibration_status)
            return group

        def _create_control(self, spec: Any) -> QtWidgets.QWidget:
            container = QtWidgets.QWidget()
            layout = QtWidgets.QGridLayout(container)
            layout.setContentsMargins(0, 5, 0, 5)
            label = QtWidgets.QLabel(f"{spec.label}\n{spec.description_zh}")
            label.setToolTip(spec.effect_zh)
            label.setWordWrap(True)
            layout.addWidget(label, 0, 0)
            value = get_path(self.session.config, spec.path)
            if spec.value_type is bool:
                checkbox = QtWidgets.QCheckBox("启用")
                checkbox.setChecked(value)
                checkbox.setToolTip(spec.effect_zh)
                checkbox.toggled.connect(
                    lambda checked, path=spec.path: self._set_value(path, checked)
                )
                layout.addWidget(checkbox, 0, 1, 1, 2)
                self._widgets[spec.path] = (spec, checkbox, None)
                return container

            factor = int(round(1.0 / spec.step))
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(round(spec.minimum * factor), round(spec.maximum * factor))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(spec.minimum, spec.maximum)
            spin.setSingleStep(spec.step)
            decimals = max(0, len(f"{spec.step:.8f}".rstrip("0").split(".")[-1]))
            spin.setDecimals(decimals)
            spin.setValue(float(value))
            slider.setValue(round(float(value) * factor))
            slider.valueChanged.connect(
                lambda raw, path=spec.path, scale=factor, target=spin: self._slider_changed(
                    path, raw / scale, target
                )
            )
            spin.valueChanged.connect(
                lambda new_value, path=spec.path, scale=factor, target=slider: self._spin_changed(
                    path, new_value, target, scale
                )
            )
            layout.addWidget(slider, 0, 1)
            layout.addWidget(spin, 0, 2)
            self._widgets[spec.path] = (spec, spin, slider)
            return container

        def _slider_changed(self, path: str, value: float, spin: Any) -> None:
            previous = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(previous)
            self._set_value(path, value)

        def _spin_changed(self, path: str, value: float, slider: Any, factor: int) -> None:
            previous = slider.blockSignals(True)
            slider.setValue(round(value * factor))
            slider.blockSignals(previous)
            self._set_value(path, value)

        def _set_value(self, path: str, value: Any) -> None:
            try:
                self.session.set_value(path, value)
            except ValueError as exc:
                self.status_label.setText(f"参数未应用: {exc}")
                self._refresh_controls()
                return
            self._update_dirty_label()
            self._apply_timer.start()

        def _apply_live_config(self) -> None:
            try:
                self.runtime.apply_config(self.session.config)
                self.status_label.setText("参数已应用到实时重定向，滤波器与 warm start 已重置。")
            except Exception as exc:
                self.status_label.setText(f"参数应用失败: {exc}")

        def _refresh_controls(self) -> None:
            config = self.session.config
            for path, (spec, primary, secondary) in self._widgets.items():
                value = get_path(config, path)
                if spec.value_type is bool:
                    was_blocked = primary.blockSignals(True)
                    primary.setChecked(value)
                    primary.blockSignals(was_blocked)
                    continue
                was_blocked = primary.blockSignals(True)
                primary.setValue(value)
                primary.blockSignals(was_blocked)
                factor = int(round(1.0 / spec.step))
                was_blocked = secondary.blockSignals(True)
                secondary.setValue(round(value * factor))
                secondary.blockSignals(was_blocked)
            self._update_dirty_label()

        def _update_dirty_label(self) -> None:
            text = "有未保存修改" if self.session.is_dirty else "配置与磁盘一致"
            self.dirty_label.setText(text)
            self.dirty_label.setStyleSheet(
                "color: #b07020; font-weight: 600;" if self.session.is_dirty else "color: #367a4b;"
            )

        def _save(self) -> None:
            try:
                self.session.save()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "保存失败", str(exc))
                return
            self._update_dirty_label()
            self.status_label.setText(
                f"已保存 YAML。默认基线: {self.session.original_path.name}"
            )

        def _restore_default(self) -> None:
            try:
                self.session.restore_default()
                self._refresh_controls()
                self._apply_live_config()
                self.status_label.setText("已载入默认基线；点击“保存 YAML”后才会写入当前配置。")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "恢复默认不可用", str(exc))

        def _start_calibration(self) -> None:
            self._calibration_samples = []
            self._calibration_started_at = time.monotonic()
            self.calibration_button.setEnabled(False)
            self.calibration_status.setText(
                f"采集中：0 / {self._calibration_target_frames}"
            )

        def _fail_calibration(self, reason: str) -> None:
            self._calibration_samples = []
            self._calibration_started_at = None
            self.calibration_button.setEnabled(True)
            message = f"标定失败：{reason} 请重新自然张开后再试。"
            self.calibration_status.setText(message)
            self.status_label.setText(message)

        def _finish_calibration(self) -> None:
            try:
                scales = compute_scales(
                    self.runtime.zero_pose_robot_vector_lengths(),
                    np.stack(self._calibration_samples),
                )
                self.session.set_segment_scalings(scales)
                self._refresh_controls()
                self._apply_live_config()
            except (CalibrationError, RuntimeError, ValueError) as exc:
                self._fail_calibration(str(exc))
                return

            self._calibration_samples = []
            self._calibration_started_at = None
            self.calibration_button.setEnabled(True)
            message = "标定成功：15 条缩放已更新，尚未保存 YAML。"
            self.calibration_status.setText(message)
            self.status_label.setText(message)

        def _capture_calibration_frame(self, fingers: np.ndarray) -> None:
            if self._calibration_started_at is None:
                return
            if time.monotonic() - self._calibration_started_at > self._calibration_timeout_seconds:
                self._fail_calibration(
                    "3 秒内未收集到 45 个有效帧，请保持姿势稳定。"
                )
                return
            try:
                lengths = human_vector_lengths(self.runtime.prepare_keypoints(fingers))
            except CalibrationError:
                return
            except Exception as exc:
                self._fail_calibration(str(exc))
                return

            self._calibration_samples.append(lengths)
            count = len(self._calibration_samples)
            self.calibration_status.setText(
                f"采集中：{count} / {self._calibration_target_frames}"
            )
            if count >= self._calibration_target_frames:
                self._finish_calibration()

        def _tick(self) -> None:
            frame = self.device.get_preview_frame()
            if frame is not None:
                image = QtGui.QImage(
                    frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                    QtGui.QImage.Format.Format_BGR888,
                ).copy()
                pixmap = QtGui.QPixmap.fromImage(image).scaled(
                    self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(pixmap)

            now = time.monotonic()
            elapsed = now - self._last_tick
            if elapsed > 0:
                self._fps = 1.0 / elapsed
            self._last_tick = now
            self.camera_status.setText(
                f"Webcam {args.camera_index} | 手侧: {args.hand} | GUI: {self._fps:.1f} FPS"
            )
            if self.debug_worker.error is not None:
                self.status_label.setText(f"MuJoCo debug 失败: {self.debug_worker.error}")

            fingers = self.device.get_fingers_data()[f"{args.hand}_fingers"]
            if np.allclose(fingers, 0):
                return
            self._capture_calibration_frame(fingers)
            try:
                qpos, diagnostics = self.runtime.process(fingers)
                self.debug_worker.submit(qpos, diagnostics, self.runtime.retargeter.optimizer)
            except Exception as exc:
                self.status_label.setText(f"重定向失败: {exc}")

        def closeEvent(self, event: Any) -> None:
            self._frame_timer.stop()
            self._apply_timer.stop()
            self.debug_worker.stop()
            self.debug_worker.join(timeout=2.0)
            self.device.cleanup()
            event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = TuningMainWindow()
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.webcam:
        parser.error("首版仅支持 --webcam USB 摄像头输入")
    try:
        return _run_gui(args)
    except RuntimeError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
