"""Desktop GUI for real-time LDJY retargeting parameter tuning."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ldjy_retargeting.simulation_timing import physics_steps_for_tick


DEBUG_CONTROL_HZ = 120
INPUT_DEVICE_TYPES = ("webcam", "webcam_wilor")
MEDIAPIPE_ONLY_PARAMETER_PREFIX = "video_input."
TUNING_RECORDS_ROOT = PROJECT_ROOT / "outputs" / "tuning_records"


class RunMode(str, Enum):
    LIVE = "live"
    REPLAY = "replay"


@dataclass(frozen=True)
class RuntimeContext:
    """The selected input is inert until the user explicitly applies it."""

    mode: RunMode
    input_device_type: str
    camera_index: int
    hand_side: str
    record_path: Path | None = None

    @property
    def hand_side_editable(self) -> bool:
        return self.mode is RunMode.LIVE

    @classmethod
    def default(cls) -> "RuntimeContext":
        return cls(RunMode.LIVE, "webcam", 0, "right")

    @classmethod
    def for_record(cls, record_info: Any) -> "RuntimeContext":
        return cls(RunMode.REPLAY, record_info.input_type, 0, record_info.hand_side, record_info.path)


@dataclass(frozen=True)
class AlgorithmChoice:
    """One GUI-selectable optimizer and its canonical editable YAML."""

    key: str
    label: str
    config_name: str
    input_types: tuple[str, ...]

    def supports_input(self, input_device_type: str) -> bool:
        return input_device_type in self.input_types


def algorithm_choices() -> tuple[AlgorithmChoice, ...]:
    """List supported algorithms in the GUI display order."""
    return (
        AlgorithmChoice(
            "adaptive_mediapipe", "Adaptive Analytical (MediaPipe)",
            "adaptive_analytical_video.yaml", ("webcam",),
        ),
        AlgorithmChoice(
            "adaptive_wilor", "Adaptive Analytical (WiLoR 21 点)",
            "adaptive_analytical_wilor.yaml", ("webcam_wilor",),
        ),
    )


def algorithm_choice(key: str) -> AlgorithmChoice:
    for choice in algorithm_choices():
        if choice.key == key:
            return choice
    raise ValueError(f"unsupported tuning algorithm: {key}")


def algorithm_key_for_config(config_path: Path, input_device_type: str) -> str:
    """Infer the initial UI selection while retaining a caller-provided YAML."""
    name = config_path.name
    for choice in algorithm_choices():
        if name == choice.config_name:
            return choice.key
    return "adaptive_wilor" if input_device_type == "webcam_wilor" else "adaptive_mediapipe"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LDJY 实时重定向调参 GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/adaptive_analytical_video.yaml",
                        help="调参 YAML，相对路径以 example 目录为基准")
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--webcam", action="store_true",
                             help="使用 USB 摄像头和 MediaPipe 输入")
    input_group.add_argument("--webcam-wilor", action="store_true",
                             help="使用 USB 摄像头和 WiLoR MANO 输入（需要 --extra wilor）")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="OpenCV USB 摄像头索引")
    parser.add_argument("--hand", choices=("left", "right"), default="right",
                        help="要重定向的手侧")
    return parser


def input_device_type_from_args(args: argparse.Namespace) -> str:
    """Resolve the GUI's intentionally small live-input device map."""
    if args.webcam:
        return "webcam"
    if args.webcam_wilor:
        return "webcam_wilor"
    return "webcam"


def parameter_specs_for_input(input_device_type: str):
    """Return only controls that affect the selected live input chain."""
    if input_device_type not in INPUT_DEVICE_TYPES:
        raise ValueError(f"unsupported tuning GUI input device: {input_device_type}")
    from ldjy_retargeting.tuning.parameters import parameter_specs

    specs = parameter_specs()
    if input_device_type == "webcam_wilor":
        return tuple(
            spec for spec in specs
            if not spec.path.startswith(MEDIAPIPE_ONLY_PARAMETER_PREFIX)
        )
    return specs


def parameter_specs_for_selection(algorithm_key: str, input_device_type: str):
    """Return only controls consumed by the selected algorithm/input pair."""
    choice = algorithm_choice(algorithm_key)
    if not choice.supports_input(input_device_type):
        raise ValueError(f"{choice.label} does not support {input_device_type}")
    return parameter_specs_for_input(input_device_type)


def supports_mano_overlay(input_device_type: str) -> bool:
    """Only WiLoR produces a MANO mesh with camera-aligned pose data."""
    if input_device_type not in INPUT_DEVICE_TYPES:
        raise ValueError(f"unsupported tuning GUI input device: {input_device_type}")
    return input_device_type == "webcam_wilor"


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

    def __init__(self, hand_side: str, mjcf_path: Path):
        super().__init__(name="ldjy-mujoco-debug", daemon=True)
        self.hand_side = hand_side
        self._mjcf_path = Path(mjcf_path)
        self._state_lock = threading.Lock()
        self._frame: tuple[np.ndarray, dict[str, Any], Any] | None = None
        self._paused = False
        self._show_skeleton = True
        self._show_rays = True
        self._stop_event = threading.Event()
        self._viewer = None
        self.error: Exception | None = None

    def reload_tip_sites(self, mjcf_path: Path) -> None:
        """Request a worker-thread update of virtual tip site positions."""
        with self._state_lock:
            self._mjcf_path = Path(mjcf_path)

    @property
    def paused(self) -> bool:
        with self._state_lock:
            return self._paused

    def set_paused(self, paused: bool) -> None:
        with self._state_lock:
            self._paused = paused

    def set_display_options(self, *, show_skeleton: bool, show_rays: bool) -> None:
        with self._state_lock:
            self._show_skeleton = show_skeleton
            self._show_rays = show_rays

    def submit(self, qpos: np.ndarray, diagnostics: dict[str, Any], optimizer: Any) -> None:
        """Replace the latest optimizer command consumed by the viewer thread."""
        keypoints = diagnostics.get("mediapipe_kp")
        if keypoints is None:
            keypoints = diagnostics["joints_task_m"]
        pinch_alphas = np.asarray(
            diagnostics.get("pinch_alphas", np.zeros(5)), dtype=np.float64
        )
        if pinch_alphas.shape == (4,):
            pinch_alphas = np.r_[pinch_alphas.max(initial=0.0), pinch_alphas]
        frame_diagnostics = {
            "mediapipe_kp": np.asarray(keypoints, dtype=np.float64).copy(),
            "pinch_alphas": pinch_alphas.copy(),
        }
        frame = (
            np.asarray(qpos, dtype=np.float64).copy(),
            frame_diagnostics,
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
            with self._state_lock:
                mjcf_path = self._mjcf_path
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

            control_tick = 0
            next_control_tick = time.monotonic()
            applied_mjcf_path = mjcf_path
            while viewer.is_running() and not self._stop_event.is_set():
                with self._state_lock:
                    frame = self._frame
                    requested_mjcf_path = self._mjcf_path
                    paused = self._paused
                    overlay.show_skeleton = self._show_skeleton
                    overlay.show_rays = self._show_rays
                if requested_mjcf_path != applied_mjcf_path:
                    source_model = mujoco.MjModel.from_xml_path(str(requested_mjcf_path))
                    for finger in ("thumb", "finger1", "finger2", "finger3", "finger4"):
                        name = f"{self.hand_side}_{finger}_link4_tip"
                        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                        source_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_SITE, name)
                        model.site_pos[target_id] = source_model.site_pos[source_id]
                    mujoco.mj_forward(model, data)
                    applied_mjcf_path = requested_mjcf_path
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
                    if frame is not None and not paused:
                        data.ctrl[:] = actuator_targets
                    if not paused:
                        physics_steps = physics_steps_for_tick(
                            control_tick, model.opt.timestep, DEBUG_CONTROL_HZ
                        )
                        for _ in range(physics_steps):
                            mujoco.mj_step(model, data)
                    if frame is not None:
                        overlay.draw(
                            viewer.user_scn,
                            data,
                            optimizer,
                            diagnostics["mediapipe_kp"],
                            diagnostics["pinch_alphas"],
                        )
                viewer.sync()

                if not paused:
                    control_tick += 1
                next_control_tick += 1.0 / DEBUG_CONTROL_HZ
                remaining = next_control_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_control_tick = time.monotonic()
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

    from ldjy_retargeting.tuning.parameters import get_path
    from ldjy_retargeting.tuning.runtime import TuningRuntime
    from ldjy_retargeting.tuning.session import TuningSession
    from ldjy_retargeting.tuning.vector_scale_calibration import (
        CalibrationError,
        compute_scales,
        human_vector_lengths,
    )
    from ldjy_retargeting.retarget_tip_frames import DEFAULT_OFFSET_FILE

    config_path = _resolve_config_path(args.config).resolve()
    initial_context = RuntimeContext(
        RunMode.LIVE, input_device_type_from_args(args), args.camera_index, args.hand
    )
    initial_algorithm_key = algorithm_key_for_config(config_path, initial_context.input_device_type)

    class TuningMainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("LDJY 实时重定向调参")
            self.resize(1440, 900)
            self.context = initial_context
            self.input_device_type = self.context.input_device_type
            self.algorithm_key = initial_algorithm_key
            self.session = TuningSession(config_path)
            self._sessions = {self.algorithm_key: self.session}
            self.runtime = None
            self.device = None
            self.replay = None
            self.debug_worker = None
            self._record_writer = None
            self._recording_requested = False
            self._widgets: dict[str, tuple[Any, Any, Any]] = {}
            self._last_tick = time.monotonic()
            self._fps = 0.0
            self._calibration_samples: list[np.ndarray] = []
            self._calibration_started_at: float | None = None
            self._calibration_target_frames = 45
            self._calibration_timeout_seconds = 3.0
            self._mano_overlay_sequence: object | None = None
            self._mano_overlay_frame: np.ndarray | None = None

            self._build_ui()
            self._apply_timer = QtCore.QTimer(self)
            self._apply_timer.setSingleShot(True)
            self._apply_timer.setInterval(100)
            self._apply_timer.timeout.connect(self._apply_live_config)
            self._frame_timer = QtCore.QTimer(self)
            self._frame_timer.setInterval(33)
            self._frame_timer.timeout.connect(self._tick)
            self._frame_timer.start()

        def _create_input_device(self, context: RuntimeContext, config: dict[str, Any]):
            if context.input_device_type == "webcam":
                from input_devices.webcam_mediapipe import WebcamMediaPipe

                return WebcamMediaPipe(
                    hand_side=context.hand_side,
                    camera_index=context.camera_index,
                    video_config=config.get("video_input", {}),
                    show_video=False,
                )
            try:
                from input_devices.webcam_wilor import WebcamWiLoR

                return WebcamWiLoR(
                    hand_side=context.hand_side,
                    camera_index=context.camera_index,
                    show_video=False,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "WiLoR 输入需要额外依赖。请执行: uv sync --extra wilor"
                ) from exc

        def _build_ui(self) -> None:
            root = QtWidgets.QWidget()
            root_layout = QtWidgets.QVBoxLayout(root)
            root_layout.setContentsMargins(8, 8, 8, 8)
            self.setCentralWidget(root)
            root_layout.addWidget(self._create_context_bar())
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            root_layout.addWidget(splitter, stretch=1)

            preview_panel = QtWidgets.QWidget()
            preview_layout = QtWidgets.QVBoxLayout(preview_panel)
            self.preview_title = QtWidgets.QLabel("OpenCV / MediaPipe 检测画面")
            title = self.preview_title
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
            self.toolbox = toolbox
            groups: dict[str, list[Any]] = defaultdict(list)
            for spec in parameter_specs_for_selection(self.algorithm_key, self.input_device_type):
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
            export_button = QtWidgets.QPushButton("导出正式资产")
            export_button.clicked.connect(self._export_tip_assets)
            self.export_button = export_button
            actions.addWidget(save_button)
            actions.addWidget(default_button)
            actions.addWidget(export_button)
            parameter_layout.addLayout(actions)
            debug_actions = QtWidgets.QHBoxLayout()
            self.skeleton_toggle = QtWidgets.QToolButton()
            self.skeleton_toggle.setText("骨架")
            self.skeleton_toggle.setCheckable(True)
            self.skeleton_toggle.setChecked(True)
            self.skeleton_toggle.toggled.connect(self._set_debug_display)
            self.rays_toggle = QtWidgets.QToolButton()
            self.rays_toggle.setText("射线")
            self.rays_toggle.setCheckable(True)
            self.rays_toggle.setChecked(True)
            self.rays_toggle.toggled.connect(self._set_debug_display)
            self.mano_toggle = QtWidgets.QToolButton()
            self.mano_toggle.setText("MANO 网格")
            self.mano_toggle.setToolTip("在原始摄像头画面叠加相机对齐的 WiLoR MANO 网格")
            self.mano_toggle.setCheckable(True)
            self.mano_toggle.setVisible(False)
            self.mano_toggle.toggled.connect(self._set_mano_overlay)
            debug_actions.addWidget(self.mano_toggle)
            self.mano_skeleton_toggle = QtWidgets.QToolButton()
            self.mano_skeleton_toggle.setText("MANO 骨架")
            self.mano_skeleton_toggle.setToolTip("在原始摄像头画面叠加 WiLoR 的 21 点 MANO 骨架")
            self.mano_skeleton_toggle.setCheckable(True)
            self.mano_skeleton_toggle.setVisible(False)
            self.mano_skeleton_toggle.toggled.connect(self._set_mano_overlay)
            debug_actions.addWidget(self.mano_skeleton_toggle)
            self.run_toggle = QtWidgets.QPushButton("暂停")
            self.run_toggle.setToolTip("暂停或继续当前实时输入 / 回放播放")
            self.run_toggle.clicked.connect(self._toggle_run)
            self.run_toggle.setEnabled(False)
            self.record_button = QtWidgets.QPushButton("开始记录")
            self.record_button.clicked.connect(self._toggle_recording)
            self.first_frame_button = QtWidgets.QToolButton()
            self.first_frame_button.setText("|<")
            self.first_frame_button.setToolTip("首帧")
            self.first_frame_button.clicked.connect(lambda: self._seek_replay(0))
            self.previous_frame_button = QtWidgets.QToolButton()
            self.previous_frame_button.setText("<")
            self.previous_frame_button.setToolTip("上一帧")
            self.previous_frame_button.clicked.connect(lambda: self._step_replay(-1))
            self.next_frame_button = QtWidgets.QToolButton()
            self.next_frame_button.setText(">")
            self.next_frame_button.setToolTip("下一帧")
            self.next_frame_button.clicked.connect(lambda: self._step_replay(1))
            self.replay_timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.replay_timeline.setMinimumWidth(150)
            self.replay_timeline.sliderMoved.connect(self._seek_replay)
            self.replay_frame_label = QtWidgets.QLabel()
            debug_actions.addWidget(self.skeleton_toggle)
            debug_actions.addWidget(self.rays_toggle)
            debug_actions.addWidget(self.run_toggle)
            debug_actions.addWidget(self.record_button)
            debug_actions.addWidget(self.first_frame_button)
            debug_actions.addWidget(self.previous_frame_button)
            debug_actions.addWidget(self.next_frame_button)
            debug_actions.addWidget(self.replay_timeline, stretch=1)
            debug_actions.addWidget(self.replay_frame_label)
            parameter_layout.addLayout(debug_actions)
            self.dirty_label = QtWidgets.QLabel()
            parameter_layout.addWidget(self.dirty_label)
            self.status_label = QtWidgets.QLabel("MuJoCo debug 窗口正在启动...")
            self.status_label.setWordWrap(True)
            parameter_layout.addWidget(self.status_label)
            splitter.addWidget(parameter_panel)
            splitter.setSizes([800, 640])
            self._update_dirty_label()
            self._set_replay_controls_visible(False)

        def _create_context_bar(self) -> QtWidgets.QWidget:
            bar = QtWidgets.QWidget()
            form = QtWidgets.QHBoxLayout(bar)
            form.setContentsMargins(0, 0, 0, 0)
            form.addWidget(QtWidgets.QLabel("重定向算法"))
            self.algorithm_combo = QtWidgets.QComboBox()
            for choice in algorithm_choices():
                self.algorithm_combo.addItem(choice.label, choice.key)
            self.algorithm_combo.setCurrentIndex(
                self.algorithm_combo.findData(self.algorithm_key)
            )
            self.algorithm_combo.currentIndexChanged.connect(self._context_controls_changed)
            form.addWidget(self.algorithm_combo)
            form.addWidget(QtWidgets.QLabel("运行模式"))
            self.mode_combo = QtWidgets.QComboBox()
            self.mode_combo.addItem("实时 USB", RunMode.LIVE.value)
            self.mode_combo.addItem("静态调参记录", RunMode.REPLAY.value)
            self.mode_combo.currentIndexChanged.connect(self._context_controls_changed)
            form.addWidget(self.mode_combo)
            form.addWidget(QtWidgets.QLabel("输入设备 / 记录类型"))
            self.input_combo = QtWidgets.QComboBox()
            self.input_combo.addItem("Webcam MediaPipe", "webcam")
            self.input_combo.addItem("Webcam WiLoR", "webcam_wilor")
            self.input_combo.setCurrentIndex(
                self.input_combo.findData(self.context.input_device_type)
            )
            self.input_combo.currentIndexChanged.connect(self._context_controls_changed)
            form.addWidget(self.input_combo)
            self.camera_caption = QtWidgets.QLabel("USB 相机")
            form.addWidget(self.camera_caption)
            self.camera_spin = QtWidgets.QSpinBox()
            self.camera_spin.setRange(0, 99)
            self.camera_spin.setValue(self.context.camera_index)
            form.addWidget(self.camera_spin)
            self.record_caption = QtWidgets.QLabel("记录")
            self.record_combo = QtWidgets.QComboBox()
            self.record_combo.setMinimumWidth(180)
            form.addWidget(self.record_caption)
            form.addWidget(self.record_combo)
            self.hand_caption = QtWidgets.QLabel("手侧")
            form.addWidget(self.hand_caption)
            self.hand_combo = QtWidgets.QComboBox()
            self.hand_combo.addItem("右手", "right")
            self.hand_combo.addItem("左手", "left")
            form.addWidget(self.hand_combo)
            self.apply_input_button = QtWidgets.QPushButton("应用输入")
            self.apply_input_button.clicked.connect(self._apply_selected_context)
            form.addWidget(self.apply_input_button)
            self.busy_indicator = QtWidgets.QProgressBar()
            self.busy_indicator.setRange(0, 0)
            self.busy_indicator.setFixedWidth(70)
            self.busy_indicator.setVisible(False)
            form.addWidget(self.busy_indicator)
            self.context_status = QtWidgets.QLabel("请选择输入后点击“应用输入”")
            form.addWidget(self.context_status, stretch=1)
            self._context_controls_changed()
            return bar

        def _context_controls_changed(self) -> None:
            self._update_algorithm_availability()
            replay = self.mode_combo.currentData() == RunMode.REPLAY.value
            self.camera_caption.setVisible(not replay)
            self.camera_spin.setVisible(not replay)
            self.record_caption.setVisible(replay)
            self.record_combo.setVisible(replay)
            self.hand_caption.setVisible(not replay)
            self.hand_combo.setVisible(not replay)
            self.hand_combo.setEnabled(not replay)
            if replay:
                self._refresh_record_choices()

        def _update_algorithm_availability(self) -> None:
            input_type = self.input_combo.currentData()
            for index, choice in enumerate(algorithm_choices()):
                item = self.algorithm_combo.model().item(index)
                if item is not None:
                    item.setEnabled(choice.supports_input(input_type))
            selected = algorithm_choice(self.algorithm_combo.currentData())
            if not selected.supports_input(input_type):
                for index, choice in enumerate(algorithm_choices()):
                    if choice.supports_input(input_type):
                        self.algorithm_combo.blockSignals(True)
                        self.algorithm_combo.setCurrentIndex(index)
                        self.algorithm_combo.blockSignals(False)
                        break

        def _selected_algorithm(self) -> AlgorithmChoice:
            return algorithm_choice(self.algorithm_combo.currentData())

        def _session_for_algorithm(self, choice: AlgorithmChoice) -> Any:
            session = self._sessions.get(choice.key)
            if session is None:
                path = EXAMPLE_DIR / "config" / choice.config_name
                session = TuningSession(path)
                self._sessions[choice.key] = session
            return session

        def _refresh_record_choices(self) -> None:
            from ldjy_retargeting.tuning.recording import list_records

            input_type = self.input_combo.currentData()
            current = self.record_combo.currentData()
            self.record_combo.blockSignals(True)
            self.record_combo.clear()
            for info in list_records(TUNING_RECORDS_ROOT, input_type):
                self.record_combo.addItem(f"{info.name} ({info.hand_side})", info)
            if current is not None:
                index = self.record_combo.findData(current)
                if index >= 0:
                    self.record_combo.setCurrentIndex(index)
            self.record_combo.blockSignals(False)

        def _set_busy(self, busy: bool, text: str) -> None:
            for widget in (
                self.algorithm_combo, self.mode_combo, self.input_combo, self.camera_spin,
                self.hand_combo, self.record_combo, self.apply_input_button,
            ):
                widget.setEnabled(not busy)
            self.busy_indicator.setVisible(busy)
            self.context_status.setText(text)
            QtWidgets.QApplication.processEvents()

        def _selected_context(self) -> RuntimeContext:
            input_type = self.input_combo.currentData()
            if self.mode_combo.currentData() == RunMode.REPLAY.value:
                record = self.record_combo.currentData()
                if record is None:
                    raise ValueError("当前类型没有可回放的完整记录")
                return RuntimeContext.for_record(record)
            return RuntimeContext(
                RunMode.LIVE, input_type, self.camera_spin.value(), self.hand_combo.currentData()
            )

        def _apply_selected_context(self) -> None:
            try:
                context = self._selected_context()
                choice = self._selected_algorithm()
                if not choice.supports_input(context.input_device_type):
                    raise ValueError(f"{choice.label} 不支持当前输入")
            except ValueError as exc:
                self.context_status.setText(str(exc))
                return
            self._set_busy(True, "正在加载输入...")
            try:
                candidate_session = self._session_for_algorithm(choice)
                if context.mode is RunMode.LIVE:
                    candidate_device = self._create_input_device(context, candidate_session.config)
                    candidate_replay = None
                elif context.input_device_type == "webcam":
                    from ldjy_retargeting.tuning.replay import MediaPipeReplay
                    candidate_device, candidate_replay = None, MediaPipeReplay(context.record_path)
                else:
                    from ldjy_retargeting.tuning.replay import WiLoRReplay
                    candidate_device, candidate_replay = None, WiLoRReplay(context.record_path)
                candidate_runtime = TuningRuntime(
                    candidate_session.config,
                    context.hand_side,
                    yaml_dir=candidate_session.config_path.parent,
                )
                candidate_debug = MuJoCoDebugWorker(context.hand_side, candidate_runtime.debug_mjcf_path)
                candidate_debug.start()
            except Exception as exc:
                if 'candidate_device' in locals() and candidate_device is not None:
                    candidate_device.cleanup()
                if 'candidate_replay' in locals() and candidate_replay is not None:
                    candidate_replay.close()
                self._set_busy(False, f"输入加载失败: {exc}")
                return
            self._stop_recording(finish=True)
            self._cleanup_active_session()
            self.context = context
            self.input_device_type = context.input_device_type
            self.algorithm_key = choice.key
            self.session = candidate_session
            self.device, self.replay = candidate_device, candidate_replay
            self.runtime, self.debug_worker = candidate_runtime, candidate_debug
            self._invalidate_mano_overlay()
            self.hand_combo.setCurrentIndex(0 if context.hand_side == "right" else 1)
            self.hand_combo.setEnabled(context.hand_side_editable)
            self.preview_title.setText(
                f"OpenCV / {'MediaPipe' if self.input_device_type == 'webcam' else 'WiLoR'} "
                f"{'回放画面' if context.mode is RunMode.REPLAY else '检测画面'}"
            )
            self.mano_toggle.setVisible(supports_mano_overlay(self.input_device_type))
            self.mano_skeleton_toggle.setVisible(supports_mano_overlay(self.input_device_type))
            self.record_button.setVisible(context.mode is RunMode.LIVE)
            self.export_button.setEnabled(True)
            self.export_button.setToolTip("将当前虚拟 tip 调整导出为正式资产。")
            self.run_toggle.setEnabled(True)
            self.run_toggle.setText("暂停")
            self._set_replay_controls_visible(context.mode is RunMode.REPLAY)
            if context.mode is RunMode.REPLAY:
                self._sync_replay_controls()
            self._rebuild_parameter_controls()
            self._set_debug_display()
            self._set_busy(False, "输入已应用" if context.mode is RunMode.LIVE else f"已载入记录: {context.record_path.name}")

        def _cleanup_active_session(self) -> None:
            if self.debug_worker is not None:
                self.debug_worker.stop()
                self.debug_worker.join(timeout=2.0)
            if self.device is not None:
                self.device.cleanup()
            if self.replay is not None:
                self.replay.close()
            self.device = self.replay = self.debug_worker = None

        def _set_replay_controls_visible(self, visible: bool) -> None:
            for widget in (
                self.first_frame_button, self.previous_frame_button, self.next_frame_button,
                self.replay_timeline, self.replay_frame_label,
            ):
                widget.setVisible(visible)

        def _sync_replay_controls(self) -> None:
            if self.replay is None:
                return
            total = len(self.replay.cursor.timestamps)
            self.replay_timeline.blockSignals(True)
            self.replay_timeline.setRange(0, total - 1)
            self.replay_timeline.setValue(self.replay.current_index)
            self.replay_timeline.blockSignals(False)
            self.replay_frame_label.setText(f"{self.replay.current_index + 1} / {total}")

        def _seek_replay(self, index: int) -> None:
            if self.replay is None:
                return
            self.replay.cursor.pause()
            self.replay.cursor.seek(index)
            self._sync_replay_controls()

        def _step_replay(self, delta: int) -> None:
            if self.replay is None:
                return
            self.replay.cursor.pause()
            self.replay.cursor.step(delta)
            self._sync_replay_controls()

        def _rebuild_parameter_controls(self) -> None:
            while self.toolbox.count():
                widget = self.toolbox.widget(0)
                self.toolbox.removeItem(0)
                widget.deleteLater()
            self._widgets = {}
            groups: dict[str, list[Any]] = defaultdict(list)
            for spec in parameter_specs_for_selection(self.algorithm_key, self.input_device_type):
                groups[spec.group].append(spec)
            for group_name, specs in groups.items():
                self.toolbox.addItem(self._create_group(specs), group_name)

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
            if self.debug_worker is not None and self.debug_worker.paused and path.startswith("tip_offsets."):
                try:
                    self.debug_worker.reload_tip_sites(
                        self.runtime.preview_tip_offsets(self.session.config)
                    )
                    self.status_label.setText(
                        "已暂停：虚拟 tip 已更新；点击“开始”后应用到优化器。"
                    )
                except Exception as exc:
                    self.status_label.setText(f"虚拟 tip 预览失败: {exc}")
                return
            self._apply_timer.start()

        def _apply_live_config(self) -> None:
            if self.runtime is None or self.debug_worker is None:
                return
            if self.debug_worker.paused:
                self.status_label.setText("已暂停：参数已修改，点击“开始”后应用到优化器与仿真。")
                return
            try:
                self.runtime.apply_config(self.session.config)
                self.debug_worker.reload_tip_sites(self.runtime.debug_mjcf_path)
                self.status_label.setText("参数已应用到实时重定向，滤波器与 warm start 已重置。")
            except Exception as exc:
                self.status_label.setText(f"参数应用失败: {exc}")

        def _set_debug_display(self) -> None:
            if self.debug_worker is None:
                return
            self.debug_worker.set_display_options(
                show_skeleton=self.skeleton_toggle.isChecked(),
                show_rays=self.rays_toggle.isChecked(),
            )

        def _set_mano_overlay(self, enabled: bool) -> None:
            del enabled
            self._invalidate_mano_overlay()

        def _invalidate_mano_overlay(self) -> None:
            self._mano_overlay_frame = None
            self._mano_overlay_sequence = None

        def _preview_frame(self) -> np.ndarray | None:
            if self.context.mode is RunMode.REPLAY:
                if self.replay is None:
                    return None
                raw_frame = self.replay.preview_at(self.replay.current_index)
                if not supports_mano_overlay(self.input_device_type):
                    return raw_frame
                overlay = self.replay.mano_overlay_at(self.replay.current_index)
                return self._render_mano_overlay(
                    raw_frame, overlay,
                    ("replay", str(self.context.record_path), self.replay.current_index),
                )
            if self.device is None:
                return None
            if not supports_mano_overlay(self.input_device_type):
                return self.device.get_preview_frame()

            raw_frame = self.device.get_raw_preview_frame()
            if not self.mano_toggle.isChecked() and not self.mano_skeleton_toggle.isChecked():
                return raw_frame
            overlay = self.device.get_mano_overlay_data()
            return self._render_mano_overlay(
                raw_frame, overlay, ("live", overlay["sequence"]) if overlay else None
            )

        def _render_mano_overlay(self, raw_frame: np.ndarray | None, overlay, sequence: object) -> np.ndarray | None:
            if raw_frame is None:
                return raw_frame
            if overlay is None:
                return raw_frame
            cache_key = (
                sequence,
                self.mano_toggle.isChecked(),
                self.mano_skeleton_toggle.isChecked(),
            )
            if not self.mano_toggle.isChecked() and not self.mano_skeleton_toggle.isChecked():
                return raw_frame
            if self._mano_overlay_sequence != cache_key:
                try:
                    from ldjy_retargeting.wilor_viewer import (
                        draw_mano_mesh_overlay,
                        draw_mano_skeleton_overlay,
                    )

                    frame = raw_frame
                    if self.mano_toggle.isChecked():
                        frame = draw_mano_mesh_overlay(
                            frame, vertices_mano=overlay["vertices_mano"],
                            camera_translation=overlay["camera_translation"],
                            faces=overlay["faces"], is_right=overlay["is_right"],
                        )
                    if self.mano_skeleton_toggle.isChecked():
                        frame = draw_mano_skeleton_overlay(
                            frame, joints_mano=overlay["joints_mano"],
                            camera_translation=overlay["camera_translation"],
                            is_right=overlay["is_right"],
                        )
                    self._mano_overlay_frame = frame
                    self._mano_overlay_sequence = cache_key
                except Exception as exc:
                    self.status_label.setText(f"MANO 预览失败: {exc}")
                    return raw_frame
            return self._mano_overlay_frame.copy()

        def _toggle_recording(self) -> None:
            if self.context.mode is not RunMode.LIVE:
                return
            if self._recording_requested:
                self._stop_recording(finish=True)
                return
            if self.device is not None:
                # A segment begins at the button press, not at device startup.
                self.device.drain_inference_samples()
            self._recording_requested = True
            self.record_button.setText("停止记录")
            self.status_label.setText("正在等待下一次完整推理后开始记录...")

        def _start_writer_for_sample(self, sample) -> None:
            from ldjy_retargeting.tuning.recording import MediaPipeRecordWriter, WiLoRRecordWriter

            height, width = sample.frame_bgr.shape[:2]
            if sample.input_type == "webcam":
                self._record_writer = MediaPipeRecordWriter.start(
                    TUNING_RECORDS_ROOT, hand_side=self.context.hand_side, width=width, height=height
                )
            else:
                faces = getattr(self.device, "_mano_faces", None)
                if faces is None:
                    raise RuntimeError("WiLoR MANO faces 尚未就绪，无法创建记录")
                self._record_writer = WiLoRRecordWriter.start(
                    TUNING_RECORDS_ROOT, hand_side=self.context.hand_side, width=width, height=height, faces=faces
                )
            self.status_label.setText(f"正在记录: {self._record_writer._final_path.name}")

        def _drain_record_samples(self) -> None:
            if not self._recording_requested or self.device is None:
                return
            from ldjy_retargeting.tuning.recording import RecordSample, WiLoRRecordSample

            for sample in self.device.drain_inference_samples():
                if self._record_writer is None:
                    self._start_writer_for_sample(sample)
                if sample.input_type == "webcam":
                    self._record_writer.append(RecordSample(
                        sample.timestamp_sec, sample.frame_bgr, sample.detected,
                        sample.payload["detector_landmarks"], sample.payload["processed_landmarks"],
                    ))
                else:
                    self._record_writer.append(WiLoRRecordSample(
                        sample.timestamp_sec, sample.frame_bgr, sample.detected, sample.payload["detection"]
                    ))

        def _stop_recording(self, *, finish: bool) -> None:
            if self._record_writer is not None:
                try:
                    info = self._record_writer.finish(config=self.session.config) if finish else None
                    if info is not None:
                        self.status_label.setText(f"记录已保存: {info.path}")
                except Exception as exc:
                    self.status_label.setText(f"记录保存失败: {exc}")
            self._record_writer = None
            self._recording_requested = False
            if hasattr(self, "record_button"):
                self.record_button.setText("开始记录")

        def _pause(self) -> None:
            if self.device is not None:
                self.device.set_paused(True)
            if self.replay is not None:
                self.replay.cursor.pause()
            if self.debug_worker is not None:
                self.debug_worker.set_paused(True)
            self.status_label.setText("已暂停：输入、重定向与 MuJoCo 保持当前帧。")
            self.run_toggle.setText("继续")

        def _resume(self) -> None:
            if self.device is not None:
                self.device.set_paused(False)
            if self.replay is not None:
                self.replay.cursor.play()
            if self.debug_worker is not None:
                self.debug_worker.set_paused(False)
            self._apply_live_config()
            self.status_label.setText("已开始：从当前摄像头最新帧继续。")
            self.run_toggle.setText("暂停")

        def _toggle_run(self) -> None:
            if self.debug_worker is not None and self.debug_worker.paused:
                self._resume()
            else:
                self._pause()

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

        def _export_tip_assets(self) -> None:
            """Write tuned tips as the canonical source and regenerate all outputs."""
            try:
                from ldjy_retargeting.tuning.tip_assets import persist_tip_offsets
                from tools.build_ldjy_mjcf import build_model
                from tools.build_ldjy_urdf import build_urdf
                from tools.build_openarm_hand_mjcf import build_mjcf
                from tools.build_openarm_hand_urdf import build_urdf as build_openarm_urdf

                offsets = persist_tip_offsets(self.session, DEFAULT_OFFSET_FILE)
                for side in ("right", "left"):
                    build_urdf(side, offsets=offsets)
                for side in ("right", "left"):
                    build_model(side, offsets=offsets)
                build_openarm_urdf(offsets=offsets)
                build_mjcf()
                self._update_dirty_label()
                self.status_label.setText("已保存 YAML，并导出正式 LDJY 与 OpenArm URDF/MJCF 资产。")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "导出资产失败", str(exc))

        def _start_calibration(self) -> None:
            if self.runtime is None:
                self.status_label.setText("请先点击“应用输入”后再标定。")
                return
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
            if self.debug_worker is None or self.runtime is None:
                return
            paused = self.debug_worker.paused
            if not paused:
                if self.context.mode is RunMode.LIVE:
                    self._drain_record_samples()
                elif self.replay is not None and self.replay.cursor.playing:
                    self.replay.cursor.advance()
                    self._sync_replay_controls()
            frame = self._preview_frame()
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
                f"{'记录回放' if self.context.mode is RunMode.REPLAY else f'Webcam {self.context.camera_index}'} | 输入: "
                f"{'MediaPipe' if self.input_device_type == 'webcam' else 'WiLoR'} | "
                f"手侧: {self.context.hand_side} | GUI: {self._fps:.1f} FPS"
            )
            if self.debug_worker.error is not None:
                self.status_label.setText(f"MuJoCo debug 失败: {self.debug_worker.error}")
            if paused:
                return

            if self.context.mode is RunMode.LIVE:
                fingers = self.device.get_fingers_data()[f"{self.context.hand_side}_fingers"]
            elif self.input_device_type == "webcam":
                fingers = self.replay.input_at(
                    self.replay.current_index, self.session.config.get("video_input", {})
                )
            else:
                fingers = self.replay.input_at(self.replay.current_index)
            if fingers is None:
                return
            if np.allclose(fingers, 0):
                return
            try:
                self._capture_calibration_frame(fingers)
                qpos, diagnostics = self.runtime.process(fingers)
                self.debug_worker.submit(qpos, diagnostics, self.runtime.retargeter.optimizer)
            except Exception as exc:
                self.status_label.setText(f"重定向失败: {exc}")

        def closeEvent(self, event: Any) -> None:
            self._frame_timer.stop()
            self._apply_timer.stop()
            self._stop_recording(finish=True)
            self._cleanup_active_session()
            event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = TuningMainWindow()
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_gui(args)
    except RuntimeError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
