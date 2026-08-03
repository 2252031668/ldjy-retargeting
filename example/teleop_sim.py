"""Teleoperation with MuJoCo Simulation.

Uses the Retargeter interface to map hand tracking input to LDJY hand joint angles,
visualized in MuJoCo simulation.

Usage:
    # Replay MediaPipe recording (default)
    mjpython teleop_sim.py --play data/avp1.pkl --hand left

    # MP4 video input with MediaPipe hand detection
    mjpython teleop_sim.py --video data/right.mp4 --hand right
    mjpython teleop_sim.py --video data/right.mp4 --hand right --show-video

    # Live USB webcam input with MediaPipe hand detection
    python teleop_sim.py --webcam --camera-index 0 --hand right --show-video

    # OpenArm bimanual model: raise and hold the selected arm, retarget its hand
    python teleop_sim.py --webcam --camera-index 0 --hand right --robot openarm

    # RealSense camera input with MediaPipe hand detection
    mjpython teleop_sim.py --realsense --hand right

    # ZED camera input with MediaPipe hand detection
    mjpython teleop_sim.py --zed --hand right

    # Live VisionPro input
    mjpython teleop_sim.py --input visionpro --ip <your-vision-pro-ip>

    # Record input data while using VisionPro
    mjpython teleop_sim.py --input visionpro --record

Input device types:
- visionpro: Live VisionPro input
- mediapipe_replay: Replay recorded MediaPipe hand tracking data
- video: MP4 video input with MediaPipe hand detection
- webcam: Live USB webcam input with MediaPipe hand detection
- realsense: RealSense camera input with MediaPipe hand detection
- zed: ZED camera input with MediaPipe hand detection
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ldjy_retargeting import Retargeter
from ldjy_retargeting.openarm_control import OPENARM_MJCF_PATH, OpenArmTeleopControl
from ldjy_retargeting.simulation_timing import physics_steps_for_tick
from ldjy_retargeting.viz.debug_overlay import (
    DebugOverlay,
    format_joint_diagnostics,
    mode_label,
)
from utils.config_paths import resolve_mjcf_path, qpos_reorder_perm
try:
    from input_devices.visionpro import VisionPro
except ImportError:
    VisionPro = None
from input_devices.mediapipe_replay import MediaPipeReplay
try:
    from input_devices.video_mediapipe import VideoMediaPipe
except ImportError:
    VideoMediaPipe = None
try:
    from input_devices.realsense_mediapipe import RealsenseMediaPipe
except ImportError:
    RealsenseMediaPipe = None
try:
    from input_devices.zed_mediapipe import ZedMediaPipe
except ImportError:
    ZedMediaPipe = None
try:
    from input_devices.webcam_mediapipe import WebcamMediaPipe
except ImportError:
    WebcamMediaPipe = None
try:
    from input_devices.webcam_wilor import WebcamWiLoR
except ImportError:
    WebcamWiLoR = None


DEBUG_MESH_ALPHA = 0.3
CONTROL_HZ = 120


def set_debug_mesh_transparency(model: mujoco.MjModel) -> None:
    """Make visible robot meshes translucent without revealing collision meshes."""
    for geom_id in range(model.ngeom):
        if (
            model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_rgba[geom_id, 3] > 0
        ):
            model.geom_rgba[geom_id, 3] = DEBUG_MESH_ALPHA


def run_teleop(
    hand_side: str = "right",
    config_path: str = "config/adaptive_analytical_avp.yaml",
    input_device_type: str = "mediapipe_replay",
    mediapipe_replay_path: str = "",
    visionpro_ip: str = "192.168.50.127",
    playback_speed: float = 1.0,
    playback_loop: bool = True,
    enable_recording: bool = False,
    video_path: str = "",
    show_video: bool = False,
    camera_index: int = 0,
    debug: bool = False,
    robot: str = "ldjy",
):
    """Run teleoperation with MuJoCo simulation.

    Args:
        hand_side: 'right' or 'left'
        config_path: Path to YAML configuration file
        input_device_type: Input device type ('visionpro' or 'mediapipe_replay')
        mediapipe_replay_path: Path to MediaPipe recording (.pkl)
        visionpro_ip: VisionPro IP address
        playback_speed: Playback speed for replay mode
        playback_loop: Whether to loop replay
        enable_recording: Whether to record raw input data
        video_path: Path to MP4 video file
        show_video: Whether to display video with MediaPipe landmarks overlay
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"
    robot = robot.lower()
    if robot not in {"ldjy", "openarm"}:
        raise ValueError("robot must be 'ldjy' or 'openarm'")

    # Load an explicitly configured MJCF, or the bundled LDJY hand for this side.
    config_file = Path(__file__).parent / config_path
    mjcf_override = resolve_mjcf_path(config_file)
    if robot == "openarm":
        mjcf_path = OPENARM_MJCF_PATH
    elif mjcf_override:
        mjcf_path = Path(mjcf_override)
    else:
        mjcf_path = (
            Path(__file__).resolve().parents[1]
            / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
            / "mjcf" / f"ldjy_{hand_side}_hand.xml"
        )
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {mjcf_path}")

    # OpenArm reuses the side-specific 20-DOF LDJY hand optimizer. Its arm
    # joints are deliberately not part of the retargeting optimization.
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if debug:
        set_debug_mesh_transparency(model)
    data = mujoco.MjData(model)
    openarm_control = (
        OpenArmTeleopControl(model, hand_side) if robot == "openarm" else None
    )

    # Start from the mechanical zero pose where it is legal, rather than from
    # each actuator's range midpoint. This keeps finger4_joint1 at 0 instead
    # of its 45-degree midpoint before the first tracked frame arrives.
    if openarm_control is None:
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                ctrl_range = model.actuator_ctrlrange[i]
                data.ctrl[i] = np.clip(0.0, ctrl_range[0], ctrl_range[1])
            else:
                data.ctrl[i] = 0.0
    else:
        openarm_control.set_initial_pose(data)
        hand_zero = np.clip(
            0.0,
            retargeter.optimizer.robot.joint_limits[:, 0],
            retargeter.optimizer.robot.joint_limits[:, 1],
        )
        data.ctrl[:] = openarm_control.targets(
            retargeter.optimizer.robot.dof_joint_names, hand_zero
        )

    # Stabilize model
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    if robot == "openarm":
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = [0, 0, 0.5]
    else:
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0, 0, 0.05]

    # Load config to get video_input settings if needed (config_file resolved above)
    with open(config_file, "r") as f:
        full_config = yaml.safe_load(f)
    video_config = full_config.get("video_input", {})

    def create_visionpro_device():
        if VisionPro is None:
            raise ImportError("visionpro mode requires avp_stream")
        return VisionPro(ip=visionpro_ip)

    # Initialize input device
    device_map = {
        "visionpro": create_visionpro_device,
        "mediapipe_replay": lambda: MediaPipeReplay(
            record_path=mediapipe_replay_path,
            playback_speed=playback_speed,
            loop=playback_loop,
        ),
        "video": lambda: VideoMediaPipe(
            video_path=video_path,
            hand_side=hand_side,
            playback_speed=playback_speed,
            loop=playback_loop,
            video_config=video_config,
            show_video=show_video,
        ),
        "realsense": lambda: RealsenseMediaPipe(
            hand_side=hand_side,
            video_config=video_config,
            show_video=show_video,
        ),
        "zed": lambda: ZedMediaPipe(
            hand_side=hand_side,
            video_config=video_config,
            show_video=show_video,
        ),
        "webcam": lambda: WebcamMediaPipe(
            hand_side=hand_side,
            camera_index=camera_index,
            video_config=video_config,
            show_video=show_video,
        ),
        "webcam_wilor": lambda: WebcamWiLoR(
            hand_side=hand_side,
            camera_index=camera_index,
            show_video=show_video,
        ),
    }
    if input_device_type not in device_map:
        raise ValueError(f"Unknown input device type: {input_device_type}")

    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        raise ValueError("mediapipe_replay_path is required for mediapipe_replay mode")
    if input_device_type == "video" and not video_path:
        raise ValueError("video_path is required for video mode")
    if input_device_type == "video" and VideoMediaPipe is None:
        raise ImportError("video mode requires mediapipe and opencv-python")
    if input_device_type == "realsense" and RealsenseMediaPipe is None:
        raise ImportError("realsense mode requires mediapipe, opencv-python, and pyrealsense2")
    if input_device_type == "zed" and ZedMediaPipe is None:
        raise ImportError("zed mode requires mediapipe, opencv-python, and pyzed")
    if input_device_type == "webcam" and WebcamMediaPipe is None:
        raise ImportError("webcam mode requires mediapipe and opencv-python")
    if input_device_type == "webcam_wilor" and WebcamWiLoR is None:
        raise ImportError("webcam_wilor mode requires: uv sync --extra wilor")

    input_device = device_map[input_device_type]()

    debug_overlay = DebugOverlay(model, hand_side) if debug else None
    last_debug_print = 0.0

    # qpos is in URDF/Pinocchio joint order; data.ctrl is in actuator order.
    # Reorder by semantic joint name before assigning actuator values.
    _act_joint_order = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        for i in range(model.nu)
    ]
    _qpos_perm = qpos_reorder_perm(
        retargeter.optimizer.robot.dof_joint_names, _act_joint_order
    )
    # A configured MJCF whose joints cannot be aligned would silently drive the
    # wrong joints, so reject it rather than falling back to identity.
    if mjcf_override and _qpos_perm is None:
        raise ValueError(
            "config declares optimizer.mjcf_path but the URDF<->MJCF joint names "
            "could not be aligned; the sim would drive the wrong joints. Check "
            "optimizer.link_naming / that urdf_path and mjcf_path describe the same hand."
        )

    # Disable recording when using replay mode
    if input_device_type == "mediapipe_replay" and enable_recording:
        print("Note: Recording disabled in replay mode")
        enable_recording = False

    # Prepare recording
    input_data_log = [] if enable_recording else None
    start_time = time.time()

    try:
        print(f"Starting teleoperation...")
        print(f"  Config: {config_path}")
        print(f"  Robot: {robot}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
        if debug:
            print("  Debug: actual=yellow/cyan, targets=green")
        print("=" * 50)

        frame_count = 0
        fps_start_time = time.time()
        control_tick = 0
        next_control_tick = time.monotonic()

        while viewer.is_running():
            # Get finger data
            fingers_data = input_device.get_fingers_data()
            fingers_pose = fingers_data[f"{hand_side}_fingers"]  # (21, 3)

            # Skip until the first valid frame arrives from the input device.
            if fingers_pose is None or np.allclose(fingers_pose, 0):
                time.sleep(0.01)
                continue

            # Record raw input data if enabled
            if enable_recording:
                input_data_log.append({
                    "t": time.time() - start_time,
                    "left_fingers": (
                        None
                        if fingers_data["left_fingers"] is None
                        else fingers_data["left_fingers"].copy()
                    ),
                    "right_fingers": (
                        None
                        if fingers_data["right_fingers"] is None
                        else fingers_data["right_fingers"].copy()
                    ),
                })

            if debug:
                qpos, debug_frame = retargeter.retarget_verbose(fingers_pose)
            else:
                qpos = retargeter.retarget(fingers_pose)
                debug_frame = None

            # FPS counter
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - fps_start_time
                fps = frame_count / elapsed
                print(f"Control FPS: {fps:.1f} / target {CONTROL_HZ}")

            # Set control signals (remap URDF qpos order -> actuator order).
            if openarm_control is not None:
                actuator_targets = openarm_control.targets(
                    retargeter.optimizer.robot.dof_joint_names, qpos
                )
            elif _qpos_perm is not None:
                actuator_targets = qpos[_qpos_perm]
            else:
                actuator_targets = qpos
            if len(actuator_targets) == model.nu:
                data.ctrl[:] = actuator_targets
            else:
                min_len = min(len(actuator_targets), model.nu)
                data.ctrl[:min_len] = actuator_targets[:min_len]

            # Step physical simulation, then draw its actual pose and current
            # adaptive target vectors.
            physics_steps = physics_steps_for_tick(
                control_tick, model.opt.timestep, CONTROL_HZ
            )
            with viewer.lock():
                for _ in range(physics_steps):
                    mujoco.mj_step(model, data)
                if debug_overlay is not None:
                    overlay_keypoints = debug_frame.get(
                        "mediapipe_kp", debug_frame.get("joints_task_m")
                    )
                    overlay_alpha = np.asarray(
                        debug_frame.get("pinch_alphas", np.zeros(5)), dtype=np.float64
                    )
                    if overlay_alpha.shape == (4,):
                        overlay_alpha = np.r_[overlay_alpha.max(initial=0.0), overlay_alpha]
                    debug_overlay.draw(
                        viewer.user_scn,
                        data,
                        retargeter.optimizer,
                        overlay_keypoints,
                        overlay_alpha,
                    )
            viewer.sync()

            if debug and time.monotonic() - last_debug_print >= 1.0:
                pinch_alphas = np.asarray(debug_frame.get("pinch_alphas", np.zeros(5)))
                if pinch_alphas.shape == (4,):
                    pinch_alphas = np.r_[pinch_alphas.max(initial=0.0), pinch_alphas]
                modes = [mode_label(alpha) for alpha in pinch_alphas]
                print(
                    format_joint_diagnostics(
                        model,
                        data,
                        actuator_targets,
                        modes,
                        debug_frame["cost"],
                    )
                )
                last_debug_print = time.monotonic()

            control_tick += 1
            next_control_tick += 1.0 / CONTROL_HZ
            remaining = next_control_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_control_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        viewer.close()
        for method_name in ("stop", "cleanup", "close"):
            method = getattr(input_device, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    return input_data_log


def main():
    parser = argparse.ArgumentParser(
        description='Teleoperation with MuJoCo Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay MediaPipe recording
  mjpython teleop_sim.py --play data/avp1.pkl --hand left

  # MP4 video input with MediaPipe hand detection
  mjpython teleop_sim.py --video data/right.mp4 --hand right
  mjpython teleop_sim.py --video data/right.mp4 --hand right --show-video

  # Live USB webcam input with MediaPipe hand detection
  python teleop_sim.py --webcam --camera-index 0 --hand right --show-video

  # OpenArm bimanual model: selected arm remains raised; only its hand retargets
  python teleop_sim.py --webcam --camera-index 0 --hand right --robot openarm

  # RealSense camera input with MediaPipe hand detection
  mjpython teleop_sim.py --realsense --hand right

  # ZED camera input with MediaPipe hand detection
  mjpython teleop_sim.py --zed --hand right

  # Live VisionPro input
  mjpython teleop_sim.py --input visionpro --ip <your-vision-pro-ip>

  # Record input data while using VisionPro
  mjpython teleop_sim.py --input visionpro --record

        """
    )

    # Config
    parser.add_argument('--config', type=str, default='config/adaptive_analytical_avp.yaml',
                        help='Path to YAML configuration file (default: config/adaptive_analytical_avp.yaml)')
    parser.add_argument('--hand', type=str, default='left', choices=['left', 'right'],
                        help='Hand side (default: left)')
    parser.add_argument('--robot', type=str, default='ldjy', choices=['ldjy', 'openarm'],
                        help='Simulation robot: standalone ldjy hand or bimanual openarm (default: ldjy)')

    # Input device options
    parser.add_argument('--input', type=str, default=None,
                        choices=['visionpro', 'mediapipe_replay', 'video', 'webcam', 'webcam_wilor', 'realsense', 'zed'],
                        help='Input device type')

    # Shortcut options
    parser.add_argument('--play', type=str, default=None, metavar='FILE',
                        help='Play MediaPipe recording file (shortcut for --input mediapipe_replay)')

    # VisionPro options
    parser.add_argument('--ip', type=str, default='192.168.50.127',
                        help='VisionPro IP address (default: 192.168.50.127)')

    # Playback options
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed for replay mode (default: 1.0)')
    parser.add_argument('--no-loop', action='store_true',
                        help='Disable looping for replay mode')

    # Recording
    parser.add_argument('--record', action='store_true',
                        help='Record input data to file')
    parser.add_argument('--output', type=str, default=None, metavar='FILE',
                        help='Output file for recording (default: auto-generated)')
    parser.add_argument('--video', type=str, default=None, metavar='FILE',
                        help='Play MP4 video file with MediaPipe hand detection (shortcut for --input video)')
    parser.add_argument('--webcam', action='store_true',
                        help='Use a live USB webcam with MediaPipe hand detection')
    parser.add_argument('--camera-index', type=int, default=0,
                        help='OpenCV camera index for --webcam (default: 0)')
    parser.add_argument('--realsense', action='store_true',
                        help='Use RealSense camera with MediaPipe hand detection (shortcut for --input realsense)')
    parser.add_argument('--zed', action='store_true',
                        help='Use ZED camera with MediaPipe hand detection (shortcut for --input zed)')
    parser.add_argument('--show-video', action='store_true',
                        help='Show input detection overlay (video/webcam/webcam_wilor/realsense/zed mode)')
    parser.add_argument('--debug', action='store_true',
                        help='Show actual-pose and target-vector overlays plus joint diagnostics')

    args = parser.parse_args()

    # Determine input device type and paths
    input_device_type = args.input
    mediapipe_replay_path = ""
    video_path = ""

    if args.webcam:
        input_device_type = "webcam"
    elif args.zed:
        input_device_type = "zed"
    elif args.realsense:
        input_device_type = "realsense"
    elif args.video:
        input_device_type = "video"
        video_path = args.video
    elif args.play:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = args.play

    # Default to mediapipe_replay with example data if no input specified
    if input_device_type is None:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = "data/avp1.pkl"

    # Auto-switch config for non-AVP input devices
    if args.config == 'config/adaptive_analytical_avp.yaml':
        if input_device_type in ("realsense", "video", "webcam", "webcam_wilor", "zed"):
            args.config = 'config/adaptive_analytical_video.yaml'

    # Validate paths
    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        parser.error("--play FILE is required for mediapipe_replay mode")

    # Run teleoperation
    log = run_teleop(
        hand_side=args.hand,
        config_path=args.config,
        input_device_type=input_device_type,
        mediapipe_replay_path=mediapipe_replay_path,
        visionpro_ip=args.ip,
        playback_speed=args.speed,
        playback_loop=not args.no_loop,
        enable_recording=args.record,
        video_path=video_path,
        show_video=args.show_video,
        camera_index=args.camera_index,
        debug=args.debug,
        robot=args.robot,
    )

    # Save recording if enabled
    if log is not None and len(log) > 0:
        if args.output:
            log_path = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = Path(__file__).parent / f"input_data_log_{timestamp}.pkl"

        with open(log_path, "wb") as f:
            pickle.dump(log, f)
        print(f"Saved input data log with {len(log)} entries to {log_path}")


if __name__ == "__main__":
    main()
