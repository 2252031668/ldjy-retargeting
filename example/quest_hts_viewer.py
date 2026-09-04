"""Quest HTS 3D hand skeleton viewer in MuJoCo.

Displays 21 MediaPipe-style landmarks received from a Meta Quest running the
Hand Tracking Streamer (HTS) app. The wrist pose is used to center the hand
in the viewer; the global wrist position is not displayed.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from hand_tracking_sdk import HTSClient, HTSClientConfig
    from hand_tracking_sdk.client import (
        ErrorPolicy,
        HandFilter,
        StreamOutput,
        TransportMode,
    )
    from hand_tracking_sdk.convert import convert_hand_frame_unity_left_to_right
    from hand_tracking_sdk.frame import HandFrame
    from hand_tracking_sdk.models import HandSide
except ImportError as exc:
    raise ImportError(
        "hand-tracking-sdk is required. Install with: uv sync --extra quest"
    ) from exc


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

# Per-finger colors (thumb, index, middle, ring, pinky)
FINGER_POINT_RGBA = np.array((
    (1.0, 0.65, 0.0, 1.0),   # thumb: orange
    (0.15, 1.0, 0.25, 1.0),  # index: green
    (0.25, 0.55, 1.0, 1.0),  # middle: blue
    (1.0, 0.25, 0.65, 1.0),  # ring: pink
    (0.75, 0.25, 1.0, 1.0),  # pinky: purple
), dtype=np.float64)

FINGER_LINK_RGBA = np.array((
    (1.0, 0.65, 0.0, 0.55),
    (0.15, 1.0, 0.25, 0.55),
    (0.25, 0.55, 1.0, 0.55),
    (1.0, 0.25, 0.65, 0.55),
    (0.75, 0.25, 1.0, 0.55),
), dtype=np.float64)

WRIST_AXIS_LENGTH_M = 0.04
WRIST_AXIS_RADIUS_M = 0.0012
AXIS_RGBA = np.array((
    (1.0, 0.2, 0.2, 1.0),  # X
    (0.2, 1.0, 0.2, 1.0),  # Y
    (0.2, 0.4, 1.0, 1.0),  # Z
), dtype=np.float64)

POINT_RADIUS_M = 0.003
LINK_RADIUS_M = 0.0012
IDENTITY_MAT = np.eye(3, dtype=np.float64).ravel()

_USAGE_EPILOG = """\
Examples:
    uv run --no-sync --extra quest --extra tuning python example/quest_hts_viewer.py
    uv run --no-sync --extra quest --extra tuning python example/quest_hts_viewer.py \\
        --transport tcp_server --port 9000 --hand right
    uv run --no-sync --extra quest --extra tuning python example/quest_hts_viewer.py \\
        --transport tcp_client --host 192.168.1.100 --port 8000 --hand left
"""

_EMPTY_MJCF = """<mujoco model="quest_hts_skeleton"><worldbody/></mujoco>"""

FINGER_INDEX_FOR_LANDMARK = (
    0,  # wrist
    0,  # thumb CMC
    0, 0, 0,  # thumb MCP/IP/TIP
    1, 1, 1, 1, 1,  # index
    2, 2, 2, 2, 2,  # middle
    3, 3, 3, 3, 3,  # ring
    4, 4, 4, 4, 4,  # pinky
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_USAGE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--transport",
        choices=("udp", "tcp_server", "tcp_client"),
        default="udp",
        help="network transport used by HTSClient; must match the setting in the Quest app",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="bind/connect host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="bind/connect port",
    )
    parser.add_argument(
        "--hand",
        choices=("left", "right", "both"),
        default="both",
        help="filter and display which hand(s) are received from Quest",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="skip Unity-left to RFU-right coordinate conversion",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="I/O timeout in seconds",
    )
    parser.add_argument(
        "--error-policy",
        choices=("strict", "tolerant"),
        default="tolerant",
        help="parser error policy",
    )
    return parser.parse_args(args)


def transport_mode_from_arg(value: str) -> TransportMode:
    return {
        "udp": TransportMode.UDP,
        "tcp_server": TransportMode.TCP_SERVER,
        "tcp_client": TransportMode.TCP_CLIENT,
    }[value]


def hand_filter_from_arg(value: str) -> HandFilter:
    return {
        "left": HandFilter.LEFT,
        "right": HandFilter.RIGHT,
        "both": HandFilter.BOTH,
    }[value]


def error_policy_from_arg(value: str) -> ErrorPolicy:
    return {
        "strict": ErrorPolicy.STRICT,
        "tolerant": ErrorPolicy.TOLERANT,
    }[value]


class HTSReceiver(threading.Thread):
    """Background thread that owns the HTSClient and stores the latest frame."""

    def __init__(self, config: HTSClientConfig) -> None:
        super().__init__(name="quest-hts-receiver", daemon=True)
        self._config = config
        self._lock = threading.Lock()
        self._latest: dict[HandSide, HandFrame] = {}
        self._running = threading.Event()
        self._running.set()
        self._stats = {"frames": 0, "errors": 0, "start_time": time.monotonic()}

    def get_latest(self) -> dict[HandSide, HandFrame]:
        with self._lock:
            return {side: frame for side, frame in self._latest.items()}

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            elapsed = time.monotonic() - self._stats["start_time"]
            return {
                "frames": self._stats["frames"],
                "errors": self._stats["errors"],
                "elapsed_sec": elapsed,
                "fps": self._stats["frames"] / elapsed if elapsed > 0 else 0.0,
            }

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        client = HTSClient(self._config)
        try:
            for event in client.iter_events():
                if not self._running.is_set():
                    break
                if not isinstance(event, HandFrame):
                    continue
                with self._lock:
                    self._latest[event.side] = event
                    self._stats["frames"] += 1
        except Exception:  # noqa: BLE001
            with self._lock:
                self._stats["errors"] += 1
            raise


def wrist_rotation_matrix(wrist) -> np.ndarray:
    """Convert wrist quaternion to a 3x3 rotation matrix."""
    q = np.array([wrist.qw, wrist.qx, wrist.qy, wrist.qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q / norm
    w, x, y, z = q
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    ), dtype=np.float64)


def _add_sphere(scene, position: np.ndarray, rgba: np.ndarray, radius: float, label: str = "") -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((radius, 0.0, 0.0), dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        IDENTITY_MAT,
        rgba,
    )
    geom.label = label
    scene.ngeom += 1


def _add_capsule(scene, start: np.ndarray, end: np.ndarray, rgba: np.ndarray, radius: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    diff = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = np.linalg.norm(diff)
    if length < 1e-12:
        return
    center = (np.asarray(start, dtype=np.float64) + np.asarray(end, dtype=np.float64)) * 0.5
    z_axis = diff / length
    # Build a rotation matrix whose z-axis aligns with the segment.
    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(z_axis, tmp)) > 0.99:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = np.cross(tmp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack((x_axis, y_axis, z_axis)).ravel()
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array((radius, length / 2.0, 0.0), dtype=np.float64),
        center,
        rot_mat,
        rgba,
    )
    scene.ngeom += 1


def _add_axis(scene, origin: np.ndarray, rotation: np.ndarray, length: float, radius: float) -> None:
    for i in range(3):
        if scene.ngeom >= scene.maxgeom:
            return
        direction = rotation[:, i]
        end = origin + direction * length
        _add_capsule(scene, origin, end, AXIS_RGBA[i], radius)


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert HandLandmarks.points to a (21, 3) numpy array."""
    return np.asarray(landmarks.points, dtype=np.float64)


def draw_hand(
    scene,
    frame: HandFrame,
    convert: bool,
    lateral_offset_m: float = 0.0,
) -> None:
    """Draw one HandFrame as a 3D skeleton centered at the wrist."""
    if convert:
        frame = convert_hand_frame_unity_left_to_right(frame)

    points = landmarks_to_array(frame.landmarks)
    wrist_pos = np.array([frame.wrist.x, frame.wrist.y, frame.wrist.z], dtype=np.float64)
    lateral = np.array([lateral_offset_m, 0.0, 0.0], dtype=np.float64)

    # Center the hand on the wrist so the viewer stays focused on hand pose.
    # A lateral offset keeps two-hand mode from overlapping in the view.
    centered = points - wrist_pos + lateral

    # Draw connection lines first so points sit on top visually.
    for start_idx, end_idx in HAND_CONNECTIONS:
        finger = FINGER_INDEX_FOR_LANDMARK[start_idx]
        _add_capsule(
            scene,
            centered[start_idx],
            centered[end_idx],
            FINGER_LINK_RGBA[finger],
            LINK_RADIUS_M,
        )

    # Draw joint spheres.
    for idx, pos in enumerate(centered):
        finger = FINGER_INDEX_FOR_LANDMARK[idx]
        label = ""
        if idx == 0:
            label = f"{frame.side.value} wrist"
        _add_sphere(scene, pos, FINGER_POINT_RGBA[finger], POINT_RADIUS_M, label=label)

    # Draw wrist coordinate axes using the wrist rotation.
    rot = wrist_rotation_matrix(frame.wrist)
    _add_axis(scene, centered[0], rot, WRIST_AXIS_LENGTH_M, WRIST_AXIS_RADIUS_M)


def draw_status_label(scene, text: str) -> None:
    """Draw a small informative label in the scene while waiting."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    # Place a tiny invisible sphere and rely on the label tooltip.
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((0.001, 0.0, 0.0), dtype=np.float64),
        np.array((0.0, 0.0, 0.05), dtype=np.float64),
        IDENTITY_MAT,
        np.array((1.0, 1.0, 1.0, 0.0), dtype=np.float64),
    )
    geom.label = text
    scene.ngeom += 1


def main() -> None:
    args = parse_args()

    config = HTSClientConfig(
        transport_mode=transport_mode_from_arg(args.transport),
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
        output=StreamOutput.FRAMES,
        hand_filter=hand_filter_from_arg(args.hand),
        error_policy=error_policy_from_arg(args.error_policy),
    )

    receiver = HTSReceiver(config)
    receiver.start()
    print(
        f"Quest HTS viewer started: transport={args.transport}, "
        f"host={args.host}, port={args.port}, hand={args.hand}, "
        f"convert={not args.no_convert}"
    )
    print("Waiting for first frame from Quest...")

    model = mujoco.MjModel.from_xml_string(_EMPTY_MJCF)
    data = mujoco.MjData(model)

    last_print_time = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.35
        viewer.cam.lookat[:] = [0, 0, 0.05]

        while viewer.is_running():
            step_start = time.time()

            latest = receiver.get_latest()
            viewer.user_scn.ngeom = 0

            displayed = 0
            offsets = {
                HandSide.LEFT: -0.12,
                HandSide.RIGHT: 0.12,
            } if args.hand == "both" else {HandSide.LEFT: 0.0, HandSide.RIGHT: 0.0}
            for side in (HandSide.LEFT, HandSide.RIGHT):
                frame = latest.get(side)
                if frame is None:
                    continue
                if args.hand in ("left", "right") and side.value.lower() != args.hand:
                    continue
                draw_hand(
                    viewer.user_scn,
                    frame,
                    convert=not args.no_convert,
                    lateral_offset_m=offsets[side],
                )
                displayed += 1

            if displayed == 0:
                draw_status_label(viewer.user_scn, "waiting for Quest HTS data...")

            viewer.sync()

            now = time.monotonic()
            if now - last_print_time >= 2.0:
                stats = receiver.get_stats()
                if stats["frames"]:
                    print(
                        f"[{stats['elapsed_sec']:.1f}s] frames={stats['frames']} "
                        f"fps={stats['fps']:.1f} errors={stats['errors']}"
                    )
                else:
                    print("still waiting for first frame...")
                last_print_time = now

            elapsed = time.time() - step_start
            time.sleep(max(0.0, 0.02 - elapsed))

    receiver.stop()


if __name__ == "__main__":
    main()
