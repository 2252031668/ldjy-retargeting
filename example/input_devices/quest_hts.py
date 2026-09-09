"""Live Quest HTS input using Hand Tracking Streamer network protocol."""

from __future__ import annotations

import threading
import time
import socket
from typing import Dict

import numpy as np

from .base import InferenceSample, InputDeviceBase


_UNITY_RIGHT_TO_RFU = np.array(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32
)


class _OpenedReceiver:
    """Keep an SDK receiver open across ``HTSClient.iter_events()`` calls."""

    def __init__(self, receiver) -> None:
        self._receiver = receiver

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def iter_lines(self):
        return self._receiver.iter_lines()


def unity_left_landmarks_to_rfu(landmarks_unity_left: np.ndarray) -> np.ndarray:
    """Map wrist-local Unity-left landmarks into the retargeter's RFU frame."""
    points = np.asarray(landmarks_unity_left, dtype=np.float32)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Quest HTS landmarks must have finite shape (21, 3)")
    # Unity-left -> Unity-right reflects Y; Unity-right -> RFU maps (x, y, z)
    # to (x, z, -y), producing (x, z, y) directly from Unity-left.
    return (points * np.array((1.0, -1.0, 1.0), dtype=np.float32)) @ _UNITY_RIGHT_TO_RFU.T


class QuestHTS(InputDeviceBase):
    """Return Quest HTS hand landmarks from Meta Quest headset."""

    def __init__(
        self,
        hand_side: str = "right",
        transport: str = "udp",
        host: str = "0.0.0.0",
        port: int = 9000,
        timeout_s: float = 1.0,
    ) -> None:
        """Initialize Quest HTS input device.

        Args:
            hand_side: 'left' or 'right'
            transport: 'udp', 'tcp_server', or 'tcp_client'
            host: bind/connect host address
            port: bind/connect port
            timeout_s: socket timeout in seconds
        """
        self.hand_side = hand_side.lower()
        if self.hand_side not in {"left", "right"}:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")

        self.transport = transport
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

        self._empty = np.zeros((21, 3), dtype=np.float32)
        self._last_valid_keypoints: Dict[str, np.ndarray] = {
            "left": self._empty.copy(),
            "right": self._empty.copy(),
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()

        self._latest_result = {
            "left_fingers": self._empty.copy(),
            "right_fingers": self._empty.copy(),
        }

        self._frame_sequence = 0
        self._fps_started_at = time.monotonic()
        self._frame_count = 0
        self._fps = 0.0
        self._connected = False
        self._last_frame_time = 0.0

        # Import and initialize HTSClient
        try:
            from hand_tracking_sdk import HTSClient, HTSClientConfig
            from hand_tracking_sdk.client import (
                ErrorPolicy,
                HandFilter,
                StreamOutput,
                TransportMode,
            )
            from hand_tracking_sdk.frame import HandFrame
            from hand_tracking_sdk.models import HandSide

            self._HandFrame = HandFrame
            self._HandSide = HandSide
        except ImportError as error:
            raise ImportError(
                "Quest HTS input requires hand-tracking-sdk. "
                "Install with: uv sync --extra quest"
            ) from error

        # Map transport string to enum
        transport_map = {
            "udp": TransportMode.UDP,
            "tcp_server": TransportMode.TCP_SERVER,
            "tcp_client": TransportMode.TCP_CLIENT,
        }
        if transport not in transport_map:
            raise ValueError(
                f"transport must be one of {list(transport_map.keys())}, got {transport!r}"
            )

        self._validate_connection_target()

        # Create HTS client config
        config = HTSClientConfig(
            transport_mode=transport_map[transport],
            host=host,
            port=port,
            timeout_s=timeout_s,
            output=StreamOutput.FRAMES,
            hand_filter=HandFilter.BOTH,
            error_policy=ErrorPolicy.TOLERANT,
        )

        try:
            # Open the actual SDK receiver here so bind/connect failures reach
            # the GUI before it tears down the current input. The facade keeps
            # HTSClient from closing that receiver until stop() does so.
            self._receiver = HTSClient(config)._make_receiver()
            self._receiver.open()
            self._client = HTSClient(
                config, receiver_factory=lambda _: _OpenedReceiver(self._receiver)
            )
        except Exception as error:
            if hasattr(self, "_receiver"):
                self._receiver.close()
            raise RuntimeError(
                f"Failed to connect to Quest HTS at {host}:{port} "
                f"via {transport}. Error: {error}"
            ) from error

        # Start receiver thread
        self._worker_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._worker_thread.start()

    def _validate_connection_target(self) -> None:
        """Surface bad binds/connects before the GUI replaces its active input."""
        socket_type = socket.SOCK_DGRAM if self.transport == "udp" else socket.SOCK_STREAM
        probe = socket.socket(socket.AF_INET, socket_type)
        try:
            if self.transport == "tcp_client":
                probe.settimeout(self.timeout_s)
                probe.connect((self.host, self.port))
            else:
                probe.bind((self.host, self.port))
                if self.transport == "tcp_server":
                    probe.listen()
        except OSError as error:
            raise RuntimeError(
                f"Failed to connect to Quest HTS at {self.host}:{self.port} via {self.transport}: {error}"
            ) from error
        finally:
            probe.close()

    def _receive_loop(self) -> None:
        """Background thread that receives HTS frames."""
        try:
            for event in self._client.iter_events():
                if self._stop_event.is_set():
                    break

                if not self._resume_event.is_set():
                    time.sleep(0.01)
                    continue

                if not isinstance(event, self._HandFrame):
                    continue

                landmarks_unity_left = np.asarray(event.landmarks.points, dtype=np.float32)
                if landmarks_unity_left.shape != (21, 3) or not np.isfinite(landmarks_unity_left).all():
                    continue
                landmarks_rfu = unity_left_landmarks_to_rfu(landmarks_unity_left)

                # Determine hand side
                side_str = "left" if event.side == self._HandSide.LEFT else "right"

                # Update result
                with self._lock:
                    self._connected = True
                    self._latest_result[f"{side_str}_fingers"] = landmarks_rfu.astype(np.float32)
                    self._last_valid_keypoints[side_str] = landmarks_rfu.astype(np.float32)
                    self._frame_count += 1
                    self._frame_sequence += 1
                    self._last_frame_time = time.monotonic()

                    # Update FPS
                    elapsed = self._last_frame_time - self._fps_started_at
                    if elapsed > 0:
                        self._fps = self._frame_count / elapsed

                    # Publish for recording
                    if side_str == self.hand_side:
                        self.publish_inference_sample(self._record_sample(
                            side_str, landmarks_unity_left, landmarks_rfu,
                            np.array([event.wrist.x, event.wrist.y, event.wrist.z]),
                            np.array([event.wrist.qx, event.wrist.qy, event.wrist.qz, event.wrist.qw]),
                            self._last_frame_time,
                        ))

        except Exception:
            with self._lock:
                self._connected = False

    @staticmethod
    def _record_sample(
        hand_side: str,
        landmarks_unity_left: np.ndarray,
        landmarks_rfu: np.ndarray,
        wrist_position_unity_left: np.ndarray,
        wrist_quaternion_unity_left: np.ndarray,
        timestamp_sec: float,
    ) -> InferenceSample:
        return InferenceSample(
            timestamp_sec=float(timestamp_sec), frame_bgr=None, input_type="quest_hts",
            hand_side=hand_side, detected=True,
            payload={
                "landmarks_unity_left": np.asarray(landmarks_unity_left, dtype=np.float32).copy(),
                "landmarks_rfu": np.asarray(landmarks_rfu, dtype=np.float32).copy(),
                "wrist_position_unity_left": np.asarray(wrist_position_unity_left, dtype=np.float32).copy(),
                "wrist_quaternion_unity_left": np.asarray(wrist_quaternion_unity_left, dtype=np.float32).copy(),
            },
        )

    def get_fingers_data(self) -> Dict[str, np.ndarray]:
        """Return the latest landmarks for both hands."""
        with self._lock:
            # Check if connection is stale (no data for 2 seconds)
            if self._last_frame_time > 0:
                elapsed = time.monotonic() - self._last_frame_time
                if elapsed > 2.0:
                    self._connected = False

            return {
                "left_fingers": self._latest_result["left_fingers"].copy(),
                "right_fingers": self._latest_result["right_fingers"].copy(),
            }

    def get_preview_frame(self) -> np.ndarray | None:
        """Quest HTS does not provide preview frames."""
        return None

    def set_paused(self, paused: bool) -> None:
        """Pause or resume frame reception."""
        if paused:
            self._resume_event.clear()
        else:
            self._resume_event.set()

    def stop(self) -> None:
        """Stop the receiver thread."""
        self._stop_event.set()
        if self.transport == "tcp_server":
            try:
                with socket.create_connection((self.host, self.port), timeout=0.2):
                    pass
            except OSError:
                pass
        if hasattr(self, "_receiver"):
            self._receiver.close()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    cleanup = stop

    @property
    def fps(self) -> float:
        """Return current reception FPS."""
        with self._lock:
            return self._fps

    @property
    def connected(self) -> bool:
        """Return connection status."""
        with self._lock:
            return self._connected

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if hasattr(self, "_stop_event"):
            self.stop()
