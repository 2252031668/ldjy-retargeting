"""Live USB webcam input using WiLoR fast MANO reconstruction."""

from __future__ import annotations

import threading
import time
from copy import deepcopy

import cv2
import numpy as np

from .base import InferenceSample, InputDeviceBase


class WebcamWiLoR(InputDeviceBase):
    """Return the newest requested-side WiLoR MANO joints from a USB webcam."""

    def __init__(
        self,
        hand_side: str = "right",
        camera_index: int = 0,
        show_video: bool = False,
    ) -> None:
        self.hand_side = hand_side.lower()
        if self.hand_side not in {"left", "right"}:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")
        self.camera_index = camera_index
        self.show_video = show_video
        self._empty = np.zeros((21, 3), dtype=np.float32)
        self._last_valid_keypoints: np.ndarray | None = None
        self._last_valid_mano = None
        self._last_selected_detection = None
        self._selected_detection_this_inference = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._latest_result = {
            "left_fingers": self._empty.copy(),
            "right_fingers": self._empty.copy(),
        }
        self._latest_preview_frame: np.ndarray | None = None
        self._latest_raw_frame: np.ndarray | None = None
        self._frame_sequence = 0
        self._fps_started_at = time.monotonic()
        self._inference_count = 0
        self._inference_fps = 0.0

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Failed to open webcam at camera index {camera_index}")
        try:
            self.runner = self._create_runner()
            self._mano_faces = np.asarray(self.runner.faces, dtype=np.int32).copy()
        except Exception:
            self.cap.release()
            raise
        self._worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_thread.start()

    @staticmethod
    def _create_runner():
        try:
            from ldjy_retargeting.wilor_runtime import WiLoRRunner, validate_wilor_assets
            return WiLoRRunner(
                assets=validate_wilor_assets(),
                device_name="cuda",
                batch_size=16,
                confidence=0.3,
                fast=True,
            )
        except ImportError as error:
            raise ImportError(
                "webcam_wilor requires the WiLoR extra; run: uv sync --extra wilor"
            ) from error

    def _select_keypoints(self, detections) -> np.ndarray | None:
        self._selected_detection_this_inference = None
        requested_is_right = self.hand_side == "right"
        candidates = [
            detection for detection in detections
            if detection.is_right == requested_is_right
            and np.asarray(detection.joints_mano).shape == (21, 3)
            and np.isfinite(detection.joints_mano).all()
        ]
        if candidates:
            selected = max(
                candidates,
                key=lambda detection: abs(
                    (detection.bbox_xyxy[2] - detection.bbox_xyxy[0])
                    * (detection.bbox_xyxy[3] - detection.bbox_xyxy[1])
                ),
            )
            self._last_valid_keypoints = np.asarray(
                selected.joints_mano, dtype=np.float32
            ).copy()
            self._last_selected_detection = selected
            self._selected_detection_this_inference = selected
            if (
                hasattr(selected, "vertices_mano")
                and hasattr(selected, "camera_translation")
                and np.asarray(selected.vertices_mano).shape == (778, 3)
                and np.asarray(selected.camera_translation).shape == (3,)
                and np.isfinite(selected.vertices_mano).all()
                and np.isfinite(selected.camera_translation).all()
            ):
                self._last_valid_mano = selected
        if self._last_valid_keypoints is None:
            return None
        return self._last_valid_keypoints.copy()

    def _record_sample_from_detection(
        self,
        frame: np.ndarray,
        detection,
        keypoints: np.ndarray | None,
        timestamp_sec: float,
    ) -> InferenceSample:
        """Build an immutable single-hand record sample after WiLoR inference."""
        detected = detection is not None and keypoints is not None
        return InferenceSample(
            timestamp_sec=float(timestamp_sec),
            frame_bgr=np.asarray(frame, dtype=np.uint8).copy(),
            input_type="webcam_wilor",
            hand_side=self.hand_side,
            detected=detected,
            payload={
                "detection": deepcopy(detection) if detected else None,
                "joints_mano": np.asarray(keypoints).copy() if detected else None,
            },
        )

    def get_fingers_data(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {
                "left_fingers": self._latest_result["left_fingers"].copy(),
                "right_fingers": self._latest_result["right_fingers"].copy(),
            }

    def set_paused(self, paused: bool) -> None:
        """Stop new CUDA inference, waiting only for an in-flight frame to finish."""
        if paused:
            self._resume_event.clear()
            with self._inference_lock:
                pass
        else:
            self._resume_event.set()

    def get_preview_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_preview_frame is None:
                return None
            return self._latest_preview_frame.copy()

    def get_raw_preview_frame(self) -> np.ndarray | None:
        """Return the unannotated camera frame for the tuning GUI."""
        with self._lock:
            if self._latest_raw_frame is None:
                return None
            return self._latest_raw_frame.copy()

    def get_mano_overlay_data(self):
        """Return the newest selected MANO mesh, preserving it across brief loss."""
        with self._lock:
            detection = self._last_valid_mano
            faces = getattr(self, "_mano_faces", None)
            frame = self._latest_raw_frame
            if detection is None or faces is None or frame is None:
                return None
            return {
                "frame_bgr": frame.copy(),
                "sequence": self._frame_sequence,
                "vertices_mano": np.asarray(detection.vertices_mano, dtype=np.float32).copy(),
                "joints_mano": np.asarray(detection.joints_mano, dtype=np.float32).copy(),
                "camera_translation": np.asarray(
                    detection.camera_translation, dtype=np.float32
                ).copy(),
                "faces": faces.copy(),
                "is_right": bool(detection.is_right),
            }

    def _annotate_frame(self, frame: np.ndarray, detections) -> np.ndarray:
        display = frame.copy()
        requested_is_right = self.hand_side == "right"
        for detection in detections:
            x0, y0, x1, y1 = np.rint(detection.bbox_xyxy).astype(int)
            selected = detection.is_right == requested_is_right
            color = (0, 220, 0) if selected else (120, 120, 120)
            cv2.rectangle(display, (x0, y0), (x1, y1), color, 2)
            label = "right" if detection.is_right else "left"
            cv2.putText(
                display, label, (x0, max(18, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
        cv2.putText(
            display, f"WiLoR fast {self._inference_fps:.1f} FPS",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            display, f"Target: {self.hand_side}",
            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA,
        )
        return display

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            self._resume_event.wait()
            if self._stop_event.is_set():
                break
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._inference_lock:
                if self._stop_event.is_set() or not self._resume_event.is_set():
                    continue
                detections = self.runner.infer(frame)
            keypoints = self._select_keypoints(detections)
            selected_detection = self._selected_detection_this_inference
            self.publish_inference_sample(self._record_sample_from_detection(
                frame, selected_detection, keypoints, time.monotonic()
            ))
            self._inference_count += 1
            elapsed = time.monotonic() - self._fps_started_at
            if elapsed > 0.0:
                self._inference_fps = self._inference_count / elapsed
            preview = self._annotate_frame(frame, detections)
            if self.show_video:
                cv2.imshow("Webcam WiLoR", preview)
                cv2.waitKey(1)

            latest = {
                "left_fingers": self._empty.copy(),
                "right_fingers": self._empty.copy(),
            }
            if keypoints is not None:
                latest[f"{self.hand_side}_fingers"] = keypoints
            with self._lock:
                self._latest_result = latest
                self._latest_preview_frame = preview
                self._latest_raw_frame = frame.copy()
                self._frame_sequence += 1

    def cleanup(self) -> None:
        if not hasattr(self, "_stop_event"):
            return
        self._stop_event.set()
        self._resume_event.set()
        worker = getattr(self, "_worker_thread", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        capture = getattr(self, "cap", None)
        if capture is not None:
            capture.release()
        if getattr(self, "show_video", False):
            cv2.destroyWindow("Webcam WiLoR")

    def __del__(self):
        self.cleanup()
