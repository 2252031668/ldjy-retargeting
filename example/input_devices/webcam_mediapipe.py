"""Live USB webcam input using MediaPipe hand landmarks."""

import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from .base import InferenceSample
from .video_mediapipe import VideoMediaPipe


class WebcamMediaPipe(VideoMediaPipe):
    """Read a live OpenCV camera and return one hand as MediaPipe keypoints."""

    def __init__(
        self,
        hand_side: str = "right",
        camera_index: int = 0,
        video_config: dict | None = None,
        show_video: bool = False,
    ):
        self.hand_side = hand_side.lower()
        self.show_video = show_video
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._worker_thread = None
        self._empty = np.zeros((21, 3), dtype=np.float32)
        self._latest_result = {
            "left_fingers": self._empty.copy(),
            "right_fingers": self._empty.copy(),
        }
        self._latest_preview_frame = None

        cfg = video_config or {}
        self.z_scale = cfg.get("z_scale", 2.5)
        self.correct_segments = cfg.get("correct_segments", True)
        self._reference_wrist_to_mid_mcp = cfg.get("reference_wrist_to_mid_mcp", 0.09)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Failed to open webcam at camera index {camera_index}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._expected_mp_label = "Left" if self.hand_side == "right" else "Right"
        self._last_valid_kp = None
        self._last_valid_raw = None

        self._worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_thread.start()

    def get_fingers_data(self) -> dict:
        with self._lock:
            return {
                "left_fingers": self._latest_result["left_fingers"].copy(),
                "right_fingers": self._latest_result["right_fingers"].copy(),
            }

    def get_preview_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_preview_frame is None:
                return None
            return self._latest_preview_frame.copy()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._resume_event.clear()
        else:
            self._resume_event.set()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            self._resume_event.wait()
            if self._stop_event.is_set():
                break
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)
            keypoints, raw_landmarks, detector_landmarks = self._extract_landmarks(results)

            if keypoints is not None:
                keypoints = self._process_landmarks(keypoints)
                self._last_valid_kp = keypoints
                self._last_valid_raw = raw_landmarks
            else:
                keypoints = self._last_valid_kp
                raw_landmarks = self._last_valid_raw

            self.publish_inference_sample(InferenceSample(
                timestamp_sec=time.monotonic(),
                frame_bgr=frame.copy(),
                input_type="webcam",
                hand_side=self.hand_side,
                detected=detector_landmarks is not None,
                payload={
                    "detector_landmarks": None if detector_landmarks is None else detector_landmarks.copy(),
                    "processed_landmarks": (
                        np.zeros((21, 3), dtype=np.float32)
                        if keypoints is None else keypoints.copy()
                    ),
                },
            ))

            preview = self._annotate_frame(frame, raw_landmarks)
            if self.show_video:
                self._show_live_frame(preview)

            latest = {
                "left_fingers": self._empty.copy(),
                "right_fingers": self._empty.copy(),
            }
            if keypoints is not None:
                latest[f"{self.hand_side}_fingers"] = keypoints.copy()
            with self._lock:
                self._latest_result = latest
                self._latest_preview_frame = preview

    def _extract_landmarks(self, results):
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return None, None, None

        for hand_landmarks, hand_classification in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if hand_classification.classification[0].label == self._expected_mp_label:
                return self._extract_hand_landmarks(hand_landmarks)

        hand_landmarks = results.multi_hand_landmarks[0]
        return self._extract_hand_landmarks(hand_landmarks)

    def _extract_hand_landmarks(self, hand_landmarks):
        detector_landmarks = np.asarray([
            (landmark.x, landmark.y, landmark.z)
            for landmark in hand_landmarks.landmark
        ], dtype=np.float32)
        return self._landmarks_to_array(hand_landmarks), [
            (landmark.x, landmark.y) for landmark in hand_landmarks.landmark
        ], detector_landmarks

    def _annotate_frame(self, frame: np.ndarray, raw_landmarks) -> np.ndarray:
        display = frame.copy()
        if raw_landmarks is not None:
            height, width = display.shape[:2]
            points = [(int(x * width), int(y * height)) for x, y in raw_landmarks]
            for start, end in self._HAND_CONNECTIONS:
                cv2.line(display, points[start], points[end], (0, 255, 0), 2)
            for index, point in enumerate(points):
                cv2.circle(display, point, 4, (0, 0, 255), -1)
                cv2.putText(display, str(index), (point[0] + 5, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.putText(display, f"Webcam {self.camera_index}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return display

    def _show_live_frame(self, annotated_frame: np.ndarray) -> None:
        display = annotated_frame
        scale = 480 / display.shape[0]
        display = cv2.resize(display, None, fx=scale, fy=scale)
        cv2.imshow("Webcam MediaPipe", display)
        cv2.waitKey(1)

    def cleanup(self) -> None:
        if not hasattr(self, "_stop_event"):
            return
        self._stop_event.set()
        self._resume_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if getattr(self, "cap", None) is not None:
            self.cap.release()
        if getattr(self, "mp_hands", None) is not None:
            self.mp_hands.close()
        if self.show_video:
            cv2.destroyAllWindows()

    def __del__(self):
        self.cleanup()
