"""Deterministic replay of completed tuning records without detector inference."""

from __future__ import annotations

import time
from pathlib import Path
import sys

import cv2
import numpy as np

from .recording import MediaPipeRecord, WiLoRRecord, load_mediapipe_record, load_wilor_record


_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
)


class ReplayCursor:
    """Frame cursor whose play cadence follows saved, potentially irregular timestamps."""

    def __init__(self, timestamps: np.ndarray) -> None:
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        if self.timestamps.ndim != 1 or not len(self.timestamps):
            raise ValueError("timestamps must be a non-empty vector")
        if not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) < 0):
            raise ValueError("timestamps must be finite and non-decreasing")
        self.index, self.playing = 0, False
        self._playhead = float(self.timestamps[0])
        self._last_clock: float | None = None

    def seek(self, index: int) -> int:
        self.index = int(np.clip(index, 0, len(self.timestamps) - 1))
        self._playhead = float(self.timestamps[self.index])
        self._last_clock = None
        return self.index

    def step(self, delta: int) -> int:
        return self.seek(self.index + delta)

    def play(self) -> None:
        self.playing, self._last_clock = True, time.monotonic()

    def pause(self) -> None:
        self.playing, self._last_clock = False, None

    def advance(self, elapsed: float | None = None) -> int:
        if elapsed is None:
            now = time.monotonic()
            elapsed = 0.0 if self._last_clock is None else now - self._last_clock
            self._last_clock = now
        if not self.playing or elapsed <= 0 or self.index >= len(self.timestamps) - 1:
            if self.index >= len(self.timestamps) - 1:
                self.pause()
            return self.index
        self._playhead += elapsed
        self.index = min(int(np.searchsorted(self.timestamps, self._playhead, side="right") - 1), len(self.timestamps) - 1)
        if self.index >= len(self.timestamps) - 1:
            self.pause()
        return self.index


class _RecordReplay:
    def __init__(self, path: str | Path, timestamps: np.ndarray) -> None:
        self.path = Path(path)
        self.cursor = ReplayCursor(timestamps)
        self._capture = cv2.VideoCapture(str(self.path / "video.mp4"))
        if not self._capture.isOpened():
            raise ValueError(f"cannot open saved video: {self.path / 'video.mp4'}")

    @property
    def current_index(self) -> int:
        return self.cursor.index

    def close(self) -> None:
        self._capture.release()

    def frame_at(self, index: int) -> np.ndarray | None:
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = self._capture.read()
        return frame if ok else None


class MediaPipeReplay(_RecordReplay):
    def __init__(self, record: str | Path | MediaPipeRecord) -> None:
        self.record = load_mediapipe_record(record) if not isinstance(record, MediaPipeRecord) else record
        super().__init__(self.record.info.path, self.record.timestamp_sec)

    @property
    def hand_side(self) -> str:
        return self.record.info.hand_side

    def input_at(self, index: int, video_config: dict) -> np.ndarray | None:
        example_dir = Path(__file__).resolve().parents[2] / "example"
        if str(example_dir) not in sys.path:
            sys.path.insert(0, str(example_dir))
        from input_devices.video_mediapipe import process_detector_landmarks

        index = int(np.clip(index, 0, self.record.frame_count - 1))
        valid = np.flatnonzero(self.record.detected[: index + 1])
        if not len(valid):
            return None
        raw = self.record.detector_landmarks[valid[-1]]
        return process_detector_landmarks(
            raw,
            width=self.record.width,
            height=self.record.height,
            z_scale=float(video_config.get("z_scale", 2.5)),
            reference_wrist_to_mid_mcp=float(video_config.get("reference_wrist_to_mid_mcp", 0.09)),
            correct_segments=bool(video_config.get("correct_segments", True)),
        )

    def preview_at(self, index: int) -> np.ndarray | None:
        frame = self.frame_at(index)
        if frame is None:
            return None
        valid = np.flatnonzero(self.record.detected[: int(index) + 1])
        if not len(valid):
            return frame
        raw = self.record.detector_landmarks[valid[-1]]
        points = np.rint(raw[:, :2] * np.array([frame.shape[1], frame.shape[0]])).astype(int)
        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame, tuple(points[start]), tuple(points[end]), (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)
        return frame


class WiLoRReplay(_RecordReplay):
    def __init__(self, record: str | Path | WiLoRRecord) -> None:
        self.record = load_wilor_record(record) if not isinstance(record, WiLoRRecord) else record
        super().__init__(self.record.info.path, self.record.timestamp_sec)

    @property
    def hand_side(self) -> str:
        return self.record.info.hand_side

    def input_at(self, index: int) -> np.ndarray | None:
        index = int(np.clip(index, 0, self.record.frame_count - 1))
        valid = np.flatnonzero(self.record.detected[: index + 1])
        return None if not len(valid) else self.record.arrays["joints_mano"][valid[-1]].copy()

    def preview_at(self, index: int) -> np.ndarray | None:
        return self.frame_at(index)

    def mano_overlay_at(self, index: int) -> dict | None:
        index = int(np.clip(index, 0, self.record.frame_count - 1))
        valid = np.flatnonzero(self.record.detected[: index + 1])
        if not len(valid):
            return None
        item = valid[-1]
        return {
            "joints_mano": self.record.arrays["joints_mano"][item].copy(),
            "vertices_mano": self.record.arrays["vertices_mano"][item].copy(),
            "camera_translation": self.record.arrays["camera_translation"][item].copy(),
            "faces": self.record.faces.copy(),
            "is_right": self.hand_side == "right",
        }


__all__ = ["MediaPipeReplay", "ReplayCursor", "WiLoRReplay"]
