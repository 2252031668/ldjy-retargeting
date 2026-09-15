from abc import ABC
from dataclasses import dataclass
import threading
from typing import Any
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class InferenceSample:
    """One completed live inference paired with its source camera frame."""

    timestamp_sec: float
    frame_bgr: np.ndarray | None
    input_type: str
    hand_side: str
    detected: bool
    payload: dict[str, Any]


class InputDeviceBase(ABC):
    def get_fingers_data(self) -> Dict[str, np.ndarray]:
        """Return a dict with `left_fingers` and `right_fingers` data."""
        raise NotImplementedError("this input does not provide 21-point hand landmarks")

    def get_latest_frame(self) -> Any | None:
        """Return the latest native frame for inputs that do not use 21 points."""
        return None

    def get_preview_frame(self) -> np.ndarray | None:
        """Return the latest annotated BGR frame when the device supports it."""
        return None

    def set_paused(self, paused: bool) -> None:
        """Optionally suspend live capture/inference while a UI is paused."""
        del paused

    def publish_inference_sample(self, sample: InferenceSample) -> None:
        """Publish one completed inference for an optional recording consumer."""
        if not hasattr(self, "_record_sample_lock"):
            self._record_sample_lock = threading.Lock()
            self._record_samples: list[InferenceSample] = []
            self._recording_enabled = False
        with self._record_sample_lock:
            if self._recording_enabled:
                self._record_samples.append(sample)

    def set_recording_enabled(self, enabled: bool) -> None:
        """Enable sample buffering only for the duration of an active recording."""
        if not hasattr(self, "_record_sample_lock"):
            self._record_sample_lock = threading.Lock()
            self._record_samples = []
        with self._record_sample_lock:
            self._recording_enabled = bool(enabled)
            if enabled:
                self._record_samples = []

    def drain_inference_samples(self) -> list[InferenceSample]:
        """Return completed samples in inference completion order."""
        if not hasattr(self, "_record_sample_lock"):
            return []
        with self._record_sample_lock:
            samples = self._record_samples
            self._record_samples = []
        return samples
