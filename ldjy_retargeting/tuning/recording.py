"""On-disk records for repeatable tuning sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


SCHEMA = "ldjy-retargeting.tuning-record.v1"
INPUT_DIRECTORY = {
    "webcam": "mediapipe",
    "webcam_wilor": "wilor",
    "quest_hts": "quest_hts",
    "manus": "manus",
}


@dataclass(frozen=True)
class RecordInfo:
    path: Path
    input_type: str
    hand_side: str
    frame_count: int

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class RecordSample:
    timestamp_sec: float
    frame_bgr: np.ndarray
    detected: bool
    detector_landmarks: np.ndarray | None
    processed_landmarks: np.ndarray


@dataclass(frozen=True)
class MediaPipeRecord:
    info: RecordInfo
    timestamp_sec: np.ndarray
    detected: np.ndarray
    detector_landmarks: np.ndarray
    processed_landmarks: np.ndarray
    width: int
    height: int

    @property
    def frame_count(self) -> int:
        return int(self.timestamp_sec.shape[0])


@dataclass(frozen=True)
class WiLoRRecordSample:
    """One selected-hand WiLoR result paired with its completed camera frame."""

    timestamp_sec: float
    frame_bgr: np.ndarray
    detected: bool
    detection: Any | None


@dataclass(frozen=True)
class WiLoRRecord:
    info: RecordInfo
    timestamp_sec: np.ndarray
    detected: np.ndarray
    faces: np.ndarray
    arrays: dict[str, np.ndarray]
    width: int
    height: int

    @property
    def frame_count(self) -> int:
        return int(self.timestamp_sec.shape[0])


@dataclass(frozen=True)
class QuestHTSRecordSample:
    """One selected-hand Quest HTS frame, kept in both source and task coordinates."""

    timestamp_sec: float
    detected: bool
    landmarks_unity_left: np.ndarray
    landmarks_rfu: np.ndarray
    wrist_position_unity_left: np.ndarray
    wrist_quaternion_unity_left: np.ndarray


@dataclass(frozen=True)
class QuestHTSRecord:
    info: RecordInfo
    timestamp_sec: np.ndarray
    detected: np.ndarray
    landmarks_unity_left: np.ndarray
    landmarks_rfu: np.ndarray
    wrist_position_unity_left: np.ndarray
    wrist_quaternion_unity_left: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.timestamp_sec.shape[0])


@dataclass(frozen=True)
class ManusRecord:
    """Lossless native MANUS frames sharing one immutable topology."""

    info: RecordInfo
    timestamp_sec: np.ndarray
    local_timestamp_sec: np.ndarray
    sdk_timestamp: np.ndarray
    glove_id: int
    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    topology: Any
    ergonomics: np.ndarray
    ergonomics_timestamp: np.ndarray
    ergonomics_valid: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.timestamp_sec.shape[0])


def _timestamp_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _metadata(path: Path) -> dict[str, Any]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _validate_info(path: Path, input_type: str) -> RecordInfo | None:
    try:
        metadata = _metadata(path)
        required = {"schema", "input_type", "hand_side", "frame_count"}
        if input_type not in {"quest_hts", "manus"}:
            required |= {"width", "height"}
        if not required <= metadata.keys() or metadata["schema"] != SCHEMA:
            return None
        if metadata["input_type"] != input_type or metadata["hand_side"] not in {"left", "right"}:
            return None
        result_name = "frames.npz" if input_type == "webcam" else "result.npz"
        needs_video = input_type not in {"quest_hts", "manus"}
        if (needs_video and not (path / "video.mp4").is_file()) or not (path / result_name).is_file():
            return None
        if not (path / "config_snapshot.yaml").is_file():
            return None
        frame_count = int(metadata["frame_count"])
        if frame_count <= 0:
            return None
        return RecordInfo(path, input_type, metadata["hand_side"], frame_count)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def list_records(root: str | Path, input_type: str) -> list[RecordInfo]:
    """List only atomically completed records for one live input type."""
    if input_type not in INPUT_DIRECTORY:
        raise ValueError(f"unsupported input_type: {input_type}")
    directory = Path(root) / INPUT_DIRECTORY[input_type]
    if not directory.is_dir():
        return []
    records = [
        info for child in directory.iterdir() if child.is_dir() and not child.name.startswith(".")
        if (info := _validate_info(child, input_type)) is not None
    ]
    return sorted(records, key=lambda item: item.name, reverse=True)


class MediaPipeRecordWriter:
    """Write one camera frame and one completed MediaPipe inference per sample."""

    def __init__(self, root: Path, hand_side: str, width: int, height: int) -> None:
        if hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be left or right")
        self.root = root
        self.hand_side = hand_side
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("record dimensions must be positive")
        self.name = _timestamp_name()
        base = self.root / INPUT_DIRECTORY["webcam"]
        base.mkdir(parents=True, exist_ok=True)
        suffix = 0
        while True:
            candidate_name = self.name if suffix == 0 else f"{self.name}_{suffix:02d}"
            candidate = base / f".{candidate_name}.tmp"
            final = base / candidate_name
            if not candidate.exists() and not final.exists():
                break
            suffix += 1
        self._temporary_path = candidate
        self._final_path = final
        self._temporary_path.mkdir()
        self._writer = cv2.VideoWriter(
            str(self._temporary_path / "video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0, (self.width, self.height),
        )
        if not self._writer.isOpened():
            self._temporary_path.rmdir()
            raise RuntimeError("could not create tuning-record MP4")
        self._timestamps: list[float] = []
        self._first_timestamp: float | None = None
        self._detected: list[bool] = []
        self._raw: list[np.ndarray] = []
        self._processed: list[np.ndarray] = []
        self._closed = False

    @classmethod
    def start(cls, root: str | Path, *, hand_side: str, width: int, height: int) -> "MediaPipeRecordWriter":
        return cls(Path(root), hand_side, width, height)

    def append(self, sample: RecordSample) -> None:
        if self._closed:
            raise RuntimeError("record writer is already closed")
        frame = np.asarray(sample.frame_bgr, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"record frame shape must be {(self.height, self.width, 3)}, got {frame.shape}")
        processed = np.asarray(sample.processed_landmarks, dtype=np.float32)
        if processed.shape != (21, 3):
            raise ValueError("processed_landmarks must have shape (21, 3)")
        if sample.detector_landmarks is None:
            raw = np.full((21, 3), np.nan, dtype=np.float32)
        else:
            raw = np.asarray(sample.detector_landmarks, dtype=np.float32)
            if raw.shape != (21, 3):
                raise ValueError("detector_landmarks must have shape (21, 3)")
        self._writer.write(frame)
        if self._first_timestamp is None:
            self._first_timestamp = float(sample.timestamp_sec)
        self._timestamps.append(float(sample.timestamp_sec) - self._first_timestamp)
        self._detected.append(bool(sample.detected))
        self._raw.append(raw.copy())
        self._processed.append(processed.copy())

    def finish(self, *, config: dict[str, Any]) -> RecordInfo:
        if self._closed:
            raise RuntimeError("record writer is already closed")
        self._closed = True
        self._writer.release()
        count = len(self._timestamps)
        np.savez_compressed(
            self._temporary_path / "frames.npz",
            timestamp_sec=np.asarray(self._timestamps, dtype=np.float64),
            detected=np.asarray(self._detected, dtype=bool),
            detector_landmarks=np.asarray(self._raw, dtype=np.float32).reshape(count, 21, 3),
            processed_landmarks=np.asarray(self._processed, dtype=np.float32).reshape(count, 21, 3),
        )
        metadata = {
            "schema": SCHEMA,
            "input_type": "webcam",
            "hand_side": self.hand_side,
            "frame_count": count,
            "width": self.width,
            "height": self.height,
            "timestamps_are_actual_inference_completion": True,
        }
        (self._temporary_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self._temporary_path / "config_snapshot.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        os.replace(self._temporary_path, self._final_path)
        return RecordInfo(self._final_path, "webcam", self.hand_side, count)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.release()


def load_mediapipe_record(path: str | Path) -> MediaPipeRecord:
    path = Path(path).resolve()
    info = _validate_info(path, "webcam")
    if info is None:
        raise ValueError(f"not a complete MediaPipe tuning record: {path}")
    metadata = _metadata(path)
    with np.load(path / "frames.npz") as arrays:
        timestamp_sec = np.asarray(arrays["timestamp_sec"], dtype=np.float64)
        detected = np.asarray(arrays["detected"], dtype=bool)
        raw = np.asarray(arrays["detector_landmarks"], dtype=np.float32)
        processed = np.asarray(arrays["processed_landmarks"], dtype=np.float32)
    expected = (info.frame_count, 21, 3)
    if timestamp_sec.shape != (info.frame_count,) or detected.shape != (info.frame_count,):
        raise ValueError("MediaPipe record timestamp/detected shape is invalid")
    if raw.shape != expected or processed.shape != expected:
        raise ValueError("MediaPipe record landmark shape is invalid")
    return MediaPipeRecord(info, timestamp_sec, detected, raw, processed, int(metadata["width"]), int(metadata["height"]))


_WILOR_FIELDS = {
    "bbox_xyxy": (4,),
    "joints_mano": (21, 3),
    "vertices_mano": (778, 3),
    "global_orient": (3, 3),
    "hand_pose": (15, 3, 3),
    "betas": (10,),
    "pred_cam": (3,),
    "camera_translation": (3,),
}


class WiLoRRecordWriter:
    """Write a selected-hand WiLoR sequence without requiring WiLoR at replay time."""

    def __init__(self, root: Path, hand_side: str, width: int, height: int, faces: np.ndarray) -> None:
        if hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be left or right")
        self.root, self.hand_side = root, hand_side
        self.width, self.height = int(width), int(height)
        self.faces = np.asarray(faces, dtype=np.int32)
        if self.faces.ndim != 2 or self.faces.shape[1:] != (3,):
            raise ValueError("faces must have shape (F, 3)")
        base = root / INPUT_DIRECTORY["webcam_wilor"]
        base.mkdir(parents=True, exist_ok=True)
        name, suffix = _timestamp_name(), 0
        while True:
            display_name = name if suffix == 0 else f"{name}_{suffix:02d}"
            temporary, final = base / f".{display_name}.tmp", base / display_name
            if not temporary.exists() and not final.exists():
                break
            suffix += 1
        self._temporary_path, self._final_path = temporary, final
        temporary.mkdir()
        self._writer = cv2.VideoWriter(
            str(temporary / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            temporary.rmdir()
            raise RuntimeError("could not create tuning-record MP4")
        self._timestamps: list[float] = []
        self._first_timestamp: float | None = None
        self._detected: list[bool] = []
        self._arrays = {name: [] for name in _WILOR_FIELDS}
        self._closed = False

    @classmethod
    def start(
        cls, root: str | Path, *, hand_side: str, width: int, height: int, faces: np.ndarray
    ) -> "WiLoRRecordWriter":
        return cls(Path(root), hand_side, width, height, faces)

    def append(self, sample: WiLoRRecordSample) -> None:
        if self._closed:
            raise RuntimeError("record writer is already closed")
        frame = np.asarray(sample.frame_bgr, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"record frame shape must be {(self.height, self.width, 3)}, got {frame.shape}")
        valid = bool(sample.detected and sample.detection is not None)
        values: dict[str, np.ndarray] = {}
        if valid:
            for name, shape in _WILOR_FIELDS.items():
                value = np.asarray(getattr(sample.detection, name), dtype=np.float32)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"WiLoR {name} must have finite shape {shape}")
                values[name] = value
        self._writer.write(frame)
        if self._first_timestamp is None:
            self._first_timestamp = float(sample.timestamp_sec)
        self._timestamps.append(float(sample.timestamp_sec) - self._first_timestamp)
        self._detected.append(valid)
        for name, shape in _WILOR_FIELDS.items():
            self._arrays[name].append(
                values.get(name, np.full(shape, np.nan, dtype=np.float32)).copy()
            )

    def finish(self, *, config: dict[str, Any]) -> RecordInfo:
        if self._closed:
            raise RuntimeError("record writer is already closed")
        self._closed = True
        self._writer.release()
        count = len(self._timestamps)
        np.savez_compressed(
            self._temporary_path / "result.npz",
            timestamp_sec=np.asarray(self._timestamps, dtype=np.float64),
            detected=np.asarray(self._detected, dtype=bool), faces=self.faces,
            **{name: np.asarray(values, dtype=np.float32).reshape((count,) + shape)
               for name, (shape, values) in ((name, (shape, self._arrays[name])) for name, shape in _WILOR_FIELDS.items())},
        )
        metadata = {
            "schema": SCHEMA, "input_type": "webcam_wilor", "hand_side": self.hand_side,
            "frame_count": count, "width": self.width, "height": self.height,
            "timestamps_are_actual_inference_completion": True,
            "coordinate_system": "wilor_mano_local",
        }
        (self._temporary_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self._temporary_path / "config_snapshot.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        os.replace(self._temporary_path, self._final_path)
        return RecordInfo(self._final_path, "webcam_wilor", self.hand_side, count)

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            self._writer.release()


def load_wilor_record(path: str | Path) -> WiLoRRecord:
    path = Path(path).resolve()
    info = _validate_info(path, "webcam_wilor")
    if info is None:
        raise ValueError(f"not a complete WiLoR tuning record: {path}")
    metadata = _metadata(path)
    with np.load(path / "result.npz") as archive:
        timestamp_sec = np.asarray(archive["timestamp_sec"], dtype=np.float64)
        detected = np.asarray(archive["detected"], dtype=bool)
        faces = np.asarray(archive["faces"], dtype=np.int32)
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in _WILOR_FIELDS}
    if timestamp_sec.shape != (info.frame_count,) or detected.shape != (info.frame_count,):
        raise ValueError("WiLoR record timestamp/detected shape is invalid")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("WiLoR record faces shape is invalid")
    for name, shape in _WILOR_FIELDS.items():
        if arrays[name].shape != (info.frame_count,) + shape:
            raise ValueError(f"WiLoR record {name} shape is invalid")
    return WiLoRRecord(info, timestamp_sec, detected, faces, arrays, int(metadata["width"]), int(metadata["height"]))


class QuestHTSRecordWriter:
    """Write Quest hand-tracking frames without a meaningless camera video."""

    _FIELDS = {
        "landmarks_unity_left": (21, 3),
        "landmarks_rfu": (21, 3),
        "wrist_position_unity_left": (3,),
        "wrist_quaternion_unity_left": (4,),
    }

    def __init__(self, root: Path, hand_side: str) -> None:
        if hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be left or right")
        self.hand_side = hand_side
        base = root / INPUT_DIRECTORY["quest_hts"]
        base.mkdir(parents=True, exist_ok=True)
        name, suffix = _timestamp_name(), 0
        while True:
            display_name = name if suffix == 0 else f"{name}_{suffix:02d}"
            temporary, final = base / f".{display_name}.tmp", base / display_name
            if not temporary.exists() and not final.exists():
                break
            suffix += 1
        self._temporary_path, self._final_path = temporary, final
        temporary.mkdir()
        self._timestamps: list[float] = []
        self._first_timestamp: float | None = None
        self._detected: list[bool] = []
        self._arrays = {name: [] for name in self._FIELDS}
        self._closed = False

    @classmethod
    def start(cls, root: str | Path, *, hand_side: str) -> "QuestHTSRecordWriter":
        return cls(Path(root), hand_side)

    def append(self, sample: QuestHTSRecordSample) -> None:
        if self._closed:
            raise RuntimeError("record writer is already closed")
        values: dict[str, np.ndarray] = {}
        if sample.detected:
            for name, shape in self._FIELDS.items():
                value = np.asarray(getattr(sample, name), dtype=np.float32)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Quest HTS {name} must have finite shape {shape}")
                values[name] = value
        if self._first_timestamp is None:
            self._first_timestamp = float(sample.timestamp_sec)
        self._timestamps.append(float(sample.timestamp_sec) - self._first_timestamp)
        self._detected.append(bool(sample.detected))
        for name, shape in self._FIELDS.items():
            self._arrays[name].append(values.get(name, np.full(shape, np.nan, dtype=np.float32)).copy())

    def finish(self, *, config: dict[str, Any]) -> RecordInfo:
        if self._closed:
            raise RuntimeError("record writer is already closed")
        self._closed = True
        count = len(self._timestamps)
        np.savez_compressed(
            self._temporary_path / "result.npz",
            timestamp_sec=np.asarray(self._timestamps, dtype=np.float64),
            detected=np.asarray(self._detected, dtype=bool),
            **{name: np.asarray(values, dtype=np.float32).reshape((count,) + self._FIELDS[name])
               for name, values in self._arrays.items()},
        )
        metadata = {
            "schema": SCHEMA, "input_type": "quest_hts", "hand_side": self.hand_side,
            "frame_count": count,
            "timestamps_are_actual_inference_completion": True,
            "coordinate_system": "unity_left_wrist_local_and_rfu_wrist_local",
        }
        (self._temporary_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self._temporary_path / "config_snapshot.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        os.replace(self._temporary_path, self._final_path)
        return RecordInfo(self._final_path, "quest_hts", self.hand_side, count)

    def abort(self) -> None:
        self._closed = True


def load_quest_hts_record(path: str | Path) -> QuestHTSRecord:
    path = Path(path).resolve()
    info = _validate_info(path, "quest_hts")
    if info is None:
        raise ValueError(f"not a complete Quest HTS tuning record: {path}")
    with np.load(path / "result.npz") as archive:
        timestamp_sec = np.asarray(archive["timestamp_sec"], dtype=np.float64)
        detected = np.asarray(archive["detected"], dtype=bool)
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in QuestHTSRecordWriter._FIELDS}
    if timestamp_sec.shape != (info.frame_count,) or detected.shape != (info.frame_count,):
        raise ValueError("Quest HTS record timestamp/detected shape is invalid")
    for name, shape in QuestHTSRecordWriter._FIELDS.items():
        if arrays[name].shape != (info.frame_count,) + shape:
            raise ValueError(f"Quest HTS record {name} shape is invalid")
    return QuestHTSRecord(info, timestamp_sec, detected, **arrays)


class ManusRecordWriter:
    """Write complete Raw Skeleton and asynchronous Ergonomics arrays."""

    def __init__(self, root: Path, hand_side: str, calibration: Any | None = None,
                 hybrid_calibration: Any | None = None, config: dict[str, Any] | None = None) -> None:
        if hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be left or right")
        self.hand_side, self.calibration, self.hybrid_calibration = hand_side, calibration, hybrid_calibration
        self._config_snapshot = config
        base = root / INPUT_DIRECTORY["manus"]
        base.mkdir(parents=True, exist_ok=True)
        name, suffix = _timestamp_name(), 0
        while True:
            display = name if suffix == 0 else f"{name}_{suffix:02d}"
            temporary, final = base / f".{display}.tmp", base / display
            if not temporary.exists() and not final.exists():
                break
            suffix += 1
        self._temporary_path, self._final_path = temporary, final
        temporary.mkdir()
        self._frames: list[Any] = []
        self._closed = False

    @classmethod
    def start(cls, root: str | Path, *, hand_side: str, calibration: Any | None = None,
              hybrid_calibration: Any | None = None, config: dict[str, Any] | None = None) -> "ManusRecordWriter":
        return cls(Path(root), hand_side, calibration, hybrid_calibration, config)

    def append(self, frame: Any) -> None:
        if self._closed:
            raise RuntimeError("record writer is already closed")
        if frame.side != self.hand_side:
            return
        if self._frames and not self._frames[0].topology.matches(frame.topology):
            raise ValueError("MANUS topology changed during recording")
        if self._frames and self._frames[0].ergonomics.shape != frame.ergonomics.shape:
            raise ValueError("MANUS Ergonomics array size changed during recording")
        self._frames.append(frame)

    def finish(self, *, config: dict[str, Any]) -> RecordInfo:
        if self._closed:
            raise RuntimeError("record writer is already closed")
        self._closed = True
        if not self._frames:
            raise ValueError("cannot save an empty MANUS recording")
        first, count = self._frames[0], len(self._frames)
        local = np.array([f.received_timestamp_sec for f in self._frames], dtype=np.float64)
        topology = first.topology
        orders = []
        for frame in self._frames:
            by_id = {int(node_id): i for i, node_id in enumerate(frame.topology.node_ids)}
            orders.append([by_id[int(node_id)] for node_id in topology.node_ids])
        arrays: dict[str, Any] = {
            "timestamp_sec": local - local[0], "local_timestamp_sec": local,
            "sdk_timestamp": np.array([f.sdk_timestamp for f in self._frames], dtype=np.uint64),
            "positions": np.stack([f.positions[order] for f, order in zip(self._frames, orders)]),
            "rotations": np.stack([f.rotations[order] for f, order in zip(self._frames, orders)]),
            "scales": np.stack([f.scales[order] for f, order in zip(self._frames, orders)]),
            "ergonomics": np.stack([f.ergonomics for f in self._frames]),
            "ergonomics_timestamp": np.array([f.ergonomics_timestamp for f in self._frames], dtype=np.uint64),
            "ergonomics_valid": np.array([f.ergonomics_valid for f in self._frames], dtype=bool),
            **{name: getattr(topology, name) for name in (
                "node_ids", "parent_ids", "chain_types", "sides", "finger_joint_types"
            )},
        }
        np.savez_compressed(
            self._temporary_path / "result.npz",
            **arrays,
        )
        metadata = {
            "schema": SCHEMA, "input_type": "manus", "hand_side": self.hand_side,
            "frame_count": count, "glove_id": first.glove_id,
            "coordinate_system": "XFromViewer_PositiveZ_RightHanded_world_meters",
            "ergonomics_is_latest_not_synchronized": True,
            "has_raw_calibration": self.calibration is not None,
            "has_ergonomics_hybrid_calibration": self.hybrid_calibration is not None,
        }
        (self._temporary_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self._temporary_path / "config_snapshot.yaml").write_text(
            yaml.safe_dump(self._config_snapshot or config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        if self.calibration is not None:
            np.savez_compressed(self._temporary_path / "calibration_snapshot.npz", **self.calibration.as_arrays())
        if self.hybrid_calibration is not None:
            np.savez_compressed(self._temporary_path / "ergonomics_hybrid_calibration_snapshot.npz", **self.hybrid_calibration.as_arrays())
        os.replace(self._temporary_path, self._final_path)
        return RecordInfo(self._final_path, "manus", self.hand_side, count)

    def abort(self) -> None:
        self._closed = True


def load_manus_record(path: str | Path) -> ManusRecord:
    from ldjy_retargeting.manus import ManusTopology

    path = Path(path).resolve()
    info = _validate_info(path, "manus")
    if info is None:
        raise ValueError(f"not a complete MANUS tuning record: {path}")
    metadata = _metadata(path)
    with np.load(path / "result.npz", allow_pickle=False) as a:
        topology = ManusTopology(*(a[name] for name in (
            "node_ids", "parent_ids", "chain_types", "sides", "finger_joint_types"
        )))
        arrays = {name: np.asarray(a[name]).copy() for name in (
            "timestamp_sec", "local_timestamp_sec", "sdk_timestamp", "positions", "rotations",
            "scales", "ergonomics", "ergonomics_timestamp", "ergonomics_valid",
        )}
    n, nodes = info.frame_count, topology.node_count
    if arrays["positions"].shape != (n, nodes, 3) or arrays["rotations"].shape != (n, nodes, 4) or arrays["scales"].shape != (n, nodes, 3):
        raise ValueError("MANUS record transform shape is invalid")
    if arrays["ergonomics"].ndim != 2 or arrays["ergonomics"].shape[0] != n:
        raise ValueError("MANUS record Ergonomics shape is invalid")
    for name in ("timestamp_sec", "local_timestamp_sec", "sdk_timestamp",
                 "ergonomics_timestamp", "ergonomics_valid"):
        if arrays[name].shape != (n,):
            raise ValueError(f"MANUS record {name} shape is invalid")
    if not np.isfinite(arrays["timestamp_sec"]).all() or np.any(np.diff(arrays["timestamp_sec"]) < 0):
        raise ValueError("MANUS record timestamps are invalid")
    return ManusRecord(info=info, glove_id=int(metadata["glove_id"]), topology=topology, **arrays)


__all__ = [
    "ManusRecord", "ManusRecordWriter", "MediaPipeRecord", "MediaPipeRecordWriter", "RecordInfo", "RecordSample",
    "QuestHTSRecord", "QuestHTSRecordSample", "QuestHTSRecordWriter",
    "WiLoRRecord", "WiLoRRecordSample", "WiLoRRecordWriter", "list_records",
    "load_manus_record", "load_mediapipe_record", "load_quest_hts_record", "load_wilor_record",
]
