"""Safe in-memory editing and persistence for tuning YAML files."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from .parameters import FINGERS, normalize_runtime_config, set_path, validate_runtime_config


class TuningSession:
    """Own one editable YAML configuration and its immutable first-save baseline."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"调参配置文件不存在: {self.config_path}")
        self._disk_config = self._load_yaml(self.config_path)
        self._config = copy.deepcopy(self._disk_config)

    @property
    def original_path(self) -> Path:
        return self.config_path.with_name(f"{self.config_path.name}.original.yaml")

    @property
    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    @property
    def is_dirty(self) -> bool:
        return self._config != self._disk_config

    def set_value(self, path: str, value: Any) -> None:
        candidate = copy.deepcopy(self._config)
        set_path(candidate, path, value)
        validate_runtime_config(candidate)
        self._config = candidate

    def set_segment_scalings(self, scales: Any) -> None:
        """Atomically replace all 15 FullHand vector scales in memory."""
        values = np.asarray(scales, dtype=np.float64)
        if values.shape != (5, 3):
            raise ValueError(f"segment scales must have shape (5, 3), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("segment scales must be finite")
        if np.any(values < 0.5) or np.any(values > 1.5):
            raise ValueError("segment scales must remain within range [0.5, 1.5]")

        candidate = copy.deepcopy(self._config)
        scaling = candidate["retarget"]["segment_scaling"]
        for finger, values_for_finger in zip(FINGERS, values):
            scaling[finger] = [float(value) for value in values_for_finger]
        validate_runtime_config(candidate)
        self._config = candidate

    def restore_default(self) -> None:
        if not self.original_path.is_file():
            raise FileNotFoundError(
                "尚未创建默认基线；请至少保存一次当前配置后再恢复默认。"
            )
        self._config = self._load_yaml(self.original_path)

    def save(self) -> None:
        validate_runtime_config(self._config)
        if not self.original_path.exists():
            shutil.copy2(self.config_path, self.original_path)

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.config_path.parent,
                prefix=f".{self.config_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                yaml.safe_dump(
                    self._config,
                    temp_file,
                    allow_unicode=True,
                    sort_keys=False,
                )
                temp_path = Path(temp_file.name)
            os.replace(temp_path, self.config_path)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

        self._disk_config = copy.deepcopy(self._config)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        if not isinstance(config, dict):
            raise ValueError(f"YAML 顶层必须是映射: {path}")
        normalize_runtime_config(config)
        validate_runtime_config(config)
        return config


__all__ = ["TuningSession"]
