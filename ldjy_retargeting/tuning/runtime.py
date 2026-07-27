"""Live retargeting state used by the desktop tuning application."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from ldjy_retargeting import Retargeter
from ldjy_retargeting.tuning.vector_scale_calibration import robot_vector_lengths


class TuningRuntime:
    """Replaceable Retargeter state with verbose frame diagnostics."""

    def __init__(
        self,
        config: dict[str, Any],
        hand_side: str,
        yaml_dir: str | Path | None = None,
    ):
        self.hand_side = hand_side.lower()
        self._yaml_dir = Path(yaml_dir).resolve() if yaml_dir is not None else Path.cwd()
        self._config: dict[str, Any] = {}
        self.retargeter: Retargeter
        self.apply_config(config)

    @property
    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    def apply_config(self, config: dict[str, Any]) -> None:
        """Apply a fresh configuration without preserving old optimizer state."""
        candidate = copy.deepcopy(config)
        candidate["__yaml_dir"] = str(self._yaml_dir)
        retargeter = Retargeter(candidate, self.hand_side)
        retargeter.reset()
        self._config = candidate
        self.retargeter = retargeter

    def prepare_keypoints(self, raw_keypoints: np.ndarray) -> np.ndarray:
        """Convert one input frame to the MANO-aligned task frame."""
        return self.retargeter._prepare_keypoints(
            np.asarray(raw_keypoints, dtype=np.float64)
        )

    def zero_pose_robot_vector_lengths(self) -> np.ndarray:
        """Return the 15 wrist task-vector lengths at the robot zero pose."""
        optimizer = self.retargeter.optimizer
        required = ("origin_link_name", "link3_names", "link4_names", "task_link_names")
        if not all(hasattr(optimizer, name) for name in required):
            raise RuntimeError(
                "自动零位标定要求具有 PIP、DIP、TIP 任务链接的自适应优化器"
            )

        robot = optimizer.robot
        robot.compute_forward_kinematics(
            np.zeros(optimizer.num_joints, dtype=np.float64)
        )

        def position(link_name: str) -> np.ndarray:
            return robot.get_link_pose(robot.get_link_index(link_name))[:3, 3]

        return robot_vector_lengths(
            position(optimizer.origin_link_name),
            np.stack([position(name) for name in optimizer.link3_names]),
            np.stack([position(name) for name in optimizer.link4_names]),
            np.stack([position(name) for name in optimizer.task_link_names]),
        )

    def process(self, raw_keypoints: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Retarget one standard `(21, 3)` hand landmark frame."""
        qpos, diagnostics = self.retargeter.retarget_verbose(raw_keypoints)
        return qpos, diagnostics


__all__ = ["TuningRuntime"]
