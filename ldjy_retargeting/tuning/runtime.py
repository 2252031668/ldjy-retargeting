"""Live retargeting state used by the desktop tuning application."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ldjy_retargeting import Retargeter
from ldjy_retargeting.retarget_tip_frames import normalize_tip_offsets
from ldjy_retargeting.tuning.vector_scale_calibration import robot_vector_lengths


TIP_ASSET_CACHE_VERSION = 2


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
        self.current_urdf_path: Path
        self.debug_mjcf_path: Path
        self.apply_config(config)

    @property
    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    def apply_config(self, config: dict[str, Any]) -> None:
        """Apply a fresh configuration without preserving old optimizer state."""
        candidate = copy.deepcopy(config)
        offsets = normalize_tip_offsets(candidate.get("tip_offsets"))
        candidate["tip_offsets"] = offsets
        urdf_path, mjcf_path = self._materialize_tip_assets(offsets)
        candidate.setdefault("optimizer", {})["urdf_path"] = str(urdf_path)
        candidate["__yaml_dir"] = str(self._yaml_dir)
        retargeter = Retargeter(candidate, self.hand_side)
        retargeter.reset()
        self._config = candidate
        self.retargeter = retargeter
        self.current_urdf_path = urdf_path
        self.debug_mjcf_path = mjcf_path

    def preview_tip_offsets(self, config: dict[str, Any]) -> Path:
        """Materialize a virtual-tip MJCF without replacing the retargeter.

        Used while the GUI is paused: only the displayed MuJoCo task sites
        move, while the frozen optimizer state and physics pose remain intact.
        """
        offsets = normalize_tip_offsets(config.get("tip_offsets"))
        _, mjcf_path = self._materialize_tip_assets(offsets)
        return mjcf_path

    def _materialize_tip_assets(
        self, offsets: dict[str, dict[str, float]]
    ) -> tuple[Path, Path]:
        """Build or reuse non-destructive standalone assets for one offset map."""
        digest = hashlib.sha256(
            json.dumps(
                {"version": TIP_ASSET_CACHE_VERSION, "offsets": offsets},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]
        asset_dir = Path(__file__).resolve().parents[1] / "assets" / "robots" / "ldjy_hand"
        cache_root = asset_dir / ".cache" / "tip_tuning" / digest
        urdf_dir = cache_root / "urdf"
        mjcf_dir = cache_root / "mjcf"
        urdf_path = urdf_dir / f"ldjy_{self.hand_side}_hand.urdf"
        mjcf_path = mjcf_dir / f"ldjy_{self.hand_side}_hand.xml"
        if urdf_path.is_file() and mjcf_path.is_file():
            return urdf_path, mjcf_path

        cache_root.mkdir(parents=True, exist_ok=True)
        meshes = cache_root / "meshes"
        if not meshes.exists():
            meshes.symlink_to(asset_dir / "meshes", target_is_directory=True)
        from tools.build_ldjy_mjcf import build_model
        from tools.build_ldjy_urdf import build_urdf

        build_urdf(self.hand_side, offsets=offsets, output_dir=urdf_dir)
        build_model(
            self.hand_side,
            offsets=offsets,
            urdf_dir=urdf_dir,
            output_dir=mjcf_dir,
        )
        return urdf_path, mjcf_path

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
