"""Live retargeting state used by the desktop tuning application."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ldjy_retargeting import Retargeter
from ldjy_retargeting.opt.base import LPFilter
from ldjy_retargeting.retarget_tip_frames import normalize_tip_offsets
from ldjy_retargeting.tuning.vector_scale_calibration import robot_vector_lengths


TIP_ASSET_CACHE_VERSION = 3
PAD_ASSET_CACHE_VERSION = 5


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
        self.retargeter: Retargeter | None
        self.optimizer: Any
        self._mano_pad_builder = None
        self._pad_filter: LPFilter | None = None
        self._is_mano_pad = False
        self._is_manus = False
        self._manus_requires_calibration = False
        self._manus_hybrid = False
        self.current_urdf_path: Path
        self.debug_mjcf_path: Path
        self.apply_config(config)

    @property
    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    def apply_config(self, config: dict[str, Any]) -> None:
        """Apply a fresh configuration without preserving old optimizer state."""
        previous_manus_calibration = (
            self.optimizer.calibration
            if getattr(self, "_manus_requires_calibration", False) and getattr(self, "optimizer", None) is not None
            else None
        )
        candidate = copy.deepcopy(config)
        offsets = normalize_tip_offsets(candidate.get("tip_offsets"))
        cache_version = (
            PAD_ASSET_CACHE_VERSION
            if candidate.get("optimizer", {}).get("type") == "ManoPadPoseOptimizer"
            else TIP_ASSET_CACHE_VERSION
        )
        candidate["tip_offsets"] = offsets
        urdf_path, mjcf_path = self._materialize_tip_assets(offsets, cache_version=cache_version)
        candidate.setdefault("optimizer", {})["urdf_path"] = str(urdf_path)
        candidate["optimizer"]["hand_side"] = self.hand_side
        candidate["__yaml_dir"] = str(self._yaml_dir)
        optimizer_type = candidate.get("optimizer", {}).get("type")
        if optimizer_type == "ManusFullSkeletonRetargeter":
            from ldjy_retargeting.manus import ManusFullSkeletonRetargeter

            optimizer = ManusFullSkeletonRetargeter(candidate)
            if previous_manus_calibration is not None:
                optimizer.set_calibration(previous_manus_calibration)
            retargeter = None
            self._mano_pad_builder = None
            self._pad_filter = None
            self._is_mano_pad = False
            self._is_manus = True
            self._manus_requires_calibration = True
            self._manus_hybrid = False
        elif optimizer_type == "ManusErgonomicsHybridRetargeter":
            from ldjy_retargeting.manus import ManusErgonomicsHybridRetargeter

            optimizer = ManusErgonomicsHybridRetargeter(candidate)
            retargeter = None
            self._mano_pad_builder = None
            self._pad_filter = None
            self._is_mano_pad = False
            self._is_manus = True
            self._manus_requires_calibration = False
            self._manus_hybrid = True
        elif optimizer_type == "ManoPadPoseOptimizer":
            from example.mano_viewer import (
                MANOModel, MANO_MODEL_PATH, PAD_3PT_VERTEX_IDS, PAD_VERTEX_IDS, TIP_ORDER,
            )
            from ldjy_retargeting.mano_pad_pose import ManoPadTargetBuilder
            from ldjy_retargeting.opt.mano_pad_pose import ManoPadPoseOptimizer

            reference = Path(candidate["optimizer"].get("mano_reference", ""))
            if not reference.is_absolute():
                reference = (self._yaml_dir / reference).resolve()
            mano = MANOModel(str(MANO_MODEL_PATH))
            self._mano_pad_builder = ManoPadTargetBuilder(
                mano,
                reference,
                self.hand_side,
                pad_vertex_ids=candidate["optimizer"].get("mano_pad_vertex_ids", PAD_VERTEX_IDS),
                pad_3pt_vertex_ids=candidate["optimizer"].get("mano_pad_3pt_vertex_ids", PAD_3PT_VERTEX_IDS),
                pad_order=TIP_ORDER,
            )
            optimizer = ManoPadPoseOptimizer(candidate)
            self._pad_filter = LPFilter(candidate.get("retarget", {}).get("lp_alpha", 0.2))
            self._is_mano_pad = True
            self._is_manus = False
            self._manus_requires_calibration = False
            self._manus_hybrid = False
            retargeter = None
        else:
            retargeter = Retargeter(candidate, self.hand_side)
            retargeter.reset()
            optimizer = retargeter.optimizer
            self._mano_pad_builder = None
            self._pad_filter = None
            self._is_mano_pad = False
            self._is_manus = False
            self._manus_requires_calibration = False
            self._manus_hybrid = False
        self._config = candidate
        self.retargeter = retargeter
        self.optimizer = optimizer
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
        self, offsets: dict[str, dict[str, float]], *, cache_version: int = TIP_ASSET_CACHE_VERSION
    ) -> tuple[Path, Path]:
        """Build or reuse non-destructive standalone assets for one offset map."""
        digest = hashlib.sha256(
            json.dumps(
                {"version": cache_version, "offsets": offsets},
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
        if self._is_mano_pad:
            raise RuntimeError("MANO 指腹算法不接受 21 点输入")
        return self.retargeter._prepare_keypoints(
            np.asarray(raw_keypoints, dtype=np.float64)
        )

    def zero_pose_robot_vector_lengths(self) -> np.ndarray:
        """Return the 15 wrist task-vector lengths at the robot zero pose."""
        optimizer = self.optimizer
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
        if self._is_mano_pad:
            raise RuntimeError("MANO 指腹算法请调用 process_mano_parameters")
        qpos, diagnostics = self.retargeter.retarget_verbose(raw_keypoints)
        return qpos, diagnostics

    def process_manus(self, frame: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._is_manus:
            raise RuntimeError("当前算法不是 MANUS Skeleton 算法")
        if self._manus_hybrid and self.optimizer.raw_calibration is None:
            from ldjy_retargeting.manus import ManusCalibration
            path = self.manus_calibration_path(frame.glove_id, frame.side)
            if not path.is_file():
                raise RuntimeError("Hybrid 需要 Raw Full 中立标定；请先用 MANUS Full Skeleton 采集一次中立姿态")
            calibration = ManusCalibration.load(path)
            if not calibration.matches(frame):
                raise ValueError("已保存的 MANUS Raw Full 标定与当前手套/拓扑不匹配")
            self.optimizer.set_raw_calibration(calibration)
        elif self._manus_requires_calibration and self.optimizer.calibration is None:
            from ldjy_retargeting.manus import ManusCalibration
            path = self.manus_calibration_path(frame.glove_id, frame.side)
            if path.is_file():
                calibration = ManusCalibration.load(path)
                if not calibration.matches(frame):
                    raise ValueError("已保存的 MANUS 标定与当前手套/拓扑不匹配")
                self.optimizer.set_calibration(calibration)
        return self.optimizer.solve(frame)

    @staticmethod
    def manus_calibration_path(glove_id: int, side: str) -> Path:
        return Path(__file__).resolve().parents[2] / "outputs" / "manus_calibrations" / f"{glove_id}_{side}.npz"

    def calibrate_manus(self, frames: list[Any], *, sdk_version: str = "unknown") -> Path:
        if self._manus_hybrid:
            from ldjy_retargeting.manus import ManusFullSkeletonRetargeter
            calibration = ManusFullSkeletonRetargeter(self._config).calibrate(frames, sdk_version=sdk_version)
            self.optimizer.set_raw_calibration(calibration)
        elif self._manus_requires_calibration:
            calibration = self.optimizer.calibrate(frames, sdk_version=sdk_version)
        else:
            raise RuntimeError("当前算法不支持 MANUS 共用零位标定")
        return calibration.save(self.manus_calibration_path(calibration.glove_id, calibration.side))

    @property
    def manus_calibrated(self) -> bool:
        if self._manus_hybrid:
            return self.optimizer.raw_calibration is not None
        return self._is_manus and (not self._manus_requires_calibration or self.optimizer.calibration is not None)

    @property
    def manus_requires_calibration(self) -> bool:
        return self._manus_requires_calibration

    def process_mano_parameters(
        self, parameters: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Retarget one absolute WiLoR MANO pose using fixed robot betas."""
        if not self._is_mano_pad or self._mano_pad_builder is None or self._pad_filter is None:
            raise RuntimeError("当前算法不是 MANO 指腹算法")
        from ldjy_retargeting.mano_pad_pose import ManoPoseInput

        target = self._mano_pad_builder.build(ManoPoseInput(
            hand_pose=np.asarray(parameters["hand_pose"]),
            global_orient=np.asarray(parameters.get("global_orient", np.eye(3))),
            translation=np.asarray(parameters.get("translation", np.zeros(3))),
            betas=np.asarray(parameters.get("betas", np.zeros(10))),
        ))
        qpos = self.optimizer.solve(target)
        filtered = self._pad_filter.next(qpos)
        diagnostics = dict(self.optimizer.last_diagnostics)
        diagnostics.update({
            "qpos_unfiltered": qpos.copy(),
            "qpos": filtered.copy(),
            "cost": float(np.sum(self.optimizer.last_diagnostics["pad_costs"])),
            "mediapipe_kp": target.mano_joints.copy(),
        })
        return filtered, diagnostics

    def reset(self) -> None:
        if self._is_manus:
            self.optimizer.reset()
        elif self._is_mano_pad:
            self.optimizer.last_qpos = None
            if self._pad_filter is not None:
                self._pad_filter.reset()
        else:
            self.retargeter.reset()

__all__ = ["TuningRuntime"]
