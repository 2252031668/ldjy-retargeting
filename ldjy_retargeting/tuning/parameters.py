"""Runtime-tunable parameter metadata and YAML configuration validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from ldjy_retargeting.retarget_tip_frames import FINGERS as ASSET_FINGERS, normalize_tip_offsets


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SEGMENTS = ("pip", "dip", "tip")


@dataclass(frozen=True)
class ParameterSpec:
    """One GUI-editable runtime parameter."""

    path: str
    group: str
    label: str
    description_zh: str
    effect_zh: str
    value_type: type
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None


_DEFAULT_RETARGET = {
    "huber_delta": 2.0,
    "huber_delta_dir": 0.5,
    "norm_delta": 0.06,
    "thumb_skip_pip": False,
    "w_hyper": 0.0,
    "soft_min": 0.0,
    "w_couple": 0.0,
    "couple_ratio": 0.7,
    "w_pos": 1.0,
    "w_dir": 10.0,
    "w_full_hand": 1.0,
    "project_tip_dir": False,
    "lp_alpha": 0.15,
    "mediapipe_rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
    "wrist_offset_cm": [0.0, 0.0, 0.0],
    "thumb_offset_cm": [0.0, 0.0, 0.0],
}
_DEFAULT_VIDEO_INPUT = {
    "z_scale": 2.5,
    "correct_segments": True,
    "reference_wrist_to_mid_mcp": 0.09,
}
_DEFAULT_SEGMENT_SCALING = {
    "thumb": [0.97, 1.0, 1.0],
    "index": [1.0, 1.03, 1.10],
    "middle": [1.0, 1.0, 1.0],
    "ring": [1.0, 1.0, 1.0],
    "pinky": [1.05, 1.15, 1.15],
}
_DEFAULT_PINCH_THRESHOLDS = {
    finger: {"d1": 2.0, "d2": 5.0}
    for finger in FINGERS[1:]
}
_DEFAULT_TIP_OFFSETS = {
    finger: {"axis_mm": 0.0, "surface_mm": 0.0} for finger in ASSET_FINGERS
}


def _spec(
    path: str,
    group: str,
    label: str,
    description_zh: str,
    effect_zh: str,
    value_type: type = float,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        path, group, label, description_zh, effect_zh,
        value_type, minimum, maximum, step,
    )


def parameter_specs() -> tuple[ParameterSpec, ...]:
    """Return all runtime controls in UI display order."""
    specs = [
        _spec("video_input.z_scale", "人手与相机尺度", "z_scale",
              "单目 MediaPipe 深度放大倍率。", "增大后前后方向变化更明显。", minimum=0.5, maximum=5.0, step=0.05),
        _spec("video_input.reference_wrist_to_mid_mcp", "人手与相机尺度", "reference_wrist_to_mid_mcp",
              "输入人手 wrist 到中指 MCP 的归一化长度，单位米。", "增大后整只目标手的尺度变大。", minimum=0.04, maximum=0.16, step=0.001),
        _spec("video_input.correct_segments", "人手与相机尺度", "correct_segments",
              "是否用标准人体比例修正 MediaPipe 的各指骨段。", "关闭后直接使用检测到的骨段比例。", bool),
    ]
    finger_labels = {"thumb": "拇指", "index": "食指", "middle": "中指", "ring": "无名指", "pinky": "小拇指"}
    segment_labels = {"pip": "PIP", "dip": "DIP", "tip": "TIP"}
    for finger in FINGERS:
        for segment in SEGMENTS:
            specs.append(_spec(
                f"retarget.segment_scaling.{finger}.{segment}", "15 条目标向量",
                f"{finger_labels[finger]} {segment_labels[segment]}",
                f"缩放 wrist 到{finger_labels[finger]} {segment_labels[segment]}关键点的目标射线，不改变 URDF 连杆长度。",
                "增大后该目标点沿当前 wrist 射线远离 wrist。",
                minimum=0.5, maximum=1.5, step=0.01,
            ))
    specs.extend([
        _spec("retarget.huber_delta", "损失权重与鲁棒性", "huber_delta",
              "位置误差 Huber 阈值，单位厘米。", "增大后对较大位置误差更宽容。", minimum=0.05, maximum=20.0, step=0.05),
        _spec("retarget.huber_delta_dir", "损失权重与鲁棒性", "huber_delta_dir",
              "末端方向误差 Huber 阈值。", "增大后对较大方向误差更宽容。", minimum=0.01, maximum=2.0, step=0.01),
        _spec("retarget.w_pos", "损失权重与鲁棒性", "w_pos",
              "TipDirVec 中 wrist 到指尖位置项的权重。", "增大后捏合时更优先匹配指尖位置。", minimum=0.0, maximum=20.0, step=0.1),
        _spec("retarget.w_dir", "损失权重与鲁棒性", "w_dir",
              "TipDirVec 中末端方向项的权重。", "增大后捏合时更优先匹配末端朝向。", minimum=0.0, maximum=50.0, step=0.1),
        _spec("retarget.w_full_hand", "损失权重与鲁棒性", "w_full_hand",
              "FullHandVec 的整体权重。", "增大后张手时更优先保持各指整体形状。", minimum=0.0, maximum=20.0, step=0.1),
    ])
    for finger, label in finger_labels.items():
        if finger == "thumb":
            continue
        specs.extend([
            _spec(f"retarget.pinch_thresholds.{finger}.d1", "捏合自适应", f"{label} d1",
                  "拇指与该指距离低于此值时进入最大捏合混合，单位厘米。", "增大后更早进入捏合模式。", minimum=0.1, maximum=15.0, step=0.1),
            _spec(f"retarget.pinch_thresholds.{finger}.d2", "捏合自适应", f"{label} d2",
                  "拇指与该指距离高于此值时回到整手模式，单位厘米。", "增大后捏合模式覆盖更大距离。", minimum=0.2, maximum=20.0, step=0.1),
        ])
    specs.extend([
        _spec("retarget.norm_delta", "稳定与滤波", "norm_delta",
              "相邻帧关节角变化的正则权重。", "增大后动作更平稳但更迟缓。", minimum=0.0, maximum=1.0, step=0.001),
        _spec("retarget.lp_alpha", "稳定与滤波", "lp_alpha",
              "输出关节角低通滤波系数。", "减小后更平滑但延迟更高。", minimum=0.01, maximum=1.0, step=0.01),
    ])
    for axis in "xyz":
        specs.append(_spec(
            f"retarget.mediapipe_rotation.{axis}", "坐标残差校正", f"rotation {axis}",
            f"MANO 任务坐标中绕 {axis.upper()} 轴的设备残余旋转，单位度。", "只用于稳定的设备残差，不用于修复资产坐标。",
            minimum=-45.0, maximum=45.0, step=0.5,
        ))
    for target, label in (("wrist_offset_cm", "wrist"), ("thumb_offset_cm", "thumb")):
        for axis in "xyz":
            specs.append(_spec(
                f"retarget.{target}.{axis}", "坐标残差校正", f"{label} offset {axis}",
                f"{label} 相关关键点沿 {axis.upper()} 轴的残余平移，单位厘米。", "用于小幅设备偏差校正。",
                minimum=-5.0, maximum=5.0, step=0.05,
            ))
    specs.extend([
        _spec("retarget.thumb_skip_pip", "高级机械约束", "thumb_skip_pip",
              "是否在 FullHandVec 中忽略拇指 PIP 位置项。", "启用后拇指仅由 DIP 和指尖项约束。", bool),
        _spec("retarget.w_hyper", "高级机械约束", "w_hyper",
              "PIP/DIP 超伸软约束权重。", "增大后更强地抑制低于 soft_min 的屈曲角。", minimum=0.0, maximum=100.0, step=0.1),
        _spec("retarget.soft_min", "高级机械约束", "soft_min",
              "超伸软约束开始作用的关节角，单位弧度。", "增大后更早抑制反向弯曲。", minimum=-1.57, maximum=1.57, step=0.01),
        _spec("retarget.w_couple", "高级机械约束", "w_couple",
              "DIP 与 PIP 耦合软约束权重。", "增大后 DIP 更接近 couple_ratio 乘以 PIP。", minimum=0.0, maximum=100.0, step=0.1),
        _spec("retarget.couple_ratio", "高级机械约束", "couple_ratio",
              "DIP 相对 PIP 的目标耦合比例。", "增大后 DIP 目标弯曲比例更高。", minimum=0.0, maximum=1.5, step=0.01),
    ])
    asset_labels = {"thumb": "拇指", "finger1": "食指", "finger2": "中指", "finger3": "无名指", "finger4": "小拇指"}
    for finger in ASSET_FINGERS:
        specs.extend([
            _spec(f"tip_offsets.{finger}.axis_mm", "末端任务点", f"{asset_labels[finger]} 纵向",
                  "沿 PIP 到 DIP 的末节射线移动虚拟 tip，单位毫米。", "正值朝指尖，负值朝指根。", minimum=-15.0, maximum=15.0, step=0.1),
            _spec(f"tip_offsets.{finger}.surface_mm", "末端任务点", f"{asset_labels[finger]} 厚度",
                  "沿指甲盖到指肚方向移动虚拟 tip，单位毫米。", "正值按模型局部指甲-指肚轴移动。", minimum=-15.0, maximum=15.0, step=0.1),
        ])
    return tuple(specs)


def _parts(path: str) -> list[str]:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid parameter path: {path!r}")
    return parts


def _segment_path(parts: list[str]) -> tuple[str, int] | None:
    if len(parts) == 4 and parts[:2] == ["retarget", "segment_scaling"]:
        try:
            return parts[2], SEGMENTS.index(parts[3])
        except ValueError as exc:
            raise ValueError(f"Invalid segment scaling path: {'.'.join(parts)}") from exc
    return None


def _offset_path(parts: list[str]) -> tuple[str, int] | None:
    if len(parts) == 3 and parts[0] == "retarget" and parts[1] in {"wrist_offset_cm", "thumb_offset_cm"}:
        try:
            return parts[1], "xyz".index(parts[2])
        except ValueError as exc:
            raise ValueError(f"Invalid offset path: {'.'.join(parts)}") from exc
    return None


def get_path(config: dict[str, Any], path: str) -> Any:
    """Read a runtime parameter using the GUI's stable dotted path."""
    parts = _parts(path)
    segment = _segment_path(parts)
    if segment is not None:
        finger, index = segment
        return config["retarget"]["segment_scaling"][finger][index]
    offset = _offset_path(parts)
    if offset is not None:
        name, index = offset
        return config["retarget"][name][index]
    value: Any = config
    for part in parts:
        value = value[part]
    return value


def set_path(config: dict[str, Any], path: str, value: Any) -> None:
    """Write a runtime parameter while retaining the YAML's list representation."""
    parts = _parts(path)
    segment = _segment_path(parts)
    if segment is not None:
        finger, index = segment
        config["retarget"]["segment_scaling"][finger][index] = value
        return
    offset = _offset_path(parts)
    if offset is not None:
        name, index = offset
        config["retarget"][name][index] = value
        return
    parent: Any = config
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value


def _merge_defaults(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"{key} must be a mapping")
            _merge_defaults(child, value)
        elif key not in target:
            target[key] = value.copy() if isinstance(value, list) else value


def normalize_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Inject supported runtime defaults into a mutable YAML configuration."""
    retarget = config.setdefault("retarget", {})
    if not isinstance(retarget, dict):
        raise ValueError("retarget must be a mapping")
    # Legacy global TipPos scaling conflicts with the calibrated TIP entry in
    # segment_scaling. Drop it on load so a subsequent save migrates the YAML.
    retarget.pop("scaling", None)
    _merge_defaults(retarget, _DEFAULT_RETARGET)
    segment_scaling = retarget.setdefault("segment_scaling", {})
    if not isinstance(segment_scaling, dict):
        raise ValueError("retarget.segment_scaling must be a mapping")
    _merge_defaults(segment_scaling, _DEFAULT_SEGMENT_SCALING)
    pinch_thresholds = retarget.setdefault("pinch_thresholds", {})
    if not isinstance(pinch_thresholds, dict):
        raise ValueError("retarget.pinch_thresholds must be a mapping")
    _merge_defaults(pinch_thresholds, _DEFAULT_PINCH_THRESHOLDS)
    video_input = config.setdefault("video_input", {})
    if not isinstance(video_input, dict):
        raise ValueError("video_input must be a mapping")
    _merge_defaults(video_input, _DEFAULT_VIDEO_INPUT)
    tip_offsets = config.setdefault("tip_offsets", {})
    if not isinstance(tip_offsets, dict):
        raise ValueError("tip_offsets must be a mapping")
    _merge_defaults(tip_offsets, _DEFAULT_TIP_OFFSETS)
    config["tip_offsets"] = normalize_tip_offsets(config["tip_offsets"])
    return config


def validate_runtime_config(config: dict[str, Any]) -> None:
    """Reject invalid values before they reach the live retargeter or disk."""
    normalize_runtime_config(config)
    specs = {spec.path: spec for spec in parameter_specs()}
    for path, spec in specs.items():
        value = get_path(config, path)
        if spec.value_type is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{path} must be boolean")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{path} must be a finite number")
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{path} must be >= {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{path} must be <= {spec.maximum}")

    for finger in FINGERS:
        scales = config["retarget"]["segment_scaling"][finger]
        if not isinstance(scales, list) or len(scales) != 3:
            raise ValueError(f"retarget.segment_scaling.{finger} must contain PIP, DIP, TIP")
    for finger in FINGERS[1:]:
        thresholds = config["retarget"]["pinch_thresholds"][finger]
        if thresholds["d1"] >= thresholds["d2"]:
            raise ValueError(f"retarget.pinch_thresholds.{finger}.d1 must be less than d2")


__all__ = [
    "FINGERS",
    "SEGMENTS",
    "ParameterSpec",
    "get_path",
    "normalize_runtime_config",
    "parameter_specs",
    "set_path",
    "validate_runtime_config",
]
