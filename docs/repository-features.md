# 仓库功能说明

`ldjy-retargeting` 将标准 MediaPipe 21 点人手姿态重定向为 LDJY 手的
20 个关节角。当前交付边界是仿真和算法验证，不发送任何真机控制命令。

## 数据流

```text
输入适配器 / pkl 回放
  -> left_fingers 或 right_fingers: (21, 3)，单位米
  -> wrist 居中、掌面坐标系和 MANO 轴约定
  -> AdaptiveOptimizerAnalytical
  -> LDJY qpos: (20,)
  -> MuJoCo 仿真或调参可视化
```

## 主要模块

| 模块 | 职责 |
| --- | --- |
| `ldjy_retargeting.Retargeter` | 坐标变换、可选校正、优化和低通滤波的统一入口。 |
| `ldjy_retargeting.mediapipe` | 以 wrist 为原点，用 landmarks 0、5、9 估计掌面朝向并转换到 MANO 约定。 |
| `ldjy_retargeting.opt` | 使用 Pinocchio FK/Jacobian 和 NLopt 在限位内求解 20 维关节角。 |
| `ldjy_retargeting.assets` | LDJY 的 URDF、左右手 MJCF 和网格。 |
| `ldjy_retargeting.viz` | MuJoCo 中的实际姿态和目标向量 debug 叠加。 |
| `ldjy_retargeting.tuning` | 图形调参所用的参数元数据、范围验证、内存会话和 YAML 安全保存。 |

## 输入

所有输入最后都返回同一字典：`left_fingers` 和 `right_fingers`，每项为
`(21, 3)` 浮点数组。支持 pkl 回放、Vision Pro、视频、USB 摄像头、RealSense
和 ZED。视频与相机适配器使用 MediaPipe 检测关键点；回放与 Vision Pro 负责
各自数据到该格式的转换。

## 实时调参 GUI

`example/tuning_gui.py` 首先支持 USB 摄像头。Webcam 输入设备在已有采集线程中同时提供
标准 `(21, 3)` 关键点和带 landmarks 的 BGR 预览帧，因此 GUI 不会第二次打开相机。PySide6
参数面板的改动以短去抖实时重建 `Retargeter`，并清空滤波与 warm start；MuJoCo 独立窗口继续
显示当前 debug 叠加。其他输入设备只要实现可选的 `get_preview_frame()`，即可复用同一 GUI
和 `(21, 3) -> Retargeter` 数据链路。

调参 GUI 将尺度统一为 15 条 `segment_scaling`：每根手指的 `PIP / DIP / TIP` 位置比例。
其中 `TIP` 比例同时供整手 FullHandVec 和捏合 TipPos 使用。MuJoCo debug 在整手模式显示
绿色目标射线，在捏合模式对参与手指显示红色 TipPos 端点和终点锚定于该端点的 TipDir 箭头。

## LDJY 模型与关节映射

LDJY 的左右 URDF 和 MJCF 都使用 `left_` 或 `right_` 前缀，例如
`right_finger1_joint1`。`joint_mapping.qpos_reorder_perm` 会先规范化此前缀，
再按名称重排 qpos，保证仿真中的关节槽位正确。

五指的算法顺序始终是：拇指、食指、中指、无名指、小指。它映射到 LDJY 的
`thumb`、`finger1`、`finger2`、`finger3`、`finger4`。默认任务帧为
`retarget_wrist`、每指 `link3`、`link4` 和 `tip`。
