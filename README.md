# ldjy-retargeting

面向 LDJY 五指灵巧手的手部姿态重定向项目。它将 MediaPipe 格式的
`(21, 3)` 人手关键点转换为 LDJY 手的 20 个关节角，并在 MuJoCo 中验证结果。

本版本仅包含算法、仿真、回放和相机输入，不包含 LDJY 真机控制。

## 快速开始

项目使用 `uv` 管理固定的 Python 3.10 环境（见 `.python-version`），并默认使用
清华 PyPI 镜像下载依赖。USB 摄像头是默认实时输入设备，OpenCV 设备索引默认是 `0`。

```bash
uv sync --extra tuning
uv run --no-sync python example/teleop_sim.py --webcam --camera-index 0 --hand right --show-video
```

实时调节相机输入和重定向参数时，使用图形调参工具。它会打开一个含 MediaPipe 检测画面的
参数窗口，以及一个独立的 MuJoCo debug 窗口：

```bash
uv sync --extra gui --extra tuning
uv run --no-sync python example/tuning_gui.py
```

在窗口顶部选择算法、`Webcam MediaPipe` 或 `Webcam WiLoR`、USB 相机和手侧，然后点击“应用输入”。
命令行
`--webcam`、`--webcam-wilor`、`--camera-index` 和 `--hand` 仅保留为初始选择兼容参数，不再是日常启动所需。
例如旧脚本可继续使用 `python example/tuning_gui.py --webcam --camera-index 0 --hand right`，但推荐直接启动 GUI。

GUI 相机预览按检测帧更新；独立 MuJoCo debug 窗口按 120 Hz 刷新，并以 2 ms 子步保持
500 Hz 平均物理积分，因此手模型的物理时间与现实时间保持同步。

不接相机时，可以使用仓库自带的 21 点回放数据验证完整链路：

```bash
uv run --no-sync python example/teleop_sim.py --play example/data/avp1.pkl --hand left
```

### WiLoR 实时 USB 摄像头

安装 `wilor` extra 后，可以直接以 WiLoR `fast` 模式从 USB 摄像头重定向。该模式使用 CUDA、FP16 与
backbone block skipping；模型在后台线程推理，MuJoCo 控制线程始终读取最近完成的一帧，不会等待模型。

```bash
uv run --extra wilor python example/teleop_sim.py \
  --input webcam_wilor --camera-index 0 --hand left --show-video
```

`--hand` 选择 WiLoR 输出的物理左右手。短暂丢失目标侧时会保持最后一帧有效 MANO 21 点；若同侧出现多个
候选，选择画面检测框最大的一个。`--show-video` 窗口显示 WiLoR 检测框、当前目标侧和实际推理 FPS。
WiLoR 输入已是米制 MANO 关键点，不经过 MediaPipe 的 `z_scale`、0.09 m 归一化或骨段修正。

## 功能

- AdaptiveOptimizerAnalytical：默认算法。在整手姿态和捏合指尖目标间连续切换。
- 输入：pkl 回放、Vision Pro、视频、USB 摄像头、RealSense、ZED。
- 实时调参：PySide6 参数面板、MediaPipe 相机预览和 MuJoCo debug 叠加。
- 虚拟末端调节：每根手指可沿末节纵向与指甲-指肚厚度方向调整 task tip，并同步到优化器和仿真。
- LDJY 资产：内置 20 自由度 URDF、左右手 MJCF 和网格，不依赖 Git 子模块。
- OpenArm 仿真：固定根 54 自由度双臂模型；选中一侧手时自动举起并保持该臂，仅重定向该手的 20 个关节。

## 资产重建

默认模型是 MANO wrist 坐标对齐的左右手生成资产，而不是原始 CAD `palm` 坐标。原始
单手 URDF、生成脚本、坐标契约、指腹标定与 MANO 零位拟合说明见
[指腹标定和 LDJY 零位拟合 MANO](%E6%8C%87%E8%85%B9%E6%A0%87%E5%AE%9A%E5%92%8Cldjy%E9%9B%B6%E4%BD%8D%E6%8B%9F%E5%90%88mano.md)。
更新原始 URDF 后：

```bash
uv run python tools/build_ldjy_urdf.py
uv run python tools/build_ldjy_mjcf.py
uv run python -m unittest tests.test_ldjy_asset_generation -v
```

另有固定根的 OpenArm 双臂模型。它保留原始 7+7+20+20 个可动关节；使用
`--robot openarm` 时，`--hand` 选择哪侧，哪侧手臂便进入预设举手姿态并持续保持，
只有该侧的 20 个手指关节接收重定向命令。它不控制实体手臂。OpenArm 的 MANO 对齐和
MJCF actuator 生成说明见 [OpenArm 手部资产](docs/openarm-hand-assets.md)：

```bash
uv run python tools/build_openarm_hand_urdf.py
uv run python tools/build_openarm_hand_mjcf.py
uv run python -m unittest tests.test_openarm_asset_generation -v
```

```bash
uv run --no-sync python example/teleop_sim.py \
  --webcam --camera-index 0 --hand right --robot openarm --show-video
```

## 项目结构

```text
ldjy_retargeting/
  retarget.py                     统一入口：关键点预处理、优化、滤波
  openarm_control.py              OpenArm 固定双臂 home 与选中手 20-DOF 控制组装
  mediapipe.py                    wrist 居中、掌面朝向估计和 MANO 坐标变换
  opt/                            AdaptiveOptimizerAnalytical 优化器
  robot.py                        Pinocchio 运动学封装和关节限位
  tuning/                         图形调参的参数 schema、验证、YAML 会话、记录与回放状态
  viz/                            MuJoCo debug 叠加绘制
  assets/robots/ldjy_hand/        MANO 对齐的左右 URDF、MJCF、网格和标定 YAML
  assets/robots/openarm_hand/     固定根 OpenArm 双臂、MANO task frame 与 54-DOF MJCF

example/
  teleop_sim.py                   常规仿真入口：回放、视频、USB 相机、RealSense、ZED、Vision Pro
  tuning_gui.py                   USB 实时调参与静态记录回放入口
  input_devices/                  各输入设备适配器，统一输出 MediaPipe (21, 3) 关键点
  config/                         自适应算法与视频输入 YAML 配置
  data/                           pkl 回放样例

tools/                            LDJY URDF/MJCF 资产生成与验证工具
tests/                            算法、资产、输入适配器、调参会话和 GUI CLI 回归测试
docs/                             中文开发者文档与设计/实施记录
指腹标定和ldjy零位拟合mano.md     LDJY/MANO 指腹标定、零位叠加拟合与参考参数保存
```

## 实时图形调参：`tuning_gui.py`

`example/tuning_gui.py` 是面向 USB 摄像头实时重定向的桌面调参工具，支持 MediaPipe 和 WiLoR 两条输入链路：

```text
WebcamMediaPipe -> MediaPipe (21, 3) -> Retargeter -> LDJY qpos (20) -> MuJoCo
WebcamWiLoR -> WiLoR MANO joints (21, 3) -> Retargeter -> LDJY qpos (20) -> MuJoCo
```

程序会打开两个窗口：

- **调参 GUI（PySide6）**：左侧嵌入 OpenCV/MediaPipe 检测画面，显示手部 landmarks、相机索引、手侧和 GUI 刷新率；右侧为可折叠的参数区域。
- **MuJoCo debug**：显示半透明 LDJY 模型、黄色实际关节球和青色实际连杆。张手时显示绿色 FullHandVec 的 15 条目标射线；参与捏合的手指改显示红色 TipPos 端点与 TipDir 方向箭头。

参数面板支持滑块和精确数值输入，并对每个参数提供英文键名与中文作用说明。可调运行时参数包括：

- 人手与相机尺度：`z_scale`、掌长归一化、输入骨段修正（仅 MediaPipe）。
- 15 条目标向量：五指各自 wrist 到 `PIP / DIP / TIP` 的目标射线缩放；其中每指 `TIP` 比例同时用于捏合指尖位置目标。
- 损失权重与鲁棒性：位置、末端方向、整手形状与 Huber 阈值。
- 捏合自适应：四根非拇指的 `d1 / d2` 阈值，GUI 保证 `d1 < d2`。
- 稳定与滤波：`norm_delta`、`lp_alpha`。
- 坐标残差校正：设备残余旋转、wrist 与拇指偏移。
- 高级机械约束：超伸、PIP-DIP 耦合和拇指 PIP 项；默认关闭，应有 LDJY 实测依据后再使用。

注意：15 条 `segment_scaling` 缩放的是从 `retarget_wrist` 到人手关键点的**目标射线**，并不修改 URDF/MJCF 的真实连杆长度。
每指的 `TIP` 比例同时用于 FullHandVec 和捏合 TipPos，因此两种模式不会再对同一指尖使用两套长度。

### 自动零位标定 15 条向量

“15 条目标向量”页顶部提供“自动零位标定”工具。将手掌朝向相机，保持手腕稳定、五指自然张开并尽量伸直，然后点击“开始采集（45 帧）”。

程序会在最多 3 秒内采集 45 个有效帧，在 MANO 坐标中计算五指的 `wrist -> PIP / DIP / TIP` 射线长度，并与 LDJY 所有关节为零时同一组 `retarget_wrist` 任务射线长度比较：

```text
segment_scaling = LDJY 零位射线长度 / 人手 45 帧长度中位数
```

15 个建议比例必须全部位于 `[0.5, 1.5]`。任一比例超出范围、关键点无效或采样超时，整次标定都会被拒绝，当前滑块与 YAML 均不会改变；请重新摆好自然张开姿势再试。

标定成功后，15 个滑块会立即更新，MuJoCo 仿真也会使用新值，但 YAML 仍不会自动写入。确认效果后点击“保存 YAML”才会持久化配置。

MuJoCo debug 窗口可独立切换“显示骨架”和“显示射线”。点击“暂停”会冻结视频、优化器和物理仿真；此时仍可移动 `tip_offsets` 滑块以静态检查虚拟 task tip，其余优化参数会在恢复运行后生效。

### 安装和启动

首次运行需要安装 GUI 和 MuJoCo 可选依赖；MediaPipe 与 OpenCV 已是默认依赖：

```bash
uv sync --extra gui --extra tuning
```

启动后在顶部选择 USB 摄像头 `0`、手侧和算法，再点击“应用输入”：

```bash
uv run --no-sync python example/tuning_gui.py
```

WiLoR 实时输入使用 `Adaptive Analytical (WiLoR 21 点)`。它不会读取或显示 MediaPipe 专用的
`video_input.z_scale`、`reference_wrist_to_mid_mcp`、`correct_segments`。WiLoR 模式的调参窗口默认显示无检测框的原始相机画面；MuJoCo 控制行的
“MANO”按钮可叠加相机对齐的半透明 MANO 网格。网格仅在 WiLoR 推理得到新帧时更新，短暂丢失目标手时
保持最后有效网格。点击“暂停”会等待正在执行的一帧 WiLoR 推理完成，然后停止后续相机读取和 CUDA
推理；恢复“开始”后才继续。首次使用 WiLoR 时安装额外依赖：

```bash
uv sync --extra gui --extra tuning --extra wilor
uv run --no-sync python example/tuning_gui.py
```

GUI 的“末端任务点”分区提供五根手指各两个偏移：纵向沿 `PIP -> DIP`，厚度沿指甲盖到指肚。
调节时会在 `.cache/tip_tuning/` 生成临时 URDF/MJCF，优化器和 MuJoCo debug 使用同一份缓存资产，
不会覆盖正式模型。`保存 YAML` 仅保存调参配置；确认后点击“导出正式资产”，才会更新
`retarget_tip_offsets.yaml` 并重建左右独立手和 OpenArm 双臂资产。

常用选项：

```bash
# 指定自己的自适应配置
uv run --no-sync \
  python example/tuning_gui.py --config config/adaptive_analytical_video.yaml
```

首版 GUI 支持 USB 实时输入与自身的静态调参记录回放。实时模式的“开始记录”会保存每个完成推理对应的一帧
视频与结果到 `outputs/tuning_records/{mediapipe,wilor}/<日期时间>/`；进入顶部的“静态调参记录”模式后可选择
同类记录。回放锁定录制手侧，不再运行检测模型；MediaPipe 会按当前 `video_input` 参数重新预处理原始点，
WiLoR 直接读取保存的 MANO 数据。底层仍依赖 `InputDeviceBase` 的标准 `(21, 3)` 接口。

### 当前 YAML 与默认 YAML

拖动控件会立即更新内存中的 `Retargeter`，并清空滤波与优化器 warm start，便于直接观察新参数效果；它不会自动写入 YAML。

- **保存 YAML**：第一次保存前创建 `adaptive_analytical_video.yaml.original.yaml` 基线备份，之后不会覆盖此备份；当前配置以原子替换写回 YAML。
- **恢复默认**：载入 `.original.yaml` 基线并立即应用到仿真，但再次点击“保存 YAML”前不会覆盖当前 YAML。

因此，GUI 只维护当前应用的 YAML 与最初的默认 YAML 两份配置。首次使用“恢复默认”前至少应先保存一次配置。

更多内容：

- [指腹标定和 LDJY 零位拟合 MANO](%E6%8C%87%E8%85%B9%E6%A0%87%E5%AE%9A%E5%92%8Cldjy%E9%9B%B6%E4%BD%8D%E6%8B%9F%E5%90%88mano.md)
- [飞书项目记录](https://zcn3p0621d6z.feishu.cn/wiki/RdHqwXGQ4iKmsIkSL3Kcd666nhg)
- [仓库功能说明](docs/repository-features.md)
- [启动方式](docs/startup-modes.md)
- [自适应算法与 LDJY 关节映射](docs/adaptive-algorithm.md)
- [重定向算法：术语、数学与 GUI 参数](docs/retargeting-algorithm-guide.md)
- [新输入设备接入](docs/new-device-integration.md)
- [OpenArm 双臂手部资产](docs/openarm-hand-assets.md)

## 常用命令

```bash
# USB 摄像头图形调参 / 静态记录回放
uv run --no-sync python example/tuning_gui.py

# MP4 视频
uv run --no-sync python example/teleop_sim.py --video <VIDEO.mp4> --hand right --show-video

# RealSense / ZED
uv sync --extra tuning --extra realsense
uv run --no-sync python example/teleop_sim.py --realsense --hand right

uv sync --extra tuning --extra zed
uv run --no-sync python example/teleop_sim.py --zed --hand right

# Vision Pro 流
uv run --no-sync python example/teleop_sim.py --input visionpro --ip <IP> --hand right
```
