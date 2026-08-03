# 启动方式

所有命令从仓库根目录执行。首次安装仿真与普通相机依赖：

```bash
uv sync --extra tuning
```

## 回放

`example/data/avp1.pkl` 是按帧存储的 21 点数据，适合不连接设备时回归算法。

```bash
uv run python example/teleop_sim.py --play example/data/avp1.pkl --hand left
```

## USB 摄像头

默认使用设备 `0`：

```bash
uv run python example/teleop_sim.py --webcam --camera-index 0 --hand right --show-video
```

## WiLoR 实时 USB 摄像头

需要先安装可选依赖：

```bash
uv sync --extra wilor
uv run --extra wilor python example/teleop_sim.py \
  --input webcam_wilor --camera-index 0 --hand left --show-video
```

此模式在后台以 WiLoR `fast`（CUDA FP16）重建 MANO 21 点。`--hand` 选择物理左右手；短暂丢失时保持
上一帧有效关键点。它不使用 MediaPipe 单目尺度修正。

## OpenArm 双臂仿真

OpenArm 是固定根的 54 自由度仿真模型。`--hand` 选择左或右手时，对应手臂会进入并保持
预设举手姿态；重定向算法仍只计算并控制这一侧的 20 个手指关节。另一侧手臂和手保持零位。
该模式不连接或控制实体 OpenArm。

仿真入口以 120 Hz 调度控制与渲染；MuJoCo 仍使用 2 ms 内部积分，并在每个调度 tick 中
执行 4 或 5 个子步，平均物理频率为 500 Hz。

```bash
uv run python example/teleop_sim.py \
  --webcam --camera-index 0 --hand left --robot openarm --show-video

uv run python example/teleop_sim.py \
  --webcam --camera-index 0 --hand right --robot openarm --debug
```

### 图形调参 GUI

图形调参工具支持 USB 摄像头的 MediaPipe 与 WiLoR 输入。它将 OpenCV 检测画面嵌入 PySide6
窗口，右侧提供可折叠的参数分区、滑块和数值输入；MuJoCo debug 作为独立窗口显示半透明
LDJY 手、黄色实际关节球、青色实际连杆。张手时的 FullHandVec 以绿色 15 条射线显示；
捏合手指改为红色 TipPos 端点与 TipDir 方向箭头，后者终点锚定在 TipPos 端点。

相机预览保持检测器自身的帧率；MuJoCo debug 以 120 Hz 调度显示与控制，并在每个 tick 中
执行 4 或 5 个 2 ms 物理子步，平均物理频率为 500 Hz。

```bash
uv sync --extra gui --extra tuning
uv run --no-sync python example/tuning_gui.py
```

窗口顶部先选择重定向算法、输入设备、USB 相机索引和手侧，再点击“应用输入”才会打开相机并创建重定向会话。
`--webcam --camera-index 0 --hand right` 仍可作为初始选择兼容参数，但不再需要每次输入。

WiLoR 使用 `Adaptive Analytical (WiLoR 21 点)`，并隐藏只影响 MediaPipe 单目预处理的 `video_input.z_scale`、
`reference_wrist_to_mid_mcp`、`correct_segments`：

```bash
uv sync --extra gui --extra tuning --extra wilor
uv run --no-sync python example/tuning_gui.py
```

WiLoR 调参预览默认使用不含检测框的原始相机画面。MuJoCo 控制行额外提供“MANO”开关：开启后在
同一画面叠加与 WiLoR 相机参数对齐的半透明网格；它只绘制 `--hand` 对应的手，并在短暂丢失时保留
最后有效网格。“暂停”会等待当前 WiLoR 推理帧结束，随后阻塞相机读取和 CUDA 推理；网格以 OpenCV
投影叠加，只在 WiLoR 产生新推理帧时更新。

### 调参记录与静态回放

实时模式点击“开始记录”后，GUI 为每一次完成的推理保存一帧 BGR 视频和同序的结果；再次点击停止并原子
完成记录。MediaPipe 记录保留原始归一化 21 点和录制时的预处理结果，WiLoR 记录保留选中手的 MANO
关节、姿态、形状、网格和相机参数。记录位于：

```text
outputs/tuning_records/mediapipe/YYYY-MM-DD_HH-MM-SS/
outputs/tuning_records/wilor/YYYY-MM-DD_HH-MM-SS/
```

选择顶部“静态调参记录”后，输入下拉框变为记录类型筛选器；只显示完整记录，且回放手侧锁定为录制时的手侧。
可用首帧、前后帧、开始/暂停和时间轴浏览。回放不运行 MediaPipe/WiLoR 推理：MediaPipe 会把已保存的原始
点重新经过当前 `z_scale`、掌长归一化和骨段修正；WiLoR 则直接使用保存的 MANO 关节。两者都会使用当前
GUI 的重定向参数重新计算 LDJY 姿态，记录目录中的 `config_snapshot.yaml` 只用于追溯，不会覆盖当前 YAML。

参数修改会立即应用到内存中的重定向器，但不会自动覆盖 YAML：

- `保存 YAML`：第一次保存前创建一次
  `adaptive_analytical_video.yaml.original.yaml` 基线，之后不会覆盖该文件；再以原子替换写入
  当前 YAML。
- `恢复默认`：读取 `.original.yaml` 基线并实时应用，但只有再次点击保存才会覆盖当前 YAML。

首次使用“恢复默认”前，必须先成功保存一次，以创建原始基线。

GUI 可独立切换“显示骨架”和“显示射线”。“暂停”会冻结相机预览、重定向优化和 MuJoCo
物理，但 `tip_offsets` 仍会立即更新缓存 MJCF 中的虚拟 task tip，方便静态观察；恢复“开始”
后其余参数和实时输入继续运行。

### 虚拟末端任务点

“末端任务点”包含五根手指各两项偏移，单位均为毫米：纵向沿 `PIP -> DIP`，厚度沿指甲盖到
指肚。滑块会生成 `.cache/tip_tuning/` 临时 URDF/MJCF，优化器与 MuJoCo debug 使用同一份缓存
资产。`保存 YAML` 不会改资产；点击“导出正式资产”才会写入
`ldjy_retargeting/assets/robots/ldjy_hand/retarget_tip_offsets.yaml` 并生成左右 LDJY 与 OpenArm 资产。

增加 `--debug` 可显示黄色球/青色连杆的物理实际姿态。绿色细线表示 FullHandVec 目标；
捏合指的红色 TipPos 红球和 TipDir 箭头表示当前捏合目标。终端会每秒输出每指的损失模式、
命令角、实际角和误差。

```bash
uv run python example/teleop_sim.py --webcam --camera-index 0 --hand right --debug
```

## 视频、RealSense、ZED、Vision Pro

```bash
uv run python example/teleop_sim.py --video <VIDEO.mp4> --hand right --show-video

uv sync --extra tuning --extra realsense
uv run python example/teleop_sim.py --realsense --hand right

uv sync --extra tuning --extra zed
uv run python example/teleop_sim.py --zed --hand right

uv run python example/teleop_sim.py --input visionpro --ip <IP> --hand right
```

相机类输入会使用 `adaptive_analytical_video.yaml`，回放与 Vision Pro 默认使用
`adaptive_analytical_avp.yaml`。可通过 `--config` 指定自己的配置文件。

本仓库不提供真机控制启动方式。请先在 MuJoCo 中完成关节方向、限位和参数验证。
