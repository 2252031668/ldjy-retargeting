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

图形调参工具目前先支持 USB 摄像头。它将 OpenCV/MediaPipe 的 2D 检测画面嵌入 PySide6
窗口，右侧提供可折叠的参数分区、滑块和数值输入；MuJoCo debug 作为独立窗口显示半透明
LDJY 手、黄色实际关节球、青色实际连杆。张手时的 FullHandVec 以绿色 15 条射线显示；
捏合手指改为红色 TipPos 端点与 TipDir 方向箭头，后者终点锚定在 TipPos 端点。

相机预览保持检测器自身的帧率；MuJoCo debug 以 120 Hz 调度显示与控制，并在每个 tick 中
执行 4 或 5 个 2 ms 物理子步，平均物理频率为 500 Hz。

```bash
uv sync --extra gui --extra tuning
uv run --no-sync \
  python example/tuning_gui.py --webcam --camera-index 0 --hand right
```

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
