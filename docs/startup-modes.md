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

### 图形调参 GUI

图形调参工具目前先支持 USB 摄像头。它将 OpenCV/MediaPipe 的 2D 检测画面嵌入 PySide6
窗口，右侧提供可折叠的参数分区、滑块和数值输入；MuJoCo debug 作为独立窗口显示半透明
LDJY 手、黄色实际关节球、青色实际连杆和绿色任务向量。

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

增加 `--debug` 可显示黄色球/青色连杆的物理实际姿态，以及绿色细球线的当前目标
向量。终端会每秒输出每指的损失模式、命令角、实际角和误差。

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
