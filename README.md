# ldjy-retargeting

面向 LDJY 五指灵巧手的手部姿态重定向项目。它将 MediaPipe/WiLoR/Quest 的 21 点输入，
或 MANUS 的原生完整手骨架，转换为 LDJY 手的 20 个关节角，并在 MuJoCo 中验证结果。

本版本仅包含算法、仿真、回放和相机输入，不包含 LDJY 真机控制。

## 快速开始

项目使用 `uv` 管理固定的 Python 3.10 环境（见 `.python-version`），并默认使用清华 PyPI 镜像下载依赖。日常入口是图形调参工具：它会打开一个检测/状态窗口和独立的 MuJoCo debug 窗口。

```bash
uv sync --extra gui --extra tuning
uv run --no-sync python example/tuning_gui.py --webcam --hand right
```

在窗口顶部选择算法、`Webcam MediaPipe`、`Webcam WiLoR`、`Quest HTS` 或 `MANUS Raw Skeleton`，再点击“应用输入”。
命令行
`--webcam`、`--webcam-wilor`、`--quest-hts`、`--manus`、`--camera-index`、`--quest-transport`、`--quest-host`、`--quest-port` 和 `--hand` 用于设置 GUI 的初始选择；其余操作均在 GUI 内完成。

GUI 相机预览按检测帧更新；独立 MuJoCo debug 窗口按 120 Hz 刷新，直接显示当前重定向
命令的运动学姿态，用于核对 Pinocchio IK 与 MuJoCo 指腹 site 是否一致。它不模拟 actuator
跟踪误差或物理时间。

### WiLoR 实时 USB 摄像头

安装 `wilor` extra 后，可以直接以 WiLoR `fast` 模式从 USB 摄像头重定向。该模式使用 CUDA、FP16 与
backbone block skipping；模型在后台线程推理，MuJoCo 控制线程始终读取最近完成的一帧，不会等待模型。

WiLoR 源码位于 Git 子模块 `third_party/WiLoR`。首次克隆仓库时使用：

```bash
git clone --recurse-submodules https://github.com/2252031668/ldjy-retargeting.git
cd ldjy-retargeting
```

已经克隆但缺少子模块时运行：

```bash
git submodule update --init --recursive
```

`uv sync --extra wilor` 只安装 Python/CUDA 依赖，**不会自动下载模型权重**。首次使用 WiLoR 前还需要：

```bash
uv sync --extra wilor
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
  -P third_party/WiLoR/pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt \
  -P third_party/WiLoR/pretrained_models/
```

此外，WiLoR 需要受 [MANO 许可证](https://mano.is.tue.mpg.de/license.html) 约束的模型；请在
[MANO 官网](https://mano.is.tue.mpg.de) 注册并下载 `mano_v*_*.zip`，将其中的右手模型放到
`third_party/WiLoR/mano_data/MANO_RIGHT.pkl`。该文件不能由本仓库自动下载或提交。上述权重和 MANO 文件均为本机未跟踪文件，不会随主仓库或子模块 push。

```bash
uv sync --extra gui --extra tuning --extra wilor
uv run --no-sync python example/tuning_gui.py --webcam-wilor --hand left
```

`--hand` 选择 WiLoR 输出的物理左右手。短暂丢失目标侧时会保持最后一帧有效 MANO 21 点；若同侧出现多个
候选，选择画面检测框最大的一个。`--show-video` 窗口显示 WiLoR 检测框、当前目标侧和实际推理 FPS。
WiLoR 输入已是米制 MANO 关键点，不经过 MediaPipe 的 `z_scale`、0.09 m 归一化或骨段修正。

## 功能

- AdaptiveOptimizerAnalytical：默认算法。在整手姿态和捏合指尖目标间连续切换。
- ManoPadPoseOptimizer：WiLoR 专用的指腹捏合模式。固定 `beta_robot`，用机器人尺度 MANO 的五个指腹位置和表面法线驱动 LDJY 20-DOF IK；捏合时联立求解拇指和参与手指。
- 输入：pkl 回放、Vision Pro、视频、USB 摄像头、RealSense、ZED、Quest HTS、MANUS Raw Skeleton。
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
  input_devices/                  各输入设备适配器；MANUS 保留原生完整骨架而非 21 点
  config/                         自适应算法与视频输入 YAML 配置
  data/                           pkl 回放样例

tools/                            LDJY URDF/MJCF 资产生成与验证工具
tests/                            算法、资产、输入适配器、调参会话和 GUI CLI 回归测试
docs/                             中文开发者文档与设计/实施记录
指腹标定和ldjy零位拟合mano.md     LDJY/MANO 指腹标定、零位叠加拟合与参考参数保存
```

## 实时图形调参：`tuning_gui.py`

`example/tuning_gui.py` 是实时重定向的桌面调参工具，支持 MediaPipe、WiLoR、Quest HTS 和 MANUS：

```text
WebcamMediaPipe -> MediaPipe (21, 3) -> Retargeter -> LDJY qpos (20) -> MuJoCo
WebcamWiLoR -> WiLoR MANO joints (21, 3) -> Retargeter -> LDJY qpos (20) -> MuJoCo
WebcamWiLoR -> WiLoR hand_pose + beta_robot -> MANO 指腹位置/法线 + 末节表面距离 -> Pad IK -> LDJY qpos (20)
MANUS Raw Skeleton -> Full Skeleton IK 或 Ergonomics Hybrid -> LDJY qpos (20) -> MuJoCo
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

### Quest HTS 实时输入

Quest 运行 Hand Tracking Streamer (HTS) 时，安装 `quest` extra 后可在 GUI 顶部选择
`Quest HTS` 与 `Adaptive Analytical (Quest HTS 21 点)`。默认以 UDP 监听
`0.0.0.0:9000`；可在选择该输入源后调整 UDP / TCP Server / TCP Client、主机和端口，再点击“应用输入”重连：

```bash
uv sync --extra gui --extra tuning --extra quest
uv run --no-sync python example/tuning_gui.py --quest-hts
```

UDP 时，`0.0.0.0` 表示电脑监听所有本机网卡，不是 Quest 的目标地址。HTS App 的默认
`255.255.255.255:9000` 是广播；网络稳定性不足时，建议改成电脑同一局域网的 IPv4 地址和端口 `9000`，GUI 仍保持 UDP、`0.0.0.0:9000`。

HTS 也支持最稳定的 USB 有线 TCP：以数据 USB-C 线连接 Quest，头显允许 USB 调试/连接后执行：

```bash
adb devices
adb reverse tcp:8000 tcp:8000
adb reverse --list
```

然后在 HTS App 选择 TCP Wired（默认 `localhost:8000`）；GUI 选择 `TCP Server`，主机填
`127.0.0.1`，端口填 `8000`。这是通过 ADB reverse 在 USB 内转发 TCP，不需要填写 Quest 或电脑的局域网 IP；`TCP Client` 不用于此有线模式。

Quest landmarks 保持 wrist 局部坐标，只转换 Unity-left 到重定向所用的 RFU 坐标，不应用 wrist 的全局 6DoF 位姿；因此它不使用或显示 `video_input` 的尺度、深度和骨段修正参数。Quest 没有相机预览，左侧状态区只显示连接与接收 FPS。短暂丢帧或断连时保持最后有效姿态；第一版不会自动重连，修改设置后点击“应用输入”即可重新建立连接。

### MANUS 手套实时输入

MANUS 使用 Core 3.2 的 Linux Integrated Mode，不经过 MediaPipe 21 点。先完成 MANUS Core 的官方手套校准并确保当前用户的 `.mcal` 已加载；运行项目时不要同时开启 Core Dashboard 或官方 SDK Client。SDK 作为外部安装依赖，不写入项目 `pyproject.toml`：

```bash
uv pip install -e /home/wxx/manus_sdk/Python
uv run --no-sync python example/tuning_gui.py --manus --hand right
```

在 GUI 中选择 `MANUS Raw Skeleton`。可选两条已实现路径：

- `MANUS Full Skeleton`：完整 Raw Skeleton 的节点位置、旋转和拓扑进入有界 IK；每副手套每只手首次使用时，保持自然张手约 2 秒采集中立姿态。
- `MANUS Ergonomics Hybrid`：食指、中指、无名指使用 MANUS 官方 Ergonomics 角度；拇指、小指使用 Raw Skeleton 任务空间 IK。首次使用点击“开始采集扩展标定”，依次完成张手、张开、握拳和四种拇指捏合；标定会自动复用。

两类项目标定都不是 MANUS `.mcal`。它们按手套 ID、手侧和机器人 URDF 指纹保存到 `outputs/manus_calibrations/` 或 `outputs/manus_ergonomics_calibrations/`。MANUS 记录保存完整 Raw Skeleton、40 维 Ergonomics 及时间戳到 `outputs/tuning_records/manus/<日期时间>/`，无需 SDK 即可回放。更多细节见 [MANUS 快速开始](docs/manus-quickstart.md) 和 [SDK 数据与映射分析](docs/manus-sdk-analysis.md)。

选择 `MANO 指腹捏合 IK (WiLoR)` 可启用第一版指腹模式。它直接使用 WiLoR 的绝对局部
`hand_pose`，但始终使用 `mano_ldjy_reference.yaml` 中固定的机器人 `betas`；不减去
`hand_pose_ref`，也不使用 WiLoR 的 `global_orient` 和 `translation` 移动机器人 wrist。
每根手指的位置、法线、权重、平滑项和是否参与求解都可独立调整；左手在 GUI 和静态回放中均受支持。

捏合距离不再量固定的两个指腹中心点。程序在启动时根据 MANO 的蒙皮权重，为拇指和四根
手指各自固定最后一个活动关节到 tip 的完整末节三角面集合；每帧以 WiLoR MANO 网格计算
拇指末节与其余末节的面-面最短距离。该距离按人手到 `beta_robot` 的尺度比例和静态 registration
scale 换算后，用于进入捏合、两指目标间距和接触面优化；指尖、侧面或指甲面接近都可触发。
张手时仍是五根手指独立的 4-DOF 求解；进入捏合的拇指-手指链则会联合优化。

GUI 的“末端任务点”分区提供五根手指各两个偏移：纵向沿 `PIP -> DIP`，厚度沿指甲盖到指肚。
调节时会在 `.cache/tip_tuning/` 生成临时 URDF/MJCF，优化器和 MuJoCo debug 使用同一份缓存资产，
不会覆盖正式模型。`保存 YAML` 仅保存调参配置；确认后点击“导出正式资产”，才会更新
`retarget_tip_offsets.yaml` 并重建左右独立手和 OpenArm 双臂资产。

需要指定预设 YAML 时，`--config` 的相对路径以 `example/` 为基准；它必须和当前选择的输入类型兼容。例如：

```bash
uv run --no-sync python example/tuning_gui.py \
  --webcam --config config/adaptive_analytical_video.yaml --hand right
```

GUI 支持实时输入与自身的静态调参记录回放。MediaPipe/WiLoR 会把每个完成推理的相机视频和结果保存到
`outputs/tuning_records/{mediapipe,wilor}/<日期时间>/`。Quest 则保存纯数据到
`outputs/tuning_records/quest_hts/<日期时间>/`：原始 Unity-left landmarks、RFU landmarks、wrist position 与 quaternion；不创建黑色视频。MANUS 保存到 `outputs/tuning_records/manus/<日期时间>/`，包含原生节点 transform、拓扑、Ergonomics 和项目标定快照。进入顶部的“静态调参记录”模式后可选择同类记录。回放锁定录制手侧，不再运行检测模型；MediaPipe 会按当前 `video_input` 参数重新预处理原始点，WiLoR、Quest 和 MANUS 直接读取保存结果。

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

以下命令均从仓库根目录运行。日常使用的唯一入口是 `tuning_gui.py`；视频、RealSense、ZED 和 Vision Pro 的 `teleop_sim.py` 旧命令不在这里作为推荐工作流。

```bash
# 1. 默认 USB 摄像头 + MediaPipe 调参
uv sync --extra gui --extra tuning
uv run --no-sync python example/tuning_gui.py --webcam --hand right

# 2. USB 摄像头 + WiLoR 调参（首次还需要按上文下载 WiLoR/MANO 资源）
uv sync --extra gui --extra tuning --extra wilor
uv run --no-sync python example/tuning_gui.py --webcam-wilor --hand right

# 3. Quest HTS：默认 UDP 监听 0.0.0.0:9000
uv sync --extra gui --extra tuning --extra quest
uv run --no-sync python example/tuning_gui.py --quest-hts --hand right

# 4. MANUS：先确认官方 SDK 已独立安装，且此时没有运行 Dashboard / SDK Client
uv pip install -e /home/wxx/manus_sdk/Python
uv run python -m example.input_devices.manus_glove --hand right --seconds 5
uv run --no-sync python example/tuning_gui.py --manus --hand right

# 5. 运行当前 MANUS 单元测试（不需要连接手套）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_manus.py tests/test_tuning_gui_cli.py tests/test_input_record_samples.py
```

启动 GUI 后再在界面中点击“应用输入”、开始/停止记录和选择“静态调参记录”回放；目前没有把这些 GUI 操作伪装成一个不存在的单行 CLI。MANUS 录制必须先有对应的项目标定：Full Skeleton 使用中立姿态标定，Ergonomics Hybrid 使用七阶段标定。
