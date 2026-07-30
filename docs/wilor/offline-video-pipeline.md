# WiLoR 离线视频 MANO 导出设计

## 目标

在不修改现有 MediaPipe 输入、实时遥操作、调参 GUI 和 LDJY 重定向逻辑的前提下，
增加一个离线工具：输入视频，逐帧运行 WiLoR，导出 21 个三维关节、MANO 参数、网格和
相机相关数据。

## 边界

- WiLoR 作为根项目的 `wilor` optional extra，使用根目录的 uv Python 3.10 `.venv`。
- 官方 WiLoR 源码位于 `third_party/WiLoR/`，固定上游提交且不修改。
- 本仓库只维护 `tools/run_wilor_video.py` 与测试代码。
- 本阶段不创建 `InputDeviceBase` 适配器，不调用 `Retargeter`，不改变任何实时命令。
- 本阶段不做跨帧身份追踪；每帧输出 WiLoR 原始检测顺序。

## 本地资产

以下文件由使用者在本地获取，不纳入 Git：

```text
third_party/WiLoR/pretrained_models/detector.pt
third_party/WiLoR/pretrained_models/wilor_final.ckpt
third_party/WiLoR/pretrained_models/model_config.yaml
third_party/WiLoR/mano_data/MANO_RIGHT.pkl
outputs/wilor/
```

工具在启动前检查这些路径，缺失时以明确错误停止。左手沿用 WiLoR 对右手 MANO 的镜像处理，
不要求额外的左手 MANO 文件。

## 输出格式

每段视频写入一个目录：

```text
outputs/wilor/<video_stem>/
├── result.npz
└── metadata.json
```

`result.npz` 采用 `float32`，其中 `T` 为抽样帧数，`H` 为该视频单帧检测手数的最大值：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `frame_index` | `(T,)` | 原视频帧号 |
| `timestamp_sec` | `(T,)` | 原视频时间戳 |
| `valid` | `(T, H)` | 当前槽位有有效检测 |
| `is_right` | `(T, H)` | WiLoR 判定的左右手 |
| `bbox_xyxy` | `(T, H, 4)` | 检测框像素坐标 |
| `joints_mano` | `(T, H, 21, 3)` | WiLoR 的 MANO 局部 21 点 |
| `vertices_mano` | `(T, H, 778, 3)` | MANO 网格顶点 |
| `global_orient` | `(T, H, 3, 3)` | MANO 全局旋转矩阵 |
| `hand_pose` | `(T, H, 15, 3, 3)` | MANO 15 个关节旋转矩阵 |
| `betas` | `(T, H, 10)` | MANO 形状参数 |
| `pred_cam` | `(T, H, 3)` | WiLoR crop camera 参数 |
| `camera_translation` | `(T, H, 3)` | 回投到全图的相机平移 |
| `faces` | `(F, 3)` | 所有帧共享的网格面索引 |

空槽位以 `valid=false` 标识，浮点数组对应位置写入 `NaN`。`metadata.json` 保存视频大小、
FPS、处理帧数、frame stride、WiLoR 上游提交、设备和坐标说明。

`joints_mano` 保持 WiLoR 输出坐标；`camera_translation` 单独保存。两者不得在这一阶段合并，
以避免把 MANO 局部手型、相机坐标和后续 LDJY/MANO 对齐混在一起。

## 命令接口

```bash
uv sync --extra wilor
uv run --extra wilor python tools/run_wilor_video.py \
  --video input.mp4 \
  --output outputs/wilor/input
```

工具还提供 `--device {auto,cpu,cuda}`、`--frame-stride`、`--max-frames`、`--batch-size`
与 `--fast`。默认保留所有检测到的手。

## 验证

无权重测试覆盖结果归档的 shape、空帧、变长多手、JSON 元数据和命令参数校验。
有权重的 smoke test 使用一个短视频，检查输出帧数、`(21, 3)` 关节、778 顶点、15 个
MANO 姿态关节和可加载的 NPZ/JSON 文件。
