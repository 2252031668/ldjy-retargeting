# MANUS 无损输入与全骨架重定向

## 数据链路

`ManusGlove` 以 Core 3.2 Integrated Mode 初始化，坐标系固定为 `XFromViewer / +Z / Right-handed / unitScale=1 / world coordinates`。它订阅 Raw Skeleton、Ergonomics、Landscape、System 和连接状态。CFFI 数据在 SDK 回调中立即复制到 `ManusFrame`；GUI 线程只读取已复制的帧，超过 0.5 秒的数据无效。

Raw Full Skeleton 与 Ergonomics Hybrid 可离线比较：

```bash
uv run python example/compare_manus_retargeting.py outputs/tuning_records/manus/<timestamp>
```

`ManusFrame` 保留本地单调时钟、SDK 时间戳、glove ID、手侧、所有节点的位置/四元数/缩放、完整 `NodeInfo` 拓扑，以及最近一份 Ergonomics 数组及其独立时间戳。节点匹配只使用 `ChainType + FingerJointType + parentId`，不依赖节点编号、数组顺序或固定节点数。

## 坐标与中立姿态

每帧先用唯一的 `Hand` 节点变换到腕部局部坐标。中立姿态采集至少 2 秒和 50 个唯一、同拓扑帧；每个节点位置中位偏差不得超过 3 mm，旋转中位偏差不得超过 5°。四元数先统一符号再平均。

标定以对应语义节点对 MANUS 与 LDJY 零位执行无缩放 Kabsch 旋转对齐，并保存每节点固定旋转偏置。原子 NPZ 路径为 `outputs/manus_calibrations/<glove_id>_<side>.npz`；glove ID、手侧或拓扑不匹配时拒绝加载。

## 全骨架目标与求解

语义映射为 `Hand`、五指 `Metacarpal/Proximal/Intermediate/Distal/Tip` 对应 LDJY wrist、link1–4 和 tip。当前人体骨段只提供方向；目标位置从腕部沿 MANUS 拓扑、使用 LDJY 零位骨段长度重建，因此不把人体骨长复制给机器人。原始位置仍完整保存在帧和 diagnostics。

节点旋转目标为 `R_target = A · R_manus_wrist_local · Aᵀ · C_node`。

`scipy.optimize.least_squares` 使用 Huber loss、URDF 硬限位、上一帧 warm start 与 Pinocchio frame Jacobian。默认损失在 `example/config/manus_full_skeleton.yaml`：位置 10 mm/1.0、方向 0.1/0.5、旋转 10°/0.5、帧间 0.1 rad/0.05、零位 0.001、最多 30 次评估和可调 `lp_alpha`。失败或非有限结果保持上一有效姿态；首帧使用零位。

## 可视化、录制与回放

MuJoCo overlay 绘制全部语义目标节点、父子边、目标到实际位置误差线和三轴旋转。状态栏显示 glove、手侧、连接、接收/GUI FPS、求解耗时和标定状态。

记录位于 `outputs/tuning_records/manus/<timestamp>`。拓扑只存一次，每帧保存完整 positions、rotations、scales、SDK/本地时间戳和完整异步 Ergonomics；同时快照 YAML 与标定。回放使用原始相对时间间隔，不导入或连接 MANUS SDK。

## 命令

```bash
uv run python -m example.input_devices.manus_glove --hand right --seconds 5
uv run python example/tuning_gui.py --manus --hand right
```

第一版不包含 Remote/Local Mode、Raw IMU、触觉控制和 MANUS `.mcal` 界面。
