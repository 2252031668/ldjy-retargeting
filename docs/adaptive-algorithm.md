# 自适应重定向与 LDJY 关节映射

## 输入和坐标处理

`Retargeter.retarget(raw_keypoints)` 接收 `(21, 3)` 的 MediaPipe 顺序关键点。
`apply_mediapipe_transformations` 先减去 landmark 0（wrist），再由 wrist、食指
MCP（5）和中指 MCP（9）建立掌面坐标系，最后转换到左右手的 MANO 轴约定。
生成的 LDJY URDF/MJCF 已将 `retarget_wrist` 固定到同一 MANO wrist 坐标，因而优化器
直接在统一的 wrist 局部坐标系中工作，不再在运行时追加 MANO 到机器人坐标的旋转。

不同输入适配器的职责是产生该统一格式：

| 输入 | 关键点来源 |
| --- | --- |
| Vision Pro | 25 个关节变换映射为 21 个位置。 |
| pkl 回放 | 直接读取保存的 `left_fingers` / `right_fingers`。 |
| 视频、USB 摄像头 | MediaPipe Hands 检测，按参考掌长归一化。 |
| RealSense、ZED | 当前同样将 RGB 图像送入 MediaPipe；深度流不参与关键点三维重建。 |

无论输入来源如何，之后都会经过同一套 wrist/MANO 变换和优化器。

## AdaptiveOptimizerAnalytical

默认优化器在 `ldjy_retargeting/opt/adaptive_analytical.py` 中实现，变量是
LDJY 的 20 维关节角 `q`。每帧使用上一帧解做 warm start，并使用 Pinocchio
计算任务帧的位置与雅可比，再由 NLopt SLSQP 在 URDF 限位内求最小值。

它混合两组目标：

- FullHand：比较 wrist 到每指 `link3`、`link4`、`tip` 的向量，保持整手形状。
- TipDir：比较 wrist 到指尖的位置，以及 `link4` 到指尖的方向，优先保证捏合时的接触位置。

对食指、中指、无名指和小指，拇指指尖距离决定连续权重 `alpha`：距离较远时
以 FullHand 为主，接近时以 TipDir 为主。目标损失还包含帧间 `norm_delta` 正则，
用于稳定 MCP 两自由度等冗余解。

## LDJY 适配

LDJY 也有 20 个独立关节，但 MCP 的两个自由度在关节顺序和轴方向上与其他模型
可能不同。因此本项目不复制某一套固定关节角映射，而是直接保留上述几何优化：

```text
MediaPipe: thumb, index, middle, ring, pinky
LDJY:      thumb, finger1, finger2, finger3, finger4
任务帧:    retarget_wrist, link3, link4, tip
```

默认配置关闭 `w_hyper`、`w_couple` 和 `thumb_skip_pip`。特别是拇指存在带符号的
屈曲范围，未经实测不能套用统一的正向屈曲软约束。

生成的左右 URDF 与对应 MJCF 都使用 `left_` 或 `right_` 前缀。更重要的是，URDF 顺序为
`finger1, finger2, finger3, thumb, finger4`，MuJoCo
MJCF 顺序为 `finger1, finger2, finger3, finger4, thumb`。因此右手和左手都必须在
写入 MuJoCo 前使用置换 `[0..11, 16..19, 12..15]`。`qpos_reorder_perm` 通过关节名
自动生成这个结果，不能以恒等映射替代。新增或改动模型时，应先验证这条映射，再调整
`segment_scaling`、旋转和滤波参数。

## 建议调参顺序

1. 使用 pkl 回放和调参工具检查掌面朝向与左右手选择。
2. 确认生成资产的 `retarget_wrist`、PIP、DIP、tip 的 URDF/MJCF FK 一致；坐标变换应在
   资产生成阶段修正，而不是通过运行时关节符号或全局旋转补丁修正。
3. 仅在资产坐标正确后，微调 `mediapipe_rotation` 以消除某个相机设备的残余误差。
4. 调整 `segment_scaling`，匹配各段相对长度。
5. 调整 `norm_delta` 与 `lp_alpha`，平衡抖动和延迟。
6. 用捏合数据检查指尖接近时的轨迹；只在得到 LDJY 实测依据后再启用关节耦合或超伸约束。
