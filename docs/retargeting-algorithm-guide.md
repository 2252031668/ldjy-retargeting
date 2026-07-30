# LDJY 手部重定向：术语、数学与代码指南

本文面向需要修改算法或调参的开发者，说明默认
`AdaptiveOptimizerAnalytical` 如何把输入设备提供的 MediaPipe `(21, 3)`
关键点转换为 LDJY 手的 20 个关节角。文中的实现对应：

- 统一入口：`ldjy_retargeting/retarget.py`
- 坐标预处理：`ldjy_retargeting/mediapipe.py`
- 目标向量与公共优化逻辑：`ldjy_retargeting/opt/base.py`
- 自适应损失和解析梯度：`ldjy_retargeting/opt/adaptive_analytical.py`
- GUI 参数定义：`ldjy_retargeting/tuning/parameters.py`
- 默认配置：`example/config/adaptive_analytical_video.yaml`

## 先给结论

`retarget.segment_scaling` 是人手到机器人任务空间的唯一位置比例表：五根手指各有
`wrist -> PIP / DIP / TIP` 三条目标射线，共 15 个比例。它同时决定：

- `FullHandVec` 的 15 条位置目标；
- 捏合 `TipPos` 的五条 `wrist -> TIP` 位置目标，复用相同手指的 `TIP` 比例。

因此同一根手指的整手 TIP 目标与捏合 TipPos 目标长度完全一致。`TipDir` 是归一化方向，
不使用任何长度比例。历史版本的 `retarget.scaling` 已移除；调参 GUI 读取旧 YAML 时会丢弃
该废弃字段，并在下次保存时完成迁移。

`TIP` 是 **fingertip**（指尖）。在本仓库中更精确地说，它是每根手指的**虚拟任务末端点**：
URDF/MJCF 中后来加入的 `{finger}_tip` / `task` frame。它不必等于网格几何最前端，也不等于
MediaPipe 的像素点；其位置可以通过 GUI 的 `tip_offsets` 调到与人手指甲盖中心更匹配的位置。

## 整体数据流

```text
InputDeviceBase
  └─ MediaPipe 格式关键点 K_raw: (21, 3), 米
       └─ wrist 平移归零 + 掌面朝向估计 + MANO 坐标变换
            └─ 可选残余旋转、wrist/拇指偏移
                 = K: (21, 3), MANO 对齐坐标，米
                   ├─ FullHandVec：15 条 wrist -> PIP/DIP/TIP 目标
                   ├─ TipPos：5 条 wrist -> TIP 位置目标
                   └─ TipDir：5 条 DIP -> TIP 单位方向目标
                         └─ 自适应加权损失 + 运动学 + 解析梯度
                              └─ NLopt SLSQP + 关节限位
                                   └─ q_raw (20)
                                        └─ 一阶低通滤波
                                             └─ q_filtered (20) -> MuJoCo / 真机接口
```

所有输入设备最终都应向 `Retargeter.retarget()` 提供同一契约：21 个三维关键点，单位米，
数组形状 `(21, 3)`。USB 摄像头、视频、回放、RealSense、ZED 和 Vision Pro 的设备获取方式
可以不同；进入 `Retargeter` 后使用相同的坐标预处理和优化器。设备侧若有深度、尺度或骨段
修正，则发生在它交出这 21 个点之前或视频适配器内部。

## 名词与关键点

### 人手解剖与 MediaPipe 索引

`wrist` 是手腕根点。`MCP`（metacarpophalangeal）是掌指关节，`PIP`
（proximal interphalangeal）是近端指间关节，`DIP`（distal interphalangeal）是远端指间关节。
拇指只有一个 IP 指间关节，不存在解剖学上的 PIP/DIP 两节；为了让五指使用同一组数组，代码
把拇指的 MediaPipe `2` 当作“PIP 角色”，把 `3` 当作“DIP 角色”。这是**算法中的统一命名**，
不要将它误解成拇指真的有 PIP 和 DIP。

| 手指 | MediaPipe 点 | 解剖/算法角色 |
| --- | --- | --- |
| 全手 | `0` | `wrist`，所有 wrist 起点向量的原点 |
| 拇指 | `1 / 2 / 3 / 4` | CMC / MCP（算法 PIP）/ IP（算法 DIP）/ TIP |
| 食指 | `5 / 6 / 7 / 8` | MCP / PIP / DIP / TIP |
| 中指 | `9 / 10 / 11 / 12` | MCP / PIP / DIP / TIP |
| 无名指 | `13 / 14 / 15 / 16` | MCP / PIP / DIP / TIP |
| 小拇指 | `17 / 18 / 19 / 20` | MCP / PIP / DIP / TIP |

代码中的数组顺序固定为 `[thumb, index, middle, ring, pinky]`。对应常量位于
`BaseOptimizer`：

```python
MP_TIP_INDICES = [4, 8, 12, 16, 20]
MP_PIP_INDICES = [2, 6, 10, 14, 18]
MP_DIP_INDICES = [3, 7, 11, 15, 19]
```

### 机器人侧任务点

优化器不直接以网格顶点作为约束，而是查询 URDF 运动学链上的语义 frame：

| 机器人语义名 | 作用 | 人手目标 |
| --- | --- | --- |
| `retarget_wrist` / `origin` | 向量起点，MANO 对齐的手腕任务坐标 | MediaPipe `wrist` |
| `link3` | 算法 PIP 位置点 | MediaPipe PIP 角色点 |
| `link4` | 算法 DIP 位置点，也是末节方向的起点 | MediaPipe DIP 角色点 |
| `{finger}_tip` / `task` | 虚拟指尖任务点 | MediaPipe TIP |

`link3`、`link4` 是资产生成后的语义链名称，不应只凭“第 3/4 个机械关节”猜测其含义；优化器会
从当前加载 URDF 的链中解析它们。虚拟 `*_tip` 的纵向和厚度偏移由 `retarget_tip_offsets.yaml`
决定，并同时生成到左右手 URDF、MJCF 与 OpenArm 资产中。

## 坐标预处理

设输入关键点为 `K_raw[i] in R^3`，单位为米。`apply_mediapipe_transformations()` 做三件事：

1. 以 wrist 归零：`K0[i] = K_raw[i] - K_raw[0]`。
2. 从 `0`（wrist）、`5`（食指 MCP）、`9`（中指 MCP）估计掌面正交坐标架，消除相机相对手掌的
   朝向变化。
3. 乘左右手各自的 `operator2mano` 旋转矩阵，得到统一 MANO 任务坐标。

之后 `Retargeter` 还会依配置执行：

```text
K = Rz(rotation.z) Ry(rotation.y) Rx(rotation.x) K_mano
K[5:21] += wrist_offset_cm / 100
K[1:5]  += thumb_offset_cm / 100
```

`wrist_offset_cm` 这个名字容易误会：代码没有移动 `K[0]`，而是移动 `5..20` 的非拇指点；
`thumb_offset_cm` 移动拇指 `1..4`。它们是围绕固定 wrist 原点的残余校正，不是机器人腕部关节
位移。坐标系根本不一致时应修正资产的 MANO 对齐，而不是大幅调这些偏移或 `rotation`。

优化器内部会将位置和 Jacobian 从米换算为厘米：`M_TO_CM = 100`。因此位置阈值、捏合距离和
`huber_delta` 的单位都是厘米；方向向量已经归一化，没有长度单位。

## 三组目标向量

记 MANO 对齐后的人手点为 `k_*`，机器人正运动学位置为 `p_*`，第 `f` 根手指的 wrist 为
`k_w` / `p_w`。注意机器人每根手指都使用同一个 `retarget_wrist`，只是实现中以批量数组存储。

### 1. FullHandVec：整手形状的 15 条射线

对每根手指建立三条由 wrist 出发的**位置向量**：

```text
t_f,PIP = s_f,PIP * (k_f,PIP - k_w)
t_f,DIP = s_f,DIP * (k_f,DIP - k_w)
t_f,TIP = s_f,TIP * (k_f,TIP - k_w)
```

这里 `s_f,*` 就是 `segment_scaling[f][pip/dip/tip]`。五根手指乘三条射线，合计 15 条。
它们既携带方向，也携带从 wrist 到该点的距离；所以不是“单纯朝向”。对应的机器人向量为：

```text
r_f,PIP = p_f,link3 - p_w
r_f,DIP = p_f,link4 - p_w
r_f,TIP = p_f,task  - p_w
```

当手张开或没有明显捏合时，优化器主要让这 15 组 `r` 接近相应 `t`。MuJoCo debug 中的绿色
细线和端点球就是这些目标向量的可视化。

### 2. TipPos：捏合时的末端位置

`TipPos` 仍然是一条**有长度的位置向量**，不是“只有方向”的向量：

```text
t_f,pos = s_f,TIP * (k_f,TIP - k_w)
r_f,pos = p_f,task - p_w
```

它把机器人虚拟指尖放到人手指尖相对 wrist 的位置。捏合时，拇指和参与捏合的手指是否能在
正确的位置接近，比整根手指每一节都像人手更重要，因此需要这项。

### 3. TipDir：捏合时的末节朝向

`TipDir` 是**单位方向向量**，长度被除掉：

```text
u_f,human = normalize(k_f,TIP - k_f,DIP)
u_f,robot = normalize(p_f,task - p_f,link4)
```

它约束机器人最后一段从 `link4` 指向虚拟 `task tip` 的朝向，等价于“末端最后一截骨段朝哪边”，
但不直接指定这段有多长。故可把捏合组合理解为：

- `TipPos`：末端要到哪里。
- `TipDir`：末端最后一节要朝哪里。

两项都在捏合时参与，互补地减少“指尖到位但末节翻转”或“方向对但指尖位置不对”。debug 中红色
细线和红球表示当前激活手指的 `TipPos` / `TipDir` 目标；同一手指的绿色 FullHand 线会隐藏，
避免把两套损失目标看成同一条线。

## 自适应模式不是硬切换

系统按“拇指 TIP 到某一非拇指 TIP 的距离”逐指计算混合系数。以食指为例，距离 `d` 的单位为厘米：

```text
alpha_index = clip((d2_index - d) / (d2_index - d1_index + 1e-8), 0, 0.7)
alpha_thumb = max(alpha_index, alpha_middle, alpha_ring, alpha_pinky)
```

每根非拇指使用自己的一组 `d1/d2`，拇指没有独立阈值，而是参与所有正在捏合的指对。这里没有
状态机、没有滞回，也不是全手同时切换：食指接近拇指时，只提高食指和拇指的 `alpha`；中指、
无名指、小拇指仍可保持整手约束。

`alpha` 最大被代码限制在 `0.7`，所以捏合分支从不会完全取代 FullHand 分支，至少保留 30% 的
整手形状项。默认 `d1=2, d2=5` cm 时，`alpha` 在距离小于等于 `2.9` cm 已达到 `0.7`；并非
等到 `2.0` cm 才达到最大。这是由上限 `0.7` 直接推得的。debug 为了视觉易读，在 `alpha > 0.05`
时就把该指显示为红色捏合目标；这是显示阈值，不是损失函数的二元模式切换。

## 损失函数与优化

令 `H_delta(x)` 为 Huber 损失：

```text
H_delta(x) = 0.5*x^2                 (|x| <= delta)
           = delta*(|x|-0.5*delta)   (|x| >  delta)
```

对第 `f` 根手指，代码计算：

```text
L_pos,f = H_huber_delta(||r_f,pos - t_f,pos||)
L_dir,f = H_huber_delta_dir(||u_f,robot - u_f,human||)

L_full,f = mean(
    H_huber_delta(||r_f,PIP - t_f,PIP||),
    H_huber_delta(||r_f,DIP - t_f,DIP||),
    H_huber_delta(||r_f,TIP - t_f,TIP||)
)

L_f = alpha_f * (w_pos * L_pos,f + w_dir * L_dir,f)
      + (1 - alpha_f) * w_full_hand * L_full,f
```

若 `thumb_skip_pip=true`，拇指 `L_full` 只平均 DIP 和 TIP 两项。总损失还包括：

```text
L_smooth = norm_delta * ||q - q_previous||^2
L_hyper  = w_hyper * sum(max(soft_min - q_PIP/DIP, 0)^2)
L_couple = w_couple * sum((q_DIP - couple_ratio * q_PIP)^2)
L_total  = sum_f L_f + L_smooth + L_hyper + L_couple
```

优化变量 `q` 是 20 个 LDJY 手指关节角。NLopt 的 SLSQP 在 URDF 关节限位内求解，每帧以上一帧
解作为 warm start。代码不靠数值差分，而是计算解析梯度：位置误差通过任务点 Jacobian，方向误差
额外使用归一化导数

```text
d(v / ||v||) / dq = (I - u*u^T) / ||v|| * dv/dq
```

这正是 `AdaptiveOptimizerAnalytical._loss_and_grad_analytical()` 中 `J_norm` 的含义。求解器的原始
输出再经过一阶低通：

```text
q_filtered[t] = q_filtered[t-1] + lp_alpha * (q_raw[t] - q_filtered[t-1])
```

## GUI 参数逐项说明

### 人手与相机尺度

| 参数 | 数学/代码作用 | 调大或启用后的结果 |
| --- | --- | --- |
| `video_input.z_scale` | 单目 MediaPipe 输入侧的深度放大倍率 | 前后（相机深度）动作更明显；不是机器人长度标定 |
| `reference_wrist_to_mid_mcp` | 输入侧 wrist 到中指 MCP 的归一化参考长度，米 | 整只人手关键点尺度增大 |
| `correct_segments` | 输入侧按标准人体比例修正各指骨段 | 减少 MediaPipe 单帧骨段比例漂移；关闭后直接采用检测结果 |

这些参数要先于优化器理解。它们改变 `K_raw` 或交给 `Retargeter` 的人手尺度；不是改变机器人的
运动学。

### 目标尺度与末端任务点

| 参数 | 作用 | 推荐用途 |
| --- | --- | --- |
| `segment_scaling.<finger>.<pip/dip/tip>` | 独立缩放 FullHandVec 的对应 wrist 射线；`tip` 同时用于捏合 TipPos | 零位张手时某一关节层级或某一手指偏近/偏远时调节 |
| `tip_offsets.<finger>.axis_mm` | 修改虚拟 `*_tip`，沿该指 PIP 到 DIP 的局部纵向移动 | 将任务点前后移到实际指甲中心 |
| `tip_offsets.<finger>.surface_mm` | 修改虚拟 `*_tip`，沿指甲盖到指肚的局部厚度方向移动 | 调整任务点在指甲/指肚之间的位置；拇指使用单独的局部轴 |

`segment_scaling` 只缩放人手目标的半径，方向不变；`tip_offsets` 改的是机器人 FK 中的虚拟任务
frame，二者不应互相代替。前者适合做“人手与机器人比例”标定，后者适合做“任务点到底在机器人
模型何处”的标定。

GUI 的“自动零位标定”会在自然张开姿势采集 45 个有效帧，取每条人手射线长度的中位数，并计算：

```text
segment_scaling[f, segment]
    = LDJY 零位对应任务射线长度 / 人手该射线长度中位数
```

15 个结果必须都落在 `[0.5, 1.5]` 才应用。它会更新 15 个滑块，但不会自动写 YAML；确认后需要
点击“保存 YAML”。

### 损失权重与鲁棒性

| 参数 | 作用 | 调参含义 |
| --- | --- | --- |
| `huber_delta`（cm） | 所有位置误差的 Huber 转折点 | 小：大误差更快被降权，抗异常点更强；大：更接近平方误差 |
| `huber_delta_dir` | 单位方向误差的 Huber 转折点 | 小：大方向偏差更稳健；它没有厘米单位 |
| `w_pos` | 捏合 `TipPos` 权重 | 大：更优先让指尖到正确位置 |
| `w_dir` | 捏合 `TipDir` 权重 | 大：更优先让末节方向一致 |
| `w_full_hand` | 整手 15 线形状项权重 | 大：张手和非捏合指的整体姿态更受重视 |

权重是相对值。将所有权重同比例放大通常不会等价于同样的动作，因为 `norm_delta`、Huber 分段和
求解器终止条件仍会参与；实际应先固定尺度和坐标，再做小范围相对调节。

### 捏合自适应

| 参数 | 作用 |
| --- | --- |
| `pinch_thresholds.{index,middle,ring,pinky}.d2`（cm） | 该指与拇指距离由远变近时，开始产生 `alpha` 的外侧阈值 |
| `...d1`（cm） | 线性插值名义上的内侧参考阈值；受 `alpha <= 0.7` 限制，默认值下不会等到 d1 才饱和 |

必须保持 `d1 < d2`。增大两者会让更远的接近动作也采用更多捏合目标。当前实现没有时间滞回，
若距离在阈值附近抖动，先检查关键点稳定性、`norm_delta` 和 `lp_alpha`，再考虑未来增加滞回。

### 稳定、坐标残差与机械先验

| 参数 | 作用 | 注意事项 |
| --- | --- | --- |
| `norm_delta` | 惩罚优化解相对上一帧的关节变化 | 大则稳但会滞后 |
| `lp_alpha` | 最终关节指令的一阶低通系数 | 小则平滑但延迟高；`1` 表示不额外平滑 |
| `mediapipe_rotation.x/y/z`（度） | MANO 变换后的残余旋转 | 只修小的稳定设备偏差；不应用来弥补资产坐标系错误 |
| `wrist_offset_cm` | 移动非拇指关键点 `5..20` | wrist 点本身不移动，适合很小的相对偏置 |
| `thumb_offset_cm` | 移动拇指关键点 `1..4` | 拇指与掌面估计存在固定偏置时使用 |
| `thumb_skip_pip` | 从拇指 FullHand 损失中去掉算法 PIP 项 | 拇指 MCP 语义不稳定时才考虑 |
| `w_hyper`, `soft_min` | 对 PIP/DIP 小于 `soft_min` 的反向弯曲施加二次软惩罚 | 默认 `w_hyper=0`，即关闭 |
| `w_couple`, `couple_ratio` | 软约束 `q_DIP ~= couple_ratio * q_PIP` | 默认 `w_couple=0`，即关闭；LDJY 拆分 MCP 的结构不应凭经验强开 |

`project_tip_dir` 仍保留在 YAML 兼容配置中，但默认解析优化器当前没有据此分支，它不是一个有效的
GUI 可调行为，修改它不会改变 `AdaptiveOptimizerAnalytical` 的现有目标计算。

## 当前实现的注意事项

### 红色 TipDir 箭头的终点锚定在 TipPos 红球

TipPos 和 TipDir 在损失函数中仍是独立残差：前者由 wrist 出发且有长度，后者只比较 DIP 到 TIP 的
单位方向。为了避免 debug 图形暗示它们构成另一套 DIP 位置目标，TipDir 显示箭头固定为 15 mm，
并让其**终点**与同一手指的 TipPos 红球重合；箭头沿目标方向向后绘制。它表达的是“指尖应到这个
位置，且最后一节应朝这个方向”，不会改变优化器的目标或权重。

## 建议的阅读和调试顺序

1. 先用 MuJoCo 观察 `retarget_wrist`、`link3`、`link4`、`*_tip` 是否与 MANO 坐标契约一致。
2. 用自然张开姿势执行 15 条 `segment_scaling` 零位标定，先确认绿色 FullHand 目标射线合理。
3. 仅在物理任务点与人手指甲中心明显不一致时调 `tip_offsets`，确认后导出正式资产。
4. 再调 `w_pos / w_dir / w_full_hand` 与捏合 `d1/d2`，观察红色 TipPos/TipDir 线以及每指 alpha。
5. 最后才启用超伸和 PIP-DIP 耦合等机械先验；它们是软约束，不应代替正确的坐标、任务点和关节限位。

GUI 的“开始/暂停”会同时冻结视频帧、优化器和 MuJoCo 物理；暂停后仍可调整 `tip_offsets`，此时只
更新缓存资产中的虚拟 tip site 以便静态检查，既不会推进仿真，也不会读取新相机帧。
