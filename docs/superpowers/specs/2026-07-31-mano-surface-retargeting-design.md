# MANO 指腹表面重定向设计

## 目标

新增一个独立的 `ManoSurfaceOptimizer`，只处理 WiLoR 输出的完整 MANO 状态。它在不改变现有 `AdaptiveOptimizerAnalytical`、MediaPipe 链路或 `*_tip` 语义的前提下，使 LDJY 手在捏合时保持两块真实指腹表面的相对位置与法线关系。

第一版成功标准：在 WiLoR 静态记录和 `webcam_wilor` 实时输入中，拇指与活动手指进入捏合后，LDJY 两个指腹中心的相对向量和相向法线明显优于现有 TipPos/TipDir；非活动手指维持原有整手形态；输出仍为符合限位的 20 维 `qpos`。

## 范围

包含：

- WiLoR 的 `vertices_mano`、`joints_mano`、15 个 `hand_pose`、`global_orient` 与逐会话稳定的 `beta_bar`。
- 五个 MANO 指腹 patch 的中心、法线、局部坐标系和椭圆边界。
- 五个 LDJY 真实 CAD 指腹 frame，生成到 MANO 对齐单手 URDF、单手 MJCF 与 OpenArm 双臂 MJCF。
- 接触激活、全手到接触的平滑交叉淡入淡出、位置/法线/姿态/限位/时间正则损失。
- 离线回放指标、MuJoCo 调试显示和单元测试。

不包含：

- MediaPipe 输入的表面重定向。MediaPipe 不具备 MANO 网格和可靠表面法线，继续使用现有算法。
- 用 778 个 MANO 顶点逐点拟合 LDJY 外观网格。
- patch 内动态最近接触点、物体分割、SDF、碰撞求解或真实抓取力闭合。
- 修改真实 LDJY 连杆长度或删除既有 `*_tip`。

## 表示与坐标

### MANO 输入

每个 WiLoR 帧已有：

```text
joints_mano:   (21, 3)
vertices_mano: (778, 3)
hand_pose:     (15, 3, 3)
global_orient: (3, 3)
betas:         (10,)
```

对一个记录或实时会话，`beta_bar` 是有效高质量帧 `betas` 的逐维中位数。它只用于稳定的人手零位几何与骨长/手掌布局标定；不得逐帧驱动机器人关节。

所有 MANO 点和法线转换到与 `retarget_wrist` 一致的 LDJY/MANO task frame。相机平移仅用于视频覆盖显示，不进入手指优化。

### 人手指腹 patch

每根手指定义固定的 MANO 三角面集合和顶点权重。对当前网格计算：

```text
p_h: patch 加权面积重心
n_h: patch 面法线的面积加权归一化平均
u_h: distal 骨向量在切平面上的投影并归一化
v_h: normalize(n_h × u_h)
a_h, b_h: patch 在 u_h、v_h 上的零位椭圆半径
```

`u_h` 用于可视化、patch 合法区域和后续滚动接触；第一版不对绕法线的滚转角施加强约束。MANO patch 索引显式存储在代码数据表中，禁止把网格顶点号假定为 MediaPipe 点号。

### LDJY 指腹 frame

每个手指新增一个固定坐标系：

```text
thumb_pad_frame
finger1_pad_frame
finger2_pad_frame
finger3_pad_frame
finger4_pad_frame
```

frame 原点位于真实指腹接触表面；局部 `z` 轴向外法线；`x` 轴沿手指远端方向的表面投影；`y = z × x`。它们定义在对应末端 link 的局部坐标中，并生成到：

- `ldjy_{left,right}_hand.urdf`
- `ldjy_{left,right}_hand.xml`
- `openarm_bimanual_mano.urdf`
- `openarm_bimanual_mano.xml`

旧 `*_tip` 保持原位，仍是旧优化器、调试和兼容接口的任务点。

## 数据流

```text
WiLoRDetection
  -> beta_bar 会话聚合
  -> MANO task-frame transform
  -> ManoSurfaceTargets
       joints / local rotations / five pad frames / pinch alpha
  -> ManoSurfaceOptimizer
       LDJY FK + position and angular Jacobians
       -> qpos[20]
  -> MuJoCo / OpenArm 控制 / 调试叠加
```

`ManoSurfaceOptimizer` 是新 `optimizer.type`，由独立 YAML 选择。它接受完整 MANO detection，而不是只接受 `(21, 3)` 关键点；旧 `Retargeter.retarget()` 行为不变，新路径提供明确的 `retarget_mano()` 接口。

## 捏合状态

对拇指和食指、中指、无名指、小指的五组人手 pad 中心距离分别计算：

```text
alpha_f = smoothstep(d2_f, d1_f, ||p_h[f] - p_h[thumb]||)
```

其中 `d1 < d2`，距离从 `d2` 降到 `d1` 时 alpha 从 0 连续升到 1。第一版只用距离激活；人手法线不作为开关条件。允许多根手指同时活动，拇指相关损失按活动 alpha 归一化，避免同一拇指被重复放大。

## 损失函数

### 非活动手指

对 alpha 很低的手指，使用 MANO 语义骨架位置、局部姿态先验、限位和速度正则，维持整手形态。

### 活动捏合对

对拇指与活动手指 `f`，使用：

```text
L_pair_pos(f) = robust(||
  (p_r[f](q) - p_r[thumb](q))
  - S_f (p_h[f] - p_h[thumb])
||)

L_pair_face(f) = (1 + n_r[f](q) dot n_r[thumb](q))^2

L_pad_normal(f) =
  (1 - n_r[thumb](q) dot n_h[thumb])^2 +
  (1 - n_r[f](q) dot n_h[f])^2
```

`S_f` 来自 beta 标定的人手零位骨架与 LDJY 零位语义骨架的比例映射。活动对的 wrist->TIP 全局目标按 `(1-alpha)` 淡出；PIP/DIP 局部形态项保留固定低权重，避免近端关节失去约束或出现不自然折叠。非活动手指不受该交叉淡出影响。

总损失为：

```text
L = L_semantic_shape
  + L_local_pose
  + Σ_f alpha_f [w_pair L_pair_pos + w_face L_pair_face + w_normal L_pad_normal]
  + L_joint_limit
  + L_velocity
```

所有位置项用 Huber 损失；法线项用单位向量点积；关节限位为软惩罚并始终由硬限位裁剪兜底。

## 姿态与比例

`beta_bar` 通过 `MANO(beta_bar, zero_pose)` 得到该操作者稳定的零位骨架。它用于计算人手骨段长度和 wrist 到 PIP/DIP/TIP 的比例，不直接输出任何 LDJY 关节角。

15 个 `hand_pose` 局部旋转提供关节姿态先验。每个旋转通过 `log(R)` 投影到经标定的人体解剖分量，并由每关节映射矩阵转换为 LDJY 的软 `q_prior`。MCP 被拆分为两自由度时，用 `2 x 3` 映射共同生成两个关节先验；PIP/DIP 用 `1 x 3` 映射。位置和接触损失仍可覆盖不可达或机械结构不同的情况。

## 实时梯度

新优化器不得为法线损失对 20 个自由度做数值差分。机器人层新增完整的世界系 frame Jacobian 支持，并经有限差分测试确认空间速度分量顺序。

给定末端 link pose `(R, t)`、局部 pad 点 `a` 与局部法线 `z`：

```text
p_r = R a + t
n_r = R z
```

由线速度 Jacobian 与角速度 Jacobian 解析得到 `dp_r/dq`、`dn_r/dq`，供 NLopt 的梯度目标使用。锚点位置和法线的计算必须共用同一 frame pose，避免 CAD 资产和优化坐标错位。

## 配置与调试

新增专用 WiLoR YAML，例如 `example/config/mano_surface_wilor.yaml`。它独立于 `adaptive_analytical_wilor.yaml`，至少包含：

- patch/anchor 配置版本与 asset 路径。
- beta 聚合窗口和有效帧条件。
- 每对 `d1/d2`、`w_pair`、`w_face`、`w_normal`。
- 非活动形态权重、活动 PIP/DIP 保形权重、局部姿态先验权重、速度和限位权重。

调试显示新增：MANO 与机器人五个 pad 中心、法线、活动捏合对、alpha、相对位置残差和法线夹角。旧绿线/红线调试语义保持不变，表面优化使用清晰独立的颜色与标签。

## 验证

单元测试：

- beta 中位数聚合忽略无效、非有限和低质量帧。
- 左右手 MANO patch 变换后法线方向正确。
- 每个 LDJY pad frame 存在于四种生成资产中，零位处位于对应视觉 mesh 表面附近。
- pad 位置与法线的解析 Jacobian 与有限差分一致。
- alpha 在 `d2`、中间、`d1` 处连续且单调。
- 活动捏合对只影响拇指和目标手指；非活动手指保留形态权重。

离线回放评估：

- 现有 WiLoR 静态记录与新的专门捏合记录。
- 人机 pad 相对位置误差、机器人双 pad 间距误差、法线相向角、语义关节误差、关节限位占用率、帧间 qpos 跳变和平均求解时间。
- 与 `AdaptiveOptimizerAnalytical` 的同一记录对比。

实时验收：WiLoR 推理暂停时优化也暂停；有效 WiLoR 帧到 `qpos` 的优化不显著拖慢现有实时输入节奏。
