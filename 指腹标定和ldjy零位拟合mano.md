# 指腹标定和 LDJY 零位拟合 MANO

本说明覆盖 LDJY 手部资产的坐标约定、指腹关键点人工标定，以及固定形状 MANO 手到右 LDJY 零位的叠加拟合。

## LDJY 手部资产

左右手 URDF 与 MJCF 都从同一份原始 CAD URDF 生成，不能手工分别修改：

```text
ldjy_retargeting/assets/robots/ldjy_hand/source/step_20_dof_hand.urdf
    原始单手 CAD URDF，SHA-256: 361f662493d4b781996f59fc247d66c9b2743e65d13dbf1440adfa4b6e0125b2
ldjy_retargeting/assets/robots/ldjy_hand/meshes/
    原始视觉与碰撞网格
ldjy_retargeting/assets/robots/ldjy_hand/urdf/ldjy_{right,left}_hand.urdf
    生成的 MANO 对齐运动学模型
ldjy_retargeting/assets/robots/ldjy_hand/mjcf/ldjy_{right,left}_hand.xml
    从对应 URDF 生成的 MuJoCo 模型
```

### 坐标约定

原始 `palm` 是机械 CAD 根，位置偏向手背，不能作为任务向量的原点。生成的模型将无质量、无网格、无碰撞的
`{side}_retarget_wrist` 作为根；原始 `{side}_palm` 是它的固定子节点，仍承载全部关节和网格。

右/左手均使用下列静态变换，其中左手先在 CAD X=0 平面镜像，再应用该根变换：

```text
p_mano = R_mano_from_cad @ (p_cad - [0, -0.015, -0.03])

R_mano_from_cad = [[ 0, -1, 0],
                   [ 1,  0, 0],
                   [ 0,  0, 1]]
```

因此零位时四个非拇指的任务射线位于 MANO wrist 坐标中，以 `+Z` 为主要伸指方向、以 `Y` 为横向分布。
MediaPipe 输入层已经把每帧关键点规范化到同一 MANO wrist 坐标；相机外参不属于模型资产，也不应写入 URDF/MJCF。
`finger4_joint1` 的范围仍是原始 CAD 的 `[0, pi/2]`，即最大 90 度。

### 重新生成

从仓库根目录执行：

```bash
uv run python tools/build_ldjy_urdf.py
uv run python tools/build_ldjy_mjcf.py
uv run python -m unittest tests.test_ldjy_asset_generation -v
```

最后一条测试会验证左右两侧在 `q=0` 时的 `retarget_wrist`、每指 PIP、DIP 与 tip 在 Pinocchio URDF 和
MuJoCo MJCF 中逐点一致。

## 指腹与 MANO-LDJY 标定

这一套流程把指腹点定义在真实网格表面，并把一份固定形状的 MANO 手配准到右 LDJY 手。所有公开的 `21x3`
MANO 关键点使用 MediaPipe/WiLoR 顺序：`0` 为 wrist，`1..4` 为拇指，`5..8` 为食指，`9..12` 为中指，
`13..16` 为无名指，`17..20` 为小指。MANO 的 `hand_pose` 仍是原生运动学顺序
`Index, Middle, Pinky, Ring, Thumb`，不要把这 15 个关节索引与 21 点索引混用。

### 1. 标定 LDJY 指腹点

运行：

```bash
uv run --extra tuning python tools/calibrate_ldjy_pads.py
```

程序为每个末节 `*_link4` 放置一个红球。球的初始位置投影在对应 STL 表面上；方向键通过三角网格的切平面移动
并跨面平行传输，因此球始终在该末节的视觉表面上。每个球旁的黄线是该三角面向外法线。

| 操作 | 含义 |
| --- | --- |
| 双击红球 | 选中要编辑的手指 |
| `1` / `2` / `3` / `4` / `5` | 选择拇指 / 食指 / 中指 / 无名指 / 小指 |
| 上 / 下 | 沿初始指尖 / 手根方向在表面移动；跨面后保持切向连续 |
| 左 / 右 | 沿指腹横向在表面移动 |
| `S` | 保存五个点 |
| `R` | 仅提示重启来重置本次临时标记；不会删除已保存的标定 |

将每个球放在从指尖向手根数第三条沟槽的中线位置后按 `S`。结果写入
`ldjy_retargeting/assets/robots/ldjy_hand/retarget_pad_points.yaml`，坐标是原始单手 CAD 的 `link4` 局部坐标。
该源标定同时用于生成左右手，无需分别重复标定左手。

保存后必须重建生成资产，才会在 MJCF 中写入 `{side}_{finger}_pad_center` site 及其法线：

```bash
uv run python tools/build_ldjy_urdf.py
uv run python tools/build_ldjy_mjcf.py
```

用下列命令检查结果。MuJoCo 的 `Control` 面板可独立调节 20 个 LDJY 关节；红球和黄线分别是指腹点和其表面法线。

```bash
uv run --extra tuning python example/ldjy_viewer.py
uv run --extra tuning python example/ldjy_viewer.py --left
```

### 2. 检查或重标 MANO 指腹点

运行：

```bash
uv run --extra wilor --extra gui --extra tuning python example/mano_viewer.py
```

这个 viewer 显示 MANO 网格、21 个关键点、橙色指腹点和从指腹出发的黄色表面法线。若要重选 MANO 指腹，先在
Qt 面板点击 `Show Candidates`，然后在 MuJoCo 窗口双击候选球：`1..5` 分别分配给拇指、食指、中指、无名指和
小指。也可用 `A`、`B`、`C` 选三个顶点，再按 `1..5` 使用三点重心。`S` 或 `Export (S)` 会将
`PAD_VERTEX_IDS` / `PAD_3PT_VERTEX_IDS` 文本输出到终端和剪贴板；将确认后的字典写回 `example/mano_viewer.py`，
它也是叠加 viewer 使用的 MANO 指腹定义。`R` 重置临时选点，`Esc` 取消当前候选点。

### 3. MANO-LDJY 叠加、拟合和保存

运行：

```bash
uv run --extra tuning --extra gui --extra wilor python example/mano_ldjy_overlay_viewer.py
```

MuJoCo 窗口控制右 LDJY 手的 20 个关节；Qt 窗口控制半透明 MANO 手和固定的 MANO-to-LDJY 配准。主界面中的
`betas` 为 MANO 形状参数，手动范围为 `[-10, 10]`；`Global orient`、`Hand pose`、`Translation` 和 `Scale`
是 MANO 当前姿态。`Static MANO -> LDJY registration` 的旋转、平移和缩放用于保存固定的模型对齐，不随每帧手势改变。

可视化开关包括 LDJY skeleton、tip sites、pad points、pad normals、MANO keypoints + skeleton、MANO active skeleton，
以及两只手的 palm normal。`MANO keypoints + skeleton` 使用上文的 MediaPipe/WiLoR 21 点顺序；
`MANO active skeleton` 仅显示 15 个 MANO 可动关节的运动学连杆。

点击 `Fit settings` 打开一次静态配准的设置，再点击其中的 `Fit` 执行。每个带手指复选框的约束可单独选择
Thumb / Index / Middle / Ring / Pinky：

| 项目 | 最小二乘残差 |
| --- | --- |
| Pad positions | 对应指腹点位置 |
| Pad normals | 对应指腹表面法线方向 |
| Finger directions | 对应指段的单位方向 |
| `same line` | 在 Finger directions 基础上，MANO 指段起点到 LDJY 对应无限直线的垂直距离 |
| Palm normal | LDJY `(thumb j2, finger1 j2, finger3 j2)` 平面法线与 MANO `(0, 5, 17)` 平面法线 |
| Straight fingers | 拇指 `2-3-4`、其余手指 `5-6-7-8`、`9-10-11-12`、`13-14-15-16`、`17-18-19-20` 的共线误差 |
| Finger directions in palm plane | 四根非拇指方向相对于通过 MANO `(0, 5, 17)` 的掌平面 |
| Hand pose prior | 约束参与拟合的局部轴角不要偏离 Current GUI pose 或 Zero pose |

方向约束对应关系为：拇指 LDJY `j2 -> j3` 对 MANO `2 -> 3`；食指 `finger1 j2 -> j3` 对 MANO `5 -> 6`；
中指 `finger2 j2 -> j3` 对 MANO `9 -> 10`；无名指 `finger3 j2 -> j3` 对 MANO `13 -> 14`；
小指 `finger4 j3 -> j4` 对 MANO `17 -> 18`。

`Straight fingers` 和 `Finger directions in palm plane` 默认是带权重的软残差。勾选它们的 `linear constraint` 后，
所选 `hand_pose` 关节不再参与普通最小二乘，而是只调整可选关节的轴角，寻求满足严格几何条件的最小改动。未勾选的
关节固定为 GUI 当前值；若选中的关节不足，界面会报告不可行。严格直指采用逐关节的局部解，避免沿骨轴自转造成手指扭曲。

`Variables to optimize` 可分别选择 betas、静态旋转、静态平移和静态缩放；`auto beta bound` 设置自动拟合 betas 的
对称边界 `[-B, B]`，默认 `[-3, 3]`。下方的 15 个 `hand_pose` 复选框按 MANO 原生手指顺序分组，只允许所选关节
以 3D 轴角变量参与拟合，内部再转换为合法旋转矩阵。

主窗口的 `Save reference` 会保存到 `ldjy_retargeting/assets/robots/ldjy_hand/mano_ldjy_reference.yaml`：MANO
`betas`、`hand_pose`、`global_orient`、`translation`、`scale`，静态 MANO-to-LDJY 旋转/平移/缩放，以及拟合设置。
叠加 viewer 每次启动都会自动加载该文件，不需要重新 build URDF/MJCF。它不保存 LDJY MuJoCo 的 20 个 Control 值；
要复现参考叠加结果，请在启动后将 LDJY 控制滑块保持在保存参考时的姿态，通常是目标零位。
