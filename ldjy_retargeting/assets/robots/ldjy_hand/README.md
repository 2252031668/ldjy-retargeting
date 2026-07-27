# LDJY 手部资产

本目录的左右手 URDF 与 MJCF 都从同一份原始 CAD URDF 生成，不能手工分别修改：

```text
source/step_20_dof_hand.urdf      原始单手 CAD URDF，SHA-256: 361f662493d4b781996f59fc247d66c9b2743e65d13dbf1440adfa4b6e0125b2
meshes/                           原始视觉与碰撞网格
urdf/ldjy_right_hand.urdf         生成的右手 MANO 对齐运动学模型
urdf/ldjy_left_hand.urdf          生成的左手 MANO 对齐运动学模型
mjcf/ldjy_right_hand.xml          从右手 URDF 生成的 MuJoCo 模型
mjcf/ldjy_left_hand.xml           从左手 URDF 生成的 MuJoCo 模型
```

## 坐标约定

原始 `palm` 是机械 CAD 根，位置偏向手背，不能作为任务向量的原点。生成的模型将
无质量、无网格、无碰撞的 `{side}_retarget_wrist` 作为根；原始 `{side}_palm` 是它的
固定子节点，仍承载全部关节和网格。

右/左手均使用下列静态变换，其中左手先在 CAD X=0 平面镜像，再应用该根变换：

```text
p_mano = R_mano_from_cad @ (p_cad - [0, -0.015, -0.03])

R_mano_from_cad = [[ 0, -1, 0],
                   [ 1,  0, 0],
                   [ 0,  0, 1]]
```

因此零位时四个非拇指的任务射线位于 MANO wrist 坐标中，以 `+Z` 为主要伸指方向、
以 `Y` 为横向分布。MediaPipe 输入层已经把每帧关键点规范化到同一 MANO wrist 坐标；
相机外参不属于模型资产，也不应写入 URDF/MJCF。

`finger4_joint1` 的范围仍是原始 CAD 的 `[0, pi/2]`，即最大 90 度。

## 重新生成

从仓库根目录执行：

```bash
uv run python tools/build_ldjy_urdf.py
uv run python tools/build_ldjy_mjcf.py
uv run python -m unittest tests.test_ldjy_asset_generation -v
```

最后一条测试会验证左右两侧在 `q=0` 时的 `retarget_wrist`、每指 PIP、DIP 与 tip 在
Pinocchio URDF 和 MuJoCo MJCF 中逐点一致。
