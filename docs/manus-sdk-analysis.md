# MANUS SDK 数据、校准与 LDJY 映射分析

本文记录本项目接入 MANUS Metaglove Pro 时已经核实的 SDK 数据、官方校准与 LDJY 手重定向边界。它是理解记录，不替代 [MANUS 接入实施计划](manus-integration-plan.md) 或 [运行快速开始](manus-quickstart.md)。

## 结论先行

MANUS 的能力可以分成三层，名称相近，但用途不同：

| 层 | 输入与产物 | 是否已经用于本项目 |
| --- | --- | --- |
| 手套校准 | 传感器数据 -> 已校准的人手姿态；档案为 MANUS `.mcal` | 是。由 MANUS Core/Integrated SDK 自动读取；备份已保存到 `calibrations/manus/` |
| MANUS 官方 Skeleton Retargeting | 已校准的人手 -> 用户定义的目标骨架节点变换 | 否，尚未接入；可作为后续 A/B 对照 |
| LDJY 重定向 | MANUS 数据 -> 20 自由度 LDJY `qpos`，带真实机械限位 | 是。当前使用完整 Raw Skeleton 的节点位置/旋转作优化目标，Ergonomics 完整保留作诊断 |

因此，重新做一次 MANUS 官方校准并不等于重新做 LDJY 标定。前者使手套的人手数据可信；后者解决 MANUS 坐标和机器人零位、机器人构型之间的对应关系。

## 当前机器已确认的校准状态

2026-09-11 在 MANUS Core 重新校准右手并保存后，Core 的本机校准存储已同时包含两副手套的 profile：

| 手侧 | Core 中的 glove ID | 本机 profile | 导出的可迁移备份 |
| --- | --- | --- | --- |
| 左手 | `0x7E0EBCDC` | 有 calibration profile 与 measurements | `calibrations/manus/WirelessUser_0x884132AALeftMetaglovePro.mcal` |
| 右手 | `0xBFD38D1E` | 有 calibration profile 与 measurements | `calibrations/manus/WirelessUser_0xD2922CDDRightMetaglovePro.mcal` |

Core 当前的本机文件是 `/home/wxx/.config/Manus/Core 3/CoreLite.Settings.3.2.0.Calibrations.json`。它属于当前 Linux 用户和当前 Core 版本的运行时设置，不应作为项目可移植配置来依赖。`.mcal` 才是用于迁移和备份的显式校准文件。

注意导出文件名中的 ID 与 SDK 运行时的 `glove_id` 表示法不一定看起来相同；导入时应让 SDK 校验手侧、手套型号和版本，不要根据文件名自行改写或拼接 ID。

## `.mcal` 的作用、读取与迁移

### 它在什么阶段参与

`.mcal` 是 MANUS 的手套/用户校准档案。它服务于从传感器读数得到稳定人手姿态、骨架和 Ergonomics 的阶段。Core 或 Integrated SDK 在连接手套时会自动从其 settings 中找到该手套对应的校准；所以正常启动项目时不需要每次重新走“放桌面、握拳、摆手、捏合”等官方校准动作。

那套多姿态流程的目的不是为 LDJY 求一个初始姿态，而是估计这副手套在这只手上的传感器偏置、手指尺寸/比例和运动模型参数。它覆盖张手、弯曲、旋转和捏合，因而比只采一个静止姿态能约束更多误差来源。这也是优先使用官方已校准数据的原因。

### SDK 的导入导出接口

SDK 传递的是字节数组，文件路径由应用自行决定：

```python
# 导出（官方 Python 示例的 CFFI 形式）
size = ffi.new("uint32_t*")
assert lib.CoreSdk_GetGloveCalibrationSize(glove_id, size) == SDKReturnCode.Success
calibration_bytes = ffi.new(f"unsigned char [{size[0]}]")
assert lib.CoreSdk_GetGloveCalibration(calibration_bytes, size[0]) == SDKReturnCode.Success
Path("backup.mcal").write_bytes(bytes(ffi.buffer(calibration_bytes, size[0])))

# 导入：除了 SDK 调用状态，也检查 result_code[0]
data = Path("backup.mcal").read_bytes()
calibration_bytes = ffi.new(f"unsigned char [{len(data)}]")
ffi.memmove(calibration_bytes, data, len(data))
result_code = ffi.new("SetGloveCalibrationReturnCode*")
status = lib.CoreSdk_SetGloveCalibration(glove_id, calibration_bytes, len(data), result_code)
```

这里的 `GetGloveCalibration` 读取的是前一次由 `GetGloveCalibrationSize(glove_id, ...)` 选定的手套校准。导入只有在 `status` 与 `result_code[0]` 都表示成功时才算成功。导入成功后，Integrated SDK 会把它写入自己的 settings 文件，之后可自动使用。

建议的跨机器操作：

1. 关闭 MANUS Core、官方 SDK Client 和本项目，避免同一接收器被多个 SDK 进程占用。
2. 将对应手套型号、对应手侧的 `.mcal` 复制到目标机器，例如本项目的 `calibrations/manus/`。
3. 在目标机器连接同一副手套，调用 SDK 的 `SetGloveCalibration` 导入一次，检查返回状态为成功。
4. 断开并重连手套，确认 Core/Integrated SDK 自动加载；之后日常使用不需要再次导入。

不能保证校准能跨不同硬件型号使用：MANUS API 会检查版本、平台、手套类型和手侧。尤其不要把 Metaglove Pro 的档案导入另一种手套，或把左手档案当右手档案使用。

### 本项目尚未做的事

当前 [`example/input_devices/manus_glove.py`](../example/input_devices/manus_glove.py) 不调用 `GetGloveCalibration` 或 `SetGloveCalibration`，也没有 `.mcal` 文件选择界面；它只初始化 Integrated SDK，后者从本机 settings 自动读取已导入的官方校准。因此，上表的备份可以迁移，但目前仍需通过 Core 或一个小型 SDK 导入工具导入到新机器。

项目的 `outputs/manus_calibrations/<glove_id>_<side>.npz` 是另一种文件：它保存 MANUS 坐标到 LDJY 的中立对齐和拓扑检查信息，**不是** MANUS `.mcal`，不能互相替代。

## SDK 能返回什么：原始性与处理层级

“Raw” 在 MANUS SDK 中不总是指未经处理的电信号。应按数据流理解：

| 数据流 | SDK 返回的主要字段 | 它相对于官方算法的位置 | 适合 LDJY 的用途 |
| --- | --- | --- | --- |
| Raw Device Data | 每手指最多 5 个传感器 `ManusTransform`，以及主 IMU rotation | 最接近设备流，但已经是位置/旋转变换；不是 ADC/原始 IMU 采样值 | 以后做传感器研究；第一版不用 |
| Raw Skeleton | 每节点 `position`、`rotation`、`scale` 与节点拓扑 | 官方校准后的人手骨架；尚未按某个目标骨架重定向 | 当前主控制真值 |
| Ergonomics | 40 个以度为单位的人体关节语义角度 | MANUS 官方人体运动/关节估计结果 | 常规四指直接角度映射、诊断、可作为软约束 |
| Processed Skeleton / 目标 Skeleton | MANUS 将人手重定向到应用加载的 Skeleton 后的节点变换 | 官方目标骨架 retargeting 结果 | 后续与当前方法 A/B 测试 |

### Raw Device Data

`RawDeviceData` 有 `id`、`sensorCount`、`sensorData[5]` 和主 `rotation`。五个传感器槽位的顺序固定为：`0=thumb`、`1=index`、`2=middle`、`3=ring`、`4=pinky`。传感器位置相对于手套主磁线圈；Pro 才支持对应的 raw feature stream。

它名字中的 Raw 表示没有被转换为人体关节角，不表示可以取得 IMU 的加速度、陀螺仪原始数值或 ADC 读数。当前范围明确不使用此流。

### Raw Skeleton

回调元信息为 `RawSkeletonInfo(gloveId, nodesCount, publishTime)`；每个节点为 `SkeletonNode(id, transform)`，其中 transform 有三维 `position`、四元数 `rotation` 和三维 `scale`。`NodeInfo` 给出：

- `nodeId`、`parentId`：拓扑；
- `chainType`：`Hand=13`，`Thumb=5`，`Index=6`，`Middle=7`，`Ring=8`，`Pinky=9`；
- `fingerJointType`：`Metacarpal=1`、`Proximal=2`、`Intermediate=3`、`Distal=4`、`Tip=5`；
- `side`：左右手。

本项目启动时明确配置为 `X from viewer / +Z / right-handed / unitScale=1 / world coordinates`，所以位置单位为米，四元数在 SDK 中是 `w, x, y, z`。当前实测右手拓扑有 25 个节点：1 个 Hand，拇指 4 个节点（Metacarpal、Proximal、Distal、Tip），其余四指各 5 个节点（Metacarpal、Proximal、Intermediate、Distal、Tip）。

节点序号与节点数不可硬编码：不同 SDK/手套/配置可能改变它们；代码必须按 `ChainType`、`FingerJointType` 和 `parentId` 查语义。当前 MANUS 输入正是按这一原则缓存拓扑。

Raw Skeleton 的 wrist 来源由 `HandMotion.Auto` 选择：有追踪器时使用追踪器，否则使用手套 IMU。它仍是 MANUS 已校准模型输出，因此比单纯传感器转换更适合作为人体空间的可靠观测。

### Ergonomics：是否就是各关节角度，如何得到

是，但它是 MANUS 定义的人体工效学关节角，而不是“机器人每个电机的角度”。单位是度。每帧 `ErgonomicsStream` 有时间戳和若干 `ErgonomicsData`；每项有 `id`、`isUserID` 和 `data[40]`。

- `isUserID=False` 时 `id` 是 glove ID，适合本项目按手套分别缓存。
- `isUserID=True` 时 `id` 是 user ID，不能把它误当作 glove ID。

当前输入模块已完整复制 40 个数字和其独立时间戳，但应在后续小修中显式跳过 `isUserID=True` 的项，以保证 per-glove 缓存语义严格正确。

排列顺序如下。每个 Stretch 是屈伸，Spread 是左右张合；拇指四项与其他手指四项的语义不同。

| 数组下标 | 左手 `0..19` | 右手 `20..39` |
| --- | --- | --- |
| `0..3` / `20..23` | Thumb：CMC Spread、CMC Stretch、MCP Stretch、IP Stretch | 同左 |
| `4..7` / `24..27` | Index：MCP Spread、MCP Stretch、PIP Stretch、DIP Stretch | 同左 |
| `8..11` / `28..31` | Middle：MCP Spread、MCP Stretch、PIP Stretch、DIP Stretch | 同左 |
| `12..15` / `32..35` | Ring：MCP Spread、MCP Stretch、PIP Stretch、DIP Stretch | 同左 |
| `16..19` / `36..39` | Pinky：MCP Spread、MCP Stretch、PIP Stretch、DIP Stretch | 同左 |

MANUS 公开了这些字段的名称、单位和顺序，但没有公开内部完整的估计器、滤波器和传感器融合公式。因此可以把它视为经过产品验证的官方人体角度输出，而不能在项目中复现其计算过程。UI 中可见的显示范围也不是硬物理范围；真实记录中，例如 PIP 可能超过一个简单 UI 滑条的上限。

## “用户提供 Skeleton”究竟能自定义到什么程度

MANUS SDK 可以创建 `SkeletonSetupInfo`，加入 `NodeSetup` 和 `ChainSetup`，再由 `CoreSdk_LoadSkeleton` 加载。每个节点可定义 ID、父节点、节点类型、零位 transform（位置和旋转）、名称和部分 NodeSettings；每条手指链带有指头语义和若干链设置。

所以答案是：**不只是骨长。** 目标 Skeleton 可以表达：

- 节点层级和父子关系；
- 零位下每个节点的三维位置，因此包含掌内安装点、各段长度和骨段方向；
- 零位下节点的旋转，因此包含目标 frame 的朝向；
- 拇指和小指相对掌心的不同起点/朝向；
- MANUS 所需的 Hand、Thumb、Index、Middle、Ring、Pinky 链语义。

对 LDJY，应从 URDF 零位正运动学取得实际 wrist、各关节和 tip 的位置与 frame，按真实几何建立目标 Skeleton，而不是拿一个通用人手骨架只改长度。

不过它**不是通用机器人运动学描述器**。目标 Skeleton 接口没有等价于 URDF 的“任意一自由度转轴 + 上下限 + 耦合/传动 + 碰撞”的通用字段。因此不能仅靠 MANUS 的 Skeleton Retargeting 精确表达：

- “这个关节只能绕这一条任意轴转动”；
- “角度必须在 `[lower, upper]`”；
- LDJY 拇指、小指的非人体轴排列或多关节耦合。

官方 retarget 的输出仍是节点 transform，不是已经合法的 LDJY 20 维 `qpos`。无论使用 Raw Skeleton 还是官方处理后的目标 Skeleton，最后都应由本项目的 URDF/Pinocchio IK 投影到 LDJY 关节限位内。这是我们不能省掉的一步。

## 官方 Skeleton Retargeting：已证实能力与最小使用流程

这里需要严谨地区分“官方已说明的能力”与“对闭源内部实现的猜测”。MANUS 官方说明 Skeleton System 的用途是把 glove/body 数据重定向到用户选择的模型；SDK Client 文档明确区分了两条流：

```text
手套传感器 + MANUS .mcal 校准
          ↓
MANUS Core 内部手部估计
          ├── RawSkeletonStream：标准 MANUS 人手骨架
          └── SkeletonStream：应用提供 Skeleton 的重定向结果
```

官方明确说 Retargeted Skeleton 是“手套数据应用到用户选择的 Skeleton”，Raw Skeleton 是“手套数据应用到标准 MANUS 手骨架”。但是官方没有公开 Retargeting 是否把公开的 Raw Skeleton 数组作为直接输入，也没有公开其目标函数、IK/Jacobian、滤波、关节权重或捏合约束。因此不应把上图理解为已经证实的 `RawSkeleton -> Retargeting` 函数调用链；两条流只是在同一个已校准 Core 中产生的不同输出。[MANUS Skeleton System](https://docs.manus-meta.com/3.2.0/Software/Skeletons/)，[MANUS SDK Client — Skeletons](https://docs.manus-meta.com/3.2.0/Plugins/SDK/C%2B%2B/SDK%20Client/#skeletons)

### 官方 Python 示例实际做了什么

`/home/wxx/manus_sdk/Python/examples/sdk_client.py` 是 C++ 官方 SDK Client 示例的 Python 包装版本。它包含完整的目标 Skeleton 例子：

1. `/home/wxx/manus_sdk/Python/examples/sdk_client.py:3240` 的 `_create_node_setup` 创建 `NodeSetup`：ID、父 ID、节点类型、position、rotation、scale、名称；示例把 rotation 设为单位四元数。
2. 同文件 `:3263` 的 `_setup_hand_nodes` 创建一个 root 加五根四节点手指的 21 节点手；节点位置是示例自定义的人手几何，而不是从手套读取。
3. 同文件 `:3369` 的 `_setup_hand_chains` 给 root、拇指、食指、中指、无名指、小指建立 `ChainSetup`，从而标注每串节点的人体语义。
4. 同文件 `:3461` 的 `load_test_skeleton` 调用 `CoreSdk_CreateSkeletonSetup`、`CoreSdk_AddNodeToSkeletonSetup`、`CoreSdk_AddChainToSkeletonSetup`、`CoreSdk_LoadSkeleton`。
5. 同文件 `:392` 的 `on_skeleton_callback` 用 `CoreSdk_GetSkeletonInfo` 与 `CoreSdk_GetSkeletonData` 接收已动画化的目标节点 transform。

官方 C++ 示例的 `SkeletonsRetargeted.cpp` 对其节点位置的注释是“平放手部的示意 world-space 关节位置；应替换成自己的模型数据”。所以“可以提供自定义目标骨架”是 SDK 公开能力，不是推测。

### 字段属于哪里、分别做什么

此前用于说明的配置不是 Makefile，也不是单个函数的参数；它是几个 SDK 结构体字段的简写。真实结构如下：

```text
SkeletonSetupInfo
├── type
└── settings: SkeletonSettings
    ├── scaleToTarget
    ├── useEndPointApproximations
    ├── targetType
    └── skeletonGloveData.gloveID

ChainSetup（Hand）
└── settings: ChainSettings
    └── hand.handMotion
```

| 简写 | SDK 真实字段 | 含义 | LDJY 第一轮建议 |
| --- | --- | --- | --- |
| `SkeletonType=Hand` | `setup.type` | 声明目标是手部 Skeleton，不是 Body/Both | `Hand` |
| `targetType=GloveData` | `setup.settings.targetType` | 指定 Skeleton 由哪类来源驱动 | `GloveData` |
| `gloveID` | `setup.settings.skeletonGloveData.gloveID` | 当 targetType 为 GloveData 时指定具体手套 | 右手 `0xBFD38D1E` |
| `scaleToTarget` | `setup.settings.scaleToTarget` | 是否将目标骨段按用户手尺寸缩放 | `False`，保持 LDJY 固定骨长 |
| `endpointApprox` | `setup.settings.useEndPointApproximations` | 是否更偏重指尖位置而非人体关节角 | 首轮 `False`，随后增加 tip leaf 再比较 `True` |
| `HandMotion` | `hand_chain.settings.hand.handMotion` | 决定 wrist/root 如何运动 | 固定机器人手先用 `None_`；需要整体朝向时用 `IMU` |

`HandMotion` 不在 `SkeletonSetupInfo.settings` 中；它属于 Hand chain。此外，同一个 `HandMotion` 枚举也用于 Raw Skeleton 的全局 wrist 设置，二者不能混淆。目标 Skeleton 中：`None_` 表示 root 不变换，`IMU` 只由手套 IMU 驱动 wrist 旋转，`Tracker` 才驱动位置和旋转，`Auto` 在可用数据源间选择。[Skeleton settings](https://docs.manus-meta.com/3.2.0/Software/Skeletons/#settings)

官方 Python 示例为了角色动画，实际使用的是 `UserIndexData`、`userIndex=0`、`scaleToTarget=True`、`useEndPointApproximations=True`、`HandMotion.IMU`。上表的 LDJY 值是针对固定连杆机器人、可控 A/B 实验的选择，不是 MANUS 规定的唯一设置。

最小设置的实际 CFFI 形态如下；其后仍要添加 Node 与 Chain，才能 load：

```python
setup = ffi.new("SkeletonSetupInfo*")
setup.type = SkeletonType.Hand
setup.settings.targetType = SkeletonTargetType.GloveData
setup.settings.skeletonGloveData.gloveID = 0xBFD38D1E
setup.settings.scaleToTarget = False
setup.settings.useEndPointApproximations = False

setup_index = ffi.new("uint32_t*")
assert lib.CoreSdk_CreateSkeletonSetup(setup[0], setup_index) == SDKReturnCode.Success
# CoreSdk_AddNodeToSkeletonSetup(...)
# CoreSdk_AddChainToSkeletonSetup(...)
# CoreSdk_LoadSkeleton(...)
```

### 从 LDJY 模型提取目标节点

优先使用当前项目已经作为运动学真值的 URDF，而不要新写一份 MJCF 解析器。URDF 同时包含 20 DoF、固定 wrist、tip 与 pad frame；本项目的 `RobotWrapper` 也已经使用 Pinocchio 从 URDF 做 FK。

不能逐项直接拷贝 URDF 的 `<joint><origin>`：它是相对父节点的局部变换。应在机器人零位进行一次 FK，再把每个目标 frame 变换到 wrist 坐标系。这与官方示例提供累计 world-space 节点位置的方式一致：

```python
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
q0 = pin.neutral(model)
pin.framesForwardKinematics(model, data, q0)

root = data.oMf[model.getFrameId("right_retarget_wrist")]
target_frame = data.oMf[model.getFrameId("right_finger1_link2")]
root_T_target = root.inverse() * target_frame
# root_T_target.translation 与 rotation 用于该 NodeSetup 的零位 transform
```

右手第一版可复用官方的 21 节点形式：root=`right_retarget_wrist`；每根手指都取 `link2 -> link3 -> link4 -> tip`。五根链的前缀分别是 `thumb`、`finger1`（食指）、`finger2`（中指）、`finger3`（无名指）、`finger4`（小指）。这里从 `link2` 开始是刻意的：常规 LDJY 指的 `joint1 + joint2` 共同承担人类 MCP 的 spread/flex，而 MANUS 手指链通常以一个 MCP 节点表达该处的三维姿态。下游 Pinocchio IK 再从这些目标节点恢复 LDJY 的四个机械关节。

第一轮不要把原始 URDF link frame 的 rotation 当作“电机转轴定义”。`NodeSetup.rotation` 可以保存 bind/frame 朝向，但它不是 URDF 的 `axis` 与 `limit` 替代品；官方手部示例也仅使用单位 rotation。应先以提取的位置、层级、Chain 语义验证 Core 是否稳定输出，再在 DevTools 中检查 frame 朝向，并继续由本项目 IK 执行关节限位。

若改用 MJCF，原理相同：在 `qpos=0` 后调用 MuJoCo forward，再使用累计后的 `data.xpos/xmat`（或 tip `site` 的全局 pose），而不是直接使用 XML 中局部的 body `pos/quat`。但现有 URDF 已有所有必要 frame，因此没有理由在第一版维护两套提取路径。

目标 Skeleton 可先通过 `CoreSdk_SaveTemporarySkeleton()` 送入 DevTools 检查，再调用 `CoreSdk_LoadSkeleton()` 开始实时重定向；其可持久化文件是 `.mskl`，与手套校准 `.mcal` 完全不同。

## LDJY 的实际构型与适合的混合映射

右手 URDF 表明 LDJY 有 20 自由度，但它们不都与人手关节逐一同构：

| 机器人部分 | 构型事实 | 推荐使用 MANUS 数据的方式 |
| --- | --- | --- |
| 食指、中指、无名指 | 每指有 MCP spread、MCP flex、PIP flex、DIP flex 的清晰四自由度结构 | 直接映射对应的 4 个 Ergonomics 角，做零位偏置、符号、增益和关节限位处理 |
| 小指 | 基部含掌内 cup/opposition 类轴，再有 spread 和两段 flex；与人体的四项不是逐项同构 | 用 Raw Skeleton 的小指 tip、骨段方向、相对无名指展开量建立任务空间目标；Ergonomics 作为软先验 |
| 拇指 | 多个轴方向与人体 CMC/MCP/IP 不一一对应 | 用 tip 位置、末段方向、pad 朝向、与食指/中指的捏合距离做任务空间 IK；Ergonomics 作为软先验 |

对三根常规手指，初始形式应是：

```text
q_robot = clamp(q_zero + sign * gain * radians(ergo - ergo_neutral), joint_lower, joint_upper)
```

其中 `sign`、`gain`、`ergo_neutral` 必须通过单关节动作确定；不能拿一次记录的最小/最大人体角度强行拉伸到机器人全范围，那会导致静止抖动和不可预测的饱和。

对拇指和小指，以任务误差求有界 IK。例如可包含：

- 目标 tip 位置；
- 指段单位方向；
- 拇指末端/指腹法向；
- 拇指到食指、中指 tip 的相对距离（捏合）；
- 小指相对无名指的展开和掌内 cup；
- 对对应 Ergonomics 的低权重软约束。

这是一条“角度优先 + 特殊指任务空间”的混合路线：它保留官方已经验证的人体关节角优势，同时不假装拇指和小指的机构与人体相同。

## 当前项目的真实处理路径

当前运行路径不是官方 Processed Skeleton，而是：

```text
Metaglove Pro
  -> MANUS Integrated SDK 自动载入本机手套校准
  -> Raw Skeleton + Ergonomics 回调
  -> 回调内深拷贝为 ManusFrame（完整节点和拓扑不降为 21 点）
  -> GUI 线程 ManusFullSkeletonRetargeter
  -> 腕部局部化 + MANUS->LDJY 中立对齐
  -> 位置、骨段方向、节点旋转、连续性构成有界最小二乘
  -> URDF 限位内的 LDJY qpos
```

原始 MANUS 节点位置、旋转、缩放、完整拓扑、40 个 Ergonomics 和各自时间戳都在录制中保存；优化时为适配机器人骨长会重建目标点，但不会覆写原始人手数据。

### 项目中“采集 MANUS 中立姿态”的含义

GUI 的“采集 MANUS 中立姿态”要求自然张手稳定约 2 秒。它采集的是同一副已完成官方手套校准的 Raw Skeleton，用来：

1. 消去每次佩戴后 MANUS wrist 在世界中的平移和朝向差；
2. 求 MANUS 手部局部坐标与 LDJY URDF 零位坐标间的固定旋转对齐；
3. 确认当前拓扑、手套 ID、手侧可匹配，并保存为项目 NPZ。

它不修改 `.mcal`，不重做传感器标定，也不替代官方的多姿态校准。官方 `.mcal` 是“一副手套/一只手可长期使用”的校准；项目中立对齐是“这副手套到这台 LDJY 机器人”的映射参数。前者跨项目可迁移，后者应随机器人模型/零位定义一起版本化。

## 建议的下一步验证，而非立刻重写

现有全骨架 IK 已经是可工作的基线。最小、可比较的后续实验是：

1. 固定同一份 `.mcal` 和同一份项目中立对齐；分别记录 Raw Skeleton、Ergonomics、官方目标 Skeleton（若接入）。
2. 对食指/中指/无名指做单关节动作，拟合每个直接角度通道的 `sign`、零偏和合理 gain。
3. 对拇指、小指分别做张开、弯曲、捏食指、捏中指动作，比较“纯全骨架 IK”“混合映射”“官方目标 Skeleton 后再 IK”的 tip 误差、越限率、延迟和主观可控性。
4. 只有当官方目标 Skeleton 在非人形拇指/小指上稳定更好时，再将它保留为可选输入；不要在未比较前替换 Raw Skeleton 基线。

## 参考位置

- 本仓库的 MANUS Integrated 输入实现：[`example/input_devices/manus_glove.py`](../example/input_devices/manus_glove.py)。
- 完整数据模型和当前优化器：[`ldjy_retargeting/manus.py`](../ldjy_retargeting/manus.py)。
- 本项目接入和运行说明：[MANUS 接入实施计划](manus-integration-plan.md)、[MANUS 快速开始](manus-quickstart.md)。
- MANUS SDK 本地示例：`/home/wxx/manus_sdk/Python/examples`，尤其是 `sdk_client.py` 与校准读写示例。
- 官方文档：[SDK getting started](https://docs.manus-meta.com/3.2.0/Plugins/SDK/getting%20started/)、[Python getting started](https://docs.manus-meta.com/3.2.0/Plugins/SDK/Python/getting%20started/)、[Integrated SDK](https://docs.manus-meta.com/3.2.0/Plugins/SDK/SDK%20Integrated/)、[C++ SDK client](https://docs.manus-meta.com/3.2.0/Plugins/SDK/C%2B%2B/SDK%20Client/)。
