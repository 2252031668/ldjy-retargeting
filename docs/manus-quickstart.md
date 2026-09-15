# MANUS Core 3.2 Linux 快速验证

本项目使用 Linux Integrated Mode。接收器上的许可证必须同时显示 `SDK (Integrated)` 和 `Linux` 可用；测试时不要同时运行官方 SDK Client 或 Dashboard。

## 已验证的官方示例

SDK 位于 `/home/wxx/manus_sdk`，项目虚拟环境中已用 editable 模式安装 Python 包。插入 Wireless Dongle、打开手套，然后运行：

```bash
cd /home/wxx/桌面/agent_loop/ldjy-retargeting
uv run python /home/wxx/manus_sdk/Python/examples/sdk_client.py
```

提示连接模式时输入 `1`（Core Integrated）。看到 Raw Skeleton 持续更新即表示许可证、USB 权限、运行库与手套配对均正常。

不要使用 `sudo python`：它会绕开项目虚拟环境，并可能让 SDK 安装和动态库路径不一致。

## 项目输入自检

官方示例退出后运行：

```bash
uv run python -m example.input_devices.manus_glove --hand right --seconds 5
```

该命令输出真实的 node ID、parent ID、`ChainType`、`FingerJointType` 和接收频率。需要为测试保存当前硬件拓扑时：

```bash
uv run python -m example.input_devices.manus_glove \
  --hand right --seconds 5 --fixture outputs/manus_topology_right.npz
```

## 启动调参 GUI

```bash
uv run python example/tuning_gui.py --manus --hand right
```

点击“应用输入”后，默认 `MANUS Full Skeleton` 需要自然张手保持至少 2 秒并采集中立姿态。成功后标定保存在 `outputs/manus_calibrations/<glove_id>_<side>.npz`；它是 MANUS→LDJY 重定向标定，不是 MANUS 自身的 `.mcal` 手套校准。

如果 Python 包丢失，当前环境已验证的安装方式是：

```bash
uv pip install -e /home/wxx/manus_sdk/Python
```

若出现 `No license found ... Integrated` 或 `does not allow receiving data on Linux`，需要回到 Windows Dashboard 检查同一个 Dongle 的 `SDK (Integrated)` 与 `Linux` 授权；代码无法绕过许可证。
