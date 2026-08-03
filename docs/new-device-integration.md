# 新输入设备接入

新设备只需实现 `InputDeviceBase.get_fingers_data()` 并返回统一的 `(21, 3)`
关键点接口：

```python
{
    "left_fingers": np.ndarray,   # (21, 3)，单位米
    "right_fingers": np.ndarray,  # (21, 3)，单位米
}
```

未检测到手时返回全零数组。数组顺序采用 MediaPipe/MANO 共用的 21 点顺序，坐标应为米。
MediaPipe 适配器提供归一化相机关键点；WiLoR 适配器提供米制 MANO 关节。`Retargeter` 会统一完成
wrist 对齐和 MANO 坐标变换。

```python
import numpy as np
from input_devices.base import InputDeviceBase


class MyDevice(InputDeviceBase):
    def get_fingers_data(self):
        left, right = self._read_and_convert()
        return {"left_fingers": left, "right_fingers": right}
```

在 `example/teleop_sim.py` 中注册该适配器到 `device_map`，并在 `--input` 的
choices 中加入名称。建议先录制为 pkl，再使用仿真验证：

```bash
uv run --no-sync python example/teleop_sim.py --play example/data/my_device.pkl --hand right --debug
```

确认关键点顺序、单位、掌面方向和关节轨迹后，再接入实时数据流。USB 摄像头输入可通过
`example/tuning_gui.py` 实时调整重定向参数。
