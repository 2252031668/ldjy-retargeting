"""MANUS Core 3 Integrated Raw Skeleton input."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from .base import InferenceSample, InputDeviceBase
from ldjy_retargeting.manus import ManusFrame, ManusTopology


_active_device: "ManusGlove | None" = None
_callbacks_defined = False


class ManusGlove(InputDeviceBase):
    """Copy MANUS CFFI buffers in callbacks and expose only native frames."""

    INSTALL_COMMAND = "uv pip install -e /home/wxx/manus_sdk/Python"

    def __init__(self, hand_side: str = "right", stale_after_sec: float = 0.5) -> None:
        global _active_device
        if hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be left or right")
        if _active_device is not None:
            _active_device.cleanup()
        try:
            from manus_sdk import ManusSDK, ffi, lib
            from manus_sdk.generated._enums import (
                AxisPolarity, AxisView, HandMotion, SDKReturnCode, Side,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MANUS Python SDK 未安装。当前已验证的安装命令: " + self.INSTALL_COMMAND
            ) from exc
        self._sdk_types = (ffi, lib, SDKReturnCode)
        self._lock = threading.Lock()
        self._frames: dict[str, ManusFrame] = {}
        self._topologies: dict[tuple[int, int], ManusTopology] = {}
        self._ergonomics: dict[int, tuple[np.ndarray, int, bool]] = {}
        self._ergonomics_size = len(ffi.new("ErgonomicsData*").data)
        self._gloves: dict[int, dict[str, Any]] = {}
        self._arrival_times: deque[float] = deque(maxlen=200)
        self._paused = False
        self._closed = False
        self.connected = False
        self.system_message = ""
        self.hand_side = hand_side
        self.stale_after_sec = float(stale_after_sec)
        self.sdk_version = "3.2.0"
        self.sdk = ManusSDK()
        result = lib.CoreSdk_InitializeIntegrated()
        if result != SDKReturnCode.Success:
            raise RuntimeError(f"CoreSdk_InitializeIntegrated failed: {SDKReturnCode(result).name}")
        _active_device = self
        try:
            self._register_callbacks()
            coordinates = ffi.new("CoordinateSystemVUH*")
            coordinates.view = AxisView.XFromViewer
            coordinates.up = AxisPolarity.PositiveZ
            coordinates.handedness = Side.Right
            coordinates.unitScale = 1.0
            self._check(lib.CoreSdk_InitializeCoordinateSystemWithVUH(coordinates[0], True), "coordinate system")
            if not self._connect():
                raise RuntimeError("MANUS Integrated host not found; do not run SDK Client/Dashboard concurrently")
            self._check(lib.CoreSdk_SetRawSkeletonHandMotion(HandMotion.Auto), "HandMotion.Auto")
        except Exception:
            self.cleanup()
            raise

    def _check(self, result: int, action: str) -> None:
        _, _, code = self._sdk_types
        if result != code.Success:
            raise RuntimeError(f"MANUS {action} failed: {code(result).name}")

    def _register_callbacks(self) -> None:
        global _callbacks_defined
        _, lib, _ = self._sdk_types
        pairs = (
            ("raw_skeleton", "_on_raw_skeleton"),
            ("ergonomics", "_on_ergonomics"),
            ("landscape", "_on_landscape"),
            ("system", "_on_system"),
            ("connected", "_on_connected"),
            ("disconnected", "_on_disconnected"),
        )
        if not _callbacks_defined:
            for name, method in pairs:
                callback = lambda *args, method=method: ManusGlove._dispatch(method, *args)
                self._check(getattr(self.sdk, f"register_{name}_callback")(callback), f"register {name} callback")
            _callbacks_defined = True
            return
        functions = {
            "raw_skeleton": "CoreSdk_RegisterCallbackForRawSkeletonStream",
            "ergonomics": "CoreSdk_RegisterCallbackForErgonomicsStream",
            "landscape": "CoreSdk_RegisterCallbackForLandscapeStream",
            "system": "CoreSdk_RegisterCallbackForSystemStream",
            "connected": "CoreSdk_RegisterCallbackForOnConnect",
            "disconnected": "CoreSdk_RegisterCallbackForOnDisconnect",
        }
        for name, _ in pairs:
            self._check(getattr(lib, functions[name])(getattr(lib, f"python_{name}_callback")), f"register {name} callback")

    def _connect(self) -> bool:
        ffi, lib, code = self._sdk_types
        if lib.CoreSdk_LookForHosts(1, False) != code.Success:
            return False
        count = ffi.new("uint32_t*")
        if lib.CoreSdk_GetNumberOfAvailableHostsFound(count) != code.Success or count[0] == 0:
            return False
        hosts = ffi.new("ManusHost[]", count[0])
        if lib.CoreSdk_GetAvailableHostsFound(hosts, count[0]) != code.Success:
            return False
        result = lib.CoreSdk_ConnectToHost(hosts[0])
        self.connected = result == code.Success
        if self.connected:
            sdk_version, core_version = ffi.new("ManusVersion*"), ffi.new("ManusVersion*")
            compatible = ffi.new("bool*")
            if lib.CoreSdk_GetVersionsAndCheckCompatibility(sdk_version, core_version, compatible) == code.Success:
                self.sdk_version = ffi.string(sdk_version.versionInfo).decode("utf-8", errors="replace")
        return self.connected

    @staticmethod
    def _dispatch(method: str, *args: Any) -> None:
        if _active_device is not None and not _active_device._closed:
            try:
                getattr(_active_device, method)(*args)
            except Exception as exc:
                _active_device.system_message = f"MANUS callback failed: {exc}"

    def _on_connected(self, host: Any) -> None:
        del host
        self.connected = True

    def _on_disconnected(self, host: Any) -> None:
        del host
        self.connected = False

    def _on_system(self, message: Any) -> None:
        ffi, _, _ = self._sdk_types
        try:
            self.system_message = ffi.string(message.infoString).decode("utf-8", errors="replace")
        except Exception:
            self.system_message = f"MANUS system message {int(message.type)}"

    def _on_landscape(self, landscape: Any) -> None:
        copied: dict[int, dict[str, Any]] = {}
        devices = landscape.gloveDevices
        for index in range(int(devices.gloveCount)):
            glove = devices.gloves[index]
            copied[int(glove.id)] = {
                "side": {1: "left", 2: "right"}.get(int(glove.side), "unknown"),
                "battery": int(glove.batteryPercentage),
                "signal": int(glove.transmissionStrength),
                "dongle_id": int(glove.dongleID),
            }
        with self._lock:
            if copied != self._gloves:
                self._topologies.clear()
            self._gloves = copied

    def _on_ergonomics(self, stream: Any) -> None:
        timestamp = int(stream.publishTime.time)
        copied = {}
        for index in range(int(stream.dataCount)):
            data = stream.data[index]
            if bool(data.isUserID):
                continue
            copied[int(data.id)] = (np.asarray(list(data.data), dtype=np.float64).copy(), timestamp, True)
        with self._lock:
            self._ergonomics.update(copied)

    @staticmethod
    def topology_from_cffi(nodes: Any, node_info: Any, count: int) -> ManusTopology:
        """Copy topology by node ID, independent of SDK array ordering."""
        info_by_id = {int(node_info[i].nodeId): node_info[i] for i in range(count)}
        ordered = [info_by_id[int(nodes[i].id)] for i in range(count)]
        return ManusTopology(
            np.array([int(nodes[i].id) for i in range(count)]),
            np.array([int(info.parentId) for info in ordered]),
            np.array([int(info.chainType) for info in ordered]),
            np.array([int(info.side) for info in ordered]),
            np.array([int(info.fingerJointType) for info in ordered]),
        )

    def _on_raw_skeleton(self, stream: Any) -> None:
        if self._paused:
            return
        ffi, lib, code = self._sdk_types
        now = time.monotonic()
        frames: list[ManusFrame] = []
        for index in range(int(stream.skeletonsCount)):
            info = ffi.new("RawSkeletonInfo*")
            if lib.CoreSdk_GetRawSkeletonInfo(index, info) != code.Success:
                continue
            count, glove_id = int(info.nodesCount), int(info.gloveId)
            nodes = ffi.new("SkeletonNode[]", count)
            if lib.CoreSdk_GetRawSkeletonData(index, nodes, count) != code.Success:
                continue
            key = (glove_id, count)
            with self._lock:
                topology = self._topologies.get(key)
            node_ids = {int(nodes[i].id) for i in range(count)}
            if topology is not None and node_ids != set(topology.node_ids.tolist()):
                topology = None
            if topology is None:
                node_info = ffi.new("NodeInfo[]", count)
                if lib.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, node_info, count) != code.Success:
                    continue
                topology = self.topology_from_cffi(nodes, node_info, count)
                with self._lock:
                    self._topologies = {cached_key: value for cached_key, value in self._topologies.items()
                                        if cached_key[0] != glove_id}
                    self._topologies[key] = topology
            node_by_id = {int(nodes[i].id): nodes[i] for i in range(count)}
            ordered_nodes = [node_by_id[int(node_id)] for node_id in topology.node_ids]
            positions = np.array([[node.transform.position.x, node.transform.position.y,
                                   node.transform.position.z] for node in ordered_nodes], dtype=np.float64)
            rotations = np.array([[node.transform.rotation.w, node.transform.rotation.x,
                                   node.transform.rotation.y, node.transform.rotation.z]
                                  for node in ordered_nodes], dtype=np.float64)
            scales = np.array([[node.transform.scale.x, node.transform.scale.y,
                                node.transform.scale.z] for node in ordered_nodes], dtype=np.float64)
            side_values = topology.sides[topology.sides != 0]
            with self._lock:
                landscape_side = self._gloves.get(glove_id, {}).get("side")
            side = ({1: "left", 2: "right"}.get(int(side_values[0]), landscape_side)
                    if len(side_values) else landscape_side)
            if side not in {"left", "right"}:
                continue
            with self._lock:
                ergonomics, ergo_time, ergo_valid = self._ergonomics.get(
                    glove_id, (np.full(self._ergonomics_size, np.nan), 0, False)
                )
            frames.append(ManusFrame(now, int(stream.publishTime.time), glove_id, side,
                                     positions, rotations, scales, topology,
                                     ergonomics, ergo_time, ergo_valid))
        if not frames:
            return
        with self._lock:
            for frame in frames:
                self._frames[frame.side] = frame
            self._arrival_times.append(now)
        for frame in frames:
            self.publish_inference_sample(InferenceSample(
                frame.received_timestamp_sec, None, "manus", frame.side, True, {"frame": frame}
            ))

    def get_latest_frame(self) -> ManusFrame | None:
        with self._lock:
            frame = self._frames.get(self.hand_side)
        if frame is None or time.monotonic() - frame.received_timestamp_sec > self.stale_after_sec:
            return None
        return frame

    @property
    def fps(self) -> float:
        with self._lock:
            values = list(self._arrival_times)
        return 0.0 if len(values) < 2 else (len(values) - 1) / max(values[-1] - values[0], 1e-9)

    @property
    def glove_status(self) -> dict[str, Any]:
        frame = self.get_latest_frame()
        if frame is None:
            with self._lock:
                glove = next((dict(values, glove_id=glove_id) for glove_id, values in self._gloves.items()
                              if values.get("side") == self.hand_side), None)
            return {"glove_id": None, "side": self.hand_side, "connected": self.connected} | (glove or {})
        return {"glove_id": frame.glove_id, "side": frame.side, "connected": self.connected,
                **self._gloves.get(frame.glove_id, {})}

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def cleanup(self) -> None:
        global _active_device
        if self._closed:
            return
        self._closed = True
        _, lib, _ = self._sdk_types
        lib.CoreSdk_ShutDown()
        if _active_device is self:
            _active_device = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MANUS Raw Skeleton Integrated self-check")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fixture", type=Path, help="save one lossless frame/topology fixture")
    args = parser.parse_args(argv)
    device = ManusGlove(args.hand)
    try:
        deadline, printed = time.monotonic() + args.seconds, False
        while time.monotonic() < deadline:
            frame = device.get_latest_frame()
            if frame is not None and not printed:
                for values in zip(frame.topology.node_ids, frame.topology.parent_ids,
                                  frame.topology.chain_types, frame.topology.finger_joint_types):
                    print("node=%d parent=%d chain=%d joint=%d" % tuple(values))
                printed = True
            time.sleep(.05)
        frame = device.get_latest_frame()
        if frame is None:
            with device._lock:
                sides = sorted(device._frames)
            detail = device.system_message or "SDK reported no additional status"
            raise RuntimeError(
                f"no fresh {args.hand} MANUS Raw Skeleton frame received; "
                f"available sides={sides}; last status={detail}"
            )
        print(f"glove={frame.glove_id} side={frame.side} nodes={frame.topology.node_count} fps={device.fps:.1f}")
        if args.fixture:
            args.fixture.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.fixture, positions=frame.positions, rotations=frame.rotations,
                                scales=frame.scales, **{name: getattr(frame.topology, name) for name in (
                                    "node_ids", "parent_ids", "chain_types", "sides", "finger_joint_types")})
    finally:
        device.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
