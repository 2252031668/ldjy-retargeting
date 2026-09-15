"""Native MANUS model, calibration, solver, and record checks."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from ldjy_retargeting.manus import ManusCalibration, ManusFrame, ManusTopology, average_quaternions_wxyz
from ldjy_retargeting.tuning.runtime import TuningRuntime
from ldjy_retargeting.tuning.session import TuningSession


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example"))


def topology(side: int = 2, self_parent_root: bool = False) -> ManusTopology:
    chains, joints, parents = [13], [0], [-1]
    for chain in (5, 6, 7, 8, 9):
        base = len(chains)
        chains += [chain] * 5
        joints += [1, 2, 3, 4, 5]
        parents += [0, base, base + 1, base + 2, base + 3]
    node_ids = np.arange(100, 126)
    return ManusTopology(
        node_ids, [node_ids[0] if parent < 0 and self_parent_root else
                   (-1 if parent < 0 else node_ids[parent]) for parent in parents],
        chains, [side] * len(chains), joints,
    )


def runtime(side: str = "right") -> TuningRuntime:
    config = TuningSession(ROOT / "example/config/manus_full_skeleton.yaml").config
    return TuningRuntime(config, side, ROOT / "example/config")


def hybrid_runtime(side: str = "right") -> TuningRuntime:
    config = TuningSession(ROOT / "example/config/manus_ergonomics_hybrid.yaml").config
    return TuningRuntime(config, side, ROOT / "example/config")


def robot_frame(optimizer, topo: ManusTopology, qpos: np.ndarray, stamp: int = 1) -> ManusFrame:
    optimizer._configure_topology(topo)
    optimizer.robot.compute_forward_kinematics(qpos)
    positions = np.zeros((topo.node_count, 3))
    rotations = np.tile([1., 0., 0., 0.], (topo.node_count, 1))
    for source_index, frame_id in zip(optimizer._frame_indices, optimizer._robot_frame_ids):
        pose = optimizer.robot.get_link_pose(frame_id)
        xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        positions[source_index] = pose[:3, 3]
        rotations[source_index] = xyzw[[3, 0, 1, 2]]
    return ManusFrame(stamp, stamp, 77, optimizer.hand_side, positions, rotations,
                      np.ones((topo.node_count, 3)), topo, np.arange(40), stamp - 1, True)


def test_frame_deep_copies_and_accepts_arbitrary_ids_and_order():
    topo = topology()
    source = np.zeros((26, 3))
    frame = ManusFrame(1, 2, 77, "right", source, np.tile([1, 0, 0, 0], (26, 1)),
                       np.ones((26, 3)), topo, np.arange(40), 1, True)
    source[:] = 9
    assert not frame.positions.any()
    permutation = np.arange(26)[::-1]
    shuffled = ManusTopology(*(getattr(topo, name)[permutation] for name in (
        "node_ids", "parent_ids", "chain_types", "sides", "finger_joint_types"
    )))
    assert topo.matches(shuffled)


def test_quaternion_sign_and_calibration_round_trip_reject_mismatch():
    np.testing.assert_allclose(average_quaternions_wxyz([[1, 0, 0, 0], [-1, 0, 0, 0]]), [1, 0, 0, 0])
    rt = runtime()
    frame = robot_frame(rt.optimizer, topology(), np.zeros(20))
    calibration = rt.optimizer.calibrate([robot_frame(rt.optimizer, topology(), np.zeros(20), i) for i in range(50)])
    with tempfile.TemporaryDirectory() as directory:
        path = calibration.save(Path(directory) / "c.npz")
        loaded = ManusCalibration.load(path)
        assert loaded.matches(frame)
        wrong = ManusFrame(1, 2, 78, "right", frame.positions, frame.rotations,
                           frame.scales, frame.topology, frame.ergonomics)
        assert not loaded.matches(wrong)


def test_synthetic_full_skeleton_recovers_qpos_within_limits():
    rt = runtime()
    original = topology()
    permutation = np.arange(original.node_count)[::-1]
    topo = ManusTopology(*(getattr(original, name)[permutation] for name in (
        "node_ids", "parent_ids", "chain_types", "sides", "finger_joint_types"
    )))
    rt.optimizer.calibrate([robot_frame(rt.optimizer, original, np.zeros(20), i) for i in range(50)])
    target = np.array([.1, .2, .3, .4] * 3 + [.2, .1, .3, .4] + [.1, .2, -.3, -.4])
    ordered = robot_frame(rt.optimizer, original, target, 100)
    observed = ManusFrame(ordered.received_timestamp_sec, ordered.sdk_timestamp, ordered.glove_id,
                          ordered.side, ordered.positions[permutation], ordered.rotations[permutation],
                          ordered.scales[permutation], topo, ordered.ergonomics,
                          ordered.ergonomics_timestamp, ordered.ergonomics_valid)
    solved, diagnostics = rt.optimizer.solve(observed)
    assert np.mean(np.abs(solved - target)) < .05
    assert np.all(solved >= rt.optimizer.limits[:, 0]) and np.all(solved <= rt.optimizer.limits[:, 1])
    assert diagnostics["manus_target_positions_m"].shape == (26, 3)
    np.testing.assert_array_equal(diagnostics["manus_raw_positions_m"],
                                  observed.positions)
    targets, parents = diagnostics["manus_target_positions_m"], diagnostics["manus_parent_indices"]
    zero = rt.optimizer._robot_zero_positions
    for index, parent in enumerate(parents):
        if parent >= 0:
            np.testing.assert_allclose(
                np.linalg.norm(targets[index] - targets[parent]),
                np.linalg.norm(zero[index] - zero[parent]), atol=1e-10,
            )


def test_manus_record_round_trip_without_sdk():
    from ldjy_retargeting.tuning.recording import ManusRecordWriter, load_manus_record
    from ldjy_retargeting.tuning.replay import ManusReplay

    rt, topo = runtime(), topology()
    frames = [robot_frame(rt.optimizer, topo, np.zeros(20), i + 10) for i in range(2)]
    calibration = rt.optimizer.calibrate([robot_frame(rt.optimizer, topo, np.zeros(20), i) for i in range(50)])
    with tempfile.TemporaryDirectory() as directory:
        writer = ManusRecordWriter.start(directory, hand_side="right", calibration=calibration)
        for frame in frames:
            writer.append(frame)
        info = writer.finish(config={"optimizer": {"type": "ManusFullSkeletonRetargeter"}})
        record = load_manus_record(info.path)
        replayed = ManusReplay(record).input_at(1)
        assert replayed.topology.matches(frames[1].topology)
        np.testing.assert_array_equal(replayed.positions, frames[1].positions)
        np.testing.assert_array_equal(replayed.rotations, frames[1].rotations)
        np.testing.assert_array_equal(replayed.scales, frames[1].scales)
        np.testing.assert_array_equal(replayed.ergonomics, frames[1].ergonomics)
        assert replayed.sdk_timestamp == frames[1].sdk_timestamp
        assert replayed.received_timestamp_sec == frames[1].received_timestamp_sec
        assert replayed.ergonomics_timestamp == frames[1].ergonomics_timestamp
        assert replayed.ergonomics_valid == frames[1].ergonomics_valid
        assert (info.path / "calibration_snapshot.npz").is_file()
        snapshot_session = TuningSession(ROOT / "example/config/manus_full_skeleton.yaml")
        snapshot_session.load_snapshot(info.path / "config_snapshot.yaml")
        assert snapshot_session.config["optimizer"]["type"] == "ManusFullSkeletonRetargeter"


def test_left_hand_uses_its_own_topology_and_limits():
    rt, topo = runtime("left"), topology(1)
    calibration = rt.optimizer.calibrate([
        robot_frame(rt.optimizer, topo, np.zeros(20), i) for i in range(50)
    ])
    assert calibration.side == "left"
    target = np.full(20, .1)
    solved, _ = rt.optimizer.solve(robot_frame(rt.optimizer, topo, target, 100))
    assert np.mean(np.abs(solved - target)) < .05
    assert np.all(solved >= rt.optimizer.limits[:, 0])
    assert np.all(solved <= rt.optimizer.limits[:, 1])


def test_cffi_topology_copy_matches_by_id_and_landscape_invalidates_cache():
    import threading
    from types import SimpleNamespace as S
    from input_devices.manus_glove import ManusGlove

    nodes = [S(id=9), S(id=3)]
    infos = [S(nodeId=3, parentId=9, chainType=6, side=2, fingerJointType=1),
             S(nodeId=9, parentId=0, chainType=13, side=2, fingerJointType=0)]
    copied = ManusGlove.topology_from_cffi(nodes, infos, 2)
    infos[1].chainType = 99
    assert copied.node_ids.tolist() == [9, 3]
    assert copied.chain_types.tolist() == [13, 6]

    device = ManusGlove.__new__(ManusGlove)
    device._lock, device._topologies = threading.Lock(), {(1, 2): copied}
    device._gloves = {}
    glove = S(id=1, side=2, batteryPercentage=50, transmissionStrength=75, dongleID=2)
    device._on_landscape(S(gloveDevices=S(gloveCount=1, gloves=[glove])))
    assert device._topologies == {}


def test_manus_self_parent_hand_node_is_a_root_not_a_cycle():
    rt, topo = runtime(), topology(self_parent_root=True)
    frames = [robot_frame(rt.optimizer, topo, np.zeros(20), i) for i in range(50)]
    calibration = rt.optimizer.calibrate(frames)
    assert calibration.topology.parent_ids[0] == calibration.topology.node_ids[0]
    root = next(i for i, source in enumerate(rt.optimizer._frame_indices)
                if topo.chain_types[source] == 13)
    assert rt.optimizer._parent_mapped_indices[root] == -1


def test_ergonomics_hybrid_uses_shared_raw_calibration_and_outputs_bounded_qpos():
    rt, source, topo = hybrid_runtime(), runtime(), topology()
    calibration = source.optimizer.calibrate([
        robot_frame(source.optimizer, topo, np.zeros(20), stamp) for stamp in range(50)
    ])
    rt.optimizer.set_raw_calibration(calibration)
    raw = robot_frame(source.optimizer, topo, np.full(20, .1), 600)
    values = raw.ergonomics.copy()
    values[[24, 28, 32]] += 20
    frame = ManusFrame(raw.received_timestamp_sec, raw.sdk_timestamp, raw.glove_id, raw.side,
                       raw.positions, raw.rotations, raw.scales, raw.topology, values, 600, True)
    qpos, diagnostics = rt.process_manus(frame)
    assert np.isfinite(qpos).all() and np.all(qpos >= rt.optimizer.limits[:, 0]) and np.all(qpos <= rt.optimizer.limits[:, 1])
    assert diagnostics["ergonomics_direct_qpos"].shape == (12,)
