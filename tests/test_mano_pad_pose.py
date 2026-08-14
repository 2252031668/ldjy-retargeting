from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from ldjy_retargeting.mano_pad_pose import (
    ManoPadTargetBuilder,
    ManoPoseInput,
    triangle_mesh_distance,
)


def test_triangle_mesh_distance_measures_parallel_surfaces():
    lower = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    upper = lower + [0.0, 0.0, 0.004]

    assert triangle_mesh_distance(lower, upper) == pytest.approx(0.004)


def test_triangle_mesh_distance_detects_intersection():
    horizontal = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    vertical = np.array([[[0.25, 0.25, -1.0], [0.25, 0.25, 1.0], [0.75, 0.25, 0.0]]])

    assert triangle_mesh_distance(horizontal, vertical) == pytest.approx(0.0)


class FakeMANO:
    def __init__(self):
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.calls = []

    def compute(self, betas, hand_pose, global_orient, translation, scale):
        self.calls.append((np.asarray(betas), np.asarray(hand_pose), np.asarray(global_orient), np.asarray(translation), scale))
        vertices = np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float64,
        )
        joints = np.zeros((21, 3), dtype=np.float64)
        joints[0] = [0.001, 0.002, 0.003]
        joints[9] = [0.001, 0.022, 0.003]
        return vertices, joints, np.zeros((5, 3), dtype=np.float64)


class ShapeDependentFakeMANO:
    def __init__(self):
        self.faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int32)

    def compute(self, betas, hand_pose, global_orient, translation, scale):
        robot_shape = np.isclose(np.asarray(betas)[0], 0.0)
        pad_distance = 0.02 if robot_shape else 0.01
        vertices = np.array(
            [[0.0, 0.0, 0.0], [pad_distance, 0.0, 0.0], [0.0, 0.01, 0.0],
             [0.0, 0.0, 0.01], [0.0, -0.01, 0.0]],
            dtype=np.float64,
        )
        joints = np.zeros((21, 3), dtype=np.float64)
        joints[9, 0] = 0.02 if robot_shape else 0.01
        return vertices, joints, np.zeros((5, 3), dtype=np.float64)


def _reference(path: Path):
    path.write_text(
        yaml.safe_dump(
            {
                "mano": {"betas": [float(i) for i in range(10)], "scale": 1.25},
                "mano_to_ldjy_registration": {
                    "rotation_axis_angle": [0.0, 0.0, 0.0],
                    "translation": [0.4, 0.5, 0.6],
                    "scale": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )


def _builder(tmp_path, side="right"):
    reference = tmp_path / "reference.yaml"
    _reference(reference)
    return ManoPadTargetBuilder(
        FakeMANO(),
        reference,
        side,
        pad_vertex_ids={name: 0 for name in ("Thumb", "Index", "Middle", "Ring", "Pinky")},
        pad_3pt_vertex_ids={},
        pad_order=("Thumb", "Index", "Middle", "Ring", "Pinky"),
    )


def _input(hand_pose=None, global_orient=None, translation=None, betas=None):
    return ManoPoseInput(
        hand_pose=np.eye(3)[None].repeat(15, axis=0) if hand_pose is None else hand_pose,
        global_orient=np.eye(3) if global_orient is None else global_orient,
        translation=np.zeros(3) if translation is None else translation,
        betas=np.full(10, 99.0) if betas is None else betas,
    )


def test_target_uses_robot_betas_and_absolute_hand_pose(tmp_path):
    builder = _builder(tmp_path)
    pose = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    current = np.repeat(pose[None], 15, axis=0)

    target = builder.build(_input(hand_pose=current))

    model = builder.mano_model
    betas, hand_pose, global_orient, translation, scale = model.calls[-2]
    np.testing.assert_allclose(betas, np.arange(10))
    np.testing.assert_allclose(hand_pose, np.repeat(Rotation.from_matrix(pose).as_rotvec()[None], 15, axis=0))
    np.testing.assert_allclose(global_orient, np.zeros(3))
    np.testing.assert_allclose(translation, np.zeros(3))
    assert scale == 1.25
    assert target.pad_positions.shape == (5, 3)
    assert target.pad_normals.shape == (5, 3)


def test_left_target_mirrors_normals_with_overlay_convention(tmp_path):
    right = _builder(tmp_path, "right").build(_input())
    left = _builder(tmp_path, "left").build(_input())
    mirror = np.diag((1.0, -1.0, 1.0))

    np.testing.assert_allclose(left.pad_positions, right.pad_positions @ mirror)
    np.testing.assert_allclose(left.pad_normals, right.pad_normals @ mirror)


def test_translation_and_global_orient_do_not_change_wrist_relative_targets(tmp_path):
    builder = _builder(tmp_path)
    first = builder.build(_input())
    second = builder.build(
        _input(
            global_orient=Rotation.from_rotvec([0.3, 0.2, -0.4]).as_matrix(),
            translation=[4.0, 5.0, 6.0],
        )
    )

    np.testing.assert_allclose(first.pad_positions, second.pad_positions)
    np.testing.assert_allclose(first.pad_normals, second.pad_normals)


def test_pinch_distances_follow_video_shape_scaled_to_robot_hand(tmp_path):
    reference = tmp_path / "reference.yaml"
    _reference(reference)
    builder = ManoPadTargetBuilder(
        ShapeDependentFakeMANO(), reference, "right",
        pad_vertex_ids={
            "Thumb": 0, "Index": 1, "Middle": 2, "Ring": 3, "Pinky": 4,
        },
        pad_3pt_vertex_ids={},
        pad_order=("Thumb", "Index", "Middle", "Ring", "Pinky"),
    )

    target = builder.build(_input())

    # 1 cm video pinch * (2 cm robot scale / 1 cm video scale) * 2x registration.
    assert target.pinch_distances_m[1] == 0.04
