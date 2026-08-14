from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ldjy_retargeting.mano_compare import (
    WILOR_LEFT_FROM_RIGHT,
    build_comparison_hands,
    load_robot_mano_reference,
)


class FakeMANO:
    faces = np.asarray(
        [[vid, 10, 11] for vid in (763, 355, 438, 573, 690, 328, 343, 350, 455, 439)],
        dtype=np.int32,
    )

    def compute(self, betas, hand_pose, global_orient, translation, scale):
        vertices = np.zeros((778, 3), dtype=np.float64)
        vertices[0] = [0.0, 0.0, 0.0]
        vertices[10] = [0.01, 0.0, 0.0]
        vertices[11] = [0.0, 0.01, 0.0]
        for vid in (763, 355, 438, 573, 690, 328, 343, 350, 455, 439):
            vertices[vid] = [0.0, 0.0, 0.01]
        joints = np.zeros((21, 3), dtype=np.float64)
        joints[0] = [0.1, 0.2, 0.3]
        vertices += joints[0]
        return vertices, joints, np.zeros((5, 3), dtype=np.float64)


class CapturingMANO(FakeMANO):
    def __init__(self):
        self.seen_betas = []

    def compute(self, betas, *args):
        self.seen_betas.append(np.asarray(betas, dtype=np.float64).copy())
        return super().compute(betas, *args)


def write_reference(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "mano": {"betas": [float(i) for i in range(10)], "scale": 1.25},
                "mano_to_ldjy_registration": {
                    "rotation_axis_angle": [0.0, 0.0, 0.2],
                    "translation": [0.4, 0.5, 0.6],
                    "scale": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )


def pose_input():
    return {
        "hand_pose": np.tile(
            Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix(), (15, 1, 1)
        ),
        "global_orient": Rotation.from_rotvec([0.2, 0.1, -0.1]).as_matrix(),
        "betas": np.full(10, 0.5),
    }


def test_reference_loads_robot_betas_and_registration_scale(tmp_path):
    path = tmp_path / "reference.yaml"
    write_reference(path)

    reference = load_robot_mano_reference(path, "right")

    np.testing.assert_allclose(reference.beta_robot, np.arange(10))
    assert reference.mesh_scale == 2.5


def test_comparison_hands_use_same_pose_but_different_betas(tmp_path):
    path = tmp_path / "reference.yaml"
    write_reference(path)

    human, robot = build_comparison_hands(
        FakeMANO(), path, "right", pose_input(), np.zeros(3), np.zeros(3)
    )

    np.testing.assert_allclose(human.joints[0], np.zeros(3))
    np.testing.assert_allclose(robot.joints[0], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(human.vertices - human.joints[0], robot.vertices - robot.joints[0])


def test_left_comparison_matches_wilor_x_mirror(tmp_path):
    path = tmp_path / "reference.yaml"
    write_reference(path)
    params = pose_input()

    right, _ = build_comparison_hands(
        FakeMANO(), path, "right", params, np.zeros(3), np.ones(3)
    )
    left, _ = build_comparison_hands(
        FakeMANO(), path, "left", params, np.zeros(3), np.ones(3)
    )

    np.testing.assert_allclose(left.vertices, right.vertices @ WILOR_LEFT_FROM_RIGHT)
    np.testing.assert_allclose(left.pad_normals, right.pad_normals @ WILOR_LEFT_FROM_RIGHT)


def test_recorded_mesh_still_builds_robot_with_beta_robot(tmp_path):
    path = tmp_path / "reference.yaml"
    write_reference(path)
    mano = CapturingMANO()
    vertices, joints, _ = mano.compute(
        pose_input()["betas"], np.zeros((15, 3)), np.zeros(3), np.zeros(3), 1.0
    )
    mano.seen_betas.clear()

    build_comparison_hands(
        mano, path, "right", pose_input(), np.zeros(3), np.ones(3),
        video_vertices=vertices, video_joints=joints,
    )

    assert len(mano.seen_betas) == 1
    np.testing.assert_allclose(mano.seen_betas[0], np.arange(10))


def test_left_recorded_mesh_flips_reflected_normals(tmp_path):
    path = tmp_path / "reference.yaml"
    write_reference(path)
    vertices, joints, _ = FakeMANO().compute(
        pose_input()["betas"], np.zeros((15, 3)), np.zeros(3), np.zeros(3), 1.0
    )

    right, _ = build_comparison_hands(
        FakeMANO(), path, "right", pose_input(), np.zeros(3), np.ones(3),
        video_vertices=vertices, video_joints=joints,
    )
    left, _ = build_comparison_hands(
        FakeMANO(), path, "left", pose_input(), np.zeros(3), np.ones(3),
        video_vertices=vertices @ WILOR_LEFT_FROM_RIGHT,
        video_joints=joints @ WILOR_LEFT_FROM_RIGHT,
    )

    np.testing.assert_allclose(left.pad_normals, right.pad_normals @ WILOR_LEFT_FROM_RIGHT)


def test_comparison_uses_the_recorded_video_mesh_when_available(tmp_path):
    path = tmp_path / "reference.yaml"
    write_reference(path)
    generated, _, _ = FakeMANO().compute(
        pose_input()["betas"], np.zeros((15, 3)), np.zeros(3), np.zeros(3), 1.0
    )
    raw_vertices = generated.copy()
    raw_vertices[500] += [0.02, 0.0, 0.0]
    raw_joints = np.zeros((21, 3), dtype=np.float64)

    from_parameters, _ = build_comparison_hands(
        FakeMANO(), path, "right", pose_input(), np.zeros(3), np.zeros(3)
    )
    from_record, _ = build_comparison_hands(
        FakeMANO(), path, "right", pose_input(), np.zeros(3), np.zeros(3),
        video_vertices=raw_vertices, video_joints=raw_joints,
    )

    assert np.linalg.norm(from_record.vertices[500] - from_parameters.vertices[500]) > 0.01
