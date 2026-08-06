import numpy as np
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ldjy_retargeting.pad_calibration import SurfacePoint, TriangleSurface


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def test_surface_walk_crosses_only_the_shared_triangle_edge():
    surface = TriangleSurface(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [1, 3, 2]]),
    )
    start = SurfacePoint(0, np.array([0.10, 0.45, 0.45]))

    end = surface.move(start, np.array([1.0, 0.0, 0.0]), 0.40)

    assert end.face == 1
    assert np.all(end.barycentric >= -1e-9)
    assert np.isclose(end.barycentric.sum(), 1.0)
    point = surface.position(end)
    triangle = surface.vertices[surface.faces[end.face]]
    np.testing.assert_allclose(point, end.barycentric @ triangle)


def test_surface_walk_transports_direction_across_a_sharp_edge():
    surface = TriangleSurface(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 3, 1]]),
    )
    start = SurfacePoint(0, np.array([0.25, 0.25, 0.50]))

    end, transport = surface.move_with_transport(start, np.array([0.0, -1.0, 0.0]), 0.70)

    assert end.face == 1
    assert surface.position(end)[2] > 0.1
    assert abs((transport @ np.array([0.0, -1.0, 0.0])) @ surface.normals[end.face]) < 1e-9


def test_calibrator_keeps_moving_up_across_the_initial_sharp_edge():
    sys.path.insert(0, str(ROOT / "tools"))
    from calibrate_ldjy_pads import PadCalibrator

    calibrator = PadCalibrator()
    start = calibrator.surfaces[calibrator.active].position(calibrator.points[calibrator.active])
    for _ in range(12):
        calibrator._move_active("up")
    end = calibrator.surfaces[calibrator.active].position(calibrator.points[calibrator.active])

    assert np.linalg.norm(end - start) > 0.002


def test_calibrator_draws_one_normal_segment_per_pad():
    import mujoco
    sys.path.insert(0, str(ROOT / "tools"))
    from calibrate_ldjy_pads import (
        MARKER_OFFSET_M,
        MARKER_RADIUS_M,
        NORMAL_LENGTH_M,
        PadCalibrator,
    )

    calibrator = PadCalibrator()
    scene = mujoco.MjvScene(calibrator.model, maxgeom=5)

    calibrator.draw_normals(scene)

    assert scene.ngeom == 5
    for index in range(scene.ngeom):
        assert scene.geoms[index].type == mujoco.mjtGeom.mjGEOM_CAPSULE
    position, normal = calibrator._display_normal("finger1")
    body_id = mujoco.mj_name2id(
        calibrator.model, mujoco.mjtObj.mjOBJ_BODY, "finger1_link4"
    )
    rotation = calibrator.data.xmat[body_id].reshape(3, 3)
    world_position = calibrator.data.xpos[body_id] + rotation @ position
    world_normal = rotation @ normal
    np.testing.assert_allclose(
        scene.geoms[0].pos,
        world_position
        + world_normal * (MARKER_OFFSET_M + MARKER_RADIUS_M + NORMAL_LENGTH_M / 2.0),
    )


def test_startup_instructions_list_selection_movement_and_save(capsys):
    sys.path.insert(0, str(ROOT / "tools"))
    from calibrate_ldjy_pads import print_instructions

    print_instructions()

    output = capsys.readouterr().out
    assert "双击红球" in output
    assert "1: 拇指" in output
    assert "方向键" in output
    assert "S: 保存" in output


def test_builders_add_fixed_pad_nodes_and_sites_from_calibration_points():
    from build_ldjy_mjcf import add_pad_sites
    from build_ldjy_urdf import add_pad_frames

    points = {
        "finger1": np.array([0.001, 0.002, 0.003]),
        "finger2": np.array([0.004, 0.005, 0.006]),
        "finger3": np.array([0.007, 0.008, 0.009]),
        "thumb": np.array([0.010, 0.011, 0.012]),
        "finger4": np.array([0.013, 0.014, 0.015]),
    }
    urdf = ET.Element("robot")
    add_pad_frames(urdf, points)

    joint = urdf.find("./joint[@name='finger2_pad_fixed']")
    assert joint.find("parent").attrib["link"] == "finger2_link4"
    assert joint.find("child").attrib["link"] == "finger2_pad"
    np.testing.assert_allclose(
        np.fromstring(joint.find("origin").attrib["xyz"], sep=" "), points["finger2"]
    )

    mjcf = ET.fromstring(
        "<mujoco><worldbody>" + "".join(
            f'<body name="right_{finger}_link4" />' for finger in points
        ) + "</worldbody></mujoco>"
    )
    add_pad_sites(mjcf, points, "right")
    site = mjcf.find(".//site[@name='right_finger2_pad_center']")
    np.testing.assert_allclose(np.fromstring(site.attrib["pos"], sep=" "), points["finger2"])
