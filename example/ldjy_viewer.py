"""Open the LDJY hand in MuJoCo with its built-in joint sliders.

Usage:
    .venv/bin/python example/ldjy_viewer.py
    .venv/bin/python example/ldjy_viewer.py --left
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ldjy_retargeting.pad_calibration import load_pad_points


ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")
PAD_MARKER_RADIUS_M = 0.002
PAD_MARKER_RGBA = np.array((1.0, 0.15, 0.05, 1.0))
NORMAL_RADIUS_M = 0.0003
NORMAL_LENGTH_M = 0.008
NORMAL_RGBA = np.array((1.0, 0.85, 0.1, 1.0))
IDENTITY_MAT = np.eye(3).ravel()


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left",
        action="store_const",
        const="left",
        dest="side",
        default="right",
        help="open the left hand (the right hand is the default)",
    )
    return parser.parse_args(args)


def mjcf_path(side: str) -> Path:
    return ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand" / "mjcf" / f"ldjy_{side}_hand.xml"


def set_zero_controls(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for index in range(model.nu):
        if model.actuator_ctrllimited[index]:
            low, high = model.actuator_ctrlrange[index]
            data.ctrl[index] = np.clip(0.0, low, high)
        else:
            data.ctrl[index] = 0.0


def draw_pad_points(scene, model: mujoco.MjModel, data: mujoco.MjData, side: str) -> None:
    """Draw calibrated pad sites without loading or scanning mesh geometry."""
    scene.ngeom = 0
    for finger in FINGERS:
        if scene.ngeom >= scene.maxgeom:
            return
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_pad_center"
        )
        if site_id < 0:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array((PAD_MARKER_RADIUS_M, 0.0, 0.0)),
            data.site_xpos[site_id],
            IDENTITY_MAT,
            PAD_MARKER_RGBA,
        )
        geom.label = f"{finger} pad"
        scene.ngeom += 1


def draw_pad_normals(scene, model: mujoco.MjModel, data: mujoco.MjData, side: str) -> None:
    """Draw stored pad-site normal axes without loading mesh geometry."""
    for finger in FINGERS:
        if scene.ngeom >= scene.maxgeom:
            return
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_pad_center"
        )
        if site_id < 0:
            continue
        normal = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        quat = np.zeros(4)
        mat = np.zeros(9)
        mujoco.mju_quatZ2Vec(quat, normal)
        mujoco.mju_quat2Mat(mat, quat)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array((NORMAL_RADIUS_M, NORMAL_LENGTH_M / 2.0, 0.0)),
            data.site_xpos[site_id] + normal * (PAD_MARKER_RADIUS_M + NORMAL_LENGTH_M / 2.0),
            mat,
            NORMAL_RGBA,
        )
        scene.ngeom += 1


def main() -> None:
    side = parse_args().side
    path = mjcf_path(side)
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {path}")

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    set_zero_controls(model, data)
    has_calibrated_pads = load_pad_points() is not None
    if not has_calibrated_pads:
        print("No pad calibration found; run tools/calibrate_ldjy_pads.py first.")
    for _ in range(100):
        mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0, 0, 0.05]
        while viewer.is_running():
            started = time.time()
            mujoco.mj_step(model, data)
            if has_calibrated_pads:
                draw_pad_points(viewer.user_scn, model, data, side)
                draw_pad_normals(viewer.user_scn, model, data, side)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - started)))


if __name__ == "__main__":
    main()
