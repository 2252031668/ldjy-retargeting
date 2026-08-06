#!/usr/bin/env python3
"""Interactively calibrate the five LDJY finger-pad surface points.

Usage:
    .venv/bin/python tools/calibrate_ldjy_pads.py

Double-click a red ball to select it.  Arrow keys move the selected point on
its own link4 mesh; 1..5 select Thumb/Index/Middle/Ring/Pinky; S saves the
five link4-local points.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from mujoco.glfw import glfw

from build_ldjy_urdf import CAD_NAIL_TO_PULP, SOURCE_URDF, distal_tip_positions, numbers
from ldjy_retargeting.pad_calibration import (
    FINGERS,
    SurfacePoint,
    TriangleSurface,
    link4_visual_surface,
    save_pad_points,
)
from ldjy_retargeting.retarget_tip_frames import task_frame_axes


MARKER_RADIUS_M = 0.002
MARKER_OFFSET_M = 0.0015
NORMAL_RADIUS_M = 0.0003
NORMAL_LENGTH_M = 0.008
STEP_M = 0.0005
FINGER_KEYS = {
    "1": "thumb",
    "2": "finger1",
    "3": "finger2",
    "4": "finger3",
    "5": "finger4",
}
ARROW_KEYS = {
    glfw.KEY_UP: "up",
    glfw.KEY_DOWN: "down",
    glfw.KEY_LEFT: "left",
    glfw.KEY_RIGHT: "right",
}


def print_instructions() -> None:
    print("=" * 58)
    print("LDJY 指腹点标定")
    print("双击红球：选择要编辑的指腹点")
    print("1: 拇指  2: 食指  3: 中指  4: 无名指  5: 小指")
    print("方向键：沿 STL 表面移动当前红球")
    print("  上/下：初始朝指尖/手根方向；跨面后方向随表面延续")
    print("  左/右：指腹横向移动")
    print("S: 保存五个点。保存后执行 URDF/MJCF 构建脚本。")
    print("=" * 58)


def surfaces_from_model(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, TriangleSurface]:
    """Extract each visual link4 mesh in its parent body coordinates."""
    return {finger: link4_visual_surface(model, data, finger) for finger in FINGERS}


def _write_marker_xml(path: Path, points: dict[str, SurfacePoint], surfaces: dict[str, TriangleSurface]) -> None:
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("fusestatic", "false")
    bodies = {body.attrib["name"]: body for body in root.findall(".//body")}
    for finger in FINGERS:
        surface = surfaces[finger]
        position = surface.position(points[finger])
        normal = surface.normals[points[finger].face]
        if normal @ (position - surface.vertices.mean(axis=0)) < 0.0:
            normal = -normal
        marker = ET.SubElement(
            bodies[f"{finger}_link4"],
            "body",
            {"name": f"calibration_{finger}", "pos": numbers(position + normal * MARKER_OFFSET_M)},
        )
        ET.SubElement(
            marker,
            "geom",
            {
                "name": f"calibration_{finger}_marker",
                "type": "sphere",
                "size": str(MARKER_RADIUS_M),
                "rgba": "1 0.1 0.05 0.75",
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
            },
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


class PadCalibrator:
    def __init__(self):
        source_model = mujoco.MjModel.from_xml_path(str(SOURCE_URDF))
        source_data = mujoco.MjData(source_model)
        mujoco.mj_forward(source_model, source_data)
        self.surfaces = surfaces_from_model(source_model, source_data)
        tips = distal_tip_positions(source_model)
        self.points = {finger: self.surfaces[finger].project(tips[finger]) for finger in FINGERS}

        with tempfile.NamedTemporaryFile(
            suffix=".xml", prefix=".pad_calibration_", dir=SOURCE_URDF.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            mujoco.mj_saveLastXML(str(temporary_path), source_model)
            _write_marker_xml(temporary_path, self.points, self.surfaces)
            self.model = mujoco.MjModel.from_xml_path(str(temporary_path))
        finally:
            temporary_path.unlink(missing_ok=True)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.marker_bodies = {
            finger: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"calibration_{finger}")
            for finger in FINGERS
        }
        self.marker_geoms = {
            finger: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"calibration_{finger}_marker")
            for finger in FINGERS
        }
        self.link4_bodies = {
            finger: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{finger}_link4")
            for finger in FINGERS
        }
        self.active = "finger1"
        self._previous_selection = 0
        self.viewer = None
        self.tangent_frames = {
            finger: self._initial_tangent_frame(finger) for finger in FINGERS
        }
        self._update_markers()

    def _initial_tangent_frame(self, finger: str) -> np.ndarray:
        axis, surface = task_frame_axes(
            self.model,
            self.data,
            finger,
            surface_reference_world=CAD_NAIL_TO_PULP,
        )
        normal = self.surfaces[finger].normals[self.points[finger].face]
        up = axis - normal * (axis @ normal)
        if np.linalg.norm(up) <= 1e-12:
            up = surface - normal * (surface @ normal)
        up /= np.linalg.norm(up)
        right = np.cross(normal, up)
        width = np.cross(axis, surface)
        if right @ width < 0.0:
            right = -right
        return np.column_stack((up, right))

    def _update_markers(self) -> None:
        for finger in FINGERS:
            surface = self.surfaces[finger]
            position, normal = self._display_normal(finger)
            self.model.body_pos[self.marker_bodies[finger]] = position + normal * MARKER_OFFSET_M
            alpha = 1.0 if finger == self.active else 0.45
            self.model.geom_rgba[self.marker_geoms[finger]] = (1.0, 0.1, 0.05, alpha)
        mujoco.mj_forward(self.model, self.data)

    def _display_normal(self, finger: str) -> tuple[np.ndarray, np.ndarray]:
        surface = self.surfaces[finger]
        position = surface.position(self.points[finger])
        normal = surface.normals[self.points[finger].face]
        if normal @ (position - surface.vertices.mean(axis=0)) < 0.0:
            normal = -normal
        return position, normal

    def draw_normals(self, scene: mujoco.MjvScene) -> None:
        """Draw one thin outward normal segment for each calibrated surface point."""
        scene.ngeom = 0
        for finger in FINGERS:
            if scene.ngeom >= scene.maxgeom:
                return
            position, normal = self._display_normal(finger)
            rotation = self.data.xmat[self.link4_bodies[finger]].reshape(3, 3)
            position = self.data.xpos[self.link4_bodies[finger]] + rotation @ position
            normal = rotation @ normal
            quat = np.zeros(4)
            mat = np.zeros(9)
            mujoco.mju_quatZ2Vec(quat, normal)
            mujoco.mju_quat2Mat(mat, quat)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array((NORMAL_RADIUS_M, NORMAL_LENGTH_M / 2.0, 0.0)),
                position
                + normal * (MARKER_OFFSET_M + MARKER_RADIUS_M + NORMAL_LENGTH_M / 2.0),
                mat,
                np.array((1.0, 0.85, 0.1, 1.0)),
            )
            scene.ngeom += 1

    def _move_active(self, direction_name: str) -> None:
        axis_index, sign = {
            "up": (0, 1.0),
            "down": (0, -1.0),
            "left": (1, -1.0),
            "right": (1, 1.0),
        }[direction_name]
        point, transport = self.surfaces[self.active].move_with_transport(
            self.points[self.active],
            sign * self.tangent_frames[self.active][:, axis_index],
            STEP_M,
        )
        self.points[self.active] = point
        self.tangent_frames[self.active] = transport @ self.tangent_frames[self.active]
        self._update_markers()

    def _check_selection(self) -> None:
        selected = int(self.viewer.perturb.select)
        if selected == self._previous_selection:
            return
        self._previous_selection = selected
        for finger, body_id in self.marker_bodies.items():
            if selected == body_id:
                self.active = finger
                self.viewer.perturb.active = 0
                self.viewer.perturb.active2 = 0
                self._update_markers()
                print(f"selected {finger}")
                return

    def _on_key_press(self, code: int) -> None:
        direction_name = ARROW_KEYS.get(code)
        if direction_name is not None:
            self._move_active(direction_name)
            return
        try:
            key = chr(code & 0x7F).upper()
        except (TypeError, ValueError, OverflowError):
            return
        if key in FINGER_KEYS:
            self.active = FINGER_KEYS[key]
            self._update_markers()
            print(f"selected {self.active}")
        elif key == "S":
            saved = save_pad_points(
                {finger: self.surfaces[finger].position(point) for finger, point in self.points.items()}
            )
            print(f"saved {saved}")
            print("run: .venv/bin/python tools/build_ldjy_urdf.py && .venv/bin/python tools/build_ldjy_mjcf.py")
        elif key == "R":
            print("R does not discard saved calibration; restart to reset the temporary markers.")

    def run(self) -> None:
        print_instructions()
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key_press) as viewer:
            self.viewer = viewer
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -25
            viewer.cam.distance = 0.38
            viewer.cam.lookat[:] = (0.0, 0.02, 0.0)
            while viewer.is_running():
                self._check_selection()
                self.draw_normals(viewer.user_scn)
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    PadCalibrator().run()
