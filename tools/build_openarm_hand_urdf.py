"""Generate a MANO-task-frame OpenArm bimanual URDF from its source asset."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
import tempfile
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

try:
    from tools.ldjy_asset_frames import RIGHT_MANO_FROM_CAD, root_palm_translation
    from tools.openarm_asset_frames import (
        J7_HOME_OFFSETS,
        PALM_MOUNT_TRANSLATION_CORRECTIONS,
    )
except ModuleNotFoundError:
    from ldjy_asset_frames import RIGHT_MANO_FROM_CAD, root_palm_translation
    from openarm_asset_frames import (
        J7_HOME_OFFSETS,
        PALM_MOUNT_TRANSLATION_CORRECTIONS,
    )

try:
    from tools.build_ldjy_urdf import build_urdf as build_ldjy_hand_urdf
except ModuleNotFoundError:
    from build_ldjy_urdf import build_urdf as build_ldjy_hand_urdf
from ldjy_retargeting.retarget_tip_frames import load_tip_offsets, normalize_tip_offsets


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "ldjy_retargeting" / "assets" / "robots" / "openarm_hand"
SOURCE_URDF = ASSET_DIR / "source" / "openarm_bimanual_20dof_hands.urdf"
OUTPUT_URDF = ASSET_DIR / "urdf" / "openarm_bimanual_mano.urdf"
LDJY_HAND_ASSET_DIR = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")


def numbers(values: np.ndarray) -> str:
    return " ".join(f"{value:.12g}" for value in np.asarray(values).ravel())


def transform_from_origin(origin: ET.Element) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    ).as_matrix()
    transform[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    return transform


def set_origin_from_transform(origin: ET.Element, transform: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    origin.set("xyz", numbers(transform[:3, 3]))
    origin.set("rpy", numbers(rpy))


def wrist_to_palm_transform(side: str) -> np.ndarray:
    """Return the shared MANO wrist -> CAD palm transform for one hand."""
    transform = np.eye(4)
    transform[:3, :3] = RIGHT_MANO_FROM_CAD
    transform[:3, 3] = root_palm_translation(side)
    return transform


def mesh_geom_for_body(model: mujoco.MjModel, body_id: int) -> int:
    candidates = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] == body_id
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    if not candidates:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        raise ValueError(f"No mesh geometry found for {body_name}")
    return candidates[0]


def distal_tip_positions(model: mujoco.MjModel, side: str) -> dict[str, np.ndarray]:
    """Find each source link-4 mesh extremity in its link-local frame."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    positions = {}
    for finger in FINGERS:
        joint3 = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{finger}_joint3"
        )
        joint4 = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{finger}_joint4"
        )
        direction = data.xanchor[joint4] - data.xanchor[joint3]
        direction /= np.linalg.norm(direction)
        body_id = model.jnt_bodyid[joint4]
        geom_id = mesh_geom_for_body(model, body_id)
        mesh_id = model.geom_dataid[geom_id]
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id]
            + model.mesh_vertnum[mesh_id]
        ]
        world_vertices = (
            vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        )
        distal_point = world_vertices[np.argmax(world_vertices @ direction)]
        positions[finger] = data.xmat[body_id].reshape(3, 3).T @ (
            distal_point - data.xpos[body_id]
        )
    return positions


def add_mujoco_compiler(root: ET.Element) -> None:
    """Keep fixed semantic bodies when the generated URDF is imported by MuJoCo."""
    mujoco_options = root.find("mujoco")
    if mujoco_options is None:
        mujoco_options = ET.Element("mujoco")
        root.insert(0, mujoco_options)
    compiler = mujoco_options.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_options, "compiler")
    compiler.set("fusestatic", "false")


def use_visual_meshes_for_collision(root: ET.Element) -> None:
    """Replace simplified collision geometry with the authored visual surface."""
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
        for index, visual in enumerate(link.findall("visual")):
            collision = ET.SubElement(
                link,
                "collision",
                {"name": f"{link.attrib['name']}_visual_surface_collision_{index}"},
            )
            for tag in ("origin", "geometry"):
                element = visual.find(tag)
                if element is not None:
                    collision.append(copy.deepcopy(element))


def calibrate_j7_zero(root: ET.Element, side: str) -> None:
    """Make generated q7=0 the source arm's natural downward wrist pose."""
    joint = root.find(f"joint[@name='openarm_{side}_joint7']")
    if joint is None:
        raise ValueError(f"Missing {side} J7 joint")
    origin = joint.find("origin")
    limit = joint.find("limit")
    if origin is None or limit is None:
        raise ValueError(f"Incomplete {side} J7 joint")
    home_offset = J7_HOME_OFFSETS[side]
    source_to_joint = transform_from_origin(origin)
    joint_offset = np.eye(4)
    joint_offset[:3, :3] = Rotation.from_rotvec((home_offset, 0.0, 0.0)).as_matrix()
    set_origin_from_transform(origin, source_to_joint @ joint_offset)
    limit.set("lower", f"{float(limit.attrib['lower']) - home_offset:.12g}")
    limit.set("upper", f"{float(limit.attrib['upper']) - home_offset:.12g}")


def visual_surface_model(root: ET.Element) -> mujoco.MjModel:
    """Compile a temporary source-local URDF so mesh paths stay relative."""
    with tempfile.NamedTemporaryFile(suffix=".urdf", dir=SOURCE_URDF.parent) as file:
        ET.ElementTree(root).write(file.name, encoding="utf-8", xml_declaration=True)
        return mujoco.MjModel.from_xml_path(file.name)


def split_hand_mount(root: ET.Element, side: str) -> None:
    """Insert a MANO wrist frame while exactly preserving the source palm pose."""
    mount = root.find(f"joint[@name='{side}_adapter_to_palm']")
    if mount is None:
        raise ValueError(f"Missing {side} hand adapter mount")
    parent = mount.find("parent")
    child = mount.find("child")
    origin = mount.find("origin")
    if parent is None or child is None or origin is None:
        raise ValueError(f"Incomplete {side} hand adapter mount")
    if child.attrib.get("link") != f"{side}_palm":
        raise ValueError(f"Unexpected {side} mount child: {child.attrib.get('link')}")

    adapter_to_palm = transform_from_origin(origin)
    mount_correction = np.eye(4)
    mount_correction[:3, 3] = PALM_MOUNT_TRANSLATION_CORRECTIONS[side]
    adapter_to_palm = mount_correction @ adapter_to_palm
    adapter_to_wrist = adapter_to_palm @ np.linalg.inv(wrist_to_palm_transform(side))
    wrist_name = f"{side}_retarget_wrist"

    ET.SubElement(root, "link", {"name": wrist_name})
    mount.set("name", f"{side}_adapter_to_retarget_wrist")
    child.set("link", wrist_name)
    set_origin_from_transform(origin, adapter_to_wrist)

    wrist_to_palm = ET.SubElement(
        root, "joint", {"name": f"{side}_retarget_wrist_fixed", "type": "fixed"}
    )
    ET.SubElement(wrist_to_palm, "parent", {"link": wrist_name})
    ET.SubElement(wrist_to_palm, "child", {"link": f"{side}_palm"})
    wrist_origin = ET.SubElement(wrist_to_palm, "origin")
    set_origin_from_transform(wrist_origin, wrist_to_palm_transform(side))


def standalone_task_points(
    offsets: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, np.ndarray]]:
    """Return canonical MANO-wrist tip points from standalone LDJY assets."""
    return {
        side: {
            finger: frame["tip"].translation.copy()
            for finger, frame in fingers.items()
        }
        for side, fingers in standalone_task_frames(offsets).items()
    }


def standalone_task_frames(
    offsets: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, dict[str, pin.SE3]]]:
    """Return canonical MANO-wrist tip frames from standalone LDJY assets."""
    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        (temporary_root / "meshes").symlink_to(
            LDJY_HAND_ASSET_DIR / "meshes", target_is_directory=True
        )
        output_dir = temporary_root / "urdf"
        result: dict[str, dict[str, np.ndarray]] = {}
        for side in ("left", "right"):
            urdf_path = build_ldjy_hand_urdf(
                side, offsets=offsets, output_dir=output_dir
            )
            model = pin.buildModelFromUrdf(str(urdf_path))
            data = model.createData()
            pin.forwardKinematics(model, data, np.zeros(model.nq))
            pin.updateFramePlacements(model, data)
            wrist = data.oMf[model.getFrameId(f"{side}_retarget_wrist", pin.BODY)].inverse()
            result[side] = {}
            for finger in FINGERS:
                result[side][finger] = {
                    "tip": wrist * data.oMf[
                        model.getFrameId(f"{side}_{finger}_tip", pin.BODY)
                    ],
                }
    return result


def openarm_wrist_pose(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return world rotation and position of the generated MANO wrist frame."""
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_palm")
    if palm_id < 0:
        raise ValueError(f"Missing {side} palm body")
    world_from_palm = data.xmat[palm_id].reshape(3, 3)
    wrist_from_palm = wrist_to_palm_transform(side)
    world_from_wrist = world_from_palm @ wrist_from_palm[:3, :3].T
    wrist_position = data.xpos[palm_id] - world_from_wrist @ wrist_from_palm[:3, 3]
    return world_from_wrist, wrist_position


def add_tip_frames(
    root: ET.Element,
    model: mujoco.MjModel,
    task_points: Mapping[str, Mapping[str, np.ndarray]],
    side: str,
) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world_from_wrist, wrist_position = openarm_wrist_pose(model, data, side)
    for finger, task_point_in_wrist in task_points[side].items():
        joint4 = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{finger}_joint4"
        )
        body_id = model.jnt_bodyid[joint4]
        world_from_link4 = data.xmat[body_id].reshape(3, 3)
        target_world = wrist_position + world_from_wrist @ task_point_in_wrist
        position = world_from_link4.T @ (target_world - data.xpos[body_id])
        tip_name = f"{side}_{finger}_tip"
        ET.SubElement(root, "link", {"name": tip_name})
        joint = ET.SubElement(
            root, "joint", {"name": f"{tip_name}_fixed", "type": "fixed"}
        )
        ET.SubElement(joint, "parent", {"link": f"{side}_{finger}_link4"})
        ET.SubElement(joint, "child", {"link": tip_name})
        ET.SubElement(joint, "origin", {"xyz": numbers(position), "rpy": "0 0 0"})


def build_urdf(
    *,
    offsets: Mapping[str, Mapping[str, float]] | None = None,
    output_path: Path = OUTPUT_URDF,
) -> Path:
    offsets = normalize_tip_offsets(load_tip_offsets() if offsets is None else offsets)
    root = ET.parse(SOURCE_URDF).getroot()
    use_visual_meshes_for_collision(root)
    for side in ("left", "right"):
        calibrate_j7_zero(root, side)
    add_mujoco_compiler(root)
    for side in ("left", "right"):
        split_hand_mount(root, side)
    source_model = visual_surface_model(root)
    task_frames = standalone_task_frames(offsets)
    task_points = {
        side: {
            finger: frame["tip"].translation.copy()
            for finger, frame in fingers.items()
        }
        for side, fingers in task_frames.items()
    }
    for side in ("left", "right"):
        add_tip_frames(root, source_model, task_points, side)
    root.set("name", "openarm_bimanual_mano")
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


if __name__ == "__main__":
    print(build_urdf())
