"""Generate MANO-aligned left/right LDJY URDFs from the original CAD URDF."""

from __future__ import annotations

from pathlib import Path
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from tools.ldjy_asset_frames import RIGHT_MANO_FROM_CAD, root_palm_translation
except ModuleNotFoundError:
    from ldjy_asset_frames import RIGHT_MANO_FROM_CAD, root_palm_translation
from ldjy_retargeting.retarget_tip_frames import (
    FINGERS,
    apply_offset,
    load_tip_offsets,
    normalize_tip_offsets,
    task_frame_axes,
)
from ldjy_retargeting.pad_calibration import load_pad_points


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
SOURCE_URDF = ASSET_DIR / "source" / "step_20_dof_hand.urdf"
OUTPUT_DIR = ASSET_DIR / "urdf"
FINGERS = ("finger1", "finger2", "finger3", "thumb", "finger4")
MIRROR_X = np.diag((-1.0, 1.0, 1.0))
# The original CAD hand lies approximately in the XZ plane.  +Y is the
# selected nail-to-pulp reference; the generated GUI labels this convention.
CAD_NAIL_TO_PULP = np.array((0.0, 1.0, 0.0))


def numbers(values: np.ndarray) -> str:
    return " ".join(f"{value:.12g}" for value in np.asarray(values).ravel())


def distal_tip_positions(
    source_model: mujoco.MjModel,
    offsets: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute task-frame locations at the distal end of each CAD fingertip."""
    data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, data)
    offsets = normalize_tip_offsets(load_tip_offsets() if offsets is None else offsets)
    positions = {}
    for finger in FINGERS:
        joint3 = mujoco.mj_name2id(
            source_model, mujoco.mjtObj.mjOBJ_JOINT, f"{finger}_joint3"
        )
        joint4 = mujoco.mj_name2id(
            source_model, mujoco.mjtObj.mjOBJ_JOINT, f"{finger}_joint4"
        )
        start, end = data.xanchor[joint3], data.xanchor[joint4]
        direction = end - start
        direction /= np.linalg.norm(direction)
        geom = mujoco.mj_name2id(
            source_model, mujoco.mjtObj.mjOBJ_GEOM, f"{finger}_link4_visual_mesh_collision"
        )
        mesh = source_model.geom_dataid[geom]
        vertices = source_model.mesh_vert[
            source_model.mesh_vertadr[mesh]: source_model.mesh_vertadr[mesh]
            + source_model.mesh_vertnum[mesh]
        ]
        world_vertices = (
            vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
        tip = end + direction * ((world_vertices - end) @ direction).max()
        body_id = source_model.jnt_bodyid[joint4]
        base_position = data.xmat[body_id].reshape(3, 3).T @ (tip - end)
        axis_local, surface_local = task_frame_axes(
            source_model,
            data,
            finger,
            surface_reference_world=CAD_NAIL_TO_PULP,
        )
        positions[finger] = apply_offset(
            base_position,
            axis_local,
            surface_local,
            **offsets[finger],
        )
    return positions


def mirror_cad_tree(root: ET.Element) -> None:
    """Mirror source-CAD geometry and kinematics across its X=0 plane."""
    for origin in root.findall(".//origin"):
        if "xyz" in origin.attrib:
            xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
            xyz[0] *= -1.0
            origin.set("xyz", numbers(xyz))
        if "rpy" in origin.attrib:
            rotation = Rotation.from_euler(
                "xyz", np.fromstring(origin.attrib["rpy"], sep=" ")
            ).as_matrix()
            mirrored = MIRROR_X @ rotation @ MIRROR_X
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Gimbal lock detected")
                rpy = Rotation.from_matrix(mirrored).as_euler("xyz")
            origin.set("rpy", numbers(rpy))

    for axis in root.findall(".//axis"):
        xyz = np.fromstring(axis.attrib["xyz"], sep=" ")
        # A rotation axis is an axial vector: after an improper reflection,
        # its sign must be reversed as well to preserve the meaning of +q.
        xyz = -MIRROR_X @ xyz
        axis.set("xyz", numbers(xyz))

    for mesh in root.findall(".//mesh"):
        scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
        scale[0] *= -1.0
        mesh.set("scale", numbers(scale))


def add_tip_frames(root: ET.Element, tip_positions: dict[str, np.ndarray]) -> None:
    for finger, position in tip_positions.items():
        ET.SubElement(root, "link", {"name": f"{finger}_tip"})
        joint = ET.SubElement(
            root,
            "joint",
            {"name": f"{finger}_tip_fixed", "type": "fixed"},
        )
        ET.SubElement(joint, "parent", {"link": f"{finger}_link4"})
        ET.SubElement(joint, "child", {"link": f"{finger}_tip"})
        ET.SubElement(joint, "origin", {"xyz": numbers(position), "rpy": "0 0 0"})


def add_pad_frames(root: ET.Element, pad_positions: Mapping[str, np.ndarray]) -> None:
    """Attach calibrated pad points as fixed children of the distal links."""
    for finger in FINGERS:
        ET.SubElement(root, "link", {"name": f"{finger}_pad"})
        joint = ET.SubElement(root, "joint", {"name": f"{finger}_pad_fixed", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": f"{finger}_link4"})
        ET.SubElement(joint, "child", {"link": f"{finger}_pad"})
        ET.SubElement(joint, "origin", {"xyz": numbers(pad_positions[finger]), "rpy": "0 0 0"})


def add_mano_root(root: ET.Element, side: str) -> None:
    """Make the CAD palm a fixed child of the MANO-aligned wrist root."""
    ET.SubElement(root, "link", {"name": "retarget_wrist"})
    joint = ET.SubElement(root, "joint", {"name": "retarget_wrist_fixed", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": "retarget_wrist"})
    ET.SubElement(joint, "child", {"link": "palm"})
    rpy = Rotation.from_matrix(RIGHT_MANO_FROM_CAD).as_euler("xyz")
    ET.SubElement(
        joint,
        "origin",
        {"xyz": numbers(root_palm_translation(side)), "rpy": numbers(rpy)},
    )


def prefix_side(root: ET.Element, side: str) -> None:
    """Prefix all link and joint names while leaving mesh file paths untouched."""
    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_names = {joint.attrib["name"] for joint in root.findall("joint")}
    for link in root.findall("link"):
        link.set("name", f"{side}_{link.attrib['name']}")
    for joint in root.findall("joint"):
        joint.set("name", f"{side}_{joint.attrib['name']}")
        for child in joint.findall("parent") + joint.findall("child"):
            link = child.attrib["link"]
            if link in link_names:
                child.set("link", f"{side}_{link}")
        mimic = joint.find("mimic")
        if mimic is not None and mimic.attrib.get("joint") in joint_names:
            mimic.set("joint", f"{side}_{mimic.attrib['joint']}")
    root.set("name", f"ldjy_{side}_hand")


def build_urdf(
    side: str,
    *,
    offsets: Mapping[str, Mapping[str, float]] | None = None,
    pad_points: Mapping[str, np.ndarray] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported side: {side}")
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_URDF))
    root = ET.parse(SOURCE_URDF).getroot()
    add_tip_frames(root, distal_tip_positions(source_model, offsets))
    calibrated_points = load_pad_points() if pad_points is None else pad_points
    if calibrated_points is not None:
        add_pad_frames(root, calibrated_points)
    if side == "left":
        mirror_cad_tree(root)
    add_mano_root(root, side)
    prefix_side(root, side)
    ET.indent(root, space="  ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"ldjy_{side}_hand.urdf"
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    for hand_side in ("right", "left"):
        print(build_urdf(hand_side))
