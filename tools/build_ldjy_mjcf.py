"""Generate LDJY MuJoCo assets from the generated MANO-aligned URDFs."""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from tools.ldjy_asset_frames import RIGHT_MANO_FROM_CAD
except ModuleNotFoundError:
    from ldjy_asset_frames import RIGHT_MANO_FROM_CAD
from ldjy_retargeting.retarget_tip_frames import (
    apply_offset,
    load_tip_offsets,
    normalize_tip_offsets,
    task_frame_axes,
)
from ldjy_retargeting.pad_calibration import link4_visual_surface, load_pad_points


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
URDF_DIR = ASSET_DIR / "urdf"
MJCF_DIR = ASSET_DIR / "mjcf"
FINGERS = ("finger1", "finger2", "finger3", "thumb", "finger4")
CAD_NAIL_TO_PULP = np.array((0.0, 1.0, 0.0))
MIRROR_X = np.diag((-1.0, 1.0, 1.0))
MESH_COLLISION_BODIES = (
    "palm", "finger1_link4", "finger2_link4", "finger3_link4", "thumb_link4", "finger4_link4"
)
CAPSULES = (
    ("finger1_link2", "-0.02233782584 0.01465 -0.07157380139", "0.005864106671017739 -3.315357052071993e-05 0.0021571807155113445 0.008643711232982262 0.0015634059923207199 0.029713695544488654", "0.007647541322"),
    ("finger1_link3", "-0.02778508462 0.01563627146 -0.1035988641", "0.005864106671017739 -3.315357052071993e-05 0.0021571807155113445 0.008643711232982262 0.0015634059923207199 0.029713695544488654", "0.007647541322"),
    ("finger2_link2", "0.005252746989 0.01465 -0.0759612939", "0.005330175945664866 -2.5459360278297892e-05 0.0032604985870525113 0.0026770375963351334 0.0016394585514782978 0.03082543317294749", "0.007647541322"),
    ("finger2_link3", "0.006162049534 0.01555589976 -0.1084359381", "0.005330175945664866 -2.5459360278297892e-05 0.0032604985870525113 0.0026770375963351334 0.0016394585514782978 0.03082543317294749", "0.007647541322"),
    ("finger3_link2", "0.03155359283 0.01465 -0.06499528464", "0.004812185541144998 0.00010615303547150366 0.003984203394555937 -0.001720734881144998 0.0029385739125284962 0.030797167725444066", "0.007647541322"),
    ("finger3_link3", "0.03707691276 0.01418050482 -0.09701906504", "0.004812185541144998 0.00010615303547150366 0.003984203394555937 -0.001720734881144998 0.0029385739125284962 0.030797167725444066", "0.007647541322"),
    ("thumb_link2", "-0.0259085607 0.01955 0.01033853323", "-0.0012674068890160713 -0.003368696520349595 0.004496985802651246 0.01835717003901607 0.0006926715583495948 0.030009979377348756", "0.007912057707"),
    ("thumb_link3", "-0.04615540993 0.02591619477 -0.02235506952", "0.0031561761038444896 0.003973012588344725 0.0003210599958376435 0.01573803159815551 0.0049888195556552755 0.025025346304162355", "0.007647541322"),
    ("finger4_link1", "0.0233 0.0136 0.00335", "-0.003267485672213381 8.358832052791327e-05 -0.009744681730066118 -0.023885122387786617 0.0004408267940720868 0.007181010764066119", "0.007121991075"),
    ("finger4_link1", "0.0233 0.0136 0.00335", "-0.023614543657127906 3.906772957551863e-05 0.02177814756457811 -0.026287592602872095 -0.00015601330679551864 0.04060698019542189", "0.007839284308"),
    ("finger4_link2", "0.04669748876 0.01834210078 -0.04005", "-0.000379027462771523 0.006169885515716631 0.0030490395542072998 -0.008689386565228477 0.004545450360283369 0.0169117013837927", "0.007571680637"),
    ("finger4_link3", "0.05103306189 0.01310607593 -0.06220472522", "-0.004436865187758943 0.0024805852068378463 0.00010139237783623675 -0.01758502777224106 -0.0011909087612378463 0.024252638022163764", "0.007647541322"),
)


def numbers(values: np.ndarray) -> str:
    return " ".join(f"{value:.12g}" for value in np.asarray(values).ravel())


def visual_transform(model: mujoco.MjModel, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    bare_name = body_name.removeprefix("right_").removeprefix("left_")
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{bare_name}_visual_mesh_collision"
    )
    if geom_id < 0:
        raise ValueError(f"No visual mesh found for {body_name}")
    quat = model.geom_quat[geom_id]
    return model.geom_pos[geom_id], Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()


def mano_nail_to_pulp(side: str) -> np.ndarray:
    cad_direction = CAD_NAIL_TO_PULP if side == "right" else MIRROR_X @ CAD_NAIL_TO_PULP
    return RIGHT_MANO_FROM_CAD @ cad_direction


def distal_tip_positions(
    model: mujoco.MjModel,
    side: str,
    offsets: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Return each fingertip mesh extremity in its link4 local coordinates."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    offsets = normalize_tip_offsets(load_tip_offsets() if offsets is None else offsets)
    positions = {}
    for finger in FINGERS:
        joint3 = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{finger}_joint3"
        )
        joint4 = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{finger}_joint4"
        )
        start, end = data.xanchor[joint3], data.xanchor[joint4]
        direction = end - start
        direction /= np.linalg.norm(direction)
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{finger}_link4_visual_mesh_collision"
        )
        mesh_id = model.geom_dataid[geom_id]
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh_id]: model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]
        ]
        world_vertices = (
            vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        )
        tip = end + direction * ((world_vertices - end) @ direction).max()
        body_id = model.jnt_bodyid[joint4]
        base_position = data.xmat[body_id].reshape(3, 3).T @ (tip - end)
        axis_local, surface_local = task_frame_axes(
            model,
            data,
            finger,
            side=side,
            surface_reference_world=mano_nail_to_pulp(side),
        )
        positions[finger] = apply_offset(
            base_position,
            axis_local,
            surface_local,
            **offsets[finger],
        )
    return positions


def restore_named_root(root: ET.Element, side: str) -> None:
    """Restore fixed root links that MuJoCo intentionally fused on import."""
    worldbody = root.find("worldbody")
    palm = ET.Element("body", {"name": f"{side}_palm"})
    for element in list(worldbody):
        worldbody.remove(element)
        palm.append(element)
    wrist = ET.Element("body", {"name": f"{side}_retarget_wrist"})
    wrist.append(palm)
    worldbody.append(wrist)


def add_collision_model(root: ET.Element, model: mujoco.MjModel, side: str) -> None:
    asset = root.find("asset")
    bodies = {body.attrib["name"]: body for body in root.findall(".//body")}
    attributes = {
        "contype": "1", "conaffinity": "1", "density": "0",
        "friction": "0.7 0.005 0.0001", "solref": "0.02 1.5",
        "solimp": "0.9 0.95 0.001 0.5 2", "condim": "3", "rgba": "0.55 0.7 0.9 0",
    }
    for body in MESH_COLLISION_BODIES:
        prefixed = f"{side}_{body}"
        mesh_name = f"{prefixed}_collision_mesh"
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": f"../meshes/collision/{body}_collision.stl"})
        position, rotation = visual_transform(model, prefixed)
        ET.SubElement(
            bodies[prefixed], "geom", attributes | {
                "name": f"{prefixed}_collision", "type": "mesh", "mesh": mesh_name,
                "pos": numbers(position),
                "quat": numbers(Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]),
            },
        )

    for index, (body, old_visual, old_fromto, radius) in enumerate(CAPSULES):
        position, rotation = visual_transform(model, f"{side}_{body}")
        visual = np.fromstring(old_visual, sep=" ")
        points = np.fromstring(old_fromto, sep=" ").reshape(2, 3)
        if side == "left":
            visual[0] *= -1.0
            points[:, 0] *= -1.0
        points = position + (rotation @ (points - visual).T).T
        ET.SubElement(
            bodies[f"{side}_{body}"], "geom", attributes | {
                "name": f"{side}_{body}_capsule_{index}", "type": "capsule",
                "fromto": numbers(points), "size": radius,
            },
        )

    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", {"body1": f"{side}_palm", "body2": f"{side}_finger4_link1"})


def add_tip_sites(
    root: ET.Element,
    model: mujoco.MjModel,
    side: str,
    offsets: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    bodies = {body.attrib["name"]: body for body in root.findall(".//body")}
    for finger, position in distal_tip_positions(model, side, offsets).items():
        link4 = bodies[f"{side}_{finger}_link4"]
        ET.SubElement(
            link4,
            "site",
            {"name": f"{side}_{finger}_link4_tip", "pos": numbers(position), "group": "4", "size": "0.003"},
        )


def add_pad_sites(
    root: ET.Element,
    pad_positions: Mapping[str, np.ndarray],
    side: str,
    pad_normals: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Add sites corresponding to the URDF's calibrated fixed pad nodes."""
    bodies = {body.attrib["name"]: body for body in root.findall(".//body")}
    for finger in FINGERS:
        position = np.asarray(pad_positions[finger], dtype=float).copy()
        if side == "left":
            position[0] *= -1.0
        attributes = {
            "name": f"{side}_{finger}_pad_center",
            "pos": numbers(position),
            "group": "4",
            "size": "0.003",
        }
        if pad_normals is not None:
            quaternion = np.zeros(4)
            mujoco.mju_quatZ2Vec(quaternion, pad_normals[finger])
            attributes["quat"] = numbers(quaternion)
        ET.SubElement(
            bodies[f"{side}_{finger}_link4"],
            "site",
            attributes,
        )


def pad_surface_normals(
    model: mujoco.MjModel, side: str, pad_positions: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return outward link4-local normals at the calibrated pad points."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    normals = {}
    for finger in FINGERS:
        position = np.asarray(pad_positions[finger], dtype=float).copy()
        if side == "left":
            position[0] *= -1.0
        surface = link4_visual_surface(model, data, finger, side)
        point = surface.project(position)
        normal = surface.normals[point.face]
        if normal @ (surface.position(point) - surface.vertices.mean(axis=0)) < 0.0:
            normal = -normal
        normals[finger] = normal
    return normals


def build_model(
    side: str,
    *,
    offsets: Mapping[str, Mapping[str, float]] | None = None,
    pad_points: Mapping[str, np.ndarray] | None = None,
    urdf_dir: Path = URDF_DIR,
    output_dir: Path = MJCF_DIR,
) -> Path:
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported side: {side}")
    source = urdf_dir / f"ldjy_{side}_hand.urdf"
    source_model = mujoco.MjModel.from_xml_path(str(source))
    with tempfile.NamedTemporaryFile(suffix=".xml") as temporary:
        mujoco.mj_saveLastXML(temporary.name, source_model)
        root = ET.parse(temporary.name).getroot()
    root.set("model", f"ldjy_{side}_hand")
    restore_named_root(root, side)
    for joint in root.findall(".//joint"):
        joint.set("armature", "0.0005")
    for geom in root.findall(".//geom"):
        geom.attrib.update({"contype": "0", "conaffinity": "0", "group": "1", "density": "0"})
    add_collision_model(root, source_model, side)
    add_tip_sites(root, source_model, side, offsets)
    calibrated_points = load_pad_points() if pad_points is None else pad_points
    if calibrated_points is not None:
        add_pad_sites(
            root,
            calibrated_points,
            side,
            pad_surface_normals(source_model, side, calibrated_points),
        )

    actuator = ET.SubElement(root, "actuator")
    for joint_id in range(source_model.njnt):
        name = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        lower, upper = source_model.jnt_range[joint_id]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{name}_actuator", "joint": name, "kp": "0.3", "kv": "0.02",
                "ctrlrange": f"{lower:.12g} {upper:.12g}", "forcerange": "-1 1",
            },
        )
    ET.indent(root, space="  ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"ldjy_{side}_hand.xml"
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    for hand_side in ("right", "left"):
        print(build_model(hand_side))
