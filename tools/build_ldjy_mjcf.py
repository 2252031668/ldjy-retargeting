"""Generate LDJY MuJoCo assets from the generated MANO-aligned URDFs."""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
URDF_DIR = ASSET_DIR / "urdf"
MJCF_DIR = ASSET_DIR / "mjcf"
FINGERS = ("finger1", "finger2", "finger3", "thumb", "finger4")
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


def distal_tip_positions(model: mujoco.MjModel, side: str) -> dict[str, np.ndarray]:
    """Return each fingertip mesh extremity in its link4 local coordinates."""
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
        positions[finger] = data.xmat[body_id].reshape(3, 3).T @ (tip - end)
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


def add_tip_sites(root: ET.Element, model: mujoco.MjModel, side: str) -> None:
    bodies = {body.attrib["name"]: body for body in root.findall(".//body")}
    for finger, position in distal_tip_positions(model, side).items():
        link4 = bodies[f"{side}_{finger}_link4"]
        ET.SubElement(
            link4,
            "site",
            {"name": f"{side}_{finger}_link4_tip", "pos": numbers(position), "group": "4", "size": "0.003"},
        )


def build_model(side: str) -> Path:
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported side: {side}")
    source = URDF_DIR / f"ldjy_{side}_hand.urdf"
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
    add_tip_sites(root, source_model, side)

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
    MJCF_DIR.mkdir(parents=True, exist_ok=True)
    output = MJCF_DIR / f"ldjy_{side}_hand.xml"
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    for hand_side in ("right", "left"):
        print(build_model(hand_side))
