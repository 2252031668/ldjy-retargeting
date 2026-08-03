"""Generate the fixed-base, 54-actuator OpenArm MJCF asset."""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "ldjy_retargeting" / "assets" / "robots" / "openarm_hand"
SOURCE_URDF = ASSET_DIR / "urdf" / "openarm_bimanual_mano.urdf"
OUTPUT_MJCF = ASSET_DIR / "mjcf" / "openarm_bimanual_mano.xml"
FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")
HAND_MESH_LINKS = (
    "palm",
    *(f"{finger}_link{index}" for finger in FINGERS for index in range(1, 5)),
)
HAND_POSITION_KP = 3.0
HAND_POSITION_KV = 0.4
HAND_JOINT_DAMPING = 0.02
HAND_JOINT_ARMATURE = 0.002
ARM_JOINT_PARAMETERS = {
    # kp, kv, joint damping, armature. The source torque limits remain intact.
    "joint1": (300.0, 30.0, 2.0, 0.02),
    "joint2": (300.0, 30.0, 2.0, 0.02),
    "joint3": (150.0, 12.0, 2.0, 0.02),
    "joint4": (300.0, 18.0, 2.0, 0.02),
    "joint5": (150.0, 6.0, 2.0, 0.02),
    "joint6": (100.0, 15.0, 5.0, 0.02),
    "joint7": (100.0, 12.0, 5.0, 0.02),
}


def actuator_parameters(joint_name: str) -> tuple[str, str]:
    """Return stable hold gains for arm links and low-inertia hand joints."""
    if "finger" in joint_name or "thumb" in joint_name:
        return f"{HAND_POSITION_KP:.12g}", f"{HAND_POSITION_KV:.12g}"
    suffix = joint_name.rsplit("_", maxsplit=1)[-1]
    kp, kv, _, _ = ARM_JOINT_PARAMETERS[suffix]
    return f"{kp:.12g}", f"{kv:.12g}"


def ensure_compiler_preserves_fixed_frames(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("fusestatic", "false")


def add_simulation_defaults(root: ET.Element) -> None:
    """Set a damped implicit simulation and a visible fixed-base floor."""
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.attrib.update({"timestep": "0.002", "integrator": "implicitfast"})

    asset = root.find("asset")
    if asset is None:
        raise ValueError("Generated MJCF has no asset section")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "floor_checker",
            "type": "2d",
            "builtin": "checker",
            "width": "512",
            "height": "512",
            "rgb1": "0.23 0.23 0.23",
            "rgb2": "0.38 0.38 0.38",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "studio_sky",
            "type": "skybox",
            "builtin": "gradient",
            "width": "512",
            "height": "512",
            "rgb1": "0.25 0.28 0.32",
            "rgb2": "0.08 0.10 0.14",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "floor_grid",
            "texture": "floor_checker",
            "texrepeat": "8 8",
            "texuniform": "true",
            "reflectance": "0.1",
        },
    )
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Generated MJCF has no worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 -0.001",
            "size": "3 3 0.1",
            "material": "floor_grid",
            "friction": "1 0.005 0.0001",
            "condim": "3",
        },
    )


def add_scene_lighting(root: ET.Element) -> None:
    """Give the dark-red robot enough fill light in the stock MuJoCo viewer."""
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(2, visual)
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(
        {
            "ambient": "0.45 0.45 0.45",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0.3 0.3 0.3",
        }
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Generated MJCF has no worldbody")
    lights = (
        ("key_light", "1.5 -1.5 2.4", "1 1 1", "0.3 0.3 0.3"),
        ("fill_light", "-1.4 1.2 1.6", "0.7 0.7 0.7", "0.15 0.15 0.15"),
        ("rim_light", "0 1.8 2.5", "0.45 0.45 0.45", "0.2 0.2 0.2"),
    )
    for name, position, diffuse, specular in lights:
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": name,
                "pos": position,
                "diffuse": diffuse,
                "specular": specular,
                "castshadow": "false",
            },
        )


def add_joint_dynamics(root: ET.Element) -> None:
    """Regularize low-inertia hand links and damp the gravity-loaded arms."""
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        if "range" not in joint.attrib:
            continue
        if "finger" in name or "thumb" in name:
            joint.attrib.update(
                {
                    "armature": f"{HAND_JOINT_ARMATURE:.12g}",
                    "damping": f"{HAND_JOINT_DAMPING:.12g}",
                }
            )
        else:
            suffix = name.rsplit("_", maxsplit=1)[-1]
            _, _, damping, armature = ARM_JOINT_PARAMETERS[suffix]
            joint.attrib.update(
                {"armature": f"{armature:.12g}", "damping": f"{damping:.12g}"}
            )


def restore_side_specific_hand_meshes(root: ET.Element) -> None:
    """Undo MuJoCo's URDF mesh-name de-duplication for the mirrored hands."""
    asset = root.find("asset")
    if asset is None:
        raise ValueError("Generated MJCF has no asset section")
    meshes = {mesh.attrib.get("name"): mesh for mesh in asset.findall("mesh")}
    bodies = {body.attrib.get("name"): body for body in root.findall(".//body")}

    for link in HAND_MESH_LINKS:
        left_mesh = meshes.get(link)
        if left_mesh is None:
            raise ValueError(f"Missing exported hand mesh asset {link}")
        left_mesh.set("name", f"left_{link}")
        left_mesh.set("file", f"../meshes/hand_mirrored/{link}.stl")

        right_attributes = dict(left_mesh.attrib)
        right_attributes.update(
            {
                "name": f"right_{link}",
                "file": f"../meshes/hand/{link}.stl",
            }
        )
        ET.SubElement(asset, "mesh", right_attributes)

        for side in ("left", "right"):
            body = bodies.get(f"{side}_{link}")
            if body is None:
                raise ValueError(f"Missing generated hand body {side}_{link}")
            for geom in body.findall("geom"):
                if geom.attrib.get("mesh") == link:
                    geom.set("mesh", f"{side}_{link}")


def compile_root(root: ET.Element) -> mujoco.MjModel:
    """Compile a generated root next to its assets so relative mesh paths resolve."""
    with tempfile.NamedTemporaryFile(
        suffix=".xml", dir=OUTPUT_MJCF.parent
    ) as temporary:
        ET.ElementTree(root).write(temporary.name, encoding="utf-8")
        return mujoco.MjModel.from_xml_path(temporary.name)


def exclude_zero_pose_visual_mesh_overlaps(root: ET.Element, model: mujoco.MjModel) -> None:
    """Exclude CAD assembly overlaps while retaining visual-surface collisions."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pairs = set()
    for contact in data.contact[: data.ncon]:
        body_ids = sorted((model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]))
        if body_ids[0] == body_ids[1]:
            continue
        names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in body_ids
        )
        if all(names):
            pairs.add(names)
    contact_element = root.find("contact")
    if contact_element is None:
        contact_element = ET.SubElement(root, "contact")
    for body1, body2 in sorted(pairs):
        ET.SubElement(contact_element, "exclude", {"body1": body1, "body2": body2})


def add_tip_sites(root: ET.Element) -> None:
    bodies = {body.attrib["name"]: body for body in root.findall(".//body")}
    for side in ("left", "right"):
        for finger in FINGERS:
            tip_name = f"{side}_{finger}_tip"
            if tip_name not in bodies:
                raise ValueError(f"Missing generated task frame {tip_name}")
            ET.SubElement(
                bodies[tip_name],
                "site",
                {
                    "name": f"{tip_name}_site",
                    "pos": "0 0 0",
                    "size": "0.003",
                    "group": "4",
                },
            )


def add_pad_sites(root: ET.Element) -> None:
    """Expose the URDF's semantic finger-pad frames in the debug MJCF."""
    bodies = {body.attrib["name"]: body for body in root.findall(".//body")}
    for side in ("left", "right"):
        for finger in FINGERS:
            frame_name = f"{side}_{finger}_pad_frame"
            if frame_name not in bodies:
                raise ValueError(f"Missing generated pad frame {frame_name}")
            ET.SubElement(
                bodies[frame_name],
                "site",
                {
                    "name": f"{side}_{finger}_pad_center",
                    "type": "ellipsoid",
                    "pos": "0 0 0",
                    "size": "0.007 0.005 0.001",
                    "group": "4",
                    "rgba": "0.9 0.3 0.1 0.5",
                },
            )


def add_position_actuators(root: ET.Element, model: mujoco.MjModel) -> None:
    existing = root.find("actuator")
    if existing is not None:
        root.remove(existing)
    actuator = ET.SubElement(root, "actuator")
    joint_ranges = {
        joint.attrib["name"]: joint.attrib["range"]
        for joint in root.findall(".//joint")
        if "range" in joint.attrib
    }
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name not in joint_ranges:
            raise ValueError(f"Missing exported range for movable joint {name}")
        kp, kv = actuator_parameters(name)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{name}_actuator",
                "joint": name,
                "kp": kp,
                "kv": kv,
                "ctrlrange": joint_ranges[name],
            },
        )


def build_mjcf() -> Path:
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_URDF))
    if source_model.nq != 54 or source_model.nv != 54:
        raise ValueError(
            f"Expected a fixed-base 54-DOF model, got nq={source_model.nq}, nv={source_model.nv}"
        )
    with tempfile.NamedTemporaryFile(suffix=".xml") as temporary:
        mujoco.mj_saveLastXML(temporary.name, source_model)
        root = ET.parse(temporary.name).getroot()

    root.set("model", "openarm_bimanual_mano")
    ensure_compiler_preserves_fixed_frames(root)
    add_simulation_defaults(root)
    add_scene_lighting(root)
    add_joint_dynamics(root)
    restore_side_specific_hand_meshes(root)
    model = compile_root(root)
    exclude_zero_pose_visual_mesh_overlaps(root, model)
    add_tip_sites(root)
    add_pad_sites(root)
    add_position_actuators(root, model)
    ET.indent(root, space="  ")
    OUTPUT_MJCF.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(OUTPUT_MJCF, encoding="utf-8", xml_declaration=True)
    return OUTPUT_MJCF


if __name__ == "__main__":
    print(build_mjcf())
