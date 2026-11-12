"""Build a MuJoCo model containing exactly one object scene."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..config import PROJECT_ROOT, SCENE_PATH
from .scenes import SceneManager
from .scenes import (
    ADJUST_CYLINDRICAL_BOTTLE_SCENE,
    ADJUST_RECTANGULAR_BOTTLE_SCENE,
    BANANA_APPLE_TRAY_SCENE,
    BOX_TO_NARROW_BOX_SCENE,
    CYLINDER_TO_NARROW_BOX_SCENE,
    GEOMETRY_TRAINING_SCENES,
    ROUND_HANDLE_POT_PEAR_SCENE,
    TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
    UNSEEN_BOX_PLATE_SCENE,
    UNSEEN_BOX_REGION_SCENE,
    UNSEEN_BUTTON_SCENE,
    UNSEEN_FRUIT_RECEPTACLES_SCENE,
    UNSEEN_POWER_STRIP_SCENE,
    UNSEEN_SPORTS_BALL_PLATE_SCENE,
)
from .robots import RobotSpec, get_robot_spec


_PANDA_INCLUDE = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
_EMPTY_SCENE = "Empty workspace"
_EMPTY_SCENE_PATH = PROJECT_ROOT / "scene_no_table_no_screen.xml"
_RUNTIME_DIR = PROJECT_ROOT / ".runtime"
_LEGACY_MESHDIR_OBJECT_PREFIX = "../../../objects/"
_ROBOT_INCLUDES = {
    "assets/mujoco_menagerie/franka_emika_panda/panda.xml",
    "assets/mujoco_menagerie/ufactory_xarm7/xarm7.xml",
    "assets/mujoco_menagerie/universal_robots_ur5e/ur5e.xml",
}
_ROUND_GRASP_PAD_GEOMS = {
    "xarm7": (
        "left_finger_pad_1",
        "left_finger_pad_2",
        "right_finger_pad_1",
        "right_finger_pad_2",
    ),
    "ur5e": (
        "rb_left_pad1",
        "rb_left_pad2",
        "rb_right_pad1",
        "rb_right_pad2",
    ),
}
_SCENE_INCLUDES = {
    **{
        scene_name: (
            "assets/objects/paper_placement/model.xml",
            "assets/objects/trays/model.xml",
        )
        for scene_name in GEOMETRY_TRAINING_SCENES
    },
    TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE: (
        "assets/objects/paper_placement/model.xml",
        "assets/objects/trays/model.xml",
    ),
    "Water bottles": "assets/objects/water_bottles/model.xml",
    "Fruits": "assets/objects/fruits/model.xml",
    "Packaged food": "assets/objects/ycb_selected/model.xml",
    "Tableware": "assets/objects/ycb_selected/model.xml",
    "Sports balls": "assets/objects/ycb_selected/model.xml",
    "Cups": "assets/objects/ycb_selected/model.xml",
    "Cylindrical containers": "assets/objects/gso_cylindrical/model.xml",
    "Rectangular objects": "assets/objects/gso_rectangular/model.xml",
    "RoboCasa saucepans": "assets/objects/robocasa_saucepans/model.xml",
    "RoboTwin bottles": "assets/objects/robotwin_bottles/model.xml",
    "Adjust Bottle": "assets/objects/robotwin_bottles/model.xml",
    ADJUST_CYLINDRICAL_BOTTLE_SCENE: "assets/objects/robotwin_bottles/model.xml",
    ADJUST_RECTANGULAR_BOTTLE_SCENE: "assets/objects/robotwin_bottles/model.xml",
    "Click Bell": "assets/objects/press_devices/model.xml",
    "Press Button": "assets/objects/press_devices/model.xml",
    "Press Power Strip": "assets/objects/press_devices/model.xml",
    UNSEEN_FRUIT_RECEPTACLES_SCENE: (
        "assets/objects/fruits/model.xml",
        "assets/objects/ycb_selected/model.xml",
    ),
    UNSEEN_SPORTS_BALL_PLATE_SCENE: "assets/objects/ycb_selected/model.xml",
    UNSEEN_BOX_PLATE_SCENE: "assets/objects/ycb_selected/model.xml",
    UNSEEN_BOX_REGION_SCENE: "assets/objects/ycb_selected/model.xml",
    UNSEEN_BUTTON_SCENE: "assets/objects/press_devices/model.xml",
    UNSEEN_POWER_STRIP_SCENE: "assets/objects/press_devices/model.xml",
    "Insert Object": "assets/objects/object_insert/model.xml",
    "Fruit Receptacles": (
        "assets/objects/fruit_receptacles/model.xml",
        "assets/objects/trays/model.xml",
    ),
    BANANA_APPLE_TRAY_SCENE: (
        "assets/objects/fruit_receptacles/model.xml",
        "assets/objects/trays/model.xml",
    ),
    ROUND_HANDLE_POT_PEAR_SCENE: (
        "assets/objects/fruit_receptacles/model.xml",
    ),
    "Paper Placement": "assets/objects/paper_placement/model.xml",
    "Ball + Paper Placement": "assets/objects/paper_placement/model.xml",
    "Color Block Stack": "assets/objects/color_block_stack/model.xml",
    "Rotate Knob": "assets/objects/rotate_knob/model.xml",
    "Whiteboard": "assets/objects/whiteboard/model.xml",
    CYLINDER_TO_NARROW_BOX_SCENE: (
        "assets/objects/cylinder_narrow_box/model.xml",
    ),
    BOX_TO_NARROW_BOX_SCENE: (
        "assets/objects/box_narrow_box/model.xml",
    ),
    "Trays": "assets/objects/trays/model.xml",
}


def _remove_inactive_bodies(root: ET.Element, active_names: set[str], known_names: set[str]) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    for body in list(worldbody.findall("body")):
        name = body.get("name")
        if name in known_names and name not in active_names:
            worldbody.remove(body)


def _prune_object_assets(root: ET.Element) -> None:
    """Remove unused mesh/material/texture declarations from an object include."""
    asset = root.find("asset")
    if asset is None:
        return
    asset_nodes = set(asset.iter())
    mesh_names: set[str] = set()
    material_names: set[str] = set()
    for element in root.iter():
        if element in asset_nodes:
            continue
        if element.get("mesh"):
            mesh_names.add(element.get("mesh", ""))
        if element.get("material"):
            material_names.add(element.get("material", ""))

    texture_names = {
        material.get("texture", "")
        for material in asset.findall("material")
        if material.get("name") in material_names and material.get("texture")
    }
    for element in list(asset):
        name = element.get("name", "")
        if element.tag == "mesh" and name not in mesh_names:
            asset.remove(element)
        elif element.tag == "material" and name not in material_names:
            asset.remove(element)
        elif element.tag == "texture" and name not in texture_names:
            asset.remove(element)


def _prune_removed_body_pairs(root: ET.Element) -> None:
    """Drop contact exclusions whose variant body was pruned from an include."""
    contact = root.find("contact")
    if contact is None:
        return
    body_names = {
        body.get("name", "")
        for body in root.iter("body")
        if body.get("name")
    }
    for pair in list(contact):
        referenced = {
            pair.get(attribute, "")
            for attribute in ("body1", "body2")
            if pair.get(attribute)
        }
        if referenced and not referenced.issubset(body_names):
            contact.remove(pair)


def _rebase_xml_paths(root: ET.Element) -> None:
    """Make project-root-relative asset paths resolve from the runtime folder."""
    for element in root.iter():
        for attribute in ("file", "meshdir", "texturedir"):
            value = element.get(attribute)
            if not value or Path(value).is_absolute():
                continue
            normalized = value.replace("\\", "/")
            if (
                element.tag == "mesh"
                and normalized.startswith(_LEGACY_MESHDIR_OBJECT_PREFIX)
            ):
                # Older object includes express mesh paths relative to the
                # Panda compiler meshdir.  Runtime XMLs clear that implicit
                # context, so preserve its resolved project location here.
                normalized = (
                    "assets/objects/"
                    + normalized.removeprefix(_LEGACY_MESHDIR_OBJECT_PREFIX)
                )
            source_path = (PROJECT_ROOT / Path(normalized)).resolve()
            runtime_path = os.path.relpath(source_path, _RUNTIME_DIR)
            element.set(attribute, Path(runtime_path).as_posix())


def _temporary_xml(root: ET.Element) -> Path:
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(
        prefix=".active_scene_", suffix=".xml", dir=_RUNTIME_DIR,
    )
    os.close(handle)
    path = Path(raw_path)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)
    return path


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"Robot model is missing body {name!r}")


def _find_geom(root: ET.Element, name: str) -> ET.Element:
    for geom in root.iter("geom"):
        if geom.get("name") == name:
            return geom
    raise RuntimeError(f"Scene is missing geom {name!r}")


def _configure_robot_pedestal(root: ET.Element, spec: RobotSpec) -> None:
    """Make the common stand physically meet the selected robot base."""
    pedestal = _find_body(root, "robot_raised_pedestal")
    base_x, base_y, mounting_height = map(float, spec.base_position_m)
    pedestal.set("pos", f"{base_x:.9g} {base_y:.9g} 0")

    # Feet and bottom rails remain on the floor. Posts extend from z=0.03 to
    # the top-rail centre; the plate's upper face exactly meets the robot's
    # calibrated mounting plane.
    top_rail_z = mounting_height - 0.030
    post_bottom_z = 0.030
    post_half_height = 0.5 * (top_rail_z - post_bottom_z)
    post_center_z = 0.5 * (top_rail_z + post_bottom_z)
    for suffix in ("fl", "fr", "rl", "rr"):
        geom = _find_geom(pedestal, f"pedestal_post_{suffix}")
        position = [float(value) for value in geom.get("pos", "").split()]
        size = [float(value) for value in geom.get("size", "").split()]
        position[2] = post_center_z
        size[2] = post_half_height
        geom.set("pos", " ".join(f"{value:.9g}" for value in position))
        geom.set("size", " ".join(f"{value:.9g}" for value in size))

    for suffix in ("x1", "x2", "y1", "y2"):
        geom = _find_geom(pedestal, f"pedestal_top_{suffix}")
        position = [float(value) for value in geom.get("pos", "").split()]
        position[2] = top_rail_z
        geom.set("pos", " ".join(f"{value:.9g}" for value in position))

    mid_z = 0.5 * mounting_height
    for suffix in ("x", "y"):
        geom = _find_geom(pedestal, f"pedestal_mid_{suffix}")
        position = [float(value) for value in geom.get("pos", "").split()]
        position[2] = mid_z
        geom.set("pos", " ".join(f"{value:.9g}" for value in position))

    brace_bottom_z = mounting_height - 0.130
    brace_coordinates = {
        "pedestal_brace_x_neg": (-0.18, 0.0, brace_bottom_z, -0.08, 0.0, top_rail_z),
        "pedestal_brace_x_pos": (0.18, 0.0, brace_bottom_z, 0.08, 0.0, top_rail_z),
        "pedestal_brace_y_neg": (0.0, -0.18, brace_bottom_z, 0.0, -0.08, top_rail_z),
        "pedestal_brace_y_pos": (0.0, 0.18, brace_bottom_z, 0.0, 0.08, top_rail_z),
    }
    for name, coordinates in brace_coordinates.items():
        _find_geom(pedestal, name).set(
            "fromto", " ".join(f"{value:.9g}" for value in coordinates),
        )

    plate = _find_geom(pedestal, "robot_mounting_plate")
    plate_size_z = float(plate.get("size", "").split()[2])
    plate.set("pos", f"0 0 {mounting_height - plate_size_z:.9g}")


def _configure_robot_workspace(root: ET.Element, spec: RobotSpec) -> None:
    """Translate the table and main camera while preserving Panda framing."""
    offset = np.asarray(spec.table_offset_m, dtype=float)
    table = next(
        (body for body in root.iter("body") if body.get("name") == "work_surface"),
        None,
    )
    if table is not None:
        position = np.asarray(
            [float(value) for value in table.get("pos", "").split()],
            dtype=float,
        )
        position += offset
        table.set("pos", " ".join(f"{value:.9g}" for value in position))

    # xArm7 and UR5e place the tabletop workspace farther along world +X.
    # Move only the main camera by the same offset so its pose relative to the
    # workspace (and therefore its framing) remains identical to Panda's.
    front_camera = next(
        (camera for camera in root.iter("camera") if camera.get("name") == "front_view"),
        None,
    )
    if front_camera is not None:
        position = np.asarray(
            [float(value) for value in front_camera.get("pos", "").split()],
            dtype=float,
        )
        position += offset
        front_camera.set("pos", " ".join(f"{value:.9g}" for value in position))


def _add_wrist_camera(root: ET.Element, spec: RobotSpec) -> None:
    if any(camera.get("name") == "wrist_camera" for camera in root.iter("camera")):
        return
    mount = _find_body(root, spec.wrist_camera_mount)
    ET.SubElement(mount, "camera", {
        "name": "wrist_camera",
        "pos": " ".join(f"{value:.9g}" for value in spec.wrist_camera_pos),
        "quat": " ".join(f"{value:.9g}" for value in spec.wrist_camera_quat),
        "fovy": "42",
    })


def _add_fill_light(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Robot model has no worldbody")
    if not any(light.get("name") == "top" for light in worldbody.findall("light")):
        worldbody.insert(0, ET.Element("light", {
            "name": "top", "pos": "0 0 2", "mode": "trackcom", "castshadow": "false",
        }))


def _add_round_grasp_contact_pairs(
    root: ET.Element,
    spec: RobotSpec,
    active_names: set[str],
) -> None:
    """Enable rolling resistance only between pads and round task objects."""
    pad_geoms = _ROUND_GRASP_PAD_GEOMS.get(spec.uid)
    cylinder_bodies = tuple(sorted(
        name
        for name in active_names
        if name.startswith("paper_object_cylinder_")
    ))
    if not pad_geoms or not cylinder_bodies:
        return
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for body_name in cylinder_bodies:
        cylinder_geom = f"{body_name}_geom"
        for pad_geom in pad_geoms:
            ET.SubElement(contact, "pair", {
                "name": f"round_grasp__{body_name}__{pad_geom}",
                "geom1": cylinder_geom,
                "geom2": pad_geom,
                "condim": "6",
                "friction": "3 3 0.05 0.01 0.01",
                "solref": "0.004 1",
                "solimp": "0.95 0.99 0.001",
            })


def _prefix_robotiq(root: ET.Element) -> None:
    """Namespace the standalone 2F-85 before merging it into UR5e."""
    for mesh in root.findall("./asset/mesh"):
        if not mesh.get("name"):
            mesh.set("name", Path(mesh.get("file", "mesh")).stem)

    named_tags = {"body", "joint", "site", "geom", "mesh", "material", "fixed", "general"}
    mappings: dict[str, str] = {}
    for element in root.iter():
        name = element.get("name")
        if name and element.tag in named_tags:
            mappings[name] = f"rb_{name}"
    class_names = {
        element.get("class")
        for element in root.iter("default")
        if element.get("class")
    }
    class_mappings = {name: f"rb_{name}" for name in class_names}

    reference_attributes = {
        "body1", "body2", "joint", "joint1", "joint2", "site", "tendon",
        "mesh", "material", "target",
    }
    for element in root.iter():
        name = element.get("name")
        if name in mappings and element.tag in named_tags:
            element.set("name", mappings[name])
        for attribute in reference_attributes:
            value = element.get(attribute)
            if value in mappings:
                element.set(attribute, mappings[value])
        for attribute in ("class", "childclass"):
            value = element.get(attribute)
            if value in class_mappings:
                element.set(attribute, class_mappings[value])

    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", f"../../robotiq_2f85/assets/{mesh.get('file')}")


def _merge_robotiq_into_ur5(root: ET.Element) -> None:
    gripper_path = PROJECT_ROOT / "assets/mujoco_menagerie/robotiq_2f85/2f85.xml"
    gripper = ET.parse(gripper_path).getroot()
    _prefix_robotiq(gripper)

    robot_asset = root.find("asset")
    robot_default = root.find("default")
    if robot_asset is None or robot_default is None:
        raise RuntimeError("UR5e model is missing asset/default sections")
    gripper_asset = gripper.find("asset")
    for child in list(gripper_asset if gripper_asset is not None else ()):
        robot_asset.append(child)
    gripper_default = gripper.find("default")
    if gripper_default is not None:
        for child in list(gripper_default):
            robot_default.append(child)

    wrist = _find_body(root, "wrist_3_link")
    attachment = ET.SubElement(wrist, "body", {
        "name": "robotiq_mount", "pos": "0 0.1 0", "quat": "-0.70710678 0.70710678 0 0",
    })
    gripper_worldbody = gripper.find("worldbody")
    if gripper_worldbody is None or not list(gripper_worldbody.findall("body")):
        raise RuntimeError("Robotiq model has no root body")
    attachment.append(list(gripper_worldbody.findall("body"))[0])

    for section_name in ("contact", "tendon", "equality", "actuator"):
        source = gripper.find(section_name)
        if source is None:
            continue
        destination = root.find(section_name)
        if destination is None:
            destination = ET.SubElement(root, section_name)
        for child in list(source):
            destination.append(child)

    # The original six-axis keyframe no longer has the articulation's full qpos.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)


def _build_robot_include(spec: RobotSpec) -> Path | None:
    if spec.uid == "panda":
        return None
    root = ET.parse(PROJECT_ROOT / spec.source_xml).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    source_folder = Path(spec.source_xml).parent
    source_asset_folder = PROJECT_ROOT / source_folder / "assets"
    # The master scene has its own compiler settings.  Absolute asset paths
    # prevent that later compiler element from rebasing an included robot.
    compiler.set("meshdir", "")
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", (source_asset_folder / mesh.get("file", "")).as_posix())

    _find_body(root, spec.base_body).set(
        "pos", " ".join(f"{value:.9g}" for value in spec.base_position_m),
    )
    if spec.uid == "ur5e":
        _merge_robotiq_into_ur5(root)
        robotiq_assets = PROJECT_ROOT / "assets/mujoco_menagerie/robotiq_2f85/assets"
        for mesh in root.findall("./asset/mesh"):
            if mesh.get("name", "").startswith("rb_"):
                mesh.set("file", (robotiq_assets / Path(mesh.get("file", "")).name).as_posix())
    _add_wrist_camera(root, spec)
    _add_fill_light(root)
    return _temporary_xml(root)


def build_scene_xml(scene_name: str, robot_uid: str = "panda") -> tuple[Path, list[Path]]:
    """Generate temporary XML in the runtime folder with valid asset paths."""
    if scene_name not in SceneManager.NAMES:
        raise ValueError(f"Unknown scene: {scene_name}")
    spec = get_robot_spec(robot_uid)
    source_path = _EMPTY_SCENE_PATH if scene_name == _EMPTY_SCENE else SCENE_PATH
    root = ET.parse(source_path).getroot()
    _configure_robot_pedestal(root, spec)
    _configure_robot_workspace(root, spec)
    active_names = set(SceneManager.body_names(scene_name))
    known_names = set(SceneManager.body_names())
    _remove_inactive_bodies(root, active_names, known_names)
    _add_round_grasp_contact_pairs(root, spec, active_names)
    include_elements = [
        (include, include.get("file", "").replace("\\", "/"))
        for include in root.findall("include")
    ]
    _rebase_xml_paths(root)

    configured_includes = _SCENE_INCLUDES.get(scene_name)
    wanted_includes = (
        set(configured_includes)
        if isinstance(configured_includes, tuple)
        else ({configured_includes} if configured_includes is not None else set())
    )
    temporary_paths: list[Path] = []
    robot_include = _build_robot_include(spec)
    if robot_include is not None:
        temporary_paths.append(robot_include)
    for include, include_file in include_elements:
        if include_file in _ROBOT_INCLUDES:
            if include_file != _PANDA_INCLUDE:
                root.remove(include)
                continue
            if robot_include is not None:
                include.set("file", robot_include.name)
            continue
        if include_file not in wanted_includes:
            root.remove(include)
            continue

        # Category files can contain several logical scenes (notably YCB).
        # Prune them too, otherwise their unused meshes and textures would
        # still consume RAM/VRAM despite the master-model body filtering.
        include_root = ET.parse(PROJECT_ROOT / include_file).getroot()
        _remove_inactive_bodies(include_root, active_names, known_names)
        _prune_removed_body_pairs(include_root)
        _prune_object_assets(include_root)
        _rebase_xml_paths(include_root)
        include_path = _temporary_xml(include_root)
        temporary_paths.append(include_path)
        include.set("file", include_path.name)

    model_path = _temporary_xml(root)
    temporary_paths.append(model_path)
    return model_path, temporary_paths


def load_scene_model(scene_name: str, robot_uid: str = "panda") -> mujoco.MjModel:
    """Compile one scene and delete its transient XML immediately afterward."""
    model_path, temporary_paths = build_scene_xml(scene_name, robot_uid)
    try:
        return mujoco.MjModel.from_xml_path(str(model_path))
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
