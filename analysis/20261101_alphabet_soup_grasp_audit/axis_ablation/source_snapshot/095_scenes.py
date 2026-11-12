"""Object-only scene variants sharing the Panda workcell, table and cameras."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..config import TABLE_ELEVATION_M


_CANONICAL_TABLE_CENTER_M = np.array((0.49, 0.0, 0.65))
# Keep authored task layouts inside the high-quality part of every robot's
# tabletop workspace.  Robot-specific table translations place this centre at
# the calibrated manipulation centre; contracting around it moves both near
# pickup objects and far receptacles toward good IK/tracking configurations
# without changing their ordering.
_WORKSPACE_PLANAR_SCALE = np.array((0.65, 0.80))
_CUP_RELEASE_FRICTION = np.array((0.32, 0.003, 0.0005))
_CUP_CONTACT_PRIORITY = 2
_INSERT_OBJECT_SIZES = (
    ("medium", 0.080, 0.030),
    ("large", 0.100, 0.050),
    ("small", 0.090, 0.040),
)
_GEOMETRY_BOX_SIZES = (
    ("medium", 0.080, 0.060),
    ("large", 0.100, 0.070),
    ("small", 0.070, 0.040),
)
_GEOMETRY_CUBE_EDGES = ((2, 0.030), (3, 0.040), (4, 0.050))
_INSERT_OBJECT_SHAPES = ("box", "cylinder")
_INSERT_OBJECT_COLORS = ("red", "green", "blue")
_INSERT_CONTAINER_MODES = ("up", "front_left", "front_right")
# Side-facing holders are intentionally kept in the model registry so scene
# activation can hide their visuals and disable their collisions.  They are
# currently not used by the task and are therefore omitted from seeded
# variants; retaining the definitions makes a future re-enable trivial.
_ACTIVE_INSERT_CONTAINER_MODES = ("up",)
_RETIRED_SCENE = "_retired"

GEOMETRY_PLATE_BOX_UPRIGHT_SCENE = "Geometry Plate - Upright Box"
GEOMETRY_PLATE_BOX_LYING_SCENE = "Geometry Plate - Lying Box"
GEOMETRY_PLATE_CUBE_SCENE = "Geometry Plate - Cube"
GEOMETRY_PLATE_BALL_SCENE = "Geometry Plate - Ball"
GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE = "Geometry Plate - Upright Cylinder"
GEOMETRY_PLATE_CYLINDER_LYING_SCENE = "Geometry Plate - Lying Cylinder"
GEOMETRY_REGION_BOX_UPRIGHT_SCENE = "Geometry Region - Upright Box"
GEOMETRY_REGION_BOX_LYING_SCENE = "Geometry Region - Lying Box"
GEOMETRY_REGION_CUBE_SCENE = "Geometry Region - Cube"
GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE = "Geometry Region - Upright Cylinder"
GEOMETRY_REGION_CYLINDER_LYING_SCENE = "Geometry Region - Lying Cylinder"
CYLINDER_TO_NARROW_BOX_SCENE = "Cylinder to Narrow Box"
BOX_TO_NARROW_BOX_SCENE = "Lying Box to Narrow Box"
TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE = (
    "Tolerance Discrimination - Lying Box"
)

GEOMETRY_TRAINING_SCENES = (
    GEOMETRY_PLATE_BOX_UPRIGHT_SCENE,
    GEOMETRY_PLATE_BOX_LYING_SCENE,
    GEOMETRY_PLATE_CUBE_SCENE,
    GEOMETRY_PLATE_BALL_SCENE,
    GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE,
    GEOMETRY_PLATE_CYLINDER_LYING_SCENE,
    GEOMETRY_REGION_BOX_UPRIGHT_SCENE,
    GEOMETRY_REGION_BOX_LYING_SCENE,
    GEOMETRY_REGION_CUBE_SCENE,
    GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE,
    GEOMETRY_REGION_CYLINDER_LYING_SCENE,
)

OPEN_CYLINDRICAL_POT_LID_SCENE = "Open Cylindrical Pot Lid"
OPEN_HANDLE_POT_LID_SCENE = "Open Handle Pot Lid"
CLOSE_CYLINDRICAL_POT_LID_SCENE = "Close Cylindrical Pot Lid"
CLOSE_HANDLE_POT_LID_SCENE = "Close Handle Pot Lid"
ROUND_HANDLE_POT_PEAR_SCENE = "Round Handle Pot with Pear"
BANANA_APPLE_TRAY_SCENE = "Banana and Apple to Tray"
WHITEBOARD_SCENE = "Whiteboard"
INDEPENDENT_POT_LID_SCENES = (
    OPEN_CYLINDRICAL_POT_LID_SCENE,
    OPEN_HANDLE_POT_LID_SCENE,
    CLOSE_CYLINDRICAL_POT_LID_SCENE,
    CLOSE_HANDLE_POT_LID_SCENE,
)

ADJUST_CYLINDRICAL_BOTTLE_SCENE = "Adjust Cylindrical Bottle"
ADJUST_RECTANGULAR_BOTTLE_SCENE = "Adjust Rectangular Bottle"

UNSEEN_FRUIT_RECEPTACLES_SCENE = "Unseen Objects - Apple and Lemon"
UNSEEN_SPORTS_BALL_PLATE_SCENE = "Unseen Objects - Sports Ball to Plate"
UNSEEN_BOX_PLATE_SCENE = "Unseen Objects - Upright Box to Plate"
UNSEEN_BOX_REGION_SCENE = "Unseen Objects - Upright Box to White Region"
UNSEEN_BUTTON_SCENE = "Unseen Objects - Press Button"
UNSEEN_POWER_STRIP_SCENE = "Unseen Objects - Four-slot Power Strip"
UNSEEN_OBJECT_SCENES = (
    UNSEEN_FRUIT_RECEPTACLES_SCENE,
    UNSEEN_SPORTS_BALL_PLATE_SCENE,
    UNSEEN_BOX_PLATE_SCENE,
    UNSEEN_BOX_REGION_SCENE,
    UNSEEN_BUTTON_SCENE,
    UNSEEN_POWER_STRIP_SCENE,
)

_POT_LID_SCENE_SPECS = {
    OPEN_CYLINDRICAL_POT_LID_SCENE: (
        "Open Pot Lid", ("teleop_pot", "teleop_pot_lid"), 0,
    ),
    OPEN_HANDLE_POT_LID_SCENE: (
        "Open Pot Lid", ("teleop_pot_lift", "teleop_pot_lift_lid"), 1,
    ),
    CLOSE_CYLINDRICAL_POT_LID_SCENE: (
        "Close Pot Lid", ("teleop_pot", "teleop_pot_lid"), 0,
    ),
    CLOSE_HANDLE_POT_LID_SCENE: (
        "Close Pot Lid", ("teleop_pot_lift", "teleop_pot_lift_lid"), 1,
    ),
}


def _paper_variant(kind: str, size: str, *, upright: bool, target: str) -> tuple[str, ...]:
    suffix = f"{kind}_{size}{'_upright' if upright else ''}"
    object_name = f"paper_object_{suffix}"
    if target == "plate":
        return object_name, "tray_khaki_medium"
    return object_name, f"paper_decal_{suffix}"


_GEOMETRY_SCENE_VARIANTS = {
    GEOMETRY_PLATE_BOX_UPRIGHT_SCENE: tuple(
        _paper_variant("box", size, upright=True, target="plate") for size in ("small", "medium", "large")
    ),
    GEOMETRY_PLATE_BOX_LYING_SCENE: tuple(
        _paper_variant("box", size, upright=False, target="plate") for size in ("small", "medium", "large")
    ),
    GEOMETRY_PLATE_CUBE_SCENE: tuple(
        (f"paper_object_cube_{size}", "tray_khaki_medium")
        for size in (2, 3, 4)
    ),
    GEOMETRY_PLATE_BALL_SCENE: tuple(
        (f"paper_object_ball_{size}", "tray_khaki_medium")
        for size in ("small", "medium", "large")
    ),
    GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE: tuple(
        _paper_variant("cylinder", size, upright=True, target="plate") for size in ("small", "medium", "large")
    ),
    GEOMETRY_PLATE_CYLINDER_LYING_SCENE: tuple(
        _paper_variant("cylinder", size, upright=False, target="plate") for size in ("small", "medium", "large")
    ),
    GEOMETRY_REGION_BOX_UPRIGHT_SCENE: tuple(
        _paper_variant("box", size, upright=True, target="region") for size in ("small", "medium", "large")
    ),
    GEOMETRY_REGION_BOX_LYING_SCENE: tuple(
        _paper_variant("box", size, upright=False, target="region") for size in ("small", "medium", "large")
    ),
    GEOMETRY_REGION_CUBE_SCENE: tuple(
        (f"paper_object_cube_{size}", f"paper_decal_cube_{size}") for size in (2, 3, 4)
    ),
    GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE: tuple(
        _paper_variant("cylinder", size, upright=True, target="region") for size in ("small", "medium", "large")
    ),
    GEOMETRY_REGION_CYLINDER_LYING_SCENE: tuple(
        _paper_variant("cylinder", size, upright=False, target="region") for size in ("small", "medium", "large")
    ),
    TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE: tuple(
        (
            f"paper_object_box_{size}",
            "tray_khaki_medium",
            f"paper_decal_box_{size}",
        )
        for size in ("small", "medium", "large")
    ),
    CYLINDER_TO_NARROW_BOX_SCENE: tuple(
        (f"narrow_box_cylinder_{size}", f"narrow_box_{size}")
        for size in ("small", "medium", "large")
    ),
    BOX_TO_NARROW_BOX_SCENE: tuple(
        (f"narrow_box_block_{size}", f"block_narrow_box_{size}")
        for size in ("small", "medium", "large")
    ),
}
_POT_LID_SCENE_VARIANTS = {
    scene_name: tuple(
        bodies for _ in range(task_variant_index + 1)
    )
    for scene_name, (
        _source_scene, bodies, task_variant_index,
    ) in _POT_LID_SCENE_SPECS.items()
}
_COMPOSED_SCENE_VARIANTS = {
    **_GEOMETRY_SCENE_VARIANTS,
    **_POT_LID_SCENE_VARIANTS,
    "Fruit Receptacles": ((
        "fruit_task_banana",
        "fruit_task_strawberry",
        "tray_khaki_medium",
    ),),
    BANANA_APPLE_TRAY_SCENE: ((
        "fruit_task_banana",
        "fruit_task_apple",
        "tray_khaki_medium",
    ),),
    ROUND_HANDLE_POT_PEAR_SCENE: ((
        "teleop_pot",
        "teleop_pot_lid",
        "fruit_task_pear",
    ),),
    ADJUST_CYLINDRICAL_BOTTLE_SCENE: (
        ("adjust_bottle_base0",),
        ("adjust_bottle_base1",),
    ),
    ADJUST_RECTANGULAR_BOTTLE_SCENE: (
        ("adjust_bottle_ganten",),
    ),
    # The fruit scene intentionally keeps both held-out fruits and both
    # receptacles visible for every one of its four language tasks.
    UNSEEN_FRUIT_RECEPTACLES_SCENE: (
        ("fruit_apple", "fruit_lemon", "ycb_plate_white", "ycb_bowl_white"),
    ),
    # Exactly one sports ball is selected per rollout; the plate is shared.
    UNSEEN_SPORTS_BALL_PLATE_SCENE: (
        ("ycb_baseball", "ycb_plate_white"),
        ("ycb_tennis_ball", "ycb_plate_white"),
        ("ycb_golf_ball", "ycb_plate_white"),
    ),
    UNSEEN_BOX_PLATE_SCENE: (
        ("ycb_sugar_box", "ycb_plate_white"),
    ),
    UNSEEN_BOX_REGION_SCENE: (
        ("ycb_sugar_box", "unseen_box_white_region"),
    ),
    UNSEEN_BUTTON_SCENE: (
        ("press_button_yellow_red",),
        ("press_button_white_red",),
        ("press_button_white_green",),
    ),
    UNSEEN_POWER_STRIP_SCENE: (
        ("power_strip_5225",),
    ),
}
_GEOMETRY_COLORS = np.array((
    (0.88, 0.12, 0.10, 1.0),
    (0.95, 0.78, 0.08, 1.0),
    (0.58, 0.22, 0.82, 1.0),
    (0.12, 0.72, 0.28, 1.0),
    (0.10, 0.56, 0.92, 1.0),
    (0.95, 0.38, 0.08, 1.0),
))
_GEOMETRY_COLOR_NAMES = (
    "red", "yellow", "purple", "green", "blue", "orange",
)
_COLOR_RANDOMIZED_GEOMETRY_SCENES = frozenset((
    *_GEOMETRY_SCENE_VARIANTS,
    "Paper Placement",
    "Ball + Paper Placement",
))


@dataclass(frozen=True)
class _ObjectState:
    body_name: str
    scene_name: str
    qpos: tuple[float, ...] | None = None


@dataclass(frozen=True)
class LayoutObject:
    """Current root pose and collision geometry for one scene object."""

    body_name: str
    position: np.ndarray
    quaternion: np.ndarray
    geom_ids: np.ndarray


class SceneManager:
    """Toggle object groups without rebuilding the common workcell XML."""

    EMPTY_SCENE = "Empty workspace"

    NAMES = (
        "Blocks", "Pot + lid", "Water bottles", "Fruits",
        "Packaged food", "Tableware", "Sports balls", "Cups",
        "Cylindrical containers", "Rectangular objects", "RoboCasa saucepans",
        "Trays",
        "RoboTwin bottles",
        "Adjust Bottle",
        ADJUST_CYLINDRICAL_BOTTLE_SCENE,
        ADJUST_RECTANGULAR_BOTTLE_SCENE,
        "Click Bell",
        "Press Button",
        "Press Power Strip",
        "Insert Object",
        "Fruit Receptacles",
        "Paper Placement",
        "Ball + Paper Placement",
        "Color Block Stack",
        "Rotate Knob",
        WHITEBOARD_SCENE,
        "Open Pot Lid",
        "Close Pot Lid",
        *GEOMETRY_TRAINING_SCENES,
        CYLINDER_TO_NARROW_BOX_SCENE,
        BOX_TO_NARROW_BOX_SCENE,
        TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
        *INDEPENDENT_POT_LID_SCENES,
        BANANA_APPLE_TRAY_SCENE,
        ROUND_HANDLE_POT_PEAR_SCENE,
        *UNSEEN_OBJECT_SCENES,
        EMPTY_SCENE,
    )
    UI_NAMES = tuple(name for name in NAMES if name != "Insert Object")
    _SEEDED_VARIANTS = {
        **_COMPOSED_SCENE_VARIANTS,
        "Adjust Bottle": (
            ("adjust_bottle_base0",),
            ("adjust_bottle_base1",),
            ("adjust_bottle_ganten",),
            ("adjust_bottle_robotwin0",),
            ("adjust_bottle_robotwin1",),
        ),
        WHITEBOARD_SCENE: (("wipe_board", "board_eraser"),),
        "Click Bell": (
            ("click_bell_base0",),
            ("click_bell_base1",),
        ),
        "Press Button": (
            ("press_button_yellow_red",),
            ("press_button_white_red",),
            ("press_button_white_green",),
        ),
        "Press Power Strip": (
            ("power_strip_525",),
            ("power_strip_555",),
            ("power_strip_5225",),
        ),
        "Insert Object": tuple(
            (
                f"insert_container_{shape}_{size_name}_{mode}",
                f"insert_object_{shape}_{size_name}_{color}",
            )
            for shape in _INSERT_OBJECT_SHAPES
            for size_name, _, _ in _INSERT_OBJECT_SIZES
            for color in _INSERT_OBJECT_COLORS
            for mode in _ACTIVE_INSERT_CONTAINER_MODES
        ),
        "Fruit Receptacles": ((
            "fruit_task_banana",
            "fruit_task_strawberry",
            "tray_khaki_medium",
        ),),
        BANANA_APPLE_TRAY_SCENE: ((
            "fruit_task_banana",
            "fruit_task_apple",
            "tray_khaki_medium",
        ),),
        ROUND_HANDLE_POT_PEAR_SCENE: ((
            "teleop_pot",
            "teleop_pot_lid",
            "fruit_task_pear",
        ),),
        "Paper Placement": (
            *(
                (
                    f"paper_object_cube_{size}",
                    f"paper_decal_cube_{size}",
                )
                for size in (2, 3, 4)
            ),
            *(
                (
                    f"paper_object_{shape}_{size_name}",
                    f"paper_decal_{shape}_{size_name}",
                )
                for shape in ("box", "cylinder")
                for size_name in ("medium", "large", "small")
            ),
            *(
                (
                    f"paper_object_{shape}_{size_name}_upright",
                    f"paper_decal_{shape}_{size_name}_upright",
                )
                for shape in ("box", "cylinder")
                for size_name in ("medium", "large", "small")
            ),
        ),
        "Ball + Paper Placement": tuple(
            (
                "teleop_ball",
                "teleop_plate",
                f"paper_object_cube_{size}",
                f"paper_decal_cube_{size}",
            )
            for size in (2, 3, 4)
        ),
        "Color Block Stack": tuple(
            ("stack_cube_yellow", "stack_cube_red", "stack_cube_purple")
            for _ in (3, 4, 5)
        ),
        "Rotate Knob": tuple(
            (f"rotate_knob_{size}",)
            for size in ("small", "medium", "large")
        ),
        "Open Pot Lid": (
            ("teleop_pot", "teleop_pot_lid"),
            ("teleop_pot_lift", "teleop_pot_lift_lid"),
        ),
        "Close Pot Lid": (
            ("teleop_pot", "teleop_pot_lid"),
            ("teleop_pot_lift", "teleop_pot_lift_lid"),
        ),
    }

    _COMPOSED_POSE_OVERRIDES = {
        UNSEEN_FRUIT_RECEPTACLES_SCENE: {
            "fruit_apple": (0.34, -0.12, 0.275, 1.0, 0.0, 0.0, 0.0),
            "fruit_lemon": (0.34, 0.12, 0.275, 1.0, 0.0, 0.0, 0.0),
            "ycb_plate_white": (0.63, -0.20, 0.275, 1.0, 0.0, 0.0, 0.0),
            "ycb_bowl_white": (0.63, 0.20, 0.275, 1.0, 0.0, 0.0, 0.0),
        },
        UNSEEN_SPORTS_BALL_PLATE_SCENE: {
            "ycb_baseball": (0.36, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0),
            "ycb_tennis_ball": (0.36, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0),
            "ycb_golf_ball": (0.36, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0),
            "ycb_plate_white": (0.63, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0),
        },
        UNSEEN_BOX_PLATE_SCENE: {
            "ycb_sugar_box": (0.36, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0),
            "ycb_plate_white": (0.63, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0),
        },
        UNSEEN_BOX_REGION_SCENE: {
            "ycb_sugar_box": (0.36, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0),
        },
    }
    _VARIANT_STREAM_ID = 0x424F54544C455641  # "BOTTLEVA"
    _OBJECTS = (
        _ObjectState("wipe_board", WHITEBOARD_SCENE),
        _ObjectState("board_eraser", WHITEBOARD_SCENE, (0.45, 0.45, 0.282, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("teleop_cube", "Blocks", (0.47, 0.0, 0.290, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("teleop_tall_block", "Blocks", (0.59, 0.12, 0.320, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("teleop_plate", "Blocks"),
        _ObjectState("teleop_ball", "Blocks", (0.48, -0.14, 0.295, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState(
            "teleop_ball", "Ball + Paper Placement",
            (0.40, -0.14, 0.295, 1.0, 0.0, 0.0, 0.0),
        ),
        _ObjectState("teleop_plate", "Ball + Paper Placement"),
        _ObjectState("teleop_cylinder", "Blocks", (0.45, 0.17, 0.315, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("teleop_pen", _RETIRED_SCENE, (0.45, -0.13, 0.293, 0.70710678, 0.0, 0.70710678, 0.0)),
        _ObjectState("teleop_pen_holder", _RETIRED_SCENE),
        _ObjectState("teleop_pot", "Pot + lid"),
        _ObjectState("teleop_pot_lid", "Pot + lid", (0.53, -0.05, 0.347, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("teleop_pot_lift", "Pot + lid"),
        _ObjectState("teleop_pot_lift_lid", "Pot + lid", (0.53, 0.18, 0.347, 1.0, 0.0, 0.0, 0.0)),
        # Task scenes contain exactly one seeded pot/lid pair.  Open starts
        # with the lid seated; Close starts with it resting beside the pot.
        _ObjectState("teleop_pot", "Open Pot Lid"),
        _ObjectState("teleop_pot_lid", "Open Pot Lid", (0.53, -0.05, 0.347, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("teleop_pot_lift", "Open Pot Lid"),
        _ObjectState("teleop_pot_lift_lid", "Open Pot Lid", (0.53, 0.18, 0.347, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("teleop_pot", "Close Pot Lid"),
        _ObjectState("teleop_pot_lid", "Close Pot Lid", (0.53, 0.30, 0.279, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("teleop_pot_lift", "Close Pot Lid"),
        _ObjectState("teleop_pot_lift_lid", "Close Pot Lid", (0.53, -0.17, 0.279, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("water_nongfu_550", "Water bottles", (0.56, -0.18, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("water_cestbon_555", "Water bottles", (0.56, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("water_ganten_570", "Water bottles", (0.56, 0.18, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("water_nongfu_550_bare", "Water bottles", (0.39, -0.18, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("water_cestbon_555_bare", "Water bottles", (0.39, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("water_ganten_570_bare", "Water bottles", (0.39, 0.18, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("fruit_banana", "Fruits", (0.42, -0.20, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("fruit_strawberry", "Fruits", (0.39, -0.04, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("fruit_apple", "Fruits", (0.50, -0.10, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("fruit_lemon", "Fruits", (0.61, -0.20, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("fruit_peach", "Fruits", (0.62, -0.07, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("fruit_pear", "Fruits", (0.50, 0.07, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("fruit_orange", "Fruits", (0.62, 0.08, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("fruit_plum", "Fruits", (0.62, 0.20, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_master_chef_can", "Packaged food", (0.42, -0.16, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_cracker_box", "Packaged food", (0.51, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_sugar_box", "Packaged food", (0.60, 0.18, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("unseen_box_white_region", "Packaged food"),
        _ObjectState("ycb_bowl_white", "Tableware", (0.62, -0.25, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_bowl_red", "Tableware", (0.62, 0.25, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_mug_white", "Tableware", (0.61, -0.075, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_mug_red", "Tableware", (0.61, 0.075, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_plate_white", "Tableware", (0.42, -0.23, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_plate_red", "Tableware", (0.42, 0.23, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_baseball", "Sports balls", (0.43, -0.12, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_tennis_ball", "Sports balls", (0.52, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_golf_ball", "Sports balls", (0.61, 0.12, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_cup_small", "Cups", (0.39, -0.18, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_cup_medium", "Cups", (0.50, -0.18, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("ycb_cup_large", "Cups", (0.62, -0.18, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("disposable_plastic_cup", "Cups", (0.43, 0.15, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("disposable_paper_cup", "Cups", (0.59, 0.15, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_tuna_can", "Cylindrical containers", (0.42, -0.28, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_allergen_bottle", "Cylindrical containers", (0.42, -0.14, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_probiotic_bottle", "Cylindrical containers", (0.42, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_tomato_can", "Cylindrical containers", (0.42, 0.14, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_5_htp", "Cylindrical containers", (0.42, 0.28, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_borage_gla", "Cylindrical containers", (0.58, -0.21, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_pet_dophilus_powder", "Cylindrical containers", (0.58, -0.07, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_beta_glucan", "Cylindrical containers", (0.58, 0.07, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("container_bifidus_balance", "Cylindrical containers", (0.58, 0.21, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("rect_cracker_box", "Rectangular objects", (0.60, -0.24, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("rect_fondant_box", "Rectangular objects", (0.60, -0.08, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("rect_invitation_kit", "Rectangular objects", (0.60, 0.08, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("rect_three_ink_pack", "Rectangular objects", (0.60, 0.24, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("rect_elderberry_box", "Rectangular objects", (0.42, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("tray_khaki_medium", "Trays"),
        _ObjectState("robocasa_saucepan_002", "RoboCasa saucepans", (0.42, -0.24, 0.320818615, 0.707106781, 0.0, 0.0, -0.707106781)),
        _ObjectState("robocasa_lid_002", "RoboCasa saucepans", (0.42, -0.24, 0.379818615, 0.707106781, 0.0, 0.0, -0.707106781)),
        _ObjectState("robocasa_saucepan_003", "RoboCasa saucepans", (0.42, 0.0, 0.313306210, 0.707106781, 0.0, 0.0, -0.707106781)),
        _ObjectState("robocasa_lid_003", "RoboCasa saucepans", (0.42, 0.0, 0.372306210, 0.707106781, 0.0, 0.0, -0.707106781)),
        _ObjectState("robocasa_saucepan_004", "RoboCasa saucepans", (0.42, 0.24, 0.319769755, 0.707106781, 0.0, 0.0, -0.707106781)),
        _ObjectState("robocasa_lid_004", "RoboCasa saucepans", (0.42, 0.24, 0.378769755, 0.707106781, 0.0, 0.0, -0.707106781)),
        _ObjectState("robocasa_saucepan_006", "RoboCasa saucepans", (0.63, -0.24, 0.306401235, 0.707106781, 0.0, 0.0, 0.707106781)),
        _ObjectState("robocasa_lid_006", "RoboCasa saucepans", (0.63, -0.24, 0.365401235, 0.707106781, 0.0, 0.0, 0.707106781)),
        _ObjectState("robocasa_saucepan_008", "RoboCasa saucepans", (0.63, 0.0, 0.315015655, 0.707106781, 0.0, 0.0, 0.707106781)),
        _ObjectState("robocasa_lid_008", "RoboCasa saucepans", (0.629990275, 0.0, 0.374015655, 0.707106781, 0.0, 0.0, 0.707106781)),
        _ObjectState("robocasa_saucepan_013", "RoboCasa saucepans", (0.63, 0.24, 0.313531610, 0.707106781, 0.0, 0.0, 0.707106781)),
        _ObjectState("robocasa_lid_013", "RoboCasa saucepans", (0.63, 0.24, 0.372531610, 0.707106781, 0.0, 0.0, 0.707106781)),
        _ObjectState("robotwin_bottle_base0", "RoboTwin bottles", (0.50, -0.07, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState("robotwin_bottle_base1", "RoboTwin bottles", (0.50, 0.07, 0.275, 1.0, 0.0, 0.0, 0.0)),
        _ObjectState(
            "adjust_bottle_base0", "Adjust Bottle",
            (0.42, 0.0, 0.307, 0.707106781, 0.0, 0.707106781, 0.0),
        ),
        _ObjectState(
            "adjust_bottle_base1", "Adjust Bottle",
            (0.42, 0.0, 0.3075, 0.707106781, 0.0, 0.707106781, 0.0),
        ),
        _ObjectState(
            "adjust_bottle_ganten", "Adjust Bottle",
            (0.42, 0.0, 0.308, 0.707106781, 0.0, 0.707106781, 0.0),
        ),
        _ObjectState(
            "adjust_bottle_robotwin0", "Adjust Bottle",
            (0.42, 0.0, 0.3074, 0.707106781, 0.0, 0.707106781, 0.0),
        ),
        _ObjectState(
            "adjust_bottle_robotwin1", "Adjust Bottle",
            (0.42, 0.0, 0.3075, 0.707106781, 0.0, 0.707106781, 0.0),
        ),
        _ObjectState("click_bell_base0", "Click Bell"),
        _ObjectState("click_bell_base1", "Click Bell"),
        _ObjectState("press_button_yellow_red", "Press Button"),
        _ObjectState("press_button_white_red", "Press Button"),
        _ObjectState("press_button_white_green", "Press Button"),
        _ObjectState("power_strip_525", "Press Power Strip"),
        _ObjectState("power_strip_555", "Press Power Strip"),
        _ObjectState("power_strip_5225", "Press Power Strip"),
        _ObjectState("insert_strip_2_round", _RETIRED_SCENE),
        _ObjectState("insert_strip_2_rocker", _RETIRED_SCENE),
        _ObjectState("insert_strip_3_round", _RETIRED_SCENE),
        _ObjectState("insert_strip_3_rocker", _RETIRED_SCENE),
        _ObjectState("insert_strip_4_round", _RETIRED_SCENE),
        _ObjectState("insert_strip_4_rocker", _RETIRED_SCENE),
        _ObjectState(
            "charger_2pin", _RETIRED_SCENE,
            (
                0.40, -0.12, 0.300,
                0.707106781, 0.0, 0.707106781, 0.0,
            ),
        ),
        _ObjectState(
            "charger_3pin", _RETIRED_SCENE,
            (
                0.40, -0.12, 0.300,
                0.707106781, 0.0, 0.707106781, 0.0,
            ),
        ),
        *(
            _ObjectState(
                f"insert_container_{shape}_{size_name}_{mode}",
                "Insert Object",
            )
            for shape in _INSERT_OBJECT_SHAPES
            for size_name, _, _ in _INSERT_OBJECT_SIZES
            for mode in _INSERT_CONTAINER_MODES
        ),
        *(
            _ObjectState(
                f"insert_object_{shape}_{size_name}_{color}",
                "Insert Object",
                (
                    0.40, -0.05, 0.275 + 0.5 * width,
                    0.707106781, 0.0, 0.707106781, 0.0,
                ),
            )
            for shape in _INSERT_OBJECT_SHAPES
            for size_name, _, width in _INSERT_OBJECT_SIZES
            for color in _INSERT_OBJECT_COLORS
        ),
        _ObjectState(
            "fruit_task_banana", "Fruit Receptacles",
            (0.22, -0.17, 0.275, 1.0, 0.0, 0.0, 0.0),
        ),
        _ObjectState(
            "fruit_task_strawberry", "Fruit Receptacles",
            (0.22, 0.17, 0.275, 1.0, 0.0, 0.0, 0.0),
        ),
        _ObjectState(
            "fruit_task_banana", BANANA_APPLE_TRAY_SCENE,
            (0.22, -0.17, 0.275, 1.0, 0.0, 0.0, 0.0),
        ),
        _ObjectState(
            "fruit_task_apple", BANANA_APPLE_TRAY_SCENE,
            (0.22, 0.17, 0.275, 1.0, 0.0, 0.0, 0.0),
        ),
        _ObjectState(
            "fruit_task_pear", ROUND_HANDLE_POT_PEAR_SCENE,
            (
                0.22, 0.17, 0.3085,
                0.707106781, 0.0, 0.707106781, 0.0,
            ),
        ),
        *(
            _ObjectState(f"fruit_task_plate_{color}", _RETIRED_SCENE)
            for color in ("white", "red")
        ),
        *(
            _ObjectState(f"fruit_task_bowl_{color}", _RETIRED_SCENE)
            for color in ("white", "red")
        ),
        *(
            _ObjectState(
                f"paper_object_cube_{legacy_size}",
                "Paper Placement",
                (
                    0.40, -0.08, 0.275 + 0.5 * edge,
                    1.0, 0.0, 0.0, 0.0,
                ),
            )
            for legacy_size, edge in _GEOMETRY_CUBE_EDGES
        ),
        *(
            _ObjectState(
                f"paper_object_ball_{size_name}",
                "Paper Placement",
                (
                    0.40, -0.08, 0.275 + radius,
                    1.0, 0.0, 0.0, 0.0,
                ),
            )
            for size_name, radius in (
                ("small", 0.020), ("medium", 0.025), ("large", 0.030),
            )
        ),
        *(
            _ObjectState(
                f"paper_object_box_{size_name}",
                "Paper Placement",
                (
                    0.40, -0.08, 0.275 + 0.5 * width,
                    0.707106781, 0.0, 0.707106781, 0.0,
                ),
            )
            for size_name, _, width in _GEOMETRY_BOX_SIZES
        ),
        *(
            _ObjectState(
                f"paper_object_cylinder_{size_name}",
                "Paper Placement",
                (
                    0.40, -0.08, 0.275 + 0.5 * width,
                    0.707106781, 0.0, 0.707106781, 0.0,
                ),
            )
            for size_name, _, width in _INSERT_OBJECT_SIZES
        ),
        *(
            _ObjectState(
                f"paper_object_box_{size_name}_upright",
                "Paper Placement",
                (
                    0.40, -0.08, 0.275 + 0.5 * length,
                    1.0, 0.0, 0.0, 0.0,
                ),
            )
            for size_name, length, _ in _GEOMETRY_BOX_SIZES
        ),
        *(
            _ObjectState(
                f"paper_object_cylinder_{size_name}_upright",
                "Paper Placement",
                (
                    0.40, -0.08, 0.275 + 0.5 * length,
                    1.0, 0.0, 0.0, 0.0,
                ),
            )
            for size_name, length, _ in _INSERT_OBJECT_SIZES
        ),
        *(
            _ObjectState(
                f"paper_object_cube_{legacy_size}",
                "Ball + Paper Placement",
                (
                    0.40, 0.14, 0.275 + 0.5 * edge,
                    1.0, 0.0, 0.0, 0.0,
                ),
            )
            for legacy_size, edge in _GEOMETRY_CUBE_EDGES
        ),
        *(
            _ObjectState(f"paper_decal_cube_{size}", "Paper Placement")
            for size in (2, 3, 4)
        ),
        *(
            _ObjectState(
                f"paper_decal_{shape}_{size_name}", "Paper Placement",
            )
            for shape in ("box", "cylinder")
            for size_name, _, _ in _INSERT_OBJECT_SIZES
        ),
        *(
            _ObjectState(
                f"paper_decal_{shape}_{size_name}_upright",
                "Paper Placement",
            )
            for shape in ("box", "cylinder")
            for size_name, _, _ in _INSERT_OBJECT_SIZES
        ),
        *(
            _ObjectState(
                f"paper_decal_cube_{size}",
                "Ball + Paper Placement",
            )
            for size in (2, 3, 4)
        ),
        *(
            _ObjectState(
                f"narrow_box_cylinder_{size_name}",
                CYLINDER_TO_NARROW_BOX_SCENE,
                (
                    0.36, -0.10, 0.275 + radius,
                    0.707106781, 0.0, 0.707106781, 0.0,
                ),
            )
            for size_name, radius in (
                ("small", 0.020),
                ("medium", 0.025),
                ("large", 0.030),
            )
        ),
        *(
            _ObjectState(
                f"narrow_box_{size_name}",
                CYLINDER_TO_NARROW_BOX_SCENE,
            )
            for size_name in ("small", "medium", "large")
        ),
        *(
            _ObjectState(
                f"narrow_box_block_{size_name}",
                BOX_TO_NARROW_BOX_SCENE,
                (
                    0.36, -0.10, 0.275 + 0.5 * width,
                    0.707106781, 0.0, 0.707106781, 0.0,
                ),
            )
            for size_name, _, width in _GEOMETRY_BOX_SIZES
        ),
        *(
            _ObjectState(
                f"block_narrow_box_{size_name}",
                BOX_TO_NARROW_BOX_SCENE,
            )
            for size_name in ("small", "medium", "large")
        ),
        *(
            _ObjectState(f"rotate_knob_{size}", "Rotate Knob")
            for size in ("small", "medium", "large")
        ),
        *(
            _ObjectState(
                f"stack_cube_{color}", "Color Block Stack",
                (x, 0.10, 0.295, 1.0, 0.0, 0.0, 0.0),
            )
            for color, x in zip(
                ("yellow", "red", "purple"),
                (0.35, 0.49, 0.63),
                strict=True,
            )
        ),
    )

    @classmethod
    def body_names(cls, scene_name: str | None = None) -> tuple[str, ...]:
        if scene_name in _COMPOSED_SCENE_VARIANTS:
            return tuple(dict.fromkeys(
                body_name
                for variant in _COMPOSED_SCENE_VARIANTS[scene_name]
                for body_name in variant
            ))
        return tuple(
            item.body_name for item in cls._OBJECTS
            if scene_name is None or item.scene_name == scene_name
        )

    @classmethod
    def _states_for_scene(cls, scene_name: str) -> tuple[_ObjectState, ...]:
        if scene_name not in _COMPOSED_SCENE_VARIANTS:
            return tuple(state for state in cls._OBJECTS if state.scene_name == scene_name)
        required = set(cls.body_names(scene_name))
        source_scene = (
            _POT_LID_SCENE_SPECS[scene_name][0]
            if scene_name in _POT_LID_SCENE_SPECS else None
        )
        pose_overrides = cls._COMPOSED_POSE_OVERRIDES.get(scene_name, {})
        result: list[_ObjectState] = []
        seen: set[str] = set()
        for state in cls._OBJECTS:
            if (
                state.body_name in required
                and state.body_name not in seen
                and (source_scene is None or state.scene_name == source_scene)
            ):
                override = pose_overrides.get(state.body_name)
                result.append(
                    _ObjectState(state.body_name, state.scene_name, override)
                    if override is not None and state.qpos is not None
                    else state
                )
                seen.add(state.body_name)
        missing = required - seen
        if missing:
            raise RuntimeError(
                f"Composed scene {scene_name!r} has unregistered bodies: {sorted(missing)}"
            )
        return tuple(result)

    def __init__(self, model: mujoco.MjModel, loaded_scene: str | None = None) -> None:
        self.model = model
        if loaded_scene is not None and loaded_scene not in self.NAMES:
            raise ValueError(f"Unknown loaded scene: {loaded_scene}")
        self.loaded_scene = loaded_scene
        self.current_name = loaded_scene or self.NAMES[0]
        self._selected_variant = 0
        self._selected_color = 0
        self._active_body_names: set[str] = set()
        self._geoms: dict[str, np.ndarray] = {}
        self._body_ids: dict[str, int] = {}
        self._body_positions: dict[str, np.ndarray] = {}
        self._body_quaternions: dict[str, np.ndarray] = {}
        self._joint_addresses: dict[str, tuple[int, int]] = {}
        self._articulated_joint_ids: dict[str, np.ndarray] = {}
        # Registry poses are authored around the canonical Panda table. Apply
        # the compiled robot-specific table translation to all tabletop
        # objects before any per-episode randomization.
        table_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        if table_id < 0 and loaded_scene != self.EMPTY_SCENE:
            raise RuntimeError("Scene model is missing work_surface")
        if table_id < 0:
            self._base_workspace_offset = np.zeros(3)
        else:
            nominal_table_delta = (
                model.body_pos[table_id] - _CANONICAL_TABLE_CENTER_M
            )
            self._base_workspace_offset = (
                np.array((0.0, 0.0, TABLE_ELEVATION_M))
                + nominal_table_delta
            )
        self._workspace_offset = self._base_workspace_offset.copy()
        self._geom_contype = model.geom_contype.copy()
        self._geom_conaffinity = model.geom_conaffinity.copy()
        self._geom_rgba = model.geom_rgba.copy()
        self._geom_friction = model.geom_friction.copy()
        self._geom_priority = model.geom_priority.copy()
        object_states = (
            self._OBJECTS
            if loaded_scene is None
            else self._states_for_scene(loaded_scene)
        )
        self._object_states = object_states
        for object_state in object_states:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_state.body_name)
            if body_id < 0:
                raise RuntimeError(f"Scene body not found: {object_state.body_name}")
            self._body_ids[object_state.body_name] = body_id
            self._body_positions[object_state.body_name] = model.body_pos[body_id].copy()
            self._body_quaternions[object_state.body_name] = model.body_quat[body_id].copy()
            # Imported assets may keep visual and collision geoms in nested
            # child bodies. Treat the complete subtree as one scene object so
            # switching scenes hides and disables every associated geom.
            descendant_ids = [
                candidate for candidate in range(model.nbody)
                if self._is_descendant(candidate, body_id)
            ]
            self._geoms[object_state.body_name] = np.flatnonzero(
                np.isin(model.geom_bodyid, descendant_ids)
            )
            if object_state.qpos is None:
                self._articulated_joint_ids[object_state.body_name] = np.array([
                    joint_id for joint_id in range(model.njnt)
                    if (
                        int(model.jnt_bodyid[joint_id]) in descendant_ids
                        and int(model.jnt_type[joint_id])
                        in (
                            int(mujoco.mjtJoint.mjJNT_HINGE),
                            int(mujoco.mjtJoint.mjJNT_SLIDE),
                        )
                    )
                ], dtype=int)
            if object_state.qpos is not None:
                joint_id = int(model.body_jntadr[body_id])
                self._joint_addresses[object_state.body_name] = (
                    int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]),
                )

    def _is_descendant(self, candidate_id: int, root_id: int) -> bool:
        while candidate_id > root_id:
            candidate_id = int(self.model.body_parentid[candidate_id])
        return candidate_id == root_id

    def activate(self, data: mujoco.MjData, scene_name: str) -> None:
        """Reset and expose one object group at the current workspace offset."""
        if scene_name not in self.NAMES:
            raise ValueError(f"Unknown scene: {scene_name}")
        if self.loaded_scene is not None and scene_name != self.loaded_scene:
            raise RuntimeError(
                f"Model contains only {self.loaded_scene!r}; scene {scene_name!r} requires a process reload."
            )
        variants = self._SEEDED_VARIANTS.get(scene_name)
        selected_names = (
            set(variants[self._selected_variant % len(variants)])
            if variants
            else None
        )
        self._active_body_names = set()
        composed_scene = scene_name in _COMPOSED_SCENE_VARIANTS
        for object_state in self._object_states:
            body_id = self._body_ids[object_state.body_name]
            geom_ids = self._geoms[object_state.body_name]
            active = (
                (composed_scene or object_state.scene_name == scene_name)
                and (
                    selected_names is None
                    or object_state.body_name in selected_names
                )
            )
            self.model.geom_friction[geom_ids] = self._geom_friction[geom_ids]
            self.model.geom_priority[geom_ids] = self._geom_priority[geom_ids]
            if active:
                self._active_body_names.add(object_state.body_name)
                self.model.geom_contype[geom_ids] = self._geom_contype[geom_ids]
                self.model.geom_conaffinity[geom_ids] = self._geom_conaffinity[geom_ids]
                self.model.geom_rgba[geom_ids] = self._geom_rgba[geom_ids]
                if scene_name == "Cups":
                    contact_ids = geom_ids[self._geom_contype[geom_ids] != 0]
                    self.model.geom_friction[contact_ids] = _CUP_RELEASE_FRICTION
                    # MuJoCo otherwise selects the larger same-priority
                    # friction value, which lets the high-friction fingers
                    # dominate and keep an opened cup hanging from the hand.
                    self.model.geom_priority[contact_ids] = _CUP_CONTACT_PRIORITY
            else:
                self.model.geom_contype[geom_ids] = 0
                self.model.geom_conaffinity[geom_ids] = 0
                self.model.geom_rgba[geom_ids] = self._geom_rgba[geom_ids]
                self.model.geom_rgba[geom_ids, 3] = 0.0
            if object_state.qpos is None:
                self.model.body_pos[body_id] = self.workspace_position(
                    self._body_positions[object_state.body_name],
                )
                self.model.body_quat[body_id] = self._body_quaternions[object_state.body_name]
                joint_ids = self._articulated_joint_ids[object_state.body_name]
                if joint_ids.size:
                    qpos_ids = self.model.jnt_qposadr[joint_ids].astype(int)
                    dof_ids = self.model.jnt_dofadr[joint_ids].astype(int)
                    data.qpos[qpos_ids] = self.model.qpos0[qpos_ids]
                    data.qvel[dof_ids] = 0.0
                continue
            qpos_address, dof_address = self._joint_addresses[object_state.body_name]
            if active:
                initial_qpos = np.asarray(object_state.qpos, dtype=float).copy()
                initial_qpos[:3] = self.workspace_position(initial_qpos[:3])
                data.qpos[qpos_address:qpos_address + 7] = initial_qpos
            else:
                # Hidden dynamic objects remain below the workcell, where
                # gravity cannot reintroduce an invisible collision.
                data.qpos[qpos_address:qpos_address + 7] = (-10.0, -10.0, -10.0, 1.0, 0.0, 0.0, 0.0)
            data.qvel[dof_address:dof_address + 6] = 0.0
        if scene_name in _COLOR_RANDOMIZED_GEOMETRY_SCENES:
            for body_name in self._active_body_names:
                if not body_name.startswith((
                    "paper_object_", "narrow_box_cylinder_",
                    "narrow_box_block_",
                )):
                    continue
                geom_ids = self._geoms[body_name]
                color = _GEOMETRY_COLORS[
                    self._selected_color % len(_GEOMETRY_COLORS)
                ]
                if body_name.startswith("narrow_box_cylinder_"):
                    visual_id = mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        f"{body_name}_geom",
                    )
                    if visual_id < 0:
                        raise RuntimeError(
                            f"Scene cylinder visual not found: {body_name}",
                        )
                    self.model.geom_rgba[visual_id] = color
                else:
                    collision_ids = geom_ids[
                        self._geom_contype[geom_ids] != 0
                    ]
                    self.model.geom_rgba[collision_ids] = color
        self.current_name = scene_name
        mujoco.mj_forward(self.model, data)

    def layout_objects(self, data: mujoco.MjData) -> tuple[LayoutObject, ...]:
        """Return active top-level objects in deterministic registry order."""
        result: list[LayoutObject] = []
        seen: set[str] = set()
        for state in self._object_states:
            if (
                state.body_name not in self._active_body_names
                or state.body_name in seen
            ):
                continue
            seen.add(state.body_name)
            body_id = self._body_ids[state.body_name]
            if state.qpos is None:
                position = self.model.body_pos[body_id].copy()
                quaternion = self.model.body_quat[body_id].copy()
            else:
                qpos_address, _ = self._joint_addresses[state.body_name]
                position = data.qpos[qpos_address:qpos_address + 3].copy()
                quaternion = data.qpos[qpos_address + 3:qpos_address + 7].copy()
            result.append(LayoutObject(
                state.body_name, position, quaternion,
                self._geoms[state.body_name].copy(),
            ))
        return tuple(result)

    @classmethod
    def variant_indices_for_bodies(
        cls,
        scene_name: str,
        body_names: tuple[str, ...],
    ) -> tuple[int, ...]:
        """Return scene variants containing any requested registered body."""
        requested = set(body_names)
        variants = cls._SEEDED_VARIANTS.get(scene_name, ())
        if not requested:
            return tuple(range(len(variants)))
        indices = tuple(
            index for index, variant in enumerate(variants)
            if requested.intersection(variant)
        )
        if not indices:
            raise ValueError(
                f"Scene {scene_name!r} has no variants for bodies {sorted(requested)}"
            )
        return indices

    def select_seeded_variant(
        self,
        seed: int,
        allowed_indices: tuple[int, ...] | None = None,
    ) -> None:
        """Choose one registered scene variant without activating it yet."""
        variants = self._SEEDED_VARIANTS.get(self.current_name)
        if not variants:
            self._selected_variant = 0
            self._selected_color = 0
            return
        if seed < -1:
            raise ValueError("Variant seed must be -1 or non-negative")
        if seed == -1:
            self._selected_variant = (
                0 if allowed_indices is None else int(allowed_indices[0])
            )
            self._selected_color = 0
            return
        choices = (
            tuple(range(len(variants)))
            if allowed_indices is None else tuple(int(index) for index in allowed_indices)
        )
        if not choices or any(not 0 <= index < len(variants) for index in choices):
            raise ValueError(
                f"Invalid allowed variant indices for {self.current_name!r}: {choices}"
            )
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(seed), self._VARIANT_STREAM_ID,
        ))))
        self._selected_variant = choices[int(rng.integers(0, len(choices)))]
        self._selected_color = int(rng.integers(0, len(_GEOMETRY_COLORS)))

    def select_variant_index(self, variant_index: int) -> None:
        """Select one scene variant explicitly for a fixed-variant task."""
        variants = self._SEEDED_VARIANTS.get(self.current_name)
        if not variants:
            if int(variant_index) != 0:
                raise ValueError(
                    f"Scene {self.current_name!r} has no selectable variants",
                )
            self._selected_variant = 0
            self._selected_color = 0
            return
        index = int(variant_index)
        if not 0 <= index < len(variants):
            raise ValueError(
                f"Variant index {index} is invalid for scene {self.current_name!r}",
            )
        self._selected_variant = index
        self._selected_color = 0

    @property
    def active_body_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_body_names))

    @property
    def selected_variant_index(self) -> int:
        """Return the currently selected deterministic scene variant."""
        return int(self._selected_variant)

    @property
    def active_geometry_color_name(self) -> str | None:
        """Return the rendered geometry color for the active training scene."""
        if self.current_name not in _COLOR_RANDOMIZED_GEOMETRY_SCENES:
            return None
        return _GEOMETRY_COLOR_NAMES[
            self._selected_color % len(_GEOMETRY_COLOR_NAMES)
        ]

    @property
    def active_geometry_color_rgba(self) -> np.ndarray | None:
        """Return the color paired with ``active_geometry_color_name``."""
        if self.current_name not in _COLOR_RANDOMIZED_GEOMETRY_SCENES:
            return None
        return _GEOMETRY_COLORS[
            self._selected_color % len(_GEOMETRY_COLORS)
        ].copy()

    def set_layout_pose(
        self,
        data: mujoco.MjData,
        body_name: str,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        """Set one fixed or free scene-object root pose without changing type."""
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion, dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Layout pose requires XYZ and a MuJoCo quaternion")
        body_id = self._body_ids[body_name]
        if body_name in self._joint_addresses:
            qpos_address, dof_address = self._joint_addresses[body_name]
            data.qpos[qpos_address:qpos_address + 3] = position
            data.qpos[qpos_address + 3:qpos_address + 7] = quaternion
            data.qvel[dof_address:dof_address + 6] = 0.0
        else:
            self.model.body_pos[body_id] = position
            self.model.body_quat[body_id] = quaternion

    def reset_layout_articulation(
        self,
        data: mujoco.MjData,
        body_name: str,
    ) -> None:
        """Restore hinge/slide joints inside one fixed layout object.

        Layout editing owns the root XY/yaw, while articulated task state
        (for example a knob angle or button travel) must return to the model's
        authored initial value whenever that layout is reapplied.
        """
        if body_name not in self._body_ids:
            raise KeyError(f"Unknown layout body: {body_name}")
        joint_ids = self._articulated_joint_ids.get(body_name)
        if joint_ids is None or joint_ids.size == 0:
            return
        qpos_ids = self.model.jnt_qposadr[joint_ids].astype(int)
        dof_ids = self.model.jnt_dofadr[joint_ids].astype(int)
        data.qpos[qpos_ids] = self.model.qpos0[qpos_ids]
        data.qvel[dof_ids] = 0.0
        data.qacc[dof_ids] = 0.0
        data.qacc_warmstart[dof_ids] = 0.0

    def set_workspace_offset(self, data: mujoco.MjData, offset: np.ndarray) -> None:
        """Move all tabletop objects with the randomized work surface."""
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,):
            raise ValueError("Workspace offset must contain X, Y and Z")
        self._workspace_offset = self._base_workspace_offset + offset
        self.activate(data, self.current_name)

    def workspace_xy(self, canonical_xy: np.ndarray) -> np.ndarray:
        """Map canonical Panda-table XY coordinates into the compact workspace."""
        canonical_xy = np.asarray(canonical_xy, dtype=float)
        if canonical_xy.shape[-1] != 2:
            raise ValueError("Workspace XY coordinates must end in two values")
        return (
            _CANONICAL_TABLE_CENTER_M[:2]
            + (canonical_xy - _CANONICAL_TABLE_CENTER_M[:2])
            * _WORKSPACE_PLANAR_SCALE
            + self._workspace_offset[:2]
        )

    def workspace_position(self, canonical_position: np.ndarray) -> np.ndarray:
        """Map a canonical tabletop XYZ position into the compact workspace."""
        canonical_position = np.asarray(canonical_position, dtype=float)
        if canonical_position.shape != (3,):
            raise ValueError("Workspace position must contain X, Y and Z")
        result = canonical_position.copy()
        result[:2] = self.workspace_xy(result[:2])
        result[2] += self._workspace_offset[2]
        return result
