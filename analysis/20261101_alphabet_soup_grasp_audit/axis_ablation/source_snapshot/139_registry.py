"""Scene/task registry shared by collectors and benchmark runners."""

from __future__ import annotations

from collections.abc import Callable

import mujoco

from .catalog import (
    ADJUST_BOTTLE_TASK,
    APPLE_TO_BOWL_TASK,
    APPLE_TO_PLATE_TASK,
    BALL_TO_PLATE_TASK,
    BALL_PLATE_THEN_BLOCK_PAPER_TASK,
    APPLE_THEN_BANANA_TO_TRAY_TASK,
    BLOCK_TO_PLATE_TASK,
    BLOCK_TO_PLATE_STABLE_TASK,
    BOX_TO_NARROW_BOX_TASK,
    CUBE_TO_PLATE_TASK,
    CYLINDER_TO_PLATE_TASK,
    CYLINDER_TO_PLATE_STABLE_TASK,
    CYLINDER_TO_NARROW_BOX_TASK,
    BANANA_TO_TRAY_TASK,
    FRUIT_TRAY_SEQUENCE_TASK_SPECS,
    CLICK_BELL_TASK,
    INSERT_OBJECT_TASK,
    LYING_BOX_TO_PLATE_TASK,
    LYING_BOX_TO_REGION_TASK,
    LEMON_TO_BOWL_TASK,
    LEMON_TO_PLATE_TASK,
    STRAWBERRY_TO_TRAY_TASK,
    PLACE_OBJECT_ON_PAPER_TASK,
    PRESS_BUTTON_TASK,
    PRESS_POWER_STRIP_TASK,
    ROTATE_KNOB_TASK,
    WIPE_WHITEBOARD_TASK,
    SPORTS_BALL_TO_PLATE_TASK,
    COLOR_BLOCK_STACK_TASK_SPECS,
    CLOSE_CYLINDRICAL_POT_LID_TASK,
    CLOSE_HANDLE_POT_LID_TASK,
    OPEN_CYLINDRICAL_POT_LID_TASK,
    OPEN_HANDLE_POT_LID_TASK,
    ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK,
    UPRIGHT_BOX_TO_PLATE_TASK,
    UPRIGHT_BOX_TO_REGION_TASK,
)
from .evaluation.adjust_bottle import AdjustBottleEvaluator
from .evaluation.cylinder_narrow_box import (
    BoxNarrowBoxEvaluator,
    CylinderNarrowBoxEvaluator,
)
from .evaluation.ball_paper_sequence import BallPaperSequenceEvaluator
from .evaluation.insert_object import InsertObjectEvaluator
from .evaluation.fruit_receptacles import (
    FruitReceptacleEvaluator,
    FruitSequenceEvaluator,
)
from .evaluation.pick_place import (
    BallToPlateEvaluator,
    BlockToPlateEvaluator,
    CubeToPlateEvaluator,
    GeometryToPlateEvaluator,
)
from ..simulation.scenes import (
    ADJUST_CYLINDRICAL_BOTTLE_SCENE,
    ADJUST_RECTANGULAR_BOTTLE_SCENE,
    BANANA_APPLE_TRAY_SCENE,
    BOX_TO_NARROW_BOX_SCENE,
    CYLINDER_TO_NARROW_BOX_SCENE,
    GEOMETRY_PLATE_BALL_SCENE,
    GEOMETRY_PLATE_BOX_LYING_SCENE,
    GEOMETRY_PLATE_BOX_UPRIGHT_SCENE,
    GEOMETRY_PLATE_CUBE_SCENE,
    GEOMETRY_PLATE_CYLINDER_LYING_SCENE,
    GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE,
    GEOMETRY_REGION_BOX_LYING_SCENE,
    GEOMETRY_REGION_BOX_UPRIGHT_SCENE,
    GEOMETRY_REGION_CUBE_SCENE,
    GEOMETRY_REGION_CYLINDER_LYING_SCENE,
    GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE,
    TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
    OPEN_CYLINDRICAL_POT_LID_SCENE,
    OPEN_HANDLE_POT_LID_SCENE,
    CLOSE_CYLINDRICAL_POT_LID_SCENE,
    CLOSE_HANDLE_POT_LID_SCENE,
    ROUND_HANDLE_POT_PEAR_SCENE,
    UNSEEN_BOX_PLATE_SCENE,
    UNSEEN_BOX_REGION_SCENE,
    UNSEEN_BUTTON_SCENE,
    UNSEEN_FRUIT_RECEPTACLES_SCENE,
    UNSEEN_POWER_STRIP_SCENE,
    UNSEEN_SPORTS_BALL_PLATE_SCENE,
    WHITEBOARD_SCENE,
)
from .evaluation.paper_placement import PaperPlacementEvaluator
from .evaluation.rotate_knob import RotateKnobEvaluator
from .evaluation.wipe_whiteboard import WipeWhiteboardEvaluator
from .evaluation.color_block_stack import ColorBlockStackEvaluator
from .evaluation.press_device import PressDeviceEvaluator
from .evaluation.pot_lid import PotLidEvaluator
from .evaluation.pot_fruit_sequence import RoundHandlePotPearSequenceEvaluator
from .evaluation.types import TaskEvaluator
from .evaluation.unseen_objects import (
    UnseenUprightBoxRegionEvaluator,
    unseen_fruit_evaluator,
    unseen_sports_ball_evaluator,
    unseen_upright_box_plate_evaluator,
)


_EVALUATOR_FACTORIES: dict[
    tuple[str, str],
    Callable[[mujoco.MjModel, mujoco.MjData | None], TaskEvaluator],
] = {
    (WHITEBOARD_SCENE, WIPE_WHITEBOARD_TASK): WipeWhiteboardEvaluator,
    ("Blocks", BLOCK_TO_PLATE_TASK): BlockToPlateEvaluator,
    ("Blocks", CUBE_TO_PLATE_TASK): CubeToPlateEvaluator,
    ("Blocks", BALL_TO_PLATE_TASK): BallToPlateEvaluator,
    (
        "Ball + Paper Placement",
        BALL_PLATE_THEN_BLOCK_PAPER_TASK,
    ): BallPaperSequenceEvaluator,
    ("Adjust Bottle", ADJUST_BOTTLE_TASK): AdjustBottleEvaluator,
    (ADJUST_CYLINDRICAL_BOTTLE_SCENE, ADJUST_BOTTLE_TASK): AdjustBottleEvaluator,
    (ADJUST_RECTANGULAR_BOTTLE_SCENE, ADJUST_BOTTLE_TASK): AdjustBottleEvaluator,
    ("Click Bell", CLICK_BELL_TASK): lambda model, data: PressDeviceEvaluator(
        model, data, root_prefix="click_bell_",
    ),
    ("Press Button", PRESS_BUTTON_TASK): lambda model, data: PressDeviceEvaluator(
        model, data, root_prefix="press_button_",
    ),
    (
        "Press Power Strip",
        PRESS_POWER_STRIP_TASK,
    ): lambda model, data: PressDeviceEvaluator(
        model, data, root_prefix="power_strip_",
    ),
    (
        "Insert Object",
        INSERT_OBJECT_TASK,
    ): InsertObjectEvaluator,
    (
        CYLINDER_TO_NARROW_BOX_SCENE,
        CYLINDER_TO_NARROW_BOX_TASK,
    ): CylinderNarrowBoxEvaluator,
    (
        BOX_TO_NARROW_BOX_SCENE,
        BOX_TO_NARROW_BOX_TASK,
    ): BoxNarrowBoxEvaluator,
    (UNSEEN_FRUIT_RECEPTACLES_SCENE, APPLE_TO_PLATE_TASK): (
        lambda model, data: unseen_fruit_evaluator(
            model, data, fruit="apple", receptacle="plate",
        )
    ),
    (UNSEEN_FRUIT_RECEPTACLES_SCENE, APPLE_TO_BOWL_TASK): (
        lambda model, data: unseen_fruit_evaluator(
            model, data, fruit="apple", receptacle="bowl",
        )
    ),
    (UNSEEN_FRUIT_RECEPTACLES_SCENE, LEMON_TO_PLATE_TASK): (
        lambda model, data: unseen_fruit_evaluator(
            model, data, fruit="lemon", receptacle="plate",
        )
    ),
    (UNSEEN_FRUIT_RECEPTACLES_SCENE, LEMON_TO_BOWL_TASK): (
        lambda model, data: unseen_fruit_evaluator(
            model, data, fruit="lemon", receptacle="bowl",
        )
    ),
    (UNSEEN_SPORTS_BALL_PLATE_SCENE, SPORTS_BALL_TO_PLATE_TASK): (
        unseen_sports_ball_evaluator
    ),
    (UNSEEN_BOX_PLATE_SCENE, UPRIGHT_BOX_TO_PLATE_TASK): (
        unseen_upright_box_plate_evaluator
    ),
    (UNSEEN_BOX_REGION_SCENE, UPRIGHT_BOX_TO_REGION_TASK): (
        UnseenUprightBoxRegionEvaluator
    ),
    (UNSEEN_BUTTON_SCENE, PRESS_BUTTON_TASK): (
        lambda model, data: PressDeviceEvaluator(
            model, data, root_prefix="press_button_",
        )
    ),
    (UNSEEN_POWER_STRIP_SCENE, PRESS_POWER_STRIP_TASK): (
        lambda model, data: PressDeviceEvaluator(
            model, data, root_prefix="power_strip_",
        )
    ),
    (
        "Fruit Receptacles",
        BANANA_TO_TRAY_TASK,
    ): lambda model, data: FruitReceptacleEvaluator(
        model, data, fruit="banana", receptacle="tray",
    ),
    (
        "Fruit Receptacles",
        STRAWBERRY_TO_TRAY_TASK,
    ): lambda model, data: FruitReceptacleEvaluator(
        model, data, fruit="strawberry", receptacle="tray",
    ),
    (
        "Paper Placement",
        PLACE_OBJECT_ON_PAPER_TASK,
    ): PaperPlacementEvaluator,
    (
        TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
        LYING_BOX_TO_PLATE_TASK,
    ): GeometryToPlateEvaluator,
    (
        TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
        LYING_BOX_TO_REGION_TASK,
    ): PaperPlacementEvaluator,
    (
        "Rotate Knob",
        ROTATE_KNOB_TASK,
    ): RotateKnobEvaluator,
    (
        "Open Pot Lid",
        OPEN_CYLINDRICAL_POT_LID_TASK,
    ): lambda model, data: PotLidEvaluator(
        model, data, close_pot=False, expected_kind="cylindrical",
    ),
    (
        "Open Pot Lid",
        OPEN_HANDLE_POT_LID_TASK,
    ): lambda model, data: PotLidEvaluator(
        model, data, close_pot=False, expected_kind="handle",
    ),
    (
        "Close Pot Lid",
        CLOSE_CYLINDRICAL_POT_LID_TASK,
    ): lambda model, data: PotLidEvaluator(
        model, data, close_pot=True, expected_kind="cylindrical",
    ),
    (
        "Close Pot Lid",
        CLOSE_HANDLE_POT_LID_TASK,
    ): lambda model, data: PotLidEvaluator(
        model, data, close_pot=True, expected_kind="handle",
    ),
    (
        ROUND_HANDLE_POT_PEAR_SCENE,
        ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK,
    ): RoundHandlePotPearSequenceEvaluator,
}

for _task_name in FRUIT_TRAY_SEQUENCE_TASK_SPECS:
    _fruit_scene = (
        BANANA_APPLE_TRAY_SCENE
        if _task_name == APPLE_THEN_BANANA_TO_TRAY_TASK
        else "Fruit Receptacles"
    )
    _EVALUATOR_FACTORIES[(_fruit_scene, _task_name)] = (
        lambda model, data, task_name=_task_name: FruitSequenceEvaluator(
            model, data, task_name,
        )
    )

for _task_name in COLOR_BLOCK_STACK_TASK_SPECS:
    _EVALUATOR_FACTORIES[("Color Block Stack", _task_name)] = (
        lambda model, data, task_name=_task_name: ColorBlockStackEvaluator(
            model, data, task_name,
        )
    )

for _scene_name, _task_name in (
    (GEOMETRY_PLATE_BOX_UPRIGHT_SCENE, BLOCK_TO_PLATE_TASK),
    (GEOMETRY_PLATE_BOX_LYING_SCENE, BLOCK_TO_PLATE_TASK),
    (GEOMETRY_PLATE_CUBE_SCENE, CUBE_TO_PLATE_TASK),
    (GEOMETRY_PLATE_BALL_SCENE, BALL_TO_PLATE_TASK),
    (GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE, CYLINDER_TO_PLATE_TASK),
    (GEOMETRY_PLATE_CYLINDER_LYING_SCENE, CYLINDER_TO_PLATE_TASK),
):
    _EVALUATOR_FACTORIES[(_scene_name, _task_name)] = GeometryToPlateEvaluator

for _scene_name, _task_name in (
    (GEOMETRY_PLATE_BOX_UPRIGHT_SCENE, BLOCK_TO_PLATE_STABLE_TASK),
    (GEOMETRY_PLATE_BOX_LYING_SCENE, BLOCK_TO_PLATE_STABLE_TASK),
    (
        GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE,
        CYLINDER_TO_PLATE_STABLE_TASK,
    ),
    (
        GEOMETRY_PLATE_CYLINDER_LYING_SCENE,
        CYLINDER_TO_PLATE_STABLE_TASK,
    ),
):
    _EVALUATOR_FACTORIES[(_scene_name, _task_name)] = (
        lambda model, data: GeometryToPlateEvaluator(
            model, data, require_target_upright=True,
        )
    )

for _scene_name in (
    GEOMETRY_REGION_BOX_UPRIGHT_SCENE,
    GEOMETRY_REGION_BOX_LYING_SCENE,
    GEOMETRY_REGION_CUBE_SCENE,
    GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE,
    GEOMETRY_REGION_CYLINDER_LYING_SCENE,
):
    _EVALUATOR_FACTORIES[
        (_scene_name, PLACE_OBJECT_ON_PAPER_TASK)
    ] = PaperPlacementEvaluator

_EVALUATOR_FACTORIES.update({
    (
        OPEN_CYLINDRICAL_POT_LID_SCENE,
        OPEN_CYLINDRICAL_POT_LID_TASK,
    ): lambda model, data: PotLidEvaluator(
        model, data, close_pot=False, expected_kind="cylindrical",
    ),
    (
        OPEN_HANDLE_POT_LID_SCENE,
        OPEN_HANDLE_POT_LID_TASK,
    ): lambda model, data: PotLidEvaluator(
        model, data, close_pot=False, expected_kind="handle",
    ),
    (
        CLOSE_CYLINDRICAL_POT_LID_SCENE,
        CLOSE_CYLINDRICAL_POT_LID_TASK,
    ): lambda model, data: PotLidEvaluator(
        model, data, close_pot=True, expected_kind="cylindrical",
    ),
    (
        CLOSE_HANDLE_POT_LID_SCENE,
        CLOSE_HANDLE_POT_LID_TASK,
    ): lambda model, data: PotLidEvaluator(
        model, data, close_pot=True, expected_kind="handle",
    ),
})


def create_task_evaluator(
    scene_name: str,
    task_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    *,
    robot_table_clearance_m: float = 0.0,
) -> TaskEvaluator:
    """Construct the authoritative evaluator for one registered task."""
    try:
        factory = _EVALUATOR_FACTORIES[(scene_name, task_name)]
    except KeyError as error:
        raise ValueError(
            f"No evaluator registered for scene={scene_name!r}, task={task_name!r}"
        ) from error
    evaluator = factory(model, data)
    from .evaluation.predicates.contact import (
        configure_robot_table_clearance_threshold,
    )
    configure_robot_table_clearance_threshold(
        evaluator, robot_table_clearance_m,
    )
    return evaluator


def registered_tasks() -> tuple[tuple[str, str], ...]:
    return tuple(_EVALUATOR_FACTORIES)
