"""Task-level rotational tolerance annotation for completed demonstrations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..tasks.catalog import (
    BANANA_TO_PLATE_TASK,
    BANANA_TO_TRAY_TASK,
    BALL_TO_PLATE_TASK,
    BLOCK_TO_PLATE_STABLE_TASK,
    CLICK_BELL_TASK,
    CYLINDER_TO_PLATE_STABLE_TASK,
    LYING_BOX_TO_PLATE_TASK,
    LYING_BOX_TO_REGION_TASK,
    PEAR_TO_BOWL_TASK,
    PEAR_TO_PLATE_TASK,
    PEAR_TO_TRAY_TASK,
    PRESS_BUTTON_TASK,
    ROTATE_KNOB_TASK,
    WIPE_WHITEBOARD_TASK,
    STRAWBERRY_TO_TRAY_TASK,
    PRESS_POWER_STRIP_TASK,
    COLOR_BLOCK_STACK_TASKS,
    STACK_COLORED_BLOCKS_TASK,
)
from ..tasks.planning.pot_lid import active_pot_lid_variant
from .specification import symmetric_rotation_tolerance_bounds
from .frame import box_tolerance_frame
from ..simulation.scenes import (
    ADJUST_CYLINDRICAL_BOTTLE_SCENE,
    ADJUST_RECTANGULAR_BOTTLE_SCENE,
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
)


POT_LID_STAGE_ROTATION_MASKS = {
    # XYZ axes in the stage tolerance frame, ordered (pregrasp, postgrasp).
    "cylindrical": ((0, 1, 1), (0, 0, 1)),
    "handle": ((1, 1, 0), (0, 0, 1)),
}

LYING_CYLINDER_STAGE_ROTATION_MASKS = {
    GEOMETRY_PLATE_CYLINDER_LYING_SCENE: (
        (1, 1, 0), (1, 1, 1),
    ),
    GEOMETRY_REGION_CYLINDER_LYING_SCENE: (
        (1, 1, 0), (1, 0, 0),
    ),
}

UPRIGHT_CYLINDER_SCENES = frozenset((
    GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE,
    GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE,
))
UPRIGHT_CYLINDER_STAGE_ROTATION_MASKS = (
    (0, 1, 1), (0, 0, 1),
)

_STATIC_SCENE_STAGE_ROTATION_MASKS = {
    CYLINDER_TO_NARROW_BOX_SCENE: ((1, 1, 0), (1, 1, 0)),
    BOX_TO_NARROW_BOX_SCENE: ((0, 1, 0), (0, 1, 0)),
    GEOMETRY_PLATE_BOX_UPRIGHT_SCENE: ((0, 1, 0), (1, 1, 1)),
    GEOMETRY_PLATE_BOX_LYING_SCENE: ((0, 1, 0), (1, 1, 1)),
    GEOMETRY_PLATE_CUBE_SCENE: ((0, 1, 0), (1, 1, 1)),
    GEOMETRY_PLATE_BALL_SCENE: ((1, 1, 1), (1, 1, 1)),
    GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE: ((0, 1, 1), (1, 1, 1)),
    GEOMETRY_PLATE_CYLINDER_LYING_SCENE: ((1, 1, 0), (1, 1, 1)),
    GEOMETRY_REGION_BOX_UPRIGHT_SCENE: ((0, 1, 0), (0, 0, 0)),
    GEOMETRY_REGION_BOX_LYING_SCENE: ((0, 1, 0), (0, 0, 0)),
    GEOMETRY_REGION_CUBE_SCENE: ((0, 1, 0), (0, 0, 0)),
    GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE: UPRIGHT_CYLINDER_STAGE_ROTATION_MASKS,
    GEOMETRY_REGION_CYLINDER_LYING_SCENE: ((1, 1, 0), (1, 0, 0)),
    ADJUST_CYLINDRICAL_BOTTLE_SCENE: ((1, 1, 0), (0, 0, 1)),
    ADJUST_RECTANGULAR_BOTTLE_SCENE: ((0, 1, 0), (0, 0, 1)),
}

_STATIC_SCENE_TASK_STAGE_ROTATION_MASKS = {
    (
        GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE,
        CYLINDER_TO_PLATE_STABLE_TASK,
    ): ((0, 1, 1), (0, 0, 1)),
    (
        GEOMETRY_PLATE_CYLINDER_LYING_SCENE,
        CYLINDER_TO_PLATE_STABLE_TASK,
    ): ((1, 1, 0), (1, 0, 1)),
    (
        GEOMETRY_PLATE_BOX_UPRIGHT_SCENE,
        BLOCK_TO_PLATE_STABLE_TASK,
    ): ((0, 1, 0), (0, 0, 1)),
    (
        GEOMETRY_PLATE_BOX_LYING_SCENE,
        BLOCK_TO_PLATE_STABLE_TASK,
    ): ((0, 1, 0), (0, 0, 1)),
}

_STATIC_TASK_STAGE_ROTATION_MASKS = {
    BALL_TO_PLATE_TASK: ((1, 1, 1), (1, 1, 1)),
    CLICK_BELL_TASK: ((1, 1, 1), (1, 1, 1)),
    PRESS_BUTTON_TASK: ((1, 1, 1), (1, 1, 1)),
    PRESS_POWER_STRIP_TASK: ((1, 1, 1), (1, 1, 1)),
    BANANA_TO_PLATE_TASK: ((1, 1, 0), (1, 1, 1)),
    BANANA_TO_TRAY_TASK: ((1, 1, 0), (1, 1, 1)),
    PEAR_TO_PLATE_TASK: ((1, 1, 1), (1, 1, 1)),
    PEAR_TO_TRAY_TASK: ((1, 1, 1), (1, 1, 1)),
    PEAR_TO_BOWL_TASK: ((1, 1, 1), (1, 1, 1)),
    STRAWBERRY_TO_TRAY_TASK: ((1, 1, 1), (1, 1, 1)),
    ROTATE_KNOB_TASK: ((0, 1, 0), (0, 0, 0)),
    WIPE_WHITEBOARD_TASK: ((0, 1, 0), (0, 0, 1)),
    # These two instructions share exactly the same visual scene.  Their
    # different Post-grasp masks are the tolerance-discrimination signal:
    # the plate accepts arbitrary final rotation while the matching white
    # region requires the authored alignment.
    LYING_BOX_TO_PLATE_TASK: ((0, 1, 0), (1, 1, 1)),
    LYING_BOX_TO_REGION_TASK: ((0, 1, 0), (0, 0, 0)),
}


def static_stage_rotation_masks(
    scene: str,
    task: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Return a model-independent mask for a registered scene or task."""
    return _STATIC_SCENE_TASK_STAGE_ROTATION_MASKS.get(
        (str(scene), str(task)),
        _STATIC_SCENE_STAGE_ROTATION_MASKS.get(
            str(scene), _STATIC_TASK_STAGE_ROTATION_MASKS.get(str(task)),
        ),
    )


def lying_cylinder_stage_rotation_masks(
    scene: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return destination-specific Pre/Post masks for a lying cylinder."""
    try:
        return LYING_CYLINDER_STAGE_ROTATION_MASKS[str(scene)]
    except KeyError as error:
        raise ValueError(f"Not a lying-cylinder training scene: {scene!r}") from error


def pot_lid_stage_rotation_masks(
    lid_kind: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the declared Pre/Post XYZ mask for one lid geometry."""
    try:
        return POT_LID_STAGE_ROTATION_MASKS[str(lid_kind)]
    except KeyError as error:
        raise ValueError(f"Unknown pot-lid geometry: {lid_kind!r}") from error


def active_pot_lid_stage_rotation_masks(
    model: Any,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Resolve the tolerance mask after seeded scene activation."""
    kind, _, _ = active_pot_lid_variant(model)
    return pot_lid_stage_rotation_masks(kind)


BOX_ROTATION_TOLERANCE_BOUNDS_RAD = {
    "pregrasp": symmetric_rotation_tolerance_bounds((0.0, math.radians(30.0), 0.0)),
    "grasp": np.zeros(6),
    "postgrasp": symmetric_rotation_tolerance_bounds((0.0, 0.0, math.radians(30.0))),
    "release": np.zeros(6),
}

STACK_ROTATION_TOLERANCE_BOUNDS_RAD = {
    "pregrasp": symmetric_rotation_tolerance_bounds((0.0, math.radians(30.0), 0.0)),
    "grasp": np.zeros(6),
    "postgrasp": np.zeros(6),
    "release": np.zeros(6),
}


def task_rotation_tolerance_profiles(task: str) -> dict[str, np.ndarray]:
    """Return declared controller bounds; these are not dataset labels."""
    profiles = (
        STACK_ROTATION_TOLERANCE_BOUNDS_RAD
        if task in (*COLOR_BLOCK_STACK_TASKS, STACK_COLORED_BLOCKS_TASK)
        else BOX_ROTATION_TOLERANCE_BOUNDS_RAD
    )
    return {name: value.copy() for name, value in profiles.items()}
