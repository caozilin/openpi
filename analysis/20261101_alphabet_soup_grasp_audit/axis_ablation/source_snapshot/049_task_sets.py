"""Curated training and unseen-object registrations for dataset collection.

This module is intentionally declarative.  A registered scene points at an
existing runtime scene and a registered task points at an existing planner;
the optional variant contract records which seeded object choices belong to
the curated dataset without duplicating trajectories or MuJoCo assets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

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
    GEOMETRY_TRAINING_SCENES,
    TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
    ROUND_HANDLE_POT_PEAR_SCENE,
    WHITEBOARD_SCENE,
    UNSEEN_BOX_PLATE_SCENE,
    UNSEEN_BOX_REGION_SCENE,
    UNSEEN_BUTTON_SCENE,
    UNSEEN_FRUIT_RECEPTACLES_SCENE,
    UNSEEN_POWER_STRIP_SCENE,
    UNSEEN_SPORTS_BALL_PLATE_SCENE,
    CLOSE_CYLINDRICAL_POT_LID_SCENE,
    CLOSE_HANDLE_POT_LID_SCENE,
    SceneManager,
)
from ..tasks.catalog import (
    ADJUST_BOTTLE_TASK,
    APPLE_TO_BOWL_TASK,
    APPLE_TO_PLATE_TASK,
    BALL_TO_PLATE_TASK,
    APPLE_THEN_BANANA_TO_TRAY_TASK,
    BLOCK_TO_PLATE_TASK,
    BLOCK_TO_PLATE_STABLE_TASK,
    BOX_TO_NARROW_BOX_TASK,
    CLOSE_CYLINDRICAL_POT_LID_TASK,
    CLOSE_HANDLE_POT_LID_TASK,
    CYLINDER_TO_PLATE_TASK,
    CYLINDER_TO_PLATE_STABLE_TASK,
    CYLINDER_TO_NARROW_BOX_TASK,
    LEMON_TO_BOWL_TASK,
    LEMON_TO_PLATE_TASK,
    LYING_BOX_TO_PLATE_TASK,
    LYING_BOX_TO_REGION_TASK,
    PLACE_OBJECT_ON_PAPER_TASK,
    PRESS_BUTTON_TASK,
    PRESS_POWER_STRIP_TASK,
    ROTATE_KNOB_TASK,
    WIPE_WHITEBOARD_TASK,
    COLOR_BLOCK_STACK_TASK_SPECS,
    ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK,
    SPORTS_BALL_TO_PLATE_TASK,
    UPRIGHT_BOX_TO_PLATE_TASK,
    UPRIGHT_BOX_TO_REGION_TASK,
)


EXISTING_SCENE_GROUP = "existing_scenes"
TRAINING_TASK_GROUP = "training_tasks"
GEOMETRIC_BASIS_SUBCATEGORY = "Geometric Basis Tasks"
CONTEXTUAL_SUBCATEGORY = "Contextual Tasks"
MULTI_SUBTASK_SUBCATEGORY = "Multi-Subtask Tasks"
TRAINING_SUBCATEGORIES = (
    GEOMETRIC_BASIS_SUBCATEGORY,
    CONTEXTUAL_SUBCATEGORY,
    MULTI_SUBTASK_SUBCATEGORY,
)
TOLERANCE_DISCRIMINATION_GROUP = "容差判别"
UNSEEN_OBJECT_GROUP = "unseen_objects"
_CURATED_SCENE_GROUPS = (
    TRAINING_TASK_GROUP,
    TOLERANCE_DISCRIMINATION_GROUP,
    UNSEEN_OBJECT_GROUP,
)


@dataclass(frozen=True)
class VariantContract:
    """Seeded choices permitted for a curated task."""

    object_names: tuple[str, ...] = ()
    sizes: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    orientation: str | None = None


@dataclass(frozen=True)
class DatasetTask:
    """One dataset task backed by an already implemented runtime task."""

    task_id: str
    runtime_task: str
    weight: int
    instructions: tuple[str, ...]
    variants: VariantContract = VariantContract()
    variant_seed_key: str | None = None

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Task weight must be positive: {self.task_id}")
        if not self.instructions or any(not value.strip() for value in self.instructions):
            raise ValueError(f"Task requires non-empty instructions: {self.task_id}")


@dataclass(frozen=True)
class DatasetScene:
    """A logical dataset scene composed from one existing runtime scene."""

    scene_id: str
    group: str
    runtime_scene: str
    tasks: tuple[DatasetTask, ...] = ()
    object_names: tuple[str, ...] = ()
    subcategory: str | None = None


_SIZES = ("small", "medium", "large")
_COLORS = ("red", "yellow", "purple", "green", "blue", "orange")


def _geometry_task(
    task_id: str,
    runtime_task: str,
    instruction: str,
    *,
    orientation: str | None = None,
    variant_seed_key: str | None = None,
) -> DatasetTask:
    return DatasetTask(
        task_id,
        runtime_task,
        1,
        (instruction,),
        VariantContract(sizes=_SIZES, colors=_COLORS, orientation=orientation),
        variant_seed_key,
    )


TRAINING_DATASET_SCENES = (
    DatasetScene(
        "geometry_plate_ball", TRAINING_TASK_GROUP,
        GEOMETRY_PLATE_BALL_SCENE,
        (_geometry_task(
            "geometry_plate_ball", BALL_TO_PLATE_TASK,
            "Place the ball on the tray",
        ),),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "geometry_plate_cylinder_upright", TRAINING_TASK_GROUP,
        GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE,
        (
            _geometry_task(
                "geometry_plate_cylinder_upright", CYLINDER_TO_PLATE_TASK,
                "Place the cylinder on the tray", orientation="upright",
            ),
            _geometry_task(
                "geometry_plate_cylinder_upright_stable",
                CYLINDER_TO_PLATE_STABLE_TASK,
                "Place the cylinder on the tray without tilting it",
                orientation="upright",
            ),
        ),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "geometry_region_cylinder_upright", TRAINING_TASK_GROUP,
        GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE,
        (_geometry_task(
            "geometry_region_cylinder_upright", PLACE_OBJECT_ON_PAPER_TASK,
            "Place the cylinder in the white target area",
            orientation="upright",
        ),),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "geometry_plate_cylinder_lying", TRAINING_TASK_GROUP,
        GEOMETRY_PLATE_CYLINDER_LYING_SCENE,
        (
            _geometry_task(
                "geometry_plate_cylinder_lying", CYLINDER_TO_PLATE_TASK,
                "Place the cylinder on the tray", orientation="lying",
            ),
            _geometry_task(
                "geometry_plate_cylinder_lying_stable",
                CYLINDER_TO_PLATE_STABLE_TASK,
                "Place the cylinder on the tray without tilting it",
                orientation="lying",
            ),
        ),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "geometry_region_cylinder_lying", TRAINING_TASK_GROUP,
        GEOMETRY_REGION_CYLINDER_LYING_SCENE,
        (_geometry_task(
            "geometry_region_cylinder_lying", PLACE_OBJECT_ON_PAPER_TASK,
            "Place the cylinder in the white target area",
            orientation="lying",
        ),),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "cylinder_to_narrow_box", TRAINING_TASK_GROUP,
        CYLINDER_TO_NARROW_BOX_SCENE,
        (_geometry_task(
            "cylinder_to_narrow_box", CYLINDER_TO_NARROW_BOX_TASK,
            "Place the cylinder in the box", orientation="lying",
        ),),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "geometry_plate_box_upright", TRAINING_TASK_GROUP,
        GEOMETRY_PLATE_BOX_UPRIGHT_SCENE,
        (
            _geometry_task(
                "geometry_plate_box_upright", BLOCK_TO_PLATE_TASK,
                "Place the rectangular block on the tray",
                orientation="upright",
            ),
            _geometry_task(
                "geometry_plate_box_upright_stable",
                BLOCK_TO_PLATE_STABLE_TASK,
                "Place the rectangular block on the tray without tilting it",
                orientation="upright",
            ),
        ),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "geometry_region_box_upright", TRAINING_TASK_GROUP,
        GEOMETRY_REGION_BOX_UPRIGHT_SCENE,
        (_geometry_task(
            "geometry_region_box_upright", PLACE_OBJECT_ON_PAPER_TASK,
            "Place the rectangular block in the white target area",
            orientation="upright",
        ),),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "geometry_plate_box_lying", TRAINING_TASK_GROUP,
        GEOMETRY_PLATE_BOX_LYING_SCENE,
        (
            _geometry_task(
                "geometry_plate_box_lying", BLOCK_TO_PLATE_TASK,
                "Place the rectangular block on the tray",
                orientation="lying",
            ),
            _geometry_task(
                "geometry_plate_box_lying_stable",
                BLOCK_TO_PLATE_STABLE_TASK,
                "Place the rectangular block on the tray without tilting it",
                orientation="lying",
            ),
        ),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "geometry_region_box_lying", TRAINING_TASK_GROUP,
        GEOMETRY_REGION_BOX_LYING_SCENE,
        (_geometry_task(
            "geometry_region_box_lying", PLACE_OBJECT_ON_PAPER_TASK,
            "Place the rectangular block in the white target area",
            orientation="lying",
        ),),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "box_to_narrow_box", TRAINING_TASK_GROUP,
        BOX_TO_NARROW_BOX_SCENE,
        (_geometry_task(
            "box_to_narrow_box", BOX_TO_NARROW_BOX_TASK,
            "Place the rectangular block in the box", orientation="lying",
        ),),
        subcategory=GEOMETRIC_BASIS_SUBCATEGORY,
    ),
    DatasetScene(
        "adjust_cylindrical_bottle", TRAINING_TASK_GROUP,
        ADJUST_CYLINDRICAL_BOTTLE_SCENE,
        (DatasetTask(
            "adjust_cylindrical_bottle", ADJUST_BOTTLE_TASK, 4,
            ("Stand the bottle upright",),
            VariantContract(object_names=(
                "adjust_bottle_base0", "adjust_bottle_base1",
            )),
        ),),
        subcategory=CONTEXTUAL_SUBCATEGORY,
    ),
    DatasetScene(
        "adjust_rectangular_bottle", TRAINING_TASK_GROUP,
        ADJUST_RECTANGULAR_BOTTLE_SCENE,
        (DatasetTask(
            "adjust_rectangular_bottle", ADJUST_BOTTLE_TASK, 2,
            ("Stand the bottle upright",),
            VariantContract(object_names=("adjust_bottle_ganten",)),
        ),),
        subcategory=CONTEXTUAL_SUBCATEGORY,
    ),
    DatasetScene(
        "button", TRAINING_TASK_GROUP, "Press Button",
        (DatasetTask(
            "press_button", PRESS_BUTTON_TASK, 2, ("Press the button",),
            VariantContract(object_names=(
                "press_button_yellow_red",
                "press_button_white_red",
                "press_button_white_green",
            )),
        ),),
        subcategory=CONTEXTUAL_SUBCATEGORY,
    ),
    DatasetScene(
        "whiteboard", TRAINING_TASK_GROUP, WHITEBOARD_SCENE,
        (DatasetTask("wipe_whiteboard", WIPE_WHITEBOARD_TASK, 1,
                     ("Wipe the whiteboard",)),),
        subcategory=CONTEXTUAL_SUBCATEGORY,
    ),
    DatasetScene(
        "rotate_knob", TRAINING_TASK_GROUP, "Rotate Knob",
        (DatasetTask(
            "rotate_knob", ROTATE_KNOB_TASK, 3,
            ("Rotate the knob 90 degrees clockwise",),
            VariantContract(
                sizes=_SIZES, colors=("red", "green", "blue"),
            ),
        ),),
        subcategory=CONTEXTUAL_SUBCATEGORY,
    ),
    DatasetScene(
        "close_cylindrical_pot_lid", TRAINING_TASK_GROUP,
        CLOSE_CYLINDRICAL_POT_LID_SCENE,
        (DatasetTask(
            "close_cylindrical_pot_lid", CLOSE_CYLINDRICAL_POT_LID_TASK, 1,
            ("Replace the lid on the pot",),
        ),),
        subcategory=CONTEXTUAL_SUBCATEGORY,
    ),
    DatasetScene(
        "close_handle_pot_lid", TRAINING_TASK_GROUP,
        CLOSE_HANDLE_POT_LID_SCENE,
        (DatasetTask(
            "close_handle_pot_lid", CLOSE_HANDLE_POT_LID_TASK, 1,
            ("Replace the lid on the pot",),
        ),),
        subcategory=CONTEXTUAL_SUBCATEGORY,
    ),
    DatasetScene(
        "banana_apple_to_tray", TRAINING_TASK_GROUP,
        BANANA_APPLE_TRAY_SCENE,
        (DatasetTask(
            "apple_then_banana_to_tray", APPLE_THEN_BANANA_TO_TRAY_TASK, 1,
            (
                "Place the apple on the tray, then place the banana on the tray",
            ),
        ),),
        subcategory=MULTI_SUBTASK_SUBCATEGORY,
    ),
    DatasetScene(
        "color_block_stack", TRAINING_TASK_GROUP, "Color Block Stack",
        tuple(
            DatasetTask(
                f"stack_{'_'.join(order)}", task, 1,
                (
                    "Place the red block squarely on the yellow block, then "
                    "place the purple block squarely on the red block",
                ),
                VariantContract(sizes=("3", "4", "5"), colors=order),
            )
            for task, order in COLOR_BLOCK_STACK_TASK_SPECS.items()
        ),
        subcategory=MULTI_SUBTASK_SUBCATEGORY,
    ),
    DatasetScene(
        "round_handle_pot_pear_sequence", TRAINING_TASK_GROUP,
        ROUND_HANDLE_POT_PEAR_SCENE,
        (DatasetTask(
            "round_handle_pot_pear_sequence",
            ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK,
            1,
            (
                "Remove the lid from the pot and place it on the table, then "
                "place the pear in the pot, and finally replace the lid on "
                "the pot",
            ),
        ),),
        subcategory=MULTI_SUBTASK_SUBCATEGORY,
    ),
)


UNSEEN_DATASET_SCENES = (
    DatasetScene(
        "unseen_fruit_receptacles",
        UNSEEN_OBJECT_GROUP,
        UNSEEN_FRUIT_RECEPTACLES_SCENE,
        (
            DatasetTask("unseen_apple_to_plate", APPLE_TO_PLATE_TASK, 1,
                        ("Place the apple on the plate",)),
            DatasetTask("unseen_apple_to_bowl", APPLE_TO_BOWL_TASK, 1,
                        ("Place the apple in the bowl",)),
            DatasetTask("unseen_lemon_to_plate", LEMON_TO_PLATE_TASK, 1,
                        ("Place the lemon on the plate",)),
            DatasetTask("unseen_lemon_to_bowl", LEMON_TO_BOWL_TASK, 1,
                        ("Place the lemon in the bowl",)),
        ),
        ("fruit_apple", "fruit_lemon", "ycb_plate_white", "ycb_bowl_white"),
    ),
    DatasetScene(
        "unseen_sports_ball_to_plate",
        UNSEEN_OBJECT_GROUP,
        UNSEEN_SPORTS_BALL_PLATE_SCENE,
        (DatasetTask(
            "unseen_sports_ball_to_plate",
            SPORTS_BALL_TO_PLATE_TASK,
            1,
            ("Place the sports ball on the plate",),
            VariantContract(object_names=(
                "ycb_baseball", "ycb_tennis_ball", "ycb_golf_ball",
            )),
        ),),
    ),
    DatasetScene(
        "unseen_upright_box_to_plate",
        UNSEEN_OBJECT_GROUP,
        UNSEEN_BOX_PLATE_SCENE,
        (DatasetTask(
            "unseen_upright_box_to_plate", UPRIGHT_BOX_TO_PLATE_TASK, 1,
            ("Place the upright box on the plate",),
            VariantContract(object_names=("ycb_sugar_box",), orientation="upright"),
        ),),
    ),
    DatasetScene(
        "unseen_upright_box_to_region",
        UNSEEN_OBJECT_GROUP,
        UNSEEN_BOX_REGION_SCENE,
        (DatasetTask(
            "unseen_upright_box_to_region", UPRIGHT_BOX_TO_REGION_TASK, 1,
            ("Align the upright box with the white region",),
            VariantContract(object_names=("ycb_sugar_box",), orientation="upright"),
        ),),
    ),
    DatasetScene(
        "unseen_press_button",
        UNSEEN_OBJECT_GROUP,
        UNSEEN_BUTTON_SCENE,
        (DatasetTask(
            "unseen_press_button", PRESS_BUTTON_TASK, 1,
            ("Press the button",),
            VariantContract(object_names=(
                "press_button_yellow_red",
                "press_button_white_red",
                "press_button_white_green",
            )),
        ),),
    ),
    DatasetScene(
        "unseen_press_four_slot_power_strip",
        UNSEEN_OBJECT_GROUP,
        UNSEEN_POWER_STRIP_SCENE,
        (DatasetTask(
            "unseen_press_four_slot_power_strip", PRESS_POWER_STRIP_TASK, 1,
            ("Press the switch on the four-slot power strip",),
            VariantContract(object_names=("power_strip_5225",)),
        ),),
    ),
)


TOLERANCE_DISCRIMINATION_DATASET_SCENES = (
    DatasetScene(
        "lying_box_plate_or_region",
        TOLERANCE_DISCRIMINATION_GROUP,
        TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
        (
            _geometry_task(
                "lying_box_to_plate",
                LYING_BOX_TO_PLATE_TASK,
                "Place the lying rectangular block on the plate",
                orientation="lying",
                variant_seed_key="lying_box_plate_or_region",
            ),
            _geometry_task(
                "lying_box_to_white_region",
                LYING_BOX_TO_REGION_TASK,
                "Align the lying rectangular block with the white region",
                orientation="lying",
                variant_seed_key="lying_box_plate_or_region",
            ),
        ),
    ),
)


DATASET_SCENES = (
    *TRAINING_DATASET_SCENES,
    *TOLERANCE_DISCRIMINATION_DATASET_SCENES,
    *UNSEEN_DATASET_SCENES,
)


def scene_groups() -> tuple[str, ...]:
    return (EXISTING_SCENE_GROUP, *_CURATED_SCENE_GROUPS)


def runtime_scenes_for_group(group: str) -> tuple[str, ...]:
    """Return the second-level UI scene list for a scene collection."""
    if group == EXISTING_SCENE_GROUP:
        return tuple(
            name for name in SceneManager.UI_NAMES
            if name not in (
                *GEOMETRY_TRAINING_SCENES,
                TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
            )
        )
    if group not in _CURATED_SCENE_GROUPS:
        raise ValueError(f"Unknown scene group: {group!r}")
    result: list[str] = []
    for registration in DATASET_SCENES:
        if registration.group == group and registration.runtime_scene not in result:
            result.append(registration.runtime_scene)
    return tuple(result)


def scene_choices_for_group(group: str) -> tuple[str, ...]:
    """Return user-facing second-level entries, including logical scenes."""
    if group == EXISTING_SCENE_GROUP:
        return runtime_scenes_for_group(group)
    if group not in _CURATED_SCENE_GROUPS:
        raise ValueError(f"Unknown scene group: {group!r}")
    return tuple(scene.scene_id for scene in DATASET_SCENES if scene.group == group)


def scene_registration(group: str, choice: str) -> DatasetScene | None:
    if group == EXISTING_SCENE_GROUP:
        return None
    for scene in DATASET_SCENES:
        if scene.group == group and scene.scene_id == choice:
            return scene
    raise ValueError(f"Unknown scene choice {choice!r} in group {group!r}")


def runtime_scene_from_choice(group: str, choice: str) -> str:
    registration = scene_registration(group, choice)
    return choice if registration is None else registration.runtime_scene


def tasks_for_scene_choice(group: str, choice: str) -> tuple[DatasetTask, ...]:
    registration = scene_registration(group, choice)
    return () if registration is None else registration.tasks


def runtime_task_from_choice(group: str, choice: str, task_id: str) -> str:
    for task in tasks_for_scene_choice(group, choice):
        if task.task_id == task_id:
            return task.runtime_task
    raise ValueError(
        f"Unknown task ID {task_id!r} for scene choice {choice!r}"
    )


def training_tasks() -> tuple[tuple[DatasetScene, DatasetTask], ...]:
    return tuple(
        (scene, task)
        for scene in TRAINING_DATASET_SCENES
        for task in scene.tasks
    )


def short_training_tasks() -> tuple[tuple[DatasetScene, DatasetTask], ...]:
    """Return training tasks with one manipulation cycle."""
    return tuple(
        (scene, task) for scene, task in training_tasks()
        if scene.subcategory != MULTI_SUBTASK_SUBCATEGORY
    )


def training_tasks_for_subcategory(
    subcategory: str,
) -> tuple[tuple[DatasetScene, DatasetTask], ...]:
    """Return the ordered training subset for one public task taxonomy."""
    if subcategory not in TRAINING_SUBCATEGORIES:
        raise ValueError(f"Unknown training subcategory: {subcategory!r}")
    return tuple(
        (scene, task)
        for scene, task in training_tasks()
        if scene.subcategory == subcategory
    )


def unseen_object_tasks() -> tuple[tuple[DatasetScene, DatasetTask], ...]:
    """Return benchmark-only tasks that intentionally have no demonstrations."""
    return tuple(
        (scene, task)
        for scene in UNSEEN_DATASET_SCENES
        for task in scene.tasks
    )


def tolerance_discrimination_tasks(
) -> tuple[tuple[DatasetScene, DatasetTask], ...]:
    """Return paired tasks that share a scene but require different tolerances."""
    return tuple(
        (scene, task)
        for scene in TOLERANCE_DISCRIMINATION_DATASET_SCENES
        for task in scene.tasks
    )


def resolve_training_task(
    scene_key: str,
    task_key: str,
) -> tuple[DatasetScene, DatasetTask]:
    """Resolve canonical IDs strictly through the curated training registry."""
    scene_text = str(scene_key).strip()
    task_text = str(task_key).strip()
    scene_matches = [
        scene for scene in TRAINING_DATASET_SCENES
        if scene_text == scene.scene_id
    ]
    if not scene_matches:
        available = ", ".join(scene.scene_id for scene in TRAINING_DATASET_SCENES)
        raise ValueError(
            f"Unknown training scene ID {scene_key!r}; "
            f"use one of: {available}"
        )
    scene = scene_matches[0]
    task_matches = [
        task for task in scene.tasks
        if task_text == task.task_id
    ]
    if not task_matches:
        available = ", ".join(task.task_id for task in scene.tasks)
        raise ValueError(
            f"Unknown training task ID {task_key!r} for "
            f"scene {scene.scene_id!r}; use one of: {available}"
        )
    return scene, task_matches[0]


def tasks_for_runtime_scene(runtime_scene: str) -> tuple[DatasetTask, ...]:
    """Return executable curated tasks shown under one runtime scene."""
    return tuple(
        task
        for scene, task in (
            *training_tasks(),
            *tolerance_discrimination_tasks(),
        )
        if scene.runtime_scene == runtime_scene
    )


def runtime_task_from_id(runtime_scene: str, task_id: str) -> str:
    """Resolve a canonical training task ID to its runtime identifier."""
    for task in tasks_for_runtime_scene(runtime_scene):
        if task.task_id == task_id:
            return task.runtime_task
    raise ValueError(
        f"Unknown curated task ID {task_id!r} for scene {runtime_scene!r}"
    )


def weighted_training_plan(multiplier: int = 1) -> tuple[tuple[DatasetScene, DatasetTask, int], ...]:
    """Return per-task collection counts for one or more weight units."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    return tuple(
        (scene, task, task.weight * multiplier)
        for scene, task in training_tasks()
    )


def task_instruction_record(
    task: DatasetTask,
    *,
    seed: int,
    episode_index: int,
    instruction_index: int | None = None,
    selection: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the instruction fields embedded in one trajectory episode.

    Callers may pass the variant selected from the episode seed (for example
    ``{"size": "small", "color": "red"}``).  Selection remains explicit in
    the dataset instead of being inferred later from the rendered image.
    """
    index = episode_index % len(task.instructions) if instruction_index is None else instruction_index
    if not 0 <= index < len(task.instructions):
        raise IndexError("instruction_index is outside the task instruction set")
    resolved_selection = (
        seeded_task_selection(task, seed)
        if selection is None else dict(selection)
    )
    return {
        "task_id": task.task_id,
        "runtime_task_id": task.runtime_task,
        "instruction_index": int(index),
        "instruction": task.instructions[index],
        "seed": int(seed),
        "selection": resolved_selection,
    }


def seeded_task_selection(task: DatasetTask, seed: int) -> dict[str, str]:
    """Resolve object, size and color choices reproducibly from one seed."""
    if seed < -1:
        raise ValueError("seed must be -1 or non-negative")

    def choose(field: str, values: tuple[str, ...]) -> str | None:
        if not values:
            return None
        if seed == -1:
            return values[0]
        digest = hashlib.sha256(
            f"tasktol-dataset-variant:"
            f"{task.variant_seed_key or task.task_id}:{field}:{seed}".encode("utf-8")
        ).digest()
        return values[int.from_bytes(digest[:8], "little") % len(values)]

    result: dict[str, str] = {}
    for field, values in (
        ("object_name", task.variants.object_names),
        ("size", task.variants.sizes),
        ("color", task.variants.colors),
    ):
        value = choose(field, values)
        if value is not None:
            result[field] = value
    if task.variants.orientation is not None:
        result["orientation"] = task.variants.orientation
    return result


__all__ = (
    "DATASET_SCENES",
    "DatasetScene",
    "DatasetTask",
    "EXISTING_SCENE_GROUP",
    "TRAINING_DATASET_SCENES",
    "GEOMETRIC_BASIS_SUBCATEGORY",
    "CONTEXTUAL_SUBCATEGORY",
    "MULTI_SUBTASK_SUBCATEGORY",
    "TRAINING_SUBCATEGORIES",
    "TRAINING_TASK_GROUP",
    "TOLERANCE_DISCRIMINATION_DATASET_SCENES",
    "TOLERANCE_DISCRIMINATION_GROUP",
    "UNSEEN_DATASET_SCENES",
    "UNSEEN_OBJECT_GROUP",
    "VariantContract",
    "runtime_scenes_for_group",
    "runtime_scene_from_choice",
    "resolve_training_task",
    "runtime_task_from_id",
    "runtime_task_from_choice",
    "scene_choices_for_group",
    "scene_groups",
    "scene_registration",
    "seeded_task_selection",
    "task_instruction_record",
    "tolerance_discrimination_tasks",
    "training_tasks",
    "short_training_tasks",
    "training_tasks_for_subcategory",
    "unseen_object_tasks",
    "tasks_for_runtime_scene",
    "tasks_for_scene_choice",
    "weighted_training_plan",
)
