"""Resolve tolerance configuration from the live scene and selected task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from ..simulation.robot import RobotHandles
from ..simulation.scenes import (
    ADJUST_CYLINDRICAL_BOTTLE_SCENE,
    ADJUST_RECTANGULAR_BOTTLE_SCENE,
)
from ..tasks.catalog import (
    ADJUST_BOTTLE_TASK,
    BALL_TO_PLATE_TASK,
    BALL_PLATE_THEN_BLOCK_PAPER_TASK,
    BANANA_PLATE_THEN_PEAR_BOWL_TASK,
    FRUIT_TRAY_SEQUENCE_TASK_SPECS,
    CLICK_BELL_TASK,
    CYLINDER_TO_PLATE_TASK,
    POT_LID_TASKS,
    pot_lid_task_spec,
    PRESS_BUTTON_TASK,
    PRESS_POWER_STRIP_TASK,
    ROTATE_KNOB_TASK,
    STACK_COLORED_BLOCKS_TASK,
    COLOR_BLOCK_STACK_TASK_SPECS,
    COLOR_BLOCK_STACK_TASKS,
    ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK,
)
from ..tasks.planning.presets import PresetTaskExecutor
from ..tasks.planning.types import TaskFrame
from ..tasks.subtasks import identify_pick_place_subtasks
from .annotations import (
    LYING_CYLINDER_STAGE_ROTATION_MASKS,
    UPRIGHT_CYLINDER_SCENES,
    UPRIGHT_CYLINDER_STAGE_ROTATION_MASKS,
    box_tolerance_frame,
    lying_cylinder_stage_rotation_masks,
    pot_lid_stage_rotation_masks,
    static_stage_rotation_masks,
)


_NON_GRASP_TASKS = {
    CLICK_BELL_TASK,
    PRESS_BUTTON_TASK,
    PRESS_POWER_STRIP_TASK,
}

_LONG_TASK_SUBTASK_STAGE_ROTATION_MASKS = {
    (BALL_PLATE_THEN_BLOCK_PAPER_TASK, "ball"): (
        (1, 1, 1), (1, 1, 1),
    ),
    (BALL_PLATE_THEN_BLOCK_PAPER_TASK, "cube"): (
        (0, 1, 0), (0, 0, 0),
    ),
    (BANANA_PLATE_THEN_PEAR_BOWL_TASK, "banana"): (
        (1, 1, 0), (1, 1, 1),
    ),
    (BANANA_PLATE_THEN_PEAR_BOWL_TASK, "pear"): (
        (1, 1, 1), (1, 1, 1),
    ),
    (STACK_COLORED_BLOCKS_TASK, "red cube"): (
        (0, 1, 0), (0, 0, 0),
    ),
    (STACK_COLORED_BLOCKS_TASK, "green cube"): (
        (0, 1, 0), (0, 0, 0),
    ),
    (STACK_COLORED_BLOCKS_TASK, "purple cube"): (
        (0, 1, 0), (0, 0, 0),
    ),
}

_LONG_TASK_SUBTASK_INDEX_STAGE_ROTATION_MASKS = {
    (ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK, 0): (
        (0, 1, 1), (1, 1, 1),
    ),
    (ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK, 1): (
        (1, 1, 1), (1, 1, 1),
    ),
    (ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK, 2): (
        (0, 1, 1), (0, 0, 1),
    ),
}

_LONG_TASK_SUBTASK_LABELS = {
    BALL_PLATE_THEN_BLOCK_PAPER_TASK: ("ball", "cube"),
    BANANA_PLATE_THEN_PEAR_BOWL_TASK: ("banana", "pear"),
    STACK_COLORED_BLOCKS_TASK: (
        "red cube", "green cube", "purple cube",
    ),
    ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK: (
        "pot lid", "pear", "pot lid",
    ),
}

for _task, _order in FRUIT_TRAY_SEQUENCE_TASK_SPECS.items():
    _LONG_TASK_SUBTASK_LABELS[_task] = _order
    for _fruit in _order:
        _LONG_TASK_SUBTASK_STAGE_ROTATION_MASKS[(_task, _fruit)] = (
            ((1, 1, 0), (1, 1, 1))
            if _fruit == "banana"
            else ((1, 1, 1), (1, 1, 1))
        )

for _task, _order in COLOR_BLOCK_STACK_TASK_SPECS.items():
    _labels = tuple(f"{color} cube" for color in _order[1:])
    _LONG_TASK_SUBTASK_LABELS[_task] = _labels
    for _label in _labels:
        _LONG_TASK_SUBTASK_STAGE_ROTATION_MASKS[(_task, _label)] = (
            (0, 1, 0), (0, 0, 0),
        )


def long_task_subtask_labels(task: str) -> tuple[str, ...]:
    """Return ordered semantic labels without consulting tolerance masks."""
    return tuple(_LONG_TASK_SUBTASK_LABELS.get(str(task), ()))


def long_task_subtask_stage_rotation_masks(
    task: str,
) -> tuple[
    tuple[str, tuple[tuple[int, int, int], tuple[int, int, int]]], ...
]:
    """Return fixed ordered Pre/Post masks for a registered long task."""
    labels = _LONG_TASK_SUBTASK_LABELS.get(str(task), ())
    def masks_for(
        index: int, label: str,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        indexed = _LONG_TASK_SUBTASK_INDEX_STAGE_ROTATION_MASKS.get((
            str(task), index,
        ))
        if indexed is not None:
            return indexed
        return _LONG_TASK_SUBTASK_STAGE_ROTATION_MASKS[(str(task), label)]

    return tuple(
        (label, masks_for(index, label))
        for index, label in enumerate(labels)
    )


def _active_adjust_bottle_variant(model: mujoco.MjModel) -> str | None:
    """Return the active Adjust Bottle variant, if one is exposed."""
    active: list[str] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or not name.startswith("adjust_bottle_"):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if geom_ids.size and np.any(model.geom_contype[geom_ids] != 0):
            active.append(name)
    return active[0] if len(active) == 1 else None


@dataclass(frozen=True)
class ResolvedStageTolerance:
    """One automatically resolved stable-stage target and tolerance axes."""

    target_position: np.ndarray
    target_rotation: np.ndarray
    tolerance_frame: np.ndarray
    ranged_axes: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.target_position, dtype=float).copy()
        target = np.asarray(self.target_rotation, dtype=float).copy()
        frame = np.asarray(self.tolerance_frame, dtype=float).copy()
        ranged = np.asarray(self.ranged_axes, dtype=bool).copy()
        if position.shape != (3,):
            raise ValueError("Resolved stage position must be a three-vector")
        if target.shape != (3, 3) or frame.shape != (3, 3):
            raise ValueError("Resolved stage rotations must be 3x3")
        if ranged.shape != (3,):
            raise ValueError("Resolved stage mask must contain three axes")
        for value in (position, target, frame):
            if not np.all(np.isfinite(value)):
                raise ValueError("Resolved stage values must be finite")
            value.setflags(write=False)
        ranged.setflags(write=False)
        object.__setattr__(self, "target_position", position)
        object.__setattr__(self, "target_rotation", target)
        object.__setattr__(self, "tolerance_frame", frame)
        object.__setattr__(self, "ranged_axes", ranged)


@dataclass(frozen=True)
class ResolvedSubtaskTolerance:
    index: int
    label: str
    pregrasp: ResolvedStageTolerance
    postgrasp: ResolvedStageTolerance


@dataclass(frozen=True)
class ResolvedTaskTolerance:
    scene: str
    task: str
    object_variant: str | None
    pregrasp: ResolvedStageTolerance
    postgrasp: ResolvedStageTolerance
    subtasks: tuple[ResolvedSubtaskTolerance, ...] = ()

    def subtask(self, index: int) -> ResolvedSubtaskTolerance:
        if not self.subtasks:
            return ResolvedSubtaskTolerance(
                0, "Subtask 1", self.pregrasp, self.postgrasp,
            )
        return self.subtasks[min(max(int(index), 0), len(self.subtasks) - 1)]


def effective_rotation_specific_axes(
    tolerance_enabled: bool,
    stage: int,
    ranged_axes: np.ndarray,
) -> np.ndarray:
    """Resolve the live Specific mask without discarding stored task masks."""
    ranged = np.asarray(ranged_axes, dtype=bool)
    if ranged.shape != (3,):
        raise ValueError("Ranged rotation mask must contain three axes")
    if not 0 <= int(stage) < 4:
        raise ValueError("Tolerance stage must be in [0, 3]")
    if not tolerance_enabled or int(stage) in (1, 3):
        return np.ones(3, dtype=bool)
    return ~ranged


def declared_stage_rotation_masks(
    model: mujoco.MjModel,
    scene: str,
    task: str,
    *,
    has_grasp: bool = True,
    subtask_label: str | None = None,
    subtask_index: int | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return Pre/Post XYZ masks for the selected task and active objects."""
    indexed_long_task_masks = _LONG_TASK_SUBTASK_INDEX_STAGE_ROTATION_MASKS.get((
        task, -1 if subtask_index is None else int(subtask_index),
    ))
    if indexed_long_task_masks is not None:
        return indexed_long_task_masks
    long_task_masks = _LONG_TASK_SUBTASK_STAGE_ROTATION_MASKS.get((
        task, (subtask_label or "").strip().lower(),
    ))
    if long_task_masks is not None:
        return long_task_masks
    static_masks = static_stage_rotation_masks(scene, task)
    if static_masks is not None:
        return static_masks
    if not has_grasp or task in _NON_GRASP_TASKS:
        return ((0, 0, 0), (0, 0, 0))
    if task in POT_LID_TASKS:
        lid_kind, _ = pot_lid_task_spec(task)
        return pot_lid_stage_rotation_masks(lid_kind)
    if scene in LYING_CYLINDER_STAGE_ROTATION_MASKS:
        return lying_cylinder_stage_rotation_masks(scene)
    if (
        scene in UPRIGHT_CYLINDER_SCENES
        or task == CYLINDER_TO_PLATE_TASK
    ):
        return UPRIGHT_CYLINDER_STAGE_ROTATION_MASKS
    if task == BALL_TO_PLATE_TASK:
        return ((1, 1, 1), (1, 1, 1))
    if task in (*COLOR_BLOCK_STACK_TASKS, STACK_COLORED_BLOCKS_TASK):
        return ((0, 1, 0), (0, 0, 0))
    if task == ROTATE_KNOB_TASK:
        return ((0, 1, 0), (0, 0, 0))
    if scene == "Adjust Bottle" and task == ADJUST_BOTTLE_TASK:
        if _active_adjust_bottle_variant(model) == "adjust_bottle_ganten":
            return ((0, 1, 0), (0, 0, 1))
        return ((1, 1, 0), (0, 0, 1))
    return ((0, 1, 0), (0, 0, 1))


def resolve_planned_subtask_tolerances(
    model: mujoco.MjModel,
    frames: Sequence[TaskFrame],
    *,
    scene: str,
    task: str,
    subtask_index_offset: int = 0,
) -> tuple[ResolvedSubtaskTolerance, ...]:
    """Resolve masks and endpoints for one already-planned task segment.

    Long tasks plan later segments only after the previous placement has
    completed.  Their local frame list therefore starts again at subtask zero;
    ``subtask_index_offset`` maps that segment back to the task-wide semantic
    index used by masks, telemetry, and the UI.
    """
    if not frames:
        raise RuntimeError("Selected task produced no reference frames")
    offset = int(subtask_index_offset)
    if offset < 0:
        raise ValueError("Subtask index offset must be non-negative")
    spans = identify_pick_place_subtasks(frames)
    semantic_grasp = bool(spans) and task not in _NON_GRASP_TASKS
    if semantic_grasp:
        stage_indices = tuple(
            (
                offset + span.index,
                span.pregrasp_index,
                span.postgrasp_index,
                span.label,
            )
            for span in spans
        )
    else:
        stage_indices = ((
            offset,
            len(frames) - 1,
            len(frames) - 1,
            "Subtask 1",
        ),)

    semantic_labels = long_task_subtask_labels(task)

    def stage(index: int, mask: tuple[int, int, int]) -> ResolvedStageTolerance:
        frame = frames[index]
        return ResolvedStageTolerance(
            frame.position,
            frame.rotation,
            box_tolerance_frame(frame.rotation),
            np.asarray(mask, dtype=bool),
        )

    resolved: list[ResolvedSubtaskTolerance] = []
    for index, pre_index, post_index, inferred_label in stage_indices:
        label = (
            semantic_labels[index].capitalize()
            if index < len(semantic_labels)
            else inferred_label
        )
        pre_mask, post_mask = declared_stage_rotation_masks(
            model,
            scene,
            task,
            has_grasp=semantic_grasp,
            subtask_label=label,
            subtask_index=index,
        )
        resolved.append(ResolvedSubtaskTolerance(
            index,
            label,
            stage(pre_index, pre_mask),
            stage(post_index, post_mask),
        ))
    return tuple(resolved)


def resolve_task_tolerance_context(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    scene: str,
    task: str,
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    *,
    episode_seed: int = -1,
    planned_frames: Sequence[TaskFrame] | None = None,
    nominal_validation_options: dict | None = None,
) -> ResolvedTaskTolerance:
    """Plan the live task and derive its stable-stage tolerance context."""
    if planned_frames is None:
        executor = PresetTaskExecutor(model, handles)
        executor.start(
            task,
            scene,
            data,
            np.asarray(current_position, dtype=float),
            np.asarray(current_rotation, dtype=float),
            episode_seed=episode_seed,
            nominal_validation_options=nominal_validation_options,
        )
        frames = executor.frames
    else:
        frames = planned_frames
    variant = None
    if scene in (
        "Adjust Bottle",
        ADJUST_CYLINDRICAL_BOTTLE_SCENE,
        ADJUST_RECTANGULAR_BOTTLE_SCENE,
    ) and task == ADJUST_BOTTLE_TASK:
        variant = _active_adjust_bottle_variant(model)
    if task in POT_LID_TASKS:
        variant = pot_lid_task_spec(task)[0]
    resolved_subtasks = resolve_planned_subtask_tolerances(
        model,
        frames,
        scene=scene,
        task=task,
    )
    first = resolved_subtasks[0]
    return ResolvedTaskTolerance(
        scene,
        task,
        variant,
        first.pregrasp,
        first.postgrasp,
        resolved_subtasks,
    )


__all__ = (
    "ResolvedStageTolerance",
    "ResolvedSubtaskTolerance",
    "ResolvedTaskTolerance",
    "declared_stage_rotation_masks",
    "effective_rotation_specific_axes",
    "long_task_subtask_labels",
    "long_task_subtask_stage_rotation_masks",
    "resolve_planned_subtask_tolerances",
    "resolve_task_tolerance_context",
)
