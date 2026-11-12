"""Scene task registry and feedback-paced execution of generated skill plans."""

from __future__ import annotations

import math
from dataclasses import asdict
from functools import lru_cache
from typing import Callable

import mujoco
import numpy as np

from ...geometry import (
    project_rotation_target,
    rotation_error_in_frame,
    rotation_matrix,
    rotation_vector,
)
from ...simulation.scenes import (
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
    SceneManager,
    WHITEBOARD_SCENE,
)
from ...simulation.robot import RobotHandles, find_handles
from .perturbations import EpisodePerturbations, grasp_height_floor, perturb_task_frames
from ..catalog import (
    ADJUST_BOTTLE_TASK,
    BALL_TO_PLATE_TASK,
    BALL_PLATE_THEN_BLOCK_PAPER_TASK,
    APPLE_THEN_BANANA_TO_TRAY_TASK,
    BLOCK_TO_PLATE_TASK,
    BLOCK_TO_PLATE_STABLE_TASK,
    BOX_TO_NARROW_BOX_TASK,
    CUBE_TO_PLATE_TASK,
    CLICK_BELL_TASK,
    CYLINDER_TO_PLATE_TASK,
    CYLINDER_TO_PLATE_STABLE_TASK,
    CYLINDER_TO_NARROW_BOX_TASK,
    FRUIT_RECEPTACLE_TASKS,
    FRUIT_TRAY_SEQUENCE_TASK_SPECS,
    INSERT_OBJECT_TASK,
    LYING_BOX_TO_PLATE_TASK,
    LYING_BOX_TO_REGION_TASK,
    PLACE_OBJECT_ON_PAPER_TASK,
    PRESS_BUTTON_TASK,
    PRESS_POWER_STRIP_TASK,
    ROTATE_KNOB_TASK,
    WIPE_WHITEBOARD_TASK,
    COLOR_BLOCK_STACK_TASKS,
    CLOSE_CYLINDRICAL_POT_LID_TASK,
    CLOSE_HANDLE_POT_LID_TASK,
    OPEN_CYLINDRICAL_POT_LID_TASK,
    OPEN_HANDLE_POT_LID_TASK,
    POT_LID_TASKS,
    pot_lid_task_spec,
    ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK,
)
from .adjust_bottle import StandFallenBottle
from .cylinder_narrow_box import PlaceBoxInNarrowBox, PlaceCylinderInNarrowBox
from .insert_object import InsertObject
from .fruit_receptacles import PlaceFruit, PlaceFruitSequence
from .ball_paper_sequence import PlaceBallThenBlockOnPaper
from .paper_placement import PlaceBlockOnPaper
from .rotate_knob import RotateKnobClockwise
from .wipe_whiteboard import WipeWhiteboard
from .color_block_stack import StackColoredBlocks
from .pick_place import (
    PickBox,
    PickCylinder,
    PickSphere,
    PlaceOnPlate,
    gripper_vertical_envelope,
)
from .press_device import PressDevice
from .pot_lid import ManipulatePotLid
from .pot_fruit_sequence import RoundHandlePotPearSequence
from .types import MotionLimits, PlanningInfeasibleError, PreparedTaskPlan, TaskAction, TaskCommand, TaskFrame
from .nominal_candidates import nominal_candidate_definitions, supports_nominal_selection


@lru_cache(maxsize=1)
def _training_runtime_tasks() -> frozenset[tuple[str, str]]:
    """Return curated runtime scene/task pairs without an import cycle."""
    from ...data.task_sets import training_tasks

    return frozenset(
        (scene.runtime_scene, task.runtime_task)
        for scene, task in training_tasks()
    )


def _end_after_final_release(frames: list[TaskFrame]) -> list[TaskFrame]:
    """Keep the final opening phase and remove all later trajectory frames."""
    release_starts = [
        index
        for index in range(1, len(frames))
        if frames[index - 1].gripper_closed
        and not frames[index].gripper_closed
    ]
    if not release_starts:
        return frames
    start = release_starts[-1]
    release_phase = frames[start].phase
    end = start + 1
    while (
        end < len(frames)
        and not frames[end].gripper_closed
        and frames[end].phase == release_phase
    ):
        end += 1
    return frames[:end]


def compile_task_actions(
    frames: tuple[TaskFrame, ...] | list[TaskFrame],
    initial_position: np.ndarray,
    initial_rotation: np.ndarray,
) -> list[TaskAction]:
    """Offline-convert absolute frames into base-frame 6+1 increments."""
    previous_position = np.asarray(initial_position, dtype=float).copy()
    previous_rotation = np.asarray(initial_rotation, dtype=float).copy()
    actions: list[TaskAction] = []
    for frame in frames:
        rotation_delta = frame.rotation @ previous_rotation.T
        cartesian = np.concatenate((
            frame.position - previous_position,
            rotation_vector(rotation_delta),
        ))
        actions.append(TaskAction(
            np.concatenate((
                cartesian,
                np.array((1.0 if frame.gripper_closed else -1.0,)),
            )),
            frame.phase,
            "",
        ))
        previous_position = frame.position.copy()
        previous_rotation = frame.rotation.copy()
    return actions


class PresetTaskExecutor:
    """Build the selected skill sequence and stream its 10 Hz commands."""

    _PICK_SPECS = {
        BLOCK_TO_PLATE_TASK: (PickBox, "teleop_tall_block", "teleop_tall_block_geom"),
        CUBE_TO_PLATE_TASK: (PickBox, "teleop_cube", "teleop_cube_geom"),
        BALL_TO_PLATE_TASK: (PickSphere, "teleop_ball", "teleop_ball_geom"),
        CYLINDER_TO_PLATE_TASK: (PickCylinder, "teleop_cylinder", "teleop_cylinder_geom"),
    }
    TASKS_BY_SCENE = {
        "Blocks": tuple(_PICK_SPECS),
        "Adjust Bottle": (ADJUST_BOTTLE_TASK,),
        ADJUST_CYLINDRICAL_BOTTLE_SCENE: (ADJUST_BOTTLE_TASK,),
        ADJUST_RECTANGULAR_BOTTLE_SCENE: (ADJUST_BOTTLE_TASK,),
        "Click Bell": (CLICK_BELL_TASK,),
        "Press Button": (PRESS_BUTTON_TASK,),
        "Press Power Strip": (PRESS_POWER_STRIP_TASK,),
        "Insert Object": (INSERT_OBJECT_TASK,),
        CYLINDER_TO_NARROW_BOX_SCENE: (CYLINDER_TO_NARROW_BOX_TASK,),
        BOX_TO_NARROW_BOX_SCENE: (BOX_TO_NARROW_BOX_TASK,),
        "Fruit Receptacles": FRUIT_RECEPTACLE_TASKS,
        BANANA_APPLE_TRAY_SCENE: (APPLE_THEN_BANANA_TO_TRAY_TASK,),
        "Paper Placement": (PLACE_OBJECT_ON_PAPER_TASK,),
        "Ball + Paper Placement": (BALL_PLATE_THEN_BLOCK_PAPER_TASK,),
        "Color Block Stack": COLOR_BLOCK_STACK_TASKS,
        "Rotate Knob": (ROTATE_KNOB_TASK,),
        WHITEBOARD_SCENE: (WIPE_WHITEBOARD_TASK,),
        "Open Pot Lid": (
            OPEN_CYLINDRICAL_POT_LID_TASK,
            OPEN_HANDLE_POT_LID_TASK,
        ),
        "Close Pot Lid": (
            CLOSE_CYLINDRICAL_POT_LID_TASK,
            CLOSE_HANDLE_POT_LID_TASK,
        ),
        GEOMETRY_PLATE_BOX_UPRIGHT_SCENE: (
            BLOCK_TO_PLATE_TASK,
            BLOCK_TO_PLATE_STABLE_TASK,
        ),
        GEOMETRY_PLATE_BOX_LYING_SCENE: (
            BLOCK_TO_PLATE_TASK,
            BLOCK_TO_PLATE_STABLE_TASK,
        ),
        GEOMETRY_PLATE_CUBE_SCENE: (CUBE_TO_PLATE_TASK,),
        GEOMETRY_PLATE_BALL_SCENE: (BALL_TO_PLATE_TASK,),
        GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE: (
            CYLINDER_TO_PLATE_TASK,
            CYLINDER_TO_PLATE_STABLE_TASK,
        ),
        GEOMETRY_PLATE_CYLINDER_LYING_SCENE: (
            CYLINDER_TO_PLATE_TASK,
            CYLINDER_TO_PLATE_STABLE_TASK,
        ),
        GEOMETRY_REGION_BOX_UPRIGHT_SCENE: (PLACE_OBJECT_ON_PAPER_TASK,),
        GEOMETRY_REGION_BOX_LYING_SCENE: (PLACE_OBJECT_ON_PAPER_TASK,),
        GEOMETRY_REGION_CUBE_SCENE: (PLACE_OBJECT_ON_PAPER_TASK,),
        GEOMETRY_REGION_CYLINDER_UPRIGHT_SCENE: (PLACE_OBJECT_ON_PAPER_TASK,),
        GEOMETRY_REGION_CYLINDER_LYING_SCENE: (PLACE_OBJECT_ON_PAPER_TASK,),
        TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE: (
            LYING_BOX_TO_PLATE_TASK,
            LYING_BOX_TO_REGION_TASK,
        ),
        OPEN_CYLINDRICAL_POT_LID_SCENE: (OPEN_CYLINDRICAL_POT_LID_TASK,),
        OPEN_HANDLE_POT_LID_SCENE: (OPEN_HANDLE_POT_LID_TASK,),
        CLOSE_CYLINDRICAL_POT_LID_SCENE: (CLOSE_CYLINDRICAL_POT_LID_TASK,),
        CLOSE_HANDLE_POT_LID_SCENE: (CLOSE_HANDLE_POT_LID_TASK,),
        ROUND_HANDLE_POT_PEAR_SCENE: (
            ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK,
        ),
    }
    _PRESS_SPECS = {
        CLICK_BELL_TASK: ("click_bell_", "bell"),
        PRESS_BUTTON_TASK: ("press_button_", "button"),
        PRESS_POWER_STRIP_TASK: ("power_strip_", "power strip button"),
    }
    _LIMITS = MotionLimits(linear_speed_m_s=0.150, angular_speed_rad_s=math.radians(45.0))
    _POSITION_GATE = 0.018
    _ROTATION_GATE = math.radians(12.0)
    _MAX_GATE_TICKS = 6
    _PLATE_TRANSFER_CLEARANCE_M = 0.040

    def __init__(self, model: mujoco.MjModel, handles: RobotHandles | None = None) -> None:
        self.model = model
        self.handles = handles
        self.gripper_envelope = gripper_vertical_envelope(model, handles)
        self.frames: list[TaskFrame] = []
        self.actions: list[TaskAction] = []
        self.index = 0
        self._gate_ticks = 0
        self._trajectory_credit = 0.0
        self._last_trajectory_command: TaskCommand | None = None
        self._last_gripper_closed = False
        self._end_at_release = False
        self._pending_subtask_planners: list[Callable[
            [mujoco.MjData, np.ndarray, np.ndarray, MotionLimits],
            tuple[TaskFrame, ...],
        ]] = []
        self.subtask_index = 0
        self.last_planning_stats: dict[str, object] = {}
        self.perturbations = EpisodePerturbations(-1)
        self._perturbation_task = ""
        self._grasp_floor = lambda rotation: -math.inf
        self.active = False
        self.status = "Preset task: idle"
        self._nominal_configuration = None

    @classmethod
    def task_names(cls, scene_name: str) -> tuple[str, ...]:
        return cls.TASKS_BY_SCENE.get(scene_name, ())

    def cancel(self, reason: str = "cancelled") -> None:
        self.frames.clear()
        self.actions.clear()
        self.index = 0
        self._gate_ticks = 0
        self._trajectory_credit = 0.0
        self._last_trajectory_command = None
        self._last_gripper_closed = False
        self._end_at_release = False
        self._pending_subtask_planners.clear()
        self.subtask_index = 0
        self.last_planning_stats = {}
        self.perturbations = EpisodePerturbations(-1)
        self.active = False
        self.status = f"Preset task: {reason}"

    def start(
        self,
        task_name: str,
        scene_name: str,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        *,
        episode_seed: int = -1,
        nominal_selection_enabled: bool = True,
        nominal_validation_options: dict | None = None,
        domain_randomization_enabled: bool = False,
        nominal_cache_only: bool = False,
        prepared_plan: PreparedTaskPlan | None = None,
        _nominal_candidate_index: int | None = None,
    ) -> None:
        if task_name not in self.TASKS_BY_SCENE.get(scene_name, ()):
            raise ValueError("The selected preset task is not available in this scene.")
        self._nominal_configuration = dict(scene=scene_name, task=task_name,
            episode_seed=episode_seed, enabled=nominal_selection_enabled,
            options=nominal_validation_options,
            domain_randomization_enabled=domain_randomization_enabled,
            cache_only=nominal_cache_only)
        configuration = dict(self._nominal_configuration)
        if prepared_plan is None and nominal_selection_enabled:
            candidates = nominal_candidate_definitions(self.model, data, scene_name, task_name)
            if candidates:
                from ...simulation.nominal_plan_table import select_nominal_candidate

                def build_candidate(index):
                    self.start(task_name, scene_name, data, current_position, current_rotation,
                        episode_seed=episode_seed, nominal_selection_enabled=False,
                        nominal_validation_options=nominal_validation_options,
                        domain_randomization_enabled=domain_randomization_enabled,
                        _nominal_candidate_index=index)
                    return dict(frames=tuple(self.frames), perturbations=dict(self.perturbations.samples))

                selected_index, table = select_nominal_candidate(
                    self.model, data, self.handles, scene=scene_name, task=task_name,
                    episode_seed=episode_seed, candidates=candidates, build_candidate=build_candidate,
                    planning_conditions=dict(motion_limits=asdict(self._LIMITS), subtask_index=0),
                    options=nominal_validation_options,
                    domain_randomization_enabled=domain_randomization_enabled,
                    cache_only=nominal_cache_only)
                try:
                    build_candidate(selected_index)
                except PlanningInfeasibleError as error:
                    error.context["nominal_plan_table"] = table
                    raise
                self._nominal_configuration = configuration
                self.last_planning_stats.update(nominal_plan_table=table,
                                                nominal_plan_tables=[dict(table, subtask_index=0)])
                return
        self.last_planning_stats = {}
        self.perturbations = EpisodePerturbations(episode_seed)
        self._perturbation_task = task_name
        mujoco.mj_forward(self.model, data)
        self._grasp_floor = (
            grasp_height_floor(self.model, data, self.handles or find_handles(self.model))
            if episode_seed != -1 and prepared_plan is None
            else lambda rotation: -math.inf
        )
        self._pending_subtask_planners.clear()
        self.subtask_index = 0
        self._end_at_release = (
            scene_name, task_name,
        ) in _training_runtime_tasks()
        mujoco.mj_forward(self.model, data)

        # The first standardized task is deliberately only a composition of
        # the two skills that are currently needed.  Additional object rules
        # are added only when a later task requires them.
        motion_limits = self._LIMITS
        if prepared_plan is not None:
            if (
                prepared_plan.scene != scene_name or prepared_plan.task != task_name
                or self.handles is None or prepared_plan.robot_uid != self.handles.spec.uid
                or not np.array_equal(prepared_plan.initial_qpos, data.qpos)
                or not prepared_plan.frames
            ):
                raise ValueError("Prepared task plan does not match the current scene/state")
            self.frames = list(prepared_plan.frames)
            self.perturbations = EpisodePerturbations(episode_seed, dict(prepared_plan.perturbations))
            self.last_planning_stats = dict(prepared_plan.planning_stats)
            self._configure_prepared_successors(task_name)
            # Already perturbed, trimmed and validated by the shared planner.
            self.actions = compile_task_actions(self.frames, current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = f"Preset task: running 1/{len(self.frames)} - {self.frames[0].phase}"
            return
        if scene_name in (
            "Adjust Bottle",
            ADJUST_CYLINDRICAL_BOTTLE_SCENE,
            ADJUST_RECTANGULAR_BOTTLE_SCENE,
        ):
            assert task_name == ADJUST_BOTTLE_TASK
            if self.handles is None:
                raise RuntimeError("Adjust Bottle requires explicit robot handles")
            bottle_planner = StandFallenBottle(
                self.model, self.handles,
            )
            self.frames = list(bottle_planner.plan(
                data,
                current_position,
                current_rotation,
                motion_limits,
                candidate_index=_nominal_candidate_index,
            ))
            self.last_planning_stats = dict(bottle_planner.last_planning_stats)
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name in self._PRESS_SPECS:
            if self.handles is None:
                raise RuntimeError(
                    f"{scene_name} requires explicit robot handles",
                )
            root_prefix, label = self._PRESS_SPECS[task_name]
            self.frames = list(PressDevice(
                self.model,
                self.handles,
                root_prefix=root_prefix,
                label=label,
            ).plan(
                data,
                current_position,
                current_rotation,
                motion_limits,
            ))
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name == INSERT_OBJECT_TASK:
            if self.handles is None:
                raise RuntimeError(
                    "Insert Object requires explicit robot handles",
                )
            self.frames = list(InsertObject(
                self.model,
                self.handles,
            ).plan(
                data,
                current_position,
                current_rotation,
                motion_limits,
            ))
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name in (
            CYLINDER_TO_NARROW_BOX_TASK,
            BOX_TO_NARROW_BOX_TASK,
        ):
            if self.handles is None:
                raise RuntimeError(
                    f"{scene_name} requires explicit robot handles",
                )
            planner_type = (
                PlaceCylinderInNarrowBox
                if task_name == CYLINDER_TO_NARROW_BOX_TASK
                else PlaceBoxInNarrowBox
            )
            self.frames = list(planner_type(
                self.model,
                self.handles,
            ).plan(
                data,
                current_position,
                current_rotation,
                motion_limits,
                candidate_index=_nominal_candidate_index,
            ))
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if (
            task_name in FRUIT_RECEPTACLE_TASKS
            or task_name in FRUIT_TRAY_SEQUENCE_TASK_SPECS
        ):
            if self.handles is None:
                raise RuntimeError(
                    "Fruit Receptacles requires explicit robot handles",
                )
            planner = (
                PlaceFruitSequence(self.model, self.handles, task_name)
                if task_name in FRUIT_TRAY_SEQUENCE_TASK_SPECS
                else PlaceFruit(self.model, self.handles, task_name)
            )
            if task_name in FRUIT_TRAY_SEQUENCE_TASK_SPECS:
                self.frames = list(planner.plan_subtask(
                    0, data, current_position, current_rotation, motion_limits,
                ))
                self._pending_subtask_planners = [
                    lambda live_data, position, rotation, live_limits, i=i, **kwargs:
                    planner.plan_subtask(
                        i, live_data, position, rotation, live_limits, **kwargs,
                    )
                    for i in range(1, len(planner.order))
                ]
            else:
                self.frames = list(planner.plan(
                    data,
                    current_position,
                    current_rotation,
                    motion_limits,
                ))
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name in (
            PLACE_OBJECT_ON_PAPER_TASK,
            LYING_BOX_TO_REGION_TASK,
        ):
            if self.handles is None:
                raise RuntimeError(
                    "Paper Placement requires explicit robot handles",
                )
            paper_planner = PlaceBlockOnPaper(
                self.model,
                self.handles,
            )
            self.frames = list(paper_planner.plan(
                data, current_position, current_rotation, motion_limits,
                candidate_index=_nominal_candidate_index,
            ))
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name == BALL_PLATE_THEN_BLOCK_PAPER_TASK:
            if self.handles is None:
                raise RuntimeError(
                    "Ball + Paper Placement requires explicit robot handles",
                )
            planner = PlaceBallThenBlockOnPaper(
                self.model, self.handles,
            )
            first_subtask = planner.plan_first_subtask(
                data,
                current_position,
                current_rotation,
                motion_limits,
            )
            self.frames = list(first_subtask)
            self._pending_subtask_planners = [planner.plan_second_subtask]
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name == WIPE_WHITEBOARD_TASK:
            if self.handles is None:
                raise RuntimeError("Whiteboard task requires explicit robot handles")
            self.frames = list(WipeWhiteboard(self.model, self.handles).plan(
                data, current_position, current_rotation, motion_limits,
                candidate_index=_nominal_candidate_index,
            ))
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name == ROTATE_KNOB_TASK:
            if self.handles is None:
                raise RuntimeError(
                    "Rotate Knob requires explicit robot handles",
                )
            self.frames = list(RotateKnobClockwise(
                self.model,
                self.handles,
            ).plan(
                data,
                current_position,
                current_rotation,
                motion_limits,
            ))
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name in COLOR_BLOCK_STACK_TASKS:
            if self.handles is None:
                raise RuntimeError(
                    "Color Block Stack requires explicit robot handles",
                )
            planner = StackColoredBlocks(
                self.model,
                self.handles,
                task_name,
            )
            self.frames = list(planner.plan_subtask(
                0, data, current_position, current_rotation, motion_limits,
                candidate_index=_nominal_candidate_index,
            ))
            self._pending_subtask_planners = [
                lambda live_data, position, rotation, live_limits, i=i, **kwargs:
                planner.plan_subtask(
                    i, live_data, position, rotation, live_limits, **kwargs,
                )
                for i in range(1, len(planner.order) - 1)
            ]
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name == ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK:
            if self.handles is None:
                raise RuntimeError(
                    "Round-handle pot sequence requires explicit robot handles",
                )
            planner = RoundHandlePotPearSequence(self.model, self.handles)
            first_subtask = planner.plan_first_subtask(
                data,
                current_position,
                current_rotation,
                motion_limits,
            )
            self.frames = list(first_subtask)
            self._pending_subtask_planners = [
                planner.plan_pear_subtask,
                planner.plan_close_subtask,
            ]
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        if task_name in POT_LID_TASKS:
            if self.handles is None:
                raise RuntimeError(f"{scene_name} requires explicit robot handles")
            lid_kind, close_pot = pot_lid_task_spec(task_name)
            self.frames = list(ManipulatePotLid(
                self.model,
                self.handles,
                close_pot=close_pot,
                expected_kind=lid_kind,
            ).plan(
                data,
                current_position,
                current_rotation,
                motion_limits,
                candidate_index=_nominal_candidate_index,
            ))
            self._compile_actions(current_position, current_rotation)
            self.index = 0
            self._gate_ticks = 0
            self._trajectory_credit = 0.0
            self._last_trajectory_command = None
            self._last_gripper_closed = False
            self.active = True
            self.status = (
                f"Preset task: running 1/{len(self.frames)}"
                f" - {self.frames[0].phase}"
            )
            return
        geometry_plate_scenes = {
            GEOMETRY_PLATE_BOX_UPRIGHT_SCENE,
            GEOMETRY_PLATE_BOX_LYING_SCENE,
            GEOMETRY_PLATE_CUBE_SCENE,
            GEOMETRY_PLATE_BALL_SCENE,
            GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE,
            GEOMETRY_PLATE_CYLINDER_LYING_SCENE,
            TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
        }
        place = None
        if scene_name in geometry_plate_scenes:
            active_objects = tuple(
                name for name in SceneManager.body_names(scene_name)
                if name.startswith("paper_object_")
                and mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name,
                ) >= 0
                and np.any(self.model.geom_contype[np.flatnonzero(
                    self.model.geom_bodyid == mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, name,
                    )
                )] != 0)
            )
            if len(active_objects) != 1:
                raise RuntimeError(
                    f"Expected one active geometry object, found {active_objects}",
                )
            object_body = active_objects[0]
            object_geom = f"{object_body}_geom"
            if scene_name == GEOMETRY_PLATE_BALL_SCENE:
                pick_type = PickSphere
                pick = pick_type(
                    self.model, body_name=object_body, geom_name=object_geom,
                    lift_clearance_m=self._PLATE_TRANSFER_CLEARANCE_M,
                    gripper_envelope=self.gripper_envelope,
                ).plan(data, current_position, current_rotation, motion_limits)
            elif scene_name in (
                GEOMETRY_PLATE_BOX_LYING_SCENE,
                GEOMETRY_PLATE_CYLINDER_LYING_SCENE,
                TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
            ):
                if self.handles is None:
                    raise RuntimeError("Lying-object task requires explicit robot handles")
                object_kind = (
                    "cylinder"
                    if scene_name == GEOMETRY_PLATE_CYLINDER_LYING_SCENE
                    else "box"
                )
                selector = PlaceBlockOnPaper(
                    self.model, self.handles, object_label=object_kind,
                )
                picks = selector._lying_pick_candidates(
                    data,
                    object_body,
                    object_geom,
                    object_kind,
                    current_position,
                    current_rotation,
                    motion_limits,
                )
                plate_planner = PlaceOnPlate(
                    self.model,
                    body_name="tray_khaki_medium",
                    bottom_geom_name="tray_khaki_medium_bottom",
                    top_geom_name="tray_khaki_medium_bottom",
                    obstacle_body_names=SceneManager.body_names(scene_name),
                    approach_clearance_m=self._PLATE_TRANSFER_CLEARANCE_M,
                    interior_site_name="tray_khaki_medium_interior_site",
                    receptacle_label="tray",
                )
                if _nominal_candidate_index is None:
                    pick, place = selector.select_lying_pick_for_suffix(
                        data, picks, lambda candidate: plate_planner.plan(data, candidate, motion_limits))
                else:
                    pick = picks[_nominal_candidate_index]
                    place = plate_planner.plan(data, pick, motion_limits)
            else:
                pick_type = (
                    PickCylinder
                    if scene_name == GEOMETRY_PLATE_CYLINDER_UPRIGHT_SCENE
                    else PickBox
                )
                pick = pick_type(
                    self.model,
                    body_name=object_body,
                    geom_name=object_geom,
                    lift_clearance_m=self._PLATE_TRANSFER_CLEARANCE_M,
                    gripper_envelope=self.gripper_envelope,
                    **(
                        {"object_label": "cube"}
                        if scene_name == GEOMETRY_PLATE_CUBE_SCENE else {}
                    ),
                ).plan(data, current_position, current_rotation, motion_limits,
                       **({"grasp_index": _nominal_candidate_index} if pick_type is PickBox else {}))
        else:
            pick_type, object_body, object_geom = self._PICK_SPECS[task_name]
            pick = pick_type(
                self.model,
                body_name=object_body,
                geom_name=object_geom,
                lift_clearance_m=self._PLATE_TRANSFER_CLEARANCE_M,
                gripper_envelope=self.gripper_envelope,
                **({"object_label": "cube"} if task_name == CUBE_TO_PLATE_TASK else {}),
            ).plan(
                data, current_position, current_rotation, motion_limits,
            )
        if place is None:
            geometry_destination = scene_name in geometry_plate_scenes
            place = PlaceOnPlate(
                self.model,
                body_name=(
                    "tray_khaki_medium" if geometry_destination
                    else "teleop_plate"
                ),
                bottom_geom_name=(
                    "tray_khaki_medium_bottom" if geometry_destination
                    else "teleop_plate_bottom"
                ),
                top_geom_name=(
                    "tray_khaki_medium_bottom" if geometry_destination
                    else "teleop_plate_top"
                ),
                obstacle_body_names=SceneManager.body_names(scene_name),
                approach_clearance_m=self._PLATE_TRANSFER_CLEARANCE_M,
                interior_site_name=(
                    "tray_khaki_medium_interior_site"
                    if geometry_destination else None
                ),
                receptacle_label=(
                    "tray" if geometry_destination else "plate"
                ),
            ).plan(
                data, pick, motion_limits,
            )
        self.frames = [*pick.frames, *place]
        self._compile_actions(current_position, current_rotation)
        self.index = 0
        self._gate_ticks = 0
        self._trajectory_credit = 0.0
        self._last_trajectory_command = None
        self._last_gripper_closed = False
        self.active = True
        self.status = f"Preset task: running 1/{len(self.frames)} - {self.frames[0].phase}"

    def _compile_actions(
        self,
        initial_position: np.ndarray,
        initial_rotation: np.ndarray,
    ) -> None:
        """Drop the opening dwell and compile base-frame 6+1 actions."""
        initial_position = np.asarray(initial_position, dtype=float)
        initial_rotation = np.asarray(initial_rotation, dtype=float)
        noise = getattr(self, "perturbations", None)
        if noise is not None:
            self.frames = perturb_task_frames(
                self.frames, initial_position, initial_rotation, noise,
                task=self._perturbation_task, subtask_index=self.subtask_index,
                grasp_floor=self._grasp_floor,
            )
        while self.frames:
            frame = self.frames[0]
            stationary = (
                np.linalg.norm(frame.position - initial_position) <= 1e-12
                and np.linalg.norm(
                    rotation_vector(initial_rotation.T @ frame.rotation),
                ) <= 1e-12
            )
            if frame.gripper_closed or not stationary:
                break
            self.frames.pop(0)
        if not self.frames:
            raise RuntimeError("Preset task contains no motion after its opening dwell")
        if (
            getattr(self, "_end_at_release", False)
            and not getattr(self, "_pending_subtask_planners", ())
        ):
            self.frames = _end_after_final_release(self.frames)
        self.actions = compile_task_actions(
            self.frames, initial_position, initial_rotation,
        )

    @property
    def needs_subtask_plan(self) -> bool:
        """Whether the current segment ended and a live successor is pending."""
        return bool(
            self.active
            and self._pending_subtask_planners
            and self.index >= len(self.actions)
        )

    def _configure_prepared_successors(self, task_name):
        """Prepared UI references must retain live, deferred task semantics."""
        if task_name in FRUIT_TRAY_SEQUENCE_TASK_SPECS:
            planner = PlaceFruitSequence(self.model, self.handles, task_name)
            self._pending_subtask_planners = [
                lambda d, p, r, limits, i=i, **kwargs: planner.plan_subtask(i, d, p, r, limits, **kwargs)
                for i in range(1, len(planner.order))]
        elif task_name in COLOR_BLOCK_STACK_TASKS:
            planner = StackColoredBlocks(self.model, self.handles, task_name)
            self._pending_subtask_planners = [
                lambda d, p, r, limits, i=i, **kwargs: planner.plan_subtask(i, d, p, r, limits, **kwargs)
                for i in range(1, len(planner.order) - 1)]
        elif task_name == ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK:
            planner = RoundHandlePotPearSequence(self.model, self.handles)
            self._pending_subtask_planners = [planner.plan_pear_subtask, planner.plan_close_subtask]
        elif task_name == BALL_PLATE_THEN_BLOCK_PAPER_TASK:
            self._pending_subtask_planners = [PlaceBallThenBlockOnPaper(self.model, self.handles).plan_second_subtask]

    def start_at_subtask(self, task, scene, data, position, rotation, *, subtask_index,
                         episode_seed, perturbations, domain_randomization_enabled,
                         nominal_validation_options=None, nominal_cache_only=False):
        """Prepare one live successor using the same boundary-time planner."""
        if not supports_nominal_selection(scene, task):
            raise ValueError("Subtask nominal preparation requires an eligible training task")
        self.cancel()
        self._configure_prepared_successors(task)
        if not 1 <= subtask_index <= len(self._pending_subtask_planners):
            raise ValueError("Invalid deferred nominal subtask index")
        self._pending_subtask_planners = self._pending_subtask_planners[subtask_index - 1:]
        self.subtask_index = subtask_index - 1
        self._nominal_configuration = dict(scene=scene, task=task, episode_seed=episode_seed,
            enabled=True, options=nominal_validation_options,
            domain_randomization_enabled=domain_randomization_enabled, cache_only=nominal_cache_only)
        self._perturbation_task = task
        self._end_at_release = (scene, task) in _training_runtime_tasks()
        self.perturbations = EpisodePerturbations(episode_seed, dict(perturbations))
        mujoco.mj_forward(self.model, data)
        self._grasp_floor = grasp_height_floor(self.model, data, self.handles) if episode_seed != -1 else lambda r: -math.inf
        self.active = True
        self.plan_next_subtask(data, position, rotation)

    def adopt_next_subtask_plan(self, prepared, data, position, rotation):
        """Adopt background references without re-solving or re-perturbing."""
        configuration = self._nominal_configuration or {}
        if (not self.needs_subtask_plan or prepared.subtask_index != self.subtask_index + 1
                or prepared.scene != configuration.get('scene') or prepared.task != configuration.get('task')
                or prepared.robot_uid != self.handles.spec.uid
                or not np.array_equal(prepared.initial_qpos, data.qpos) or not prepared.frames):
            raise ValueError("Prepared subtask does not match the current boundary/state")
        self._pending_subtask_planners.pop(0)
        self.subtask_index = prepared.subtask_index
        self.frames = list(prepared.frames)
        self.actions = compile_task_actions(self.frames, position, rotation)
        self.perturbations = EpisodePerturbations(configuration['episode_seed'], dict(prepared.perturbations))
        self.last_planning_stats.setdefault('nominal_plan_tables', []).extend(prepared.planning_stats.get('nominal_plan_tables', []))
        self.index = self._gate_ticks = 0
        self._trajectory_credit = 0.
        self._last_trajectory_command = None
        self.status = f"Preset task: subtask {self.subtask_index + 1}, running 1/{len(self.frames)}"

    def plan_next_subtask(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
    ) -> None:
        """Build the next segment at its boundary from the current nominal EE pose."""
        if not self.needs_subtask_plan:
            raise RuntimeError("No subtask is waiting for boundary-time planning")
        planner = self._pending_subtask_planners[0]
        mujoco.mj_forward(self.model, data)
        configuration = self._nominal_configuration or {}
        next_index = self.subtask_index + 1
        candidates = (nominal_candidate_definitions(self.model, data,
            configuration['scene'], configuration['task'], next_index)
            if configuration.get('enabled') else [])
        samples = dict(self.perturbations.samples)

        def build_candidate(index=None):
            self.perturbations = EpisodePerturbations(self.perturbations.seed, dict(samples))
            self.frames = list(planner(data, np.asarray(current_position).copy(),
                np.asarray(current_rotation).copy(), self._LIMITS,
                **({"candidate_index": index} if index is not None else {})))
            self.subtask_index = next_index
            # Preserve intermediate retraction; trim only the final segment.
            pending = self._pending_subtask_planners
            self._pending_subtask_planners = pending[1:]
            try:
                self._compile_actions(current_position, current_rotation)
            finally:
                self._pending_subtask_planners = pending
            return dict(frames=tuple(self.frames), perturbations=dict(self.perturbations.samples))

        if candidates:
            from ...simulation.nominal_plan_table import select_nominal_candidate
            selected, table = select_nominal_candidate(self.model, data, self.handles,
                **{key: configuration[key] for key in ('scene', 'task', 'episode_seed',
                   'options', 'domain_randomization_enabled', 'cache_only')},
                candidates=candidates, build_candidate=build_candidate,
                planning_conditions=dict(motion_limits=asdict(self._LIMITS), subtask_index=next_index))
            try:
                build_candidate(selected)
            except PlanningInfeasibleError as error:
                error.context["nominal_plan_table"] = table
                raise
            self.last_planning_stats.setdefault('nominal_plan_tables', []).append(dict(table, subtask_index=next_index))
        else:
            build_candidate()
        self._pending_subtask_planners.pop(0)
        self.index = 0
        self._gate_ticks = 0
        self._trajectory_credit = 0.0
        self._last_trajectory_command = None
        self.status = (
            f"Preset task: subtask {self.subtask_index + 1}, "
            f"running 1/{len(self.frames)} - {self.frames[0].phase}"
        )

    def command(
        self,
        actual_position: np.ndarray,
        actual_rotation: np.ndarray,
        active_dofs: np.ndarray | None = None,
        tolerance_rotation: np.ndarray | None = None,
    ) -> TaskCommand | None:
        if not self.active or not self.frames:
            return None
        constrained = np.ones(6, dtype=bool) if active_dofs is None else np.asarray(active_dofs, dtype=bool)
        if constrained.shape != (6,):
            raise ValueError("active_dofs must contain six Cartesian-axis flags")
        frame = self.frames[self.index]
        frame_rotation = frame.rotation if tolerance_rotation is None else project_rotation_target(frame.rotation, tolerance_rotation, constrained[3:])
        position_error = float(np.linalg.norm((frame.position - actual_position)[constrained[:3]]))
        if tolerance_rotation is None:
            rotation_components = rotation_vector(frame_rotation.T @ actual_rotation)
        else:
            rotation_components = rotation_error_in_frame(actual_rotation, frame_rotation, tolerance_rotation)
        rotation_error = float(np.linalg.norm(rotation_components[constrained[3:]]))
        reached = position_error <= self._POSITION_GATE and rotation_error <= self._ROTATION_GATE
        self._gate_ticks += 1
        if reached or self._gate_ticks >= self._MAX_GATE_TICKS:
            self.index += 1
            self._gate_ticks = 0
            if self.index >= len(self.frames):
                self.active = False
                self.status = "Preset task: completed"
                return TaskCommand(frame.position.copy(), frame_rotation.copy(), frame.gripper_closed, self.status)
            frame = self.frames[self.index]
            frame_rotation = frame.rotation if tolerance_rotation is None else project_rotation_target(frame.rotation, tolerance_rotation, constrained[3:])
        self.status = f"Preset task: running {self.index + 1}/{len(self.frames)} - {frame.phase}"
        return TaskCommand(frame.position.copy(), frame_rotation.copy(), frame.gripper_closed, self.status)

    def trajectory_command(self) -> TaskCommand | None:
        """Return exactly one precomputed 10 Hz trajectory frame.

        Unlike :meth:`command`, this path is deliberately independent of the
        measured robot state.  Dataset actions therefore describe the scripted
        expert trajectory rather than feedback-paced controller holds.
        """
        if not self.active or not self.frames:
            return None
        frame = self.frames[self.index]
        self.index += 1
        if self.index >= len(self.frames):
            self.active = False
            self.status = "Preset task: completed"
        else:
            next_frame = self.frames[self.index]
            self.status = (
                f"Preset task: running {self.index + 1}/{len(self.frames)}"
                f" - {next_frame.phase}"
            )
        return TaskCommand(
            frame.position.copy(), frame.rotation.copy(),
            frame.gripper_closed, self.status,
        )

    def trajectory_action(self) -> TaskAction | None:
        """Return one precomputed 10 Hz 6+1 incremental action."""
        if not self.active or not self.actions or self.index >= len(self.actions):
            return None
        action = self.actions[self.index]
        self.index += 1
        self._last_gripper_closed = action.gripper_closed
        if self.index >= len(self.actions):
            if self._pending_subtask_planners:
                self.status = "Preset task: waiting to plan next subtask"
            else:
                self.active = False
                self.status = "Preset task: completed"
        else:
            next_action = self.actions[self.index]
            self.status = (
                f"Preset task: running {self.index + 1}/{len(self.actions)}"
                f" - {next_action.phase}"
            )
        return TaskAction(
            action.value, action.phase, self.status,
        )

    def peek_trajectory_actions(
        self, limit: int = 10,
    ) -> tuple[TaskAction, ...]:
        """Read future 10 Hz actions without advancing the live trajectory."""
        if limit <= 0 or not self.active or not self.actions:
            return ()
        end = self.index + limit
        return tuple(self.actions[self.index:end])

    def peek_rate_scaled_trajectory_actions(
        self,
        speed_scale: float,
        limit: int = 10,
    ) -> tuple[TaskAction, ...]:
        """Preview future controller-tick actions without consuming them."""
        speed_scale = float(speed_scale)
        if (
            limit <= 0
            or not np.isfinite(speed_scale)
            or speed_scale <= 0.0
            or not self.active
            or not self.actions
        ):
            return ()
        index = self.index
        credit = self._trajectory_credit
        last_closed = self._last_gripper_closed
        preview: list[TaskAction] = []
        for _ in range(limit):
            if index >= len(self.actions):
                break
            if index == 0:
                action = self.actions[index]
                index += 1
                last_closed = action.gripper_closed
                preview.append(action)
                continue
            credit += speed_scale
            steps = int(math.floor(credit + 1e-12))
            if steps <= 0:
                previous = self.actions[max(0, index - 1)]
                preview.append(TaskAction(
                    np.array((
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        1.0 if last_closed else -1.0,
                    )),
                    previous.phase,
                    "Preset task: preview",
                ))
                continue
            consumed = list(
                self.actions[index:min(
                    index + steps,
                    len(self.actions),
                )]
            )
            if not consumed:
                break
            index += len(consumed)
            credit -= len(consumed)
            last_closed = consumed[-1].gripper_closed
            preview.append(TaskAction(
                self._merge_actions(consumed),
                consumed[-1].phase,
                "Preset task: preview",
            ))
        return tuple(preview)

    @staticmethod
    def _merge_actions(actions: list[TaskAction]) -> np.ndarray:
        translation = np.sum(
            [action.value[:3] for action in actions], axis=0,
        )
        rotation = np.eye(3)
        for action in actions:
            delta = rotation_matrix(action.value[3:6])
            rotation = delta @ rotation
        return np.concatenate((
            translation,
            rotation_vector(rotation),
            np.array((actions[-1].value[6],)),
        ))

    def rate_scaled_trajectory_action(
        self,
        speed_scale: float,
    ) -> TaskAction | None:
        """Stream the offline action plan using a fractional live clock."""
        speed_scale = float(speed_scale)
        if not np.isfinite(speed_scale) or speed_scale <= 0.0:
            raise ValueError("Trajectory speed scale must be positive and finite")
        if not self.active or not self.actions:
            return None

        if self.index == 0:
            return self.trajectory_action()

        self._trajectory_credit += speed_scale
        steps = int(math.floor(self._trajectory_credit + 1e-12))
        if steps <= 0:
            return TaskAction(
                np.array((
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0 if self._last_gripper_closed else -1.0,
                )),
                self.actions[max(0, self.index - 1)].phase,
                self.status,
            )

        consumed: list[TaskAction] = []
        while (
            len(consumed) < steps
            and self.active
            and not self.needs_subtask_plan
        ):
            action = self.trajectory_action()
            if action is None:
                break
            consumed.append(action)
        self._trajectory_credit -= len(consumed)
        if not consumed:
            return None
        return TaskAction(
            self._merge_actions(consumed),
            consumed[-1].phase,
            self.status,
        )

    def first_moving_trajectory_action(
        self,
        atol: float = 1e-12,
    ) -> TaskAction | None:
        """Skip leading pose holds and return the first moving 6+1 action."""
        while self.active:
            action = self.trajectory_action()
            assert action is not None
            if np.linalg.norm(action.value[:6]) > atol:
                return action
        return None

    def rate_scaled_trajectory_command(
        self,
        speed_scale: float,
    ) -> TaskCommand | None:
        """Stream the 10 Hz plan using a live dimensionless playback rate."""
        speed_scale = float(speed_scale)
        if not np.isfinite(speed_scale) or speed_scale <= 0.0:
            raise ValueError("Trajectory speed scale must be positive and finite")
        if not self.active or not self.frames:
            return None

        # Present frame zero exactly once. Later calls advance a fractional
        # trajectory clock, so runtime speed edits need no plan rebuild.
        if self._last_trajectory_command is None:
            command = self.trajectory_command()
            assert command is not None
            self._last_trajectory_command = command
            return command

        self._trajectory_credit += speed_scale
        steps = int(math.floor(self._trajectory_credit + 1e-12))
        if steps <= 0:
            command = self._last_trajectory_command
            return TaskCommand(
                command.position.copy(),
                command.rotation.copy(),
                command.gripper_closed,
                self.status,
            )

        consumed = 0
        command = self._last_trajectory_command
        while consumed < steps and self.active:
            next_command = self.trajectory_command()
            assert next_command is not None
            command = next_command
            consumed += 1
        self._trajectory_credit -= consumed
        self._last_trajectory_command = command
        return command

    def first_moving_trajectory_command(
        self,
        reference_position: np.ndarray,
        reference_rotation: np.ndarray,
        atol: float = 1e-12,
    ) -> TaskCommand | None:
        """Skip leading pose holds and return the first Cartesian motion."""
        while self.active:
            command = self.trajectory_command()
            assert command is not None
            translation = command.position - reference_position
            rotation = rotation_vector(reference_rotation.T @ command.rotation)
            if np.linalg.norm(translation) > atol or np.linalg.norm(rotation) > atol:
                return command
        return None


__all__ = (
    "compile_task_actions", "PresetTaskExecutor", "TaskAction",
    "TaskCommand", "TaskFrame",
)
