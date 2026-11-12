"""Shared live/headless rotational-tolerance control operations."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
from typing import Hashable
import time
import numpy as np

from ..control.ipopt import IpoptIK
from ..control.pose import TargetPose
from ..control.task_space.types import AxisTask, TaskKind
from ..geometry import rotation_matrix
from .frame import (
    TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
    TOLERANCE_FRAME_MODE_STAGE_ENDPOINT,
    TOLERANCE_FRAME_MODES,
    cycle_tolerance_frame,
)
from .release_state import RotationReleaseState
from .state import RotationalToleranceState, STRICT_ROTATION_STAGES


def advance_reference_pose(
    position: np.ndarray,
    rotation: np.ndarray,
    action: np.ndarray,
    *,
    tolerance: RotationalToleranceState | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the independent nominal planner reference by one increment.

    Rotation is unconditionally left-multiplied. Absolute planner frames and
    source-frame switches are intentionally absent from this API. Runtime
    control, simulation and benchmarks must all integrate the same incremental
    action from the previous nominal reference. When tolerance is active,
    this records rotational intent before applying the nominal increment; the
    physical FK feedback remains an observation input and never replaces this
    nominal reference.
    """
    position_array = np.asarray(position, dtype=float)
    rotation_array = np.asarray(rotation, dtype=float)
    action_array = np.asarray(action, dtype=float)
    if position_array.shape != (3,) or rotation_array.shape != (3, 3):
        raise ValueError("Reference pose must contain a 3-vector and 3x3 rotation")
    if action_array.shape not in ((6,), (7,)) or not np.all(np.isfinite(action_array)):
        raise ValueError("Reference action must contain six or seven finite values")
    rotation_action = action_array[3:6]
    if tolerance is not None:
        if dt is None or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("A positive finite dt is required with tolerance")
        tolerance.update_intent(rotation_action, float(dt))
        advanced_rotation = tolerance.apply_increment(
            rotation_array, rotation_action,
        )
    else:
        delta = rotation_matrix(rotation_action)
        advanced_rotation = delta @ rotation_array
    return position_array + action_array[:3], advanced_rotation


def configure_rotation_axis_tasks(
    tolerance: RotationalToleranceState,
    controllers: Iterable[object],
) -> tuple[tuple[AxisTask, AxisTask, AxisTask], np.ndarray]:
    """Install requested tasks and derive their requested release mask."""
    tasks = (
        tolerance.task(0),
        tolerance.task(1),
        tolerance.task(2),
    )
    for controller in controllers:
        for axis, task in enumerate(tasks):
            controller.set_axis_task(axis + 3, task)
    release_mask = np.fromiter(
        (task.kind is TaskKind.FLAT_RANGE for task in tasks),
        dtype=bool,
        count=3,
    )
    return tasks, release_mask


def rebase_rotation_subtask(
    target: TargetPose,
    tolerance: RotationalToleranceState,
    release_state: RotationReleaseState,
    *,
    stage_key: Hashable,
    accepted_rotation: np.ndarray,
) -> np.ndarray:
    """Atomically make an accepted optimized rotation the new nominal zero.

    Every simulation entry point must use this operation at a real subtask
    boundary. It resets the action-integrated nominal branch, the persistent
    release coordinates, and all stage-local tolerance filter/intent history
    as one state transition.
    """
    anchor = np.asarray(accepted_rotation, dtype=float)
    if anchor.shape != (3, 3) or not np.all(np.isfinite(anchor)):
        raise ValueError("Accepted rotation must be a finite 3x3 matrix")
    target.rotation = anchor.copy()
    tolerance.reset_subtask_history()
    release_state.rebase_subtask(stage_key, anchor)
    return anchor.copy()


def preset_stage_lookahead(executor, speed_scale: float, phase, horizon: int):
    """Preview stable commands without predicting a future observed phase.

    Finger settling during Grasp/Release is unknown until observed, so those
    stages execute one control interval at a time. Stable stages latch until
    the next gripper command edge; action labels have no execution semantics.
    """
    from ..tasks.phases import ManipulationPhase
    if phase in (ManipulationPhase.GRASP, ManipulationPhase.RELEASE):
        return ()
    closed = phase is ManipulationPhase.POSTGRASP
    preview = executor.peek_rate_scaled_trajectory_actions(speed_scale, limit=horizon - 1)
    actions = []
    for action in preview:
        if action.gripper_closed != closed:
            break
        actions.append(action.value)
    return tuple(actions)


def solve_stage_relative_target(
    controller: IpoptIK,
    data: object,
    target: TargetPose,
    tolerance: RotationalToleranceState,
    release_state: RotationReleaseState,
    active_dofs: np.ndarray,
    *,
    tolerance_frame_mode: str = TOLERANCE_FRAME_MODE_STAGE_ENDPOINT,
    future_actions: tuple[np.ndarray, ...] = (),
):
    """Execute one canonical stage-relative rotational IPOPT cycle.

    This is the single backend used by interactive, benchmark, and
    counterfactual simulation. It owns tolerance-frame branch resolution,
    task installation, the frozen release snapshot, the solve, and the
    feasible-state commit. Real-time mode constructs the frame directly from
    the current nominal target without descriptor EMA.
    """
    started = time.perf_counter()
    if tolerance_frame_mode not in TOLERANCE_FRAME_MODES:
        raise ValueError(
            f"Unknown tolerance-frame mode: {tolerance_frame_mode!r}",
        )
    if (release_state.maximum_release_coordinate_step_rad is not None or release_state.so3_preprocessor is not None) and (
        future_actions or getattr(controller, "has_pending_block", False)
    ):
        raise ValueError("Prediction boundary governor currently requires single-step control")
    if tolerance_frame_mode == TOLERANCE_FRAME_MODE_REALTIME_NOMINAL:
        tolerance.frame = cycle_tolerance_frame(
            tolerance_frame_mode,
            target.rotation,
            tolerance.raw_tolerance_frame,
        )
    rotation_tasks, release_mask = configure_rotation_axis_tasks(
        tolerance, (controller,),
    )
    active_dofs[:3] = True
    active_dofs[3:] = ~release_mask
    use_governor = (
        (release_state.maximum_release_coordinate_step_rad is not None or release_state.so3_preprocessor is not None)
        and tolerance.active_stage not in STRICT_ROTATION_STAGES
    )
    # A state-aware boundary governor replaces magnitude EMA for this opt-in
    # path: its target is the latest prediction, rather than yesterday's box.
    negative = tolerance.raw_negative_limits if use_governor else tolerance.negative_limits
    positive = tolerance.raw_positive_limits if use_governor else tolerance.positive_limits
    release_step = release_state.prepare_cycle(
        target.rotation,
        tolerance.frame,
        release_mask,
        -negative,
        positive,
        allow_boundary_governor=tolerance.active_stage not in STRICT_ROTATION_STAGES,
    )
    release_mask = release_step.mask.copy()
    active_dofs[3:] = ~release_mask
    if use_governor:
        # Constraint installation, objective metadata, and the returned mask
        # must all describe the same final box, including temporarily relaxed
        # axes whose newest prediction was exactly zero.
        rotation_tasks = tuple(
            replace(
                rotation_tasks[axis], kind=TaskKind.FLAT_RANGE,
                lower=float(release_step.effective_lower_limits[axis]),
                upper=float(release_step.effective_upper_limits[axis]),
                ema_nominal_action=float(tolerance.nominal_action_ema[axis]),
            ) if release_mask[axis] else AxisTask(TaskKind.SPECIFIC, goal=0.)
            for axis in range(3)
        )
        for axis, task in enumerate(rotation_tasks):
            controller.set_axis_task(axis + 3, task)
    solve_options = {}
    if future_actions and not controller.has_pending_block:
        from ..config import TELEOP_DT
        preview_tolerance = deepcopy(tolerance)
        targets = [TargetPose(target.position.copy(), target.rotation.copy())]
        tasks = [tuple(controller._resolve_axis_tasks(active_dofs))]
        steps = [release_step]
        position, rotation = target.position.copy(), target.rotation.copy()
        for action in future_actions:
            position, rotation = advance_reference_pose(
                position, rotation, action, tolerance=preview_tolerance, dt=TELEOP_DT,
            )
            frame = cycle_tolerance_frame(
                tolerance_frame_mode, rotation, preview_tolerance.frame,
            )
            preview_tolerance.frame = frame
            _, mask = configure_rotation_axis_tasks(preview_tolerance, ())
            targets.append(TargetPose(position.copy(), rotation.copy()))
            tasks.append(tuple(controller._resolve_axis_tasks(active_dofs)[:3]) + tuple(
                preview_tolerance.task(axis) for axis in range(3)
            ))
            steps.append(release_state.prepare_cycle(
                rotation, frame, mask, -preview_tolerance.negative_limits,
                preview_tolerance.positive_limits,
            ))
        solve_options = dict(
            horizon_targets=tuple(targets), horizon_tasks=tuple(tasks),
            horizon_release_steps=tuple(steps),
        )
    candidate_angle = None
    budget_exceeded = None
    if release_step.so3_result is not None:
        from .so3_preprocessor import rotation_distance
        settings = release_state.so3_preprocessor
        def accept_candidate(q):
            nonlocal candidate_angle, budget_exceeded
            rotation = controller.kinematics.evaluate(
                q, include_manipulability=False, include_link_points=False,
            ).rotation
            candidate_angle = rotation_distance(rotation, release_step.previous_accepted_rotation)
            budget_exceeded = bool(candidate_angle > settings.maximum_step_rad + settings.publication_tolerance_rad)
            if not release_step.so3_result.usable:
                return 'so3_preprocessing:' + release_step.so3_result.status
            if settings.acceptance_mode == 'reject' and budget_exceeded:
                return 'so3_command_budget_exceeded'
            return None
        solve_options['candidate_acceptance'] = accept_candidate
    joint_target, diagnostics = controller.solve(
        data, target, active_dofs, tolerance.frame,
        rotation_release_step=release_step, **solve_options,
    )
    if diagnostics.feasible:
        release_state.commit(joint_target, release_step)
    if release_step.so3_result is not None:
        diagnostics = replace(diagnostics, candidate_rotation_step_rad=candidate_angle,
                              so3_budget_exceeded=budget_exceeded,
                              control_wall_ms=(time.perf_counter()-started)*1000.)
        if not release_step.so3_result.usable and diagnostics.prepublication_rejection_reason is None:
            reason = 'so3_preprocessing:' + release_step.so3_result.status
            diagnostics = replace(diagnostics, prepublication_rejection_reason=reason,
                                  status=f'{diagnostics.status}; {reason}')
    return (
        joint_target,
        diagnostics,
        rotation_tasks,
        release_mask,
        release_step,
    )


__all__ = (
    "advance_reference_pose",
    "configure_rotation_axis_tasks",
    "rebase_rotation_subtask",
    "solve_stage_relative_target",
    "preset_stage_lookahead",
)
