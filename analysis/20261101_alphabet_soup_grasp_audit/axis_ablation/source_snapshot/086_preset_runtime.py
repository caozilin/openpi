"""Canonical execution backend for every preset simulation entry point."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from ..config import TELEOP_DT
from ..control.joint_tracker import JointReferenceTracker, JointTrackingDiagnostics
from ..control.ipopt import MAX_CONSECUTIVE_SOLVER_FAILURES
from ..tasks.evaluation.types import TaskEvaluator
from ..tasks.phases import PhaseClassifier, PhaseObservation
from .robot import RobotHandles


@dataclass(frozen=True)
class PresetPhysicsStep:
    """State produced by one canonical simulator physics step."""

    joint_command: np.ndarray
    gripper_ctrl: float
    interpolation_elapsed: float
    interpolation_alpha: float
    tracking: JointTrackingDiagnostics
    aborted: bool


@dataclass(frozen=True)
class PresetControlInterval:
    """State produced by one complete planner/controller interval."""

    joint_command: np.ndarray
    gripper_ctrl: float
    interpolation_elapsed: float
    physics_steps_executed: int
    aborted: bool


class PresetSimulationBackend:
    """The sole physics/controller/evaluator backend for preset tasks.

    Interactive UI code calls :meth:`step_physics` at the display loop rate.
    Offline collection, annotation, video, and benchmark code call
    :meth:`run_control_interval`. Both paths therefore use exactly the same
    interpolation, tracking controller, gripper actuation, MuJoCo step, and
    evaluator hooks.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        handles: RobotHandles,
    ) -> None:
        self.model = model
        self.data = data
        self.handles = handles
        self.phase_classifier = PhaseClassifier(model, handles)

    def reset_manipulation_phase(self) -> None:
        """Start a new command stream without carrying phase history."""
        self.phase_classifier.reset()

    def observe_manipulation_phase(self, gripper_closed: bool) -> PhaseObservation:
        """Classify once before each 10 Hz solve from current state and command.

        Presets and model rollouts use the same causal inputs. Planned action
        labels, stored phase labels, and future commands never override this
        observation. Physics stepping does not sample again.
        """
        return self.phase_classifier.update_from_data(self.data, gripper_closed)

    @staticmethod
    def optimizer_should_abort(diagnostics) -> bool:
        """Use one failure rule for interactive and offline presets."""
        return bool(
            (diagnostics.controller_mode == "ipopt_block" and not diagnostics.feasible)
            or diagnostics.consecutive_failures >= MAX_CONSECUTIVE_SOLVER_FAILURES
        )

    def settle(self, duration_s: float) -> None:
        """Advance an uncontrolled scene-settling interval."""
        if not np.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("settling duration must be finite and non-negative")
        steps = max(1, int(round(duration_s / float(self.model.opt.timestep))))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    @staticmethod
    def track_joint_reference(
        *,
        tracker: JointReferenceTracker,
        interpolation_start: np.ndarray,
        joint_target: np.ndarray,
        interpolation_elapsed: float,
        interpolation_duration_s: float,
        physics_dt: float,
        actual_q: np.ndarray,
        actual_velocity_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, JointTrackingDiagnostics, float, float]:
        """Shared interpolation and tracking calculation; never steps physics.

        The canonical simulator and native LIBERO actuator adapter call this
        same calculation at their own physical timestep. The outer command
        interval is explicit, so 20Hz LIBERO does not acquire a 100ms ramp.
        """
        if not np.isfinite(interpolation_duration_s) or interpolation_duration_s <= 0:
            raise ValueError("Interpolation duration must be finite and positive")
        elapsed = min(interpolation_duration_s, interpolation_elapsed + physics_dt)
        alpha = elapsed / interpolation_duration_s
        reference = (1.0 - alpha) * interpolation_start + alpha * joint_target
        command, tracking = tracker.update(reference, actual_q, physics_dt, actual_velocity_q)
        return reference, command, tracking, elapsed, alpha

    def step_physics(
        self,
        *,
        tracker: JointReferenceTracker,
        joint_command: np.ndarray,
        interpolation_start: np.ndarray,
        joint_target: np.ndarray,
        interpolation_elapsed: float,
        gripper_ctrl: float,
        gripper_closed: bool,
        gripper_speed: float,
        speed_scale: float,
        evaluator: TaskEvaluator | None = None,
        frame_index: int = 0,
        tracking_observer: Callable[[JointTrackingDiagnostics], None] | None = None,
    ) -> PresetPhysicsStep:
        """Apply one canonical interpolation, control, physics, and evaluation step."""
        physics_dt = float(self.model.opt.timestep)
        joint_command, actuator_command, tracking, interpolation_elapsed, alpha = self.track_joint_reference(
            tracker=tracker,
            interpolation_start=interpolation_start,
            joint_target=joint_target,
            interpolation_elapsed=interpolation_elapsed,
            interpolation_duration_s=TELEOP_DT,
            physics_dt=physics_dt,
            actual_q=self.data.qpos[self.handles.qpos_ids],
            actual_velocity_q=self.data.qvel[self.handles.dof_ids],
        )
        if tracking_observer is not None:
            tracking_observer(tracking)
        self.data.ctrl[self.handles.actuator_ids] = actuator_command
        # Position servos otherwise need a persistent joint error to support
        # gravity. Feed forward the current gravity/Coriolis/centrifugal bias
        # at the physics rate; the servos and tracker correct residual error.
        # Only arm DOFs are owned here, leaving gripper/object forces intact.
        self.data.qfrc_applied[self.handles.dof_ids] = (
            self.data.qfrc_bias[self.handles.dof_ids]
        )
        gripper_goal = (
            self.handles.spec.gripper_closed_ctrl
            if gripper_closed else self.handles.spec.gripper_open_ctrl
        )
        max_step = gripper_speed * physics_dt * speed_scale
        gripper_ctrl += float(np.clip(
            gripper_goal - gripper_ctrl, -max_step, max_step,
        ))
        self.data.ctrl[self.handles.gripper_id] = gripper_ctrl
        mujoco.mj_step(self.model, self.data)

        aborted = False
        if evaluator is not None:
            evaluator.observe_contacts(
                self.data, frame_index, float(self.data.time),
            )
            aborted = bool(evaluator.should_abort)
        return PresetPhysicsStep(
            joint_command=joint_command,
            gripper_ctrl=gripper_ctrl,
            interpolation_elapsed=interpolation_elapsed,
            interpolation_alpha=alpha,
            tracking=tracking,
            aborted=aborted,
        )

    def observe_frame(
        self,
        evaluator: TaskEvaluator | None,
        gripper_closed: bool,
        frame_index: int,
    ) -> bool:
        """Run the canonical 10 Hz task-evaluator hook."""
        if evaluator is None:
            return False
        evaluator.observe_frame(
            self.data, gripper_closed, frame_index, float(self.data.time),
        )
        return bool(evaluator.should_abort)

    def run_control_interval(
        self,
        *,
        tracker: JointReferenceTracker,
        joint_command: np.ndarray,
        joint_target: np.ndarray,
        gripper_ctrl: float,
        gripper_closed: bool,
        gripper_speed: float,
        speed_scale: float,
        physics_steps: int,
        evaluator: TaskEvaluator | None = None,
        frame_index: int = 0,
        tracking_observer: Callable[[JointTrackingDiagnostics], None] | None = None,
        physics_observer: Callable[[PresetPhysicsStep], None] | None = None,
    ) -> PresetControlInterval:
        """Execute one complete planner interval through the canonical backend."""
        if physics_steps < 1:
            raise ValueError("physics_steps must be positive")
        interpolation_start = joint_command.copy()
        interpolation_elapsed = 0.0
        executed = 0
        aborted = False
        for _ in range(physics_steps):
            step = self.step_physics(
                tracker=tracker,
                joint_command=joint_command,
                interpolation_start=interpolation_start,
                joint_target=joint_target,
                interpolation_elapsed=interpolation_elapsed,
                gripper_ctrl=gripper_ctrl,
                gripper_closed=gripper_closed,
                gripper_speed=gripper_speed,
                speed_scale=speed_scale,
                evaluator=evaluator,
                frame_index=frame_index,
                tracking_observer=tracking_observer,
            )
            joint_command = step.joint_command
            gripper_ctrl = step.gripper_ctrl
            interpolation_elapsed = step.interpolation_elapsed
            executed += 1
            if physics_observer is not None:
                physics_observer(step)
            if step.aborted:
                aborted = True
                break

        if not aborted:
            aborted = self.observe_frame(evaluator, gripper_closed, frame_index)
        return PresetControlInterval(
            joint_command=joint_command,
            gripper_ctrl=gripper_ctrl,
            interpolation_elapsed=interpolation_elapsed,
            physics_steps_executed=executed,
            aborted=aborted,
        )
