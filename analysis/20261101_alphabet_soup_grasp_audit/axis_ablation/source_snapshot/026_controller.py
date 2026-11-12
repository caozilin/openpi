from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from typing import Callable

import mujoco
import numpy as np

from ..pose import TargetPose
from ..task_space.kinematics import RobotKinematics
from ..task_space.conditioning import (
    conditioning_activation,
    conditioning_penalty,
    conditioning_region,
)
from ..task_space.singularity import SIGMA_MIN_PENALTY_PROFILES
from ..task_space.objective import (
    DEFAULT_EMA_RELEASE_LOSS_WEIGHT,
    MotionState,
    ObjectiveSettings,
    TaskObjective,
    make_controller_objective,
    release_loss_multipliers_for_tasks,
)
from ..task_space.types import (
    AxisTask,
    TaskKind,
    resolve_axis_tasks,
    validate_axis_task,
)
from ...geometry import (
    nominal_rotation_coordinate_jacobian,
)
from ...simulation.robot import RobotHandles
from ...tolerance.release_state import (
    RotationReleaseCandidate,
    RotationReleaseStep,
)

from .adapter import (
    ACCEPTABLE_IPOPT_STATUS_CODES,
    IpoptAdapter,
    IpoptSettings,
)
from .validation import ConstraintValues, validate_candidate


MAX_CONSECUTIVE_SOLVER_FAILURES = 3


@dataclass(frozen=True)
class _SingleStepNLP:
    seed: np.ndarray
    last_command: np.ndarray
    previous: np.ndarray
    previous2: np.ndarray
    tasks: tuple[AxisTask, ...]
    chart_rotation: np.ndarray | None
    objective: TaskObjective
    constraint_evaluation: Callable[
        [np.ndarray],
        tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]],
    ]


@dataclass(frozen=True)
class _PublishedResult:
    q: np.ndarray
    cost: float
    iterations: int
    optimality: float
    elapsed_ms: float
    converged: bool
    constraints: ConstraintValues
    task_costs: dict[str, float]
    status: str


@dataclass(frozen=True)
class IpoptDiagnostics:
    sigma_min_raw: float
    sigma_min_normalized: float
    sigma_max_normalized: float
    sigma_min_gap_normalized: float
    inverse_condition_index: float
    conditioning_enabled: bool
    conditioning_eta_low: float
    conditioning_eta_high: float
    conditioning_penalty: float
    conditioning_activation: float
    conditioning_region: str
    conditioning_active: bool
    manipulability: float | None
    cost: float
    iterations: int
    optimality: float
    elapsed_ms: float
    task_costs: dict[str, float]
    task_penalties: dict[str, float]
    objective_gradient_norms: dict[str, float]
    objective_tangent_gradient_norms: dict[str, float]
    objective_gradient_cancellation_ratio: float
    objective_tangent_gradient_cancellation_ratio: float
    feasible: bool
    converged: bool
    position_residual: float
    strict_rotation_residual: float
    effective_constraint_violation: float
    status: str
    consecutive_failures: int = 0
    conditioning_gradient_norm: float | None = None
    conditioning_gradient_relative_error_half_step: float | None = None
    conditioning_gradient_relative_error_double_step: float | None = None
    conditioning_gradient_cosine_half_step: float | None = None
    conditioning_gradient_cosine_double_step: float | None = None
    conditioning_diagnostic_inverse_condition_index: float | None = None
    conditioning_diagnostic_sigma_min_gap_normalized: float | None = None
    controller_mode: str = "ipopt"
    solver_called: bool = True
    block_knot_index: int | None = None
    block_horizon: int = 1
    block_window_feasible: bool | None = None
    block_first_knot_feasible: bool | None = None
    block_warm_start_used: bool = False
    block_seed_source: str | None = None
    release_loss_multiplier: float = 0.05
    stage_relative_tolerance_release_rad: np.ndarray | None = None
    stage_relative_rotation_error_rad: np.ndarray | None = None
    incremental_release_rad: np.ndarray | None = None
    predicted_tolerance_violation_rad: np.ndarray | None = None
    prepublication_rejection_reason: str | None = None
    candidate_rotation_step_rad: float | None = None
    so3_budget_exceeded: bool | None = None
    control_wall_ms: float | None = None

    @property
    def sigma_min(self) -> float:
        """Current velocity-capability-normalized full-Jacobian diagnostic."""
        return self.sigma_min_normalized


def rotation_release_error_and_jacobian(
    kinematics: RobotKinematics,
    state,
    step: RotationReleaseStep,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total stage-relative coordinates and their analytic Jacobian.

    The maintained IPOPT backend uses this stage-relative chart.
    """
    reference = step.stage_reference_rotation()
    angular_world = state.rotation @ state.jacobian[3:]
    error = step.stage_relative_coordinates(state.rotation)
    jacobian = nominal_rotation_coordinate_jacobian(
        state.rotation, reference, angular_world, step.tolerance_frame,
    )
    return error, jacobian


class IpoptIK:
    """IPOPT inverse kinematics with hard Cartesian constraints.

    Position and Specific rotation are hard equalities; ranged rotation and
    joint limits are hard inequalities. Within that feasible set, the active
    profile uses squared joint motion, conditioning, and nominal-action EMA
    release losses. Acceleration and jerk are evaluation metrics only.
    """

    _RANGED_KINDS = {
        TaskKind.RANGE,
        TaskKind.PREFERRED_RANGE,
        TaskKind.FLAT_RANGE,
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        settings: IpoptSettings = IpoptSettings(),
        objective_settings: ObjectiveSettings | None = None,
        *,
        diagnose_conditioning_gradient: bool = False,
        block_warm_start: bool = True,
    ) -> None:
        self.block_warm_start = bool(block_warm_start)
        self.model = model
        self.handles = handles
        self.kinematics = RobotKinematics(model, handles)
        self.settings = settings
        self.objective_settings = (
            objective_settings
            or ObjectiveSettings.delta_q_squared_conditioning_safeguard()
        )
        self.diagnose_conditioning_gradient = bool(
            diagnose_conditioning_gradient
        )
        self._conditioning_config = replace(
            self.objective_settings.conditioning,
            enabled=True,
        )
        quadratic_initial = (
            self.objective_settings.squared_delta_q_weight != 0.0
            or self.objective_settings.sigma_min_weight != 0.0
            or self.objective_settings.conditioning.enabled
        )
        self._reference_objective_settings = (
            ObjectiveSettings.reference_defaults()
            if quadratic_initial
            else self.objective_settings
        )
        self._loss_profile = "reference"
        self._set_loss_profile_from_initial_settings()
        self._axis_tasks: list[AxisTask | None] = [None] * 6
        self._xopt: np.ndarray | None = None
        self._previous_state: np.ndarray | None = None
        self._previous_state2: np.ndarray | None = None
        self._warm_start: np.ndarray | None = None
        self._consecutive_failures = 0
        self._block_plan = None
        self._pending_block_plan = None
        self._pending_block_targets = ()
        self._pending_block_index = 0
        self._pending_block_diagnostics = None

    def fork(self, *, objective_settings: ObjectiveSettings | None = None) -> "IpoptIK":
        """Copy solver history for an isolated one-step counterfactual solve.

        Model and robot handles are read-only shared inputs. FK scratch data,
        command history, warm starts and task/block state are independent.
        """
        result = copy.copy(self)
        for name, value in vars(self).items():
            if name not in ("model", "handles", "kinematics"):
                setattr(result, name, copy.deepcopy(value))
        result.kinematics = RobotKinematics(self.model, self.handles)
        if objective_settings is not None:
            result.objective_settings = objective_settings
            result._set_loss_profile_from_initial_settings()
        return result

    def reset(self) -> None:
        self._xopt = None
        self._previous_state = None
        self._previous_state2 = None
        self._warm_start = None
        self._consecutive_failures = 0
        self._block_plan = None
        self._pending_block_plan = None
        self._pending_block_targets = ()
        self._pending_block_index = 0
        self._pending_block_diagnostics = None

    @property
    def has_pending_block(self) -> bool:
        """A solved block still has commands waiting for execution."""
        return (self._pending_block_plan is not None
                and self._pending_block_index < len(self._pending_block_plan))

    def discard_block_plan(self) -> None:
        """Cancel unexecuted knots while preserving physical command history."""
        self._block_plan = None
        self._pending_block_plan = None
        self._pending_block_targets = ()
        self._pending_block_index = 0
        self._pending_block_diagnostics = None

    def set_warm_start(self, q: np.ndarray) -> None:
        """Replace the next IPOPT seed while preserving accepted history."""
        candidate = np.asarray(q, dtype=float)
        expected_shape = (self.handles.spec.arm_dof,)
        if candidate.shape != expected_shape or not np.all(np.isfinite(candidate)):
            raise ValueError(f"IPOPT warm start must have shape {expected_shape} and be finite")
        if np.any(candidate < self.handles.joint_lower) or np.any(
            candidate > self.handles.joint_upper
        ):
            raise ValueError("IPOPT warm start must satisfy joint limits")
        self._warm_start = candidate.copy()

    def set_objective_weights(self, weights: np.ndarray) -> None:
        weights = np.clip(np.asarray(weights, dtype=float), 0.0, 1000.0)
        if weights.shape != (7,):
            raise ValueError("Expected seven objective weights.")
        updated = replace(
            self.objective_settings,
            position_weight=weights[0],
            rotation_weight=weights[1],
            velocity_weight=weights[2],
            jerk_weight=weights[3],
            manipulability_weight=weights[4],
            self_collision_weight=weights[5],
            release_loss_weight=weights[6],
        )
        if updated != self.objective_settings:
            self.objective_settings = updated
            if self._loss_profile == "reference":
                self._reference_objective_settings = updated

    @property
    def loss_profile(self) -> str:
        """Name of the currently active secondary objective family."""
        return self._loss_profile

    def _set_loss_profile_from_initial_settings(self) -> None:
        """Detect a constructor-supplied quadratic profile without changing it."""
        if self.objective_settings.conditioning.enabled:
            self._loss_profile = "delta_q_squared_conditioning_safeguard"
        elif self.objective_settings.sigma_min_weight != 0.0 or self.objective_settings.sigma_min_loss_kind != "negative":
            self._loss_profile = next(
                (profile for profile, kind in SIGMA_MIN_PENALTY_PROFILES.items()
                 if kind == self.objective_settings.sigma_min_loss_kind),
                "delta_q_squared_sigma_min_legacy",
            )
        elif self.objective_settings.squared_delta_q_weight != 0.0:
            self._loss_profile = "delta_q_squared_only"

    def set_loss_profile(
        self,
        profile: str,
        weights: np.ndarray | None = None,
        *,
        sigma_min_warning_threshold: float | None = None,
        sigma_min_reciprocal_epsilon: float | None = None,
    ) -> None:
        """Switch among the maintained reference and quadratic profiles.

        Legacy weights are kept in a separate snapshot, so changing profiles
        in the UI does not destroy the user's previous reference settings.
        Maintained-profile ``weights`` are ordered as
        ``(delta_q_squared, ema_release)``. Conditioning is suspended and a
        nonzero legacy conditioning entry is rejected instead of silently
        changing the optimization problem.
        A/B protection profiles use ``(delta_q_squared, sigma_min, ema)``;
        their warning threshold and reciprocal regularization are optional
        explicit overrides of the current settings.
        """
        if profile == "delta_q_squared_sigma_min":
            profile = "delta_q_squared_sigma_min_legacy"
        profiles = (
            "reference",
            "delta_q_squared_only",
            "delta_q_squared_sigma_min_legacy",
            *SIGMA_MIN_PENALTY_PROFILES,
            "delta_q_squared_ema_release",
            "delta_q_squared_conditioning_safeguard",
        )
        if profile not in profiles:
            raise ValueError(f"Unknown optimizer loss profile: {profile}")
        if profile not in SIGMA_MIN_PENALTY_PROFILES and (
            sigma_min_warning_threshold is not None or sigma_min_reciprocal_epsilon is not None
        ):
            raise ValueError("Sigma-min threshold overrides require an A/B protection profile")
        if profile == "reference":
            updated = self._reference_objective_settings
        else:
            if weights is None:
                weights = np.array((1.0, DEFAULT_EMA_RELEASE_LOSS_WEIGHT), dtype=float)
            values = np.asarray(weights, dtype=float)
            if values.shape == (3,):
                if (
                    profile in (
                        "delta_q_squared_ema_release",
                        "delta_q_squared_conditioning_safeguard",
                    )
                    and abs(float(values[1])) > 0.0
                ):
                    raise ValueError(
                        "The conditioning (w_c) objective is suspended and "
                        "cannot be enabled",
                    )
                motion_weight, secondary_weight, ema_weight = values
            elif values.shape == (2,):
                motion_weight, ema_weight = values
                secondary_weight = 0.0
            else:
                raise ValueError(
                    "Maintained loss weights must contain two finite values",
                )
            if not np.all(np.isfinite(values)):
                raise ValueError("Loss weights must be finite")
            values = np.clip(values, 0.0, 1000.0)
            motion_weight = float(np.clip(motion_weight, 0.0, 1000.0))
            secondary_weight = float(np.clip(secondary_weight, 0.0, 1000.0))
            ema_weight = float(np.clip(ema_weight, 0.0, 1000.0))
            if self._loss_profile == "reference":
                self._reference_objective_settings = self.objective_settings
            if profile == "delta_q_squared_only":
                updated = replace(
                    ObjectiveSettings.delta_q_squared_only(),
                    squared_delta_q_weight=motion_weight,
                    ema_release_loss_weight=ema_weight,
                )
            elif profile in SIGMA_MIN_PENALTY_PROFILES:
                updated = ObjectiveSettings.delta_q_squared_sigma_min_penalty(
                    kind=SIGMA_MIN_PENALTY_PROFILES[profile],
                    warning_threshold=(self.objective_settings.sigma_min_warning_threshold
                        if sigma_min_warning_threshold is None else sigma_min_warning_threshold),
                    reciprocal_epsilon=(self.objective_settings.sigma_min_reciprocal_epsilon
                        if sigma_min_reciprocal_epsilon is None else sigma_min_reciprocal_epsilon),
                    motion_weight=motion_weight, sigma_min_weight=secondary_weight,
                    ema_release_loss_weight=ema_weight,
                )
            elif profile == "delta_q_squared_sigma_min_legacy":
                updated = ObjectiveSettings.delta_q_squared_sigma_min_legacy(
                    motion_weight=motion_weight,
                    sigma_min_weight=secondary_weight,
                    length_scale_m=(
                        self.objective_settings.sigma_min_length_scale_m
                    ),
                    ema_release_loss_weight=ema_weight,
                )
            else:
                config = self._conditioning_config
                updated = (
                    ObjectiveSettings
                    .delta_q_squared_conditioning_safeguard(
                        motion_weight=motion_weight,
                        conditioning_weight=0.0,
                        eta_low=config.eta_low,
                        eta_high=config.eta_high,
                        sigma_epsilon=config.sigma_epsilon,
                        finite_difference_relative_step=(
                            config.finite_difference_relative_step
                        ),
                        ema_release_loss_weight=ema_weight,
                    )
                )
                self._conditioning_config = updated.conditioning
        if profile != self._loss_profile or updated != self.objective_settings:
            self.objective_settings = updated
            self._loss_profile = profile
            self._consecutive_failures = 0

    def set_release_loss_schedule(
        self, offset: float, gain: float, power: float,
    ) -> None:
        """Set the intent/activity release multiplier schedule."""
        values = np.asarray((offset, gain, power), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Release-loss parameters must be finite and non-negative.")
        if power <= 0.0:
            raise ValueError("Release-loss power must be positive.")
        updated = replace(
            self.objective_settings,
            release_loss_offset=float(offset),
            release_loss_gain=float(gain),
            release_loss_power=float(power),
        )
        if updated != self.objective_settings:
            self.objective_settings = updated

    def release_loss_multipliers(
        self,
        tasks: tuple[AxisTask, ...] | list[AxisTask],
    ) -> np.ndarray:
        """Return the live release weights used by this IPOPT instance."""
        return release_loss_multipliers_for_tasks(
            tasks, self.objective_settings,
        )

    def set_axis_task(self, axis: int, task: AxisTask | None) -> None:
        validate_axis_task(axis, task)
        self._axis_tasks[axis] = task

    def _resolve_axis_tasks(
        self, active_dofs: np.ndarray,
    ) -> tuple[AxisTask, ...]:
        return resolve_axis_tasks(self._axis_tasks, active_dofs)

    @staticmethod
    def _project_to_constraint_tangent(
        vector: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        """Project a diagnostic vector into the local constraint tangent."""
        if rows.size == 0:
            return vector.copy()
        correction, *_ = np.linalg.lstsq(
            rows @ rows.T,
            rows @ vector,
            rcond=None,
        )
        return vector - rows.T @ correction

    @staticmethod
    def _component_norm_diagnostics(
        components: dict[str, np.ndarray],
        tangent_rows: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float], float, float]:
        raw = {name: float(np.linalg.norm(value)) for name, value in components.items()}
        projected_vectors = {
            name: IpoptIK._project_to_constraint_tangent(
                value, tangent_rows,
            )
            for name, value in components.items()
        }
        tangent = {
            name: float(np.linalg.norm(value))
            for name, value in projected_vectors.items()
        }

        def cancellation(vectors: dict[str, np.ndarray]) -> float:
            denominator = sum(float(np.linalg.norm(value)) for value in vectors.values())
            if denominator <= 1.0e-15:
                return 1.0
            return float(np.linalg.norm(sum(vectors.values()))) / denominator

        return raw, tangent, cancellation(components), cancellation(projected_vectors)

    @staticmethod
    def _objective_tasks(
        tasks: tuple[AxisTask, ...],
    ) -> tuple[AxisTask, ...]:
        """Remove hard pose rows while retaining ranged preference losses."""
        result = [AxisTask(TaskKind.FREE) for _ in range(6)]
        for index in range(3, 6):
            if tasks[index].kind in IpoptIK._RANGED_KINDS:
                result[index] = tasks[index]
        return tuple(result)

    def _constraint_values(
        self,
        q: np.ndarray,
        target: TargetPose,
        tasks: tuple[AxisTask, ...],
        tolerance_rotation: np.ndarray | None,
        rotation_release_step: RotationReleaseStep | None = None,
    ) -> ConstraintValues:
        return self._constraint_values_and_jacobian(
            q, target, tasks, tolerance_rotation, rotation_release_step,
        )[0]

    def _constraint_values_and_jacobian(
        self,
        q: np.ndarray,
        target: TargetPose,
        tasks: tuple[AxisTask, ...],
        tolerance_rotation: np.ndarray | None,
        rotation_release_step: RotationReleaseStep | None = None,
    ) -> tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]]:
        state = self.kinematics.evaluate(
            q,
            include_manipulability=False,
            include_link_points=False,
        )
        if rotation_release_step is None:
            error = self.kinematics.pose_error(state, target.position, target.rotation, tolerance_rotation)
            pose_jacobian = self.kinematics.pose_error_jacobian(state, target.rotation, tolerance_rotation)
        else:
            error = np.zeros(6, dtype=float)
            error[:3] = state.position - target.position
            error[3:], rotation_jacobian = rotation_release_error_and_jacobian(
                self.kinematics, state, rotation_release_step,
            )
            pose_jacobian = np.vstack((state.jacobian[:3], rotation_jacobian))
        equality_values: list[float] = list(error[:3])
        equality_rows: list[np.ndarray] = [
            pose_jacobian[index] for index in range(3)
        ]
        strict_rotation_values: list[float] = []
        inequality_values: list[float] = []
        inequality_rows: list[np.ndarray] = []
        for index in range(3, 6):
            task = tasks[index]
            value = float(error[index])
            row = pose_jacobian[index]
            if rotation_release_step is None and task.absolute_rotation_reference is not None:
                absolute_reference = np.asarray(
                    task.absolute_rotation_reference, dtype=float,
                )
                absolute_error = self.kinematics.pose_error(
                    state,
                    target.position,
                    absolute_reference,
                    tolerance_rotation,
                )
                absolute_jacobian = self.kinematics.pose_error_jacobian(
                    state,
                    absolute_reference,
                    tolerance_rotation,
                )
                value = float(absolute_error[index])
                row = absolute_jacobian[index]
            if rotation_release_step is not None and rotation_release_step.mask[index - 3]:
                axis = index - 3
                lower = float(
                    rotation_release_step.effective_lower_limits[axis]
                )
                upper = float(
                    rotation_release_step.effective_upper_limits[axis]
                )
                if np.isfinite(lower):
                    inequality_values.append(value - lower)
                    inequality_rows.append(row)
                if np.isfinite(upper):
                    inequality_values.append(upper - value)
                    inequality_rows.append(-row)
            elif rotation_release_step is not None:
                # Specific axes remain strict in this cycle's stage-relative chart.
                residual = value
                equality_values.append(residual)
                equality_rows.append(row)
                strict_rotation_values.append(residual)
            elif task.kind is TaskKind.SPECIFIC:
                residual = value - task.goal
                equality_values.append(residual)
                equality_rows.append(row)
                strict_rotation_values.append(residual)
            elif task.kind in self._RANGED_KINDS:
                if np.isfinite(task.lower):
                    inequality_values.append(value - task.lower)
                    inequality_rows.append(row)
                if np.isfinite(task.upper):
                    inequality_values.append(task.upper - value)
                    inequality_rows.append(-row)

        if (rotation_release_step is not None
                and rotation_release_step.maximum_release_coordinate_step_rad is not None):
            budget = rotation_release_step.maximum_release_coordinate_step_rad
            delta = error[3:] - rotation_release_step.effective_box_baseline_error
            # Same frozen chart for candidate and transported carry. The
            # nominal action is already in the reference, outside this budget.
            for axis in range(3):
                inequality_values.extend((1. + delta[axis] / budget,
                                          1. - delta[axis] / budget))
                inequality_rows.extend((pose_jacobian[axis + 3] / budget,
                                        -pose_jacobian[axis + 3] / budget))

        equality = np.asarray(equality_values, dtype=float)
        inequality = np.asarray(inequality_values, dtype=float)
        values = ConstraintValues(
            equality=equality,
            inequality=inequality,
            position_residual=float(np.max(np.abs(error[:3]))),
            rotation_residual=(
                float(np.max(np.abs(strict_rotation_values)))
                if strict_rotation_values
                else 0.0
            ),
            inequality_violation=(
                float(np.max(np.maximum(-inequality, 0.0)))
                if inequality.size
                else 0.0
            ),
        )
        return values, (
            np.asarray(equality_rows, dtype=float).reshape(-1, q.size),
            np.asarray(inequality_rows, dtype=float).reshape(-1, q.size),
        )

    def _prepare_single_step_nlp(
        self,
        data: mujoco.MjData,
        target: TargetPose,
        active_dofs: np.ndarray,
        tolerance_rotation: np.ndarray | None = None,
        rotation_release_step: RotationReleaseStep | None = None,
    ) -> _SingleStepNLP:
        measured_q = data.qpos[self.handles.qpos_ids].copy()
        last_command = measured_q if self._xopt is None else self._xopt.copy()
        seed = (
            last_command.copy()
            if self._warm_start is None
            else self._warm_start.copy()
        )
        previous = (
            last_command
            if self._previous_state is None
            else self._previous_state
        )
        previous2 = previous if self._previous_state2 is None else self._previous_state2
        tasks = self._resolve_axis_tasks(np.asarray(active_dofs, dtype=bool))
        chart_rotation = (
            tolerance_rotation
            if any(
                task.kind is not TaskKind.SPECIFIC
                or task.absolute_rotation_reference is not None
                for task in tasks[3:]
            )
            else None
        )
        if rotation_release_step is not None:
            chart_rotation = None
        objective = make_controller_objective(
            self.kinematics,
            target,
            active_dofs,
            MotionState(last_command, previous, previous2),
            self.objective_settings,
            self._objective_tasks(tasks),
            chart_rotation,
            rotation_release_step=rotation_release_step,
        )

        constraint_cache: dict[
            bytes,
            tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]],
        ] = {}

        def constraint_evaluation(
            q: np.ndarray,
        ) -> tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]]:
            key = np.ascontiguousarray(q).tobytes()
            cached = constraint_cache.get(key)
            if cached is None:
                cached = self._constraint_values_and_jacobian(
                    q, target, tasks, chart_rotation, rotation_release_step,
                )
                constraint_cache[key] = cached
            return cached

        return _SingleStepNLP(
            seed=seed,
            last_command=last_command,
            previous=previous,
            previous2=previous2,
            tasks=tasks,
            chart_rotation=chart_rotation,
            objective=objective,
            constraint_evaluation=constraint_evaluation,
        )

    def solve(
        self,
        data: mujoco.MjData,
        target: TargetPose,
        active_dofs: np.ndarray,
        tolerance_rotation: np.ndarray | None = None,
        rotation_release_step: RotationReleaseStep | None = None,
        *,
        horizon_targets: tuple[TargetPose, ...] | None = None,
        horizon_tasks: tuple[tuple[AxisTask, ...], ...] | None = None,
        horizon_release_steps: tuple[RotationReleaseStep, ...] | None = None,
        candidate_acceptance: Callable[[np.ndarray], str | None] | None = None,
    ) -> tuple[np.ndarray, IpoptDiagnostics]:
        """Solve one constrained command with IPOPT and publish if feasible."""
        problem = self._prepare_single_step_nlp(
            data,
            target,
            active_dofs,
            tolerance_rotation,
            rotation_release_step,
        )
        seed = problem.seed
        last_command = problem.last_command
        previous = problem.previous
        previous2 = problem.previous2
        tasks = problem.tasks
        chart_rotation = problem.chart_rotation
        objective = problem.objective
        constraint_evaluation = problem.constraint_evaluation

        objective_gradient = lambda q: objective.gradient(
            q,
            self.settings.derivative_epsilon,
            self.handles.joint_lower,
            self.handles.joint_upper,
            self.settings.expensive_gradient_refresh_rad,
        )
        solver_called = True
        controller_mode = "ipopt"
        block_knot_index = None
        block_horizon = 1
        block_window_feasible = None
        block_first_knot_feasible = None
        block_warm_start_used = False
        block_seed_source = None
        if self.has_pending_block:
            expected = self._pending_block_targets[self._pending_block_index]
            if not (np.allclose(target.position, expected.position, atol=1e-10, rtol=0)
                    and np.allclose(target.rotation, expected.rotation, atol=1e-10, rtol=0)):
                raise ValueError("Pending Block target changed; reset before replanning")
            solver_called = False
            controller_mode = "ipopt_block"
            block_knot_index = self._pending_block_index
            block_horizon = len(self._pending_block_plan)
            command_q = self._pending_block_plan[block_knot_index].copy()
            cached = self._pending_block_diagnostics
            current_constraints = constraint_evaluation(command_q)[0]
            current_validation = validate_candidate(
                command_q, self.handles.joint_lower, self.handles.joint_upper,
                current_constraints, position_tolerance=self.settings.position_tolerance,
                rotation_tolerance=self.settings.rotation_tolerance,
                inequality_tolerance=self.settings.inequality_tolerance,
            )
            current_cost, current_parts = objective.evaluate(command_q, include_breakdown=True)
            result = _PublishedResult(
                q=command_q, cost=float(current_cost), task_costs=current_parts,
                iterations=0, optimality=cached.optimality, elapsed_ms=0.0,
                converged=cached.converged,
                status=f"block execution: knot {block_knot_index + 1}/{block_horizon}",
                constraints=current_constraints,
            )
            block_window_feasible = True
            block_first_knot_feasible = current_validation.feasible
            publishable = current_validation.feasible
            self._pending_block_index += 1
            if not publishable:
                self._pending_block_plan = None
                self._block_plan = None
        elif horizon_targets is not None and len(horizon_targets) > 1:
            from ..block_sqp import BlockSQPIK, BlockSQPSettings
            horizon = len(horizon_targets)
            controller_mode = "ipopt_block"
            block_knot_index = 0
            block_horizon = horizon
            if horizon_tasks is None or horizon_release_steps is None:
                raise ValueError("Block execution requires tasks and release steps for every knot")
            block = BlockSQPIK(
                self.model, self.handles, BlockSQPSettings(horizon, self.settings),
                self.objective_settings,
            )
            block.kinematics = self.kinematics
            block_data = mujoco.MjData(self.model)
            block_data.qpos[self.handles.qpos_ids] = last_command
            plan_seed = np.repeat(last_command[None, :], horizon, axis=0)
            block_seed_source = "last_accepted_command"
            if self.block_warm_start and self._block_plan is not None:
                block_warm_start_used = True
                block_seed_source = "extrapolated_previous_block"
                delta = self._block_plan[-1] - self._block_plan[-2]
                plan_seed = np.clip(
                    last_command[None, :] + np.arange(1, horizon + 1)[:, None] * delta,
                    self.handles.joint_lower, self.handles.joint_upper,
                )
            plan, block_diagnostics = block.solve_epoch(
                block_data, horizon_targets,
                tuple(active_dofs.copy() for _ in range(horizon)),
                tuple(step.tolerance_frame for step in horizon_release_steps),
                plan_seed, previous_q=previous, previous_q2=previous2,
                axis_tasks=horizon_tasks, rotation_release_steps=horizon_release_steps,
            )
            first_constraints = constraint_evaluation(plan[0])[0]
            first_cost, first_parts = objective.evaluate(plan[0], include_breakdown=True)
            result = _PublishedResult(
                q=plan[0], cost=float(first_cost), task_costs=first_parts,
                iterations=block_diagnostics.iterations, optimality=block_diagnostics.optimality,
                elapsed_ms=block_diagnostics.elapsed_ms, converged=block_diagnostics.converged,
                status=block_diagnostics.status, constraints=first_constraints,
            )
            # All H commands will execute: require a feasible complete block.
            first_validation = validate_candidate(
                plan[0], self.handles.joint_lower, self.handles.joint_upper,
                first_constraints,
                position_tolerance=self.settings.position_tolerance,
                rotation_tolerance=self.settings.rotation_tolerance,
                inequality_tolerance=self.settings.inequality_tolerance,
            )
            block_window_feasible = block_diagnostics.feasible
            block_first_knot_feasible = first_validation.feasible
            publishable = bool(block_diagnostics.step_succeeded and first_validation.feasible)
            self._block_plan = plan.copy() if publishable and self.block_warm_start else None
            if publishable:
                self._pending_block_plan = plan.copy()
                self._pending_block_targets = tuple(
                    TargetPose(t.position.copy(), t.rotation.copy()) for t in horizon_targets
                )
                self._pending_block_index = 1
                self._pending_block_diagnostics = block_diagnostics
            else:
                self._pending_block_plan = None
        else:
            self._block_plan = None
            ipopt_result = IpoptAdapter(self.settings).solve(
                lambda q: float(objective.evaluate(q)),
                objective_gradient,
                lambda q: constraint_evaluation(q)[0],
                lambda q, _values: constraint_evaluation(q)[1],
                seed,
                self.handles.joint_lower,
                self.handles.joint_upper,
                position_tolerance=self.settings.position_tolerance,
                rotation_tolerance=self.settings.rotation_tolerance,
                inequality_tolerance=self.settings.inequality_tolerance,
            )
            ipopt_cost, ipopt_parts = objective.evaluate(
                ipopt_result.q, include_breakdown=True,
            )
            result = _PublishedResult(
                q=ipopt_result.q,
                cost=float(ipopt_cost),
                iterations=ipopt_result.iterations,
                optimality=ipopt_result.optimality,
                elapsed_ms=ipopt_result.elapsed_ms,
                converged=ipopt_result.solver_succeeded,
                status=f"ipopt: {ipopt_result.status}",
                constraints=ipopt_result.constraints,
                task_costs=ipopt_parts,
            )
            publishable = bool(
                ipopt_result.status_code in ACCEPTABLE_IPOPT_STATUS_CODES
                and ipopt_result.validation.feasible
            )
        prepublication_rejection_reason = None
        if publishable and candidate_acceptance is not None:
            prepublication_rejection_reason = candidate_acceptance(result.q)
            if prepublication_rejection_reason is not None:
                publishable = False
                result = replace(result, status=f'{result.status}; {prepublication_rejection_reason}')
        # Commit only a nonlinear-feasible result with an explicitly accepted
        # termination. Algorithmic, numerical, and unknown failures may leave
        # a feasible intermediate iterate, but never publish it.
        self._consecutive_failures = (
            0 if publishable else self._consecutive_failures + 1
        )
        command = result.q if publishable else last_command
        # History records commands returned by this controller, including a
        # repeated command after a rejected solve. A warm start is only a
        # numerical seed and never enters the physical motion history.
        self._previous_state2 = previous.copy()
        self._previous_state = last_command.copy()
        self._xopt = command.copy()
        self._warm_start = None
        command_constraints = (
            result.constraints
            if publishable
            else self._constraint_values(
                command, target, tasks, chart_rotation, rotation_release_step,
            )
        )
        command_state = self.kinematics.evaluate(
            command,
            include_manipulability=True,
            include_link_points=False,
        )
        if publishable:
            command_cost = result.cost
            command_parts = result.task_costs
        else:
            command_cost, command_parts = objective.evaluate(
                command, include_breakdown=True,
            )
        if rotation_release_step is None:
            release_candidate = None
        else:
            release_error = rotation_release_step.stage_relative_coordinates(
                command_state.rotation,
            )
            release_candidate = RotationReleaseCandidate(
                rotation=command_state.rotation.copy(),
                stage_relative_error_fixed_xyz_rad=release_error.copy(),
                stage_relative_tolerance_release_rad=np.where(
                    rotation_release_step.mask, release_error, 0.0,
                ),
                predicted_tolerance_violation_rad=(
                    rotation_release_step.predicted_violation(release_error)
                ),
                incremental_release_rad=(
                    rotation_release_step.incremental_release_coordinates(
                        command_state.rotation,
                    )
                ),
            )
        condition = objective.conditioning_metrics(command_state)
        condition_penalty = conditioning_penalty(
            condition, objective.settings.conditioning,
        )
        condition_activation = conditioning_activation(
            condition, objective.settings.conditioning,
        )
        condition_region = conditioning_region(
            condition, objective.settings.conditioning,
        )
        component_gradients = {
            "delta_q_squared": (
                2.0 * objective.settings.squared_delta_q_weight
                * (command - last_command)
            ),
            "conditioning": objective.conditioning_gradient(
                command,
                self.handles.joint_lower,
                self.handles.joint_upper,
                current_metrics=condition,
            ),
        }
        _, command_jacobians = constraint_evaluation(command)
        equality_rows, inequality_rows = command_jacobians
        active_tolerance = max(
            10.0 * self.settings.inequality_tolerance,
            1.0e-6,
        )
        active = command_constraints.inequality <= active_tolerance
        tangent_rows = equality_rows
        if np.any(active):
            tangent_rows = np.vstack((
                equality_rows,
                inequality_rows[active],
            ))
        (
            gradient_norms,
            tangent_gradient_norms,
            gradient_cancellation,
            tangent_gradient_cancellation,
        ) = self._component_norm_diagnostics(
            component_gradients,
            tangent_rows,
        )
        gradient_consistency = None
        diagnostic_condition = None
        if (
            self.diagnose_conditioning_gradient
            and objective.settings.conditioning.enabled
        ):
            diagnostic_state = self.kinematics.evaluate(
                result.q,
                include_manipulability=False,
                include_link_points=False,
            )
            diagnostic_condition = objective.conditioning_metrics(
                diagnostic_state
            )
            gradient_consistency = objective.conditioning_gradient_consistency(
                result.q,
                self.handles.joint_lower,
                self.handles.joint_upper,
            )
        diagnostics = IpoptDiagnostics(
            sigma_min_raw=condition.sigma_min_raw,
            sigma_min_normalized=condition.sigma_min_normalized,
            sigma_max_normalized=condition.sigma_max_normalized,
            sigma_min_gap_normalized=condition.sigma_min_gap_normalized,
            inverse_condition_index=condition.inverse_condition_index,
            conditioning_enabled=bool(
                objective.settings.conditioning.enabled
            ),
            conditioning_eta_low=float(
                objective.settings.conditioning.eta_low
            ),
            conditioning_eta_high=float(
                objective.settings.conditioning.eta_high
            ),
            conditioning_penalty=condition_penalty,
            conditioning_activation=condition_activation,
            conditioning_region=condition_region,
            conditioning_active=bool(
                objective.settings.conditioning.enabled
                and condition.inverse_condition_index
                < objective.settings.conditioning.eta_high
            ),
            manipulability=command_state.manipulability,
            cost=float(command_cost),
            iterations=result.iterations,
            optimality=result.optimality,
            elapsed_ms=result.elapsed_ms,
            task_costs=command_parts,
            task_penalties=objective.penalty_breakdown(command_parts),
            objective_gradient_norms=gradient_norms,
            objective_tangent_gradient_norms=tangent_gradient_norms,
            objective_gradient_cancellation_ratio=gradient_cancellation,
            objective_tangent_gradient_cancellation_ratio=(
                tangent_gradient_cancellation
            ),
            feasible=publishable,
            converged=result.converged,
            position_residual=command_constraints.position_residual,
            strict_rotation_residual=command_constraints.rotation_residual,
            effective_constraint_violation=(
                command_constraints.inequality_violation
            ),
            status=result.status,
            consecutive_failures=self._consecutive_failures,
            conditioning_gradient_norm=(
                None if gradient_consistency is None
                else gradient_consistency.gradient_norm
            ),
            conditioning_gradient_relative_error_half_step=(
                None if gradient_consistency is None
                else gradient_consistency.relative_error_half_step
            ),
            conditioning_gradient_relative_error_double_step=(
                None if gradient_consistency is None
                else gradient_consistency.relative_error_double_step
            ),
            conditioning_gradient_cosine_half_step=(
                None if gradient_consistency is None
                else gradient_consistency.cosine_half_step
            ),
            conditioning_gradient_cosine_double_step=(
                None if gradient_consistency is None
                else gradient_consistency.cosine_double_step
            ),
            conditioning_diagnostic_inverse_condition_index=(
                None if diagnostic_condition is None
                else diagnostic_condition.inverse_condition_index
            ),
            conditioning_diagnostic_sigma_min_gap_normalized=(
                None if diagnostic_condition is None
                else diagnostic_condition.sigma_min_gap_normalized
            ),
            solver_called=solver_called,
            block_knot_index=block_knot_index,
            block_horizon=block_horizon,
            block_window_feasible=block_window_feasible,
            block_first_knot_feasible=block_first_knot_feasible,
            block_warm_start_used=block_warm_start_used,
            block_seed_source=block_seed_source,
            release_loss_multiplier=objective.release_loss_multiplier,
            stage_relative_tolerance_release_rad=None if release_candidate is None else release_candidate.stage_relative_tolerance_release_rad,
            stage_relative_rotation_error_rad=None if release_candidate is None else release_candidate.stage_relative_error_fixed_xyz_rad,
            incremental_release_rad=None if release_candidate is None else release_candidate.incremental_release_rad,
            predicted_tolerance_violation_rad=None if release_candidate is None else release_candidate.predicted_tolerance_violation_rad,
            controller_mode=controller_mode,
            prepublication_rejection_reason=prepublication_rejection_reason,
        )
        return command, diagnostics
