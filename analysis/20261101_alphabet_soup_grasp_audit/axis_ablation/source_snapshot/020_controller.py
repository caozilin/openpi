from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from ..ipopt import (
    ACCEPTABLE_IPOPT_STATUS_CODES,
    ConstraintValues,
    IpoptAdapter,
    validate_candidate,
)
from ..pose import TargetPose
from ..task_space.kinematics import RobotKinematics
from ..task_space.objective import ObjectiveSettings
from ..task_space.types import AxisTask, TaskKind, resolve_axis_tasks, validate_axis_task
from ...geometry import stage_reference_rotation
from ...simulation.robot import RobotHandles
from .objective import BlockObjective
from .types import BlockSQPDiagnostics, BlockSQPSettings, BlockStageReference


class BlockSQPIK:
    """Direct-joint Block optimizer used by the shared IPOPT runtime."""

    MAINTAINED = True

    _RANGED_KINDS = {
        TaskKind.RANGE,
        TaskKind.PREFERRED_RANGE,
        TaskKind.FLAT_RANGE,
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        settings: BlockSQPSettings,
        objective_settings: ObjectiveSettings | None = None,
    ) -> None:
        self.handles = handles
        self.settings = settings
        self.kinematics = RobotKinematics(model, handles)
        self.objective_settings = (
            objective_settings or ObjectiveSettings.reference_defaults()
        )
        self._axis_tasks: list[AxisTask | None] = [None] * 6

    def reset(self) -> None:
        pass

    def set_axis_task(self, axis: int, task: AxisTask | None) -> None:
        validate_axis_task(axis, task)
        self._axis_tasks[axis] = task

    def _tasks(self, active_dofs: np.ndarray) -> tuple[AxisTask, ...]:
        return resolve_axis_tasks(
            self._axis_tasks,
            np.asarray(active_dofs, dtype=bool),
        )

    @classmethod
    def _objective_tasks(
        cls,
        tasks: tuple[AxisTask, ...],
    ) -> tuple[AxisTask, ...]:
        """Hard Cartesian rows cannot be traded against secondary losses."""
        return tuple(
            task
            if index >= 3 and task.kind in cls._RANGED_KINDS
            else AxisTask(TaskKind.FREE)
            for index, task in enumerate(tasks)
        )

    def _constraints(
        self,
        flat: np.ndarray,
        targets: tuple[TargetPose, ...],
        tasks: tuple[tuple[AxisTask, ...], ...],
        frames: tuple[np.ndarray | None, ...],
        stage_references: tuple[BlockStageReference | None, ...],
        *,
        jacobian: bool,
        rotation_release_steps: tuple | None = None,
    ) -> tuple[
        ConstraintValues,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        dof = self.handles.spec.arm_dof
        blocks = np.asarray(flat, dtype=float).reshape(len(targets), dof)
        equality_values: list[float] = []
        equality_rows: list[np.ndarray] = []
        inequality_values: list[float] = []
        inequality_rows: list[np.ndarray] = []
        position_residual = 0.0
        rotation_residual = 0.0

        states = tuple(
            self.kinematics.evaluate(
                q,
                include_manipulability=False,
                include_link_points=False,
            )
            for q in blocks
        )
        anchor_rotation_probes: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray, float]
        ] = {}
        if jacobian:
            for anchor in sorted({
                reference.anchor_knot
                for reference in stage_references
                if reference is not None
            }):
                anchor_q = blocks[anchor]
                for joint in range(dof):
                    step = self.settings.ipopt.derivative_epsilon * max(
                        1.0, abs(float(anchor_q[joint])),
                    )
                    plus = min(
                        float(anchor_q[joint]) + step,
                        float(self.handles.joint_upper[joint]),
                    )
                    minus = max(
                        float(anchor_q[joint]) - step,
                        float(self.handles.joint_lower[joint]),
                    )
                    if plus <= minus:
                        continue
                    q_plus = anchor_q.copy()
                    q_minus = anchor_q.copy()
                    q_plus[joint] = plus
                    q_minus[joint] = minus
                    anchor_rotation_probes[(anchor, joint)] = (
                        self.kinematics.evaluate(
                            q_plus,
                            include_manipulability=False,
                            include_link_points=False,
                        ).rotation,
                        self.kinematics.evaluate(
                            q_minus,
                            include_manipulability=False,
                            include_link_points=False,
                        ).rotation,
                        plus - minus,
                    )

        for knot, (q, target, knot_tasks, frame, reference, state) in enumerate(
            zip(
                blocks,
                targets,
                tasks,
                frames,
                stage_references,
                states,
                strict=True,
            ),
        ):
            target_rotation = (
                target.rotation
                if reference is None
                else stage_reference_rotation(
                    states[reference.anchor_knot].rotation,
                    reference.anchor_nominal_rotation,
                    target.rotation,
                )
            )
            error = self.kinematics.pose_error(
                state, target.position, target_rotation, frame,
            )
            pose_jacobian = (
                self.kinematics.pose_error_jacobian(
                    state, target_rotation, frame,
                )
                if jacobian else None
            )
            release_step = None if rotation_release_steps is None else rotation_release_steps[knot]
            if release_step is not None:
                from ..ipopt.controller import rotation_release_error_and_jacobian
                error[3:], rotation_jacobian = rotation_release_error_and_jacobian(
                    self.kinematics, state, release_step,
                )
                if jacobian:
                    pose_jacobian = np.vstack((state.jacobian[:3], rotation_jacobian))
            anchor_jacobian = None
            if jacobian and reference is not None:
                anchor = reference.anchor_knot
                anchor_jacobian = np.zeros((3, dof), dtype=float)
                for joint in range(dof):
                    probe = anchor_rotation_probes.get((anchor, joint))
                    if probe is None:
                        continue
                    anchor_rotation_plus, anchor_rotation_minus, denominator = probe
                    rotation_plus = stage_reference_rotation(
                        anchor_rotation_plus,
                        reference.anchor_nominal_rotation,
                        target.rotation,
                    )
                    rotation_minus = stage_reference_rotation(
                        anchor_rotation_minus,
                        reference.anchor_nominal_rotation,
                        target.rotation,
                    )
                    error_plus = self.kinematics.pose_error(
                        state, target.position, rotation_plus, frame,
                    )
                    error_minus = self.kinematics.pose_error(
                        state, target.position, rotation_minus, frame,
                    )
                    anchor_jacobian[:, joint] = (
                        error_plus[3:] - error_minus[3:]
                    ) / denominator
            position_residual = max(
                position_residual,
                float(np.max(np.abs(error[:3]))),
            )
            for axis in range(3):
                equality_values.append(float(error[axis]))
                if pose_jacobian is not None:
                    row = np.zeros(flat.size)
                    row[knot * dof:(knot + 1) * dof] = pose_jacobian[axis]
                    equality_rows.append(row)

            for axis in range(3, 6):
                task = knot_tasks[axis]
                value = float(error[axis])
                if task.kind is TaskKind.SPECIFIC or (release_step is not None and not release_step.mask[axis - 3]):
                    residual = value if release_step is not None else value - task.goal
                    equality_values.append(residual)
                    rotation_residual = max(rotation_residual, abs(residual))
                    if pose_jacobian is not None:
                        row = np.zeros(flat.size)
                        row[knot * dof:(knot + 1) * dof] = pose_jacobian[axis]
                        if anchor_jacobian is not None:
                            anchor = reference.anchor_knot
                            row[anchor * dof:(anchor + 1) * dof] += (
                                anchor_jacobian[axis - 3]
                            )
                        equality_rows.append(row)
                elif task.kind in self._RANGED_KINDS:
                    lower = task.lower if release_step is None else release_step.effective_lower_limits[axis - 3]
                    upper = task.upper if release_step is None else release_step.effective_upper_limits[axis - 3]
                    if np.isfinite(lower):
                        inequality_values.append(value - lower)
                        if pose_jacobian is not None:
                            row = np.zeros(flat.size)
                            row[knot * dof:(knot + 1) * dof] = pose_jacobian[axis]
                            if anchor_jacobian is not None:
                                anchor = reference.anchor_knot
                                row[anchor * dof:(anchor + 1) * dof] += (
                                    anchor_jacobian[axis - 3]
                                )
                            inequality_rows.append(row)
                    if np.isfinite(upper):
                        inequality_values.append(upper - value)
                        if pose_jacobian is not None:
                            row = np.zeros(flat.size)
                            row[knot * dof:(knot + 1) * dof] = -pose_jacobian[axis]
                            if anchor_jacobian is not None:
                                anchor = reference.anchor_knot
                                row[anchor * dof:(anchor + 1) * dof] -= (
                                    anchor_jacobian[axis - 3]
                                )
                            inequality_rows.append(row)

        inequality = np.asarray(inequality_values, dtype=float)
        values = ConstraintValues(
            equality=np.asarray(equality_values, dtype=float),
            inequality=inequality,
            position_residual=position_residual,
            rotation_residual=rotation_residual,
            inequality_violation=(
                float(np.max(np.maximum(-inequality, 0.0)))
                if inequality.size else 0.0
            ),
        )
        if not jacobian:
            return values, None
        return values, (
            np.asarray(equality_rows, dtype=float).reshape(-1, flat.size),
            np.asarray(inequality_rows, dtype=float).reshape(-1, flat.size),
        )

    def solve_epoch(
        self,
        data: mujoco.MjData,
        targets: Sequence[TargetPose],
        active_dofs: Sequence[np.ndarray],
        tolerance_frames: Sequence[np.ndarray | None],
        initial_plan: np.ndarray,
        *,
        previous_q: np.ndarray | None = None,
        previous_q2: np.ndarray | None = None,
        axis_tasks: Sequence[Sequence[AxisTask]] | None = None,
        stage_references: Sequence[BlockStageReference | None] | None = None,
        rotation_release_steps: Sequence | None = None,
    ) -> tuple[np.ndarray, BlockSQPDiagnostics]:
        """Run one configured Block IPOPT epoch from an explicit plan."""
        horizon = len(targets)
        if horizon != self.settings.horizon:
            raise ValueError(
                f"Expected horizon {self.settings.horizon}, got {horizon}",
            )
        if len(active_dofs) != horizon or len(tolerance_frames) != horizon:
            raise ValueError("Every BlockSQP target needs tasks and a frame")
        target_tuple = tuple(targets)
        active_tuple = tuple(
            np.asarray(value, dtype=bool).copy() for value in active_dofs
        )
        if any(value.shape != (6,) for value in active_tuple):
            raise ValueError("BlockSQP active DOFs must have shape (6,)")
        frame_tuple = tuple(
            None if value is None else np.asarray(value, dtype=float).copy()
            for value in tolerance_frames
        )
        dof = self.handles.spec.arm_dof
        plan = np.asarray(initial_plan, dtype=float)
        expected_shape = (horizon, dof)
        if plan.shape != expected_shape or not np.all(np.isfinite(plan)):
            raise ValueError(f"initial_plan must be a finite array of shape {expected_shape}")

        entry_q = data.qpos[self.handles.qpos_ids].copy()
        history1 = entry_q if previous_q is None else np.asarray(previous_q, dtype=float)
        history2 = history1 if previous_q2 is None else np.asarray(previous_q2, dtype=float)
        if axis_tasks is None:
            tasks = tuple(self._tasks(active) for active in active_tuple)
        else:
            if len(axis_tasks) != horizon:
                raise ValueError("Every BlockSQP target needs six axis tasks")
            tasks = tuple(tuple(value) for value in axis_tasks)
            if any(len(value) != 6 for value in tasks):
                raise ValueError("Every BlockSQP knot needs six axis tasks")
            for knot_tasks in tasks:
                for axis, task in enumerate(knot_tasks):
                    validate_axis_task(axis, task)
        objective_tasks = tuple(self._objective_tasks(value) for value in tasks)
        references = (
            tuple(None for _ in range(horizon))
            if stage_references is None
            else tuple(stage_references)
        )
        if len(references) != horizon:
            raise ValueError("Every BlockSQP knot needs a stage reference entry")
        for knot, reference in enumerate(references):
            if reference is None:
                continue
            if not isinstance(reference, BlockStageReference):
                raise TypeError("BlockSQP stage references must use BlockStageReference")
            if reference.anchor_knot >= knot:
                raise ValueError(
                    "BlockSQP stage anchors must precede the referenced knot",
                )
        release_steps = None if rotation_release_steps is None else tuple(rotation_release_steps)
        if release_steps is not None:
            if len(release_steps) != horizon:
                raise ValueError("Every Block knot needs a release step")
            if any(reference is not None for reference in references):
                raise ValueError("Frozen release steps cannot use decision-dependent stage anchors")
        lower = np.tile(self.handles.joint_lower, horizon)
        upper = np.tile(self.handles.joint_upper, horizon)
        seed = np.clip(plan.reshape(-1), lower, upper)
        objective = BlockObjective(
            self.kinematics,
            target_tuple,
            objective_tasks,
            frame_tuple,
            entry_q,
            history1,
            history2,
            self.objective_settings,
            self.handles.joint_lower,
            self.handles.joint_upper,
            self.settings.ipopt.derivative_epsilon,
            self.settings.ipopt.expensive_gradient_refresh_rad,
            references,
            release_steps,
        )
        seed_cost, seed_parts = objective.evaluate(seed, include_breakdown=True)
        cache: dict[
            bytes,
            tuple[ConstraintValues, tuple[np.ndarray, np.ndarray] | None],
        ] = {}

        def evaluate_constraints(value: np.ndarray):
            key = np.ascontiguousarray(value).tobytes()
            if key not in cache:
                cache[key] = self._constraints(
                    value,
                    target_tuple,
                    tasks,
                    frame_tuple,
                    references,
                    jacobian=True,
                    rotation_release_steps=release_steps,
                )
            return cache[key]

        result = IpoptAdapter(self.settings.ipopt).solve(
            lambda value: float(objective.evaluate(value)),
            objective.gradient,
            lambda value: evaluate_constraints(value)[0],
            lambda value, _evaluated: evaluate_constraints(value)[1],
            seed,
            lower,
            upper,
            position_tolerance=self.settings.ipopt.position_tolerance,
            rotation_tolerance=self.settings.ipopt.rotation_tolerance,
            inequality_tolerance=self.settings.ipopt.inequality_tolerance,
        )
        candidate = result.q.reshape(expected_shape)
        seed_constraints = evaluate_constraints(seed)[0]
        seed_validation = validate_candidate(
            seed,
            lower,
            upper,
            seed_constraints,
            position_tolerance=self.settings.ipopt.position_tolerance,
            rotation_tolerance=self.settings.ipopt.rotation_tolerance,
            inequality_tolerance=self.settings.ipopt.inequality_tolerance,
        )
        candidate_cost, candidate_parts = objective.evaluate(
            result.q, include_breakdown=True,
        )
        accepted_status = result.status_code in ACCEPTABLE_IPOPT_STATUS_CODES
        diagnostics = BlockSQPDiagnostics(
            cost=float(candidate_cost),
            seed_cost=float(seed_cost),
            seed_feasible=seed_validation.feasible,
            seed_position_residual=seed_constraints.position_residual,
            seed_strict_rotation_residual=seed_constraints.rotation_residual,
            seed_tolerance_violation=seed_constraints.inequality_violation,
            iterations=result.iterations,
            optimality=result.optimality,
            elapsed_ms=result.elapsed_ms,
            feasible=result.validation.feasible,
            iteration_succeeded=accepted_status,
            step_succeeded=bool(
                accepted_status and result.validation.feasible
            ),
            converged=result.solver_succeeded,
            position_residual=result.constraints.position_residual,
            strict_rotation_residual=result.constraints.rotation_residual,
            tolerance_violation=result.constraints.inequality_violation,
            status=f"ipopt: {result.status}",
            horizon=horizon,
            task_costs=candidate_parts,
            seed_task_costs=seed_parts,
        )
        return candidate, diagnostics
