"""Whole-stage objective shared by the IPOPT-backed Block optimizer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..pose import TargetPose
from ..task_space.kinematics import RobotKinematics
from ..task_space.losses import groove, groove_derivative
from ..task_space.objective import MotionState, ObjectiveSettings, TaskObjective
from ..task_space.types import AxisTask
from ...geometry import stage_reference_rotation
from .types import BlockStageReference


class BlockObjective:
    """Whole-stage objective over a fixed horizon of joint configurations."""

    def __init__(
        self,
        kinematics: RobotKinematics,
        targets: tuple[TargetPose, ...],
        tasks: tuple[tuple[AxisTask, ...], ...],
        tolerance_frames: tuple[np.ndarray | None, ...],
        entry_q: np.ndarray,
        previous_q: np.ndarray,
        previous_q2: np.ndarray,
        settings: ObjectiveSettings,
        lower: np.ndarray,
        upper: np.ndarray,
        derivative_epsilon: float,
        expensive_refresh_distance: float,
        stage_references: tuple[BlockStageReference | None, ...] | None = None,
        rotation_release_steps: tuple | None = None,
    ) -> None:
        self.kinematics = kinematics
        self.targets = tuple(
            TargetPose(
                np.asarray(target.position, dtype=float).copy(),
                np.asarray(target.rotation, dtype=float).copy(),
            )
            for target in targets
        )
        self.entry_q = np.asarray(entry_q, dtype=float).copy()
        self.previous_q = np.asarray(previous_q, dtype=float).copy()
        self.previous_q2 = np.asarray(previous_q2, dtype=float).copy()
        self.settings = settings
        self.lower = lower
        self.upper = upper
        self.derivative_epsilon = derivative_epsilon
        self.expensive_refresh_distance = expensive_refresh_distance
        self.stage_references = (
            tuple(None for _ in targets)
            if stage_references is None
            else tuple(stage_references)
        )
        if len(self.stage_references) != len(targets):
            raise ValueError("Every Block objective knot needs a stage reference")
        spatial_settings = replace(
            settings,
            velocity_weight=0.0,
            jerk_weight=0.0,
            squared_delta_q_weight=0.0,
            squared_acceleration_weight=0.0,
        )
        release_steps = rotation_release_steps or tuple(None for _ in targets)
        self.spatial = tuple(
            TaskObjective(
                kinematics,
                target,
                np.ones(6, dtype=bool),
                MotionState(self.entry_q, self.entry_q, self.entry_q),
                spatial_settings,
                knot_tasks,
                frame,
                dynamic_tolerance_penalty=release_step is not None,
                rotation_release_step=release_step,
            )
            for target, knot_tasks, frame, release_step in zip(
                self.targets, tasks, tolerance_frames, release_steps, strict=True,
            )
        )
        self.groove_loss = self.spatial[0].groove_loss

    @property
    def horizon(self) -> int:
        return len(self.targets)

    def _blocks(self, flat: np.ndarray) -> np.ndarray:
        return np.asarray(flat, dtype=float).reshape(
            self.horizon, self.entry_q.size,
        )

    def _state(self, blocks: np.ndarray, index: int) -> np.ndarray:
        if index >= 1:
            return blocks[index - 1]
        if index == 0:
            return self.entry_q
        if index == -1:
            return self.previous_q
        if index == -2:
            return self.previous_q2
        raise IndexError(index)

    def _difference(
        self,
        blocks: np.ndarray,
        knot: int,
        coefficients: tuple[float, ...],
    ) -> np.ndarray:
        return sum(
            coefficient * self._state(blocks, knot - offset)
            for offset, coefficient in enumerate(coefficients)
        )

    def _reference_rotation(
        self,
        blocks: np.ndarray,
        index: int,
        *,
        anchor_override: np.ndarray | None = None,
        anchor_rotation_override: np.ndarray | None = None,
    ) -> np.ndarray:
        target = self.targets[index]
        reference = self.stage_references[index]
        if reference is None:
            return target.rotation
        if anchor_rotation_override is None:
            anchor_q = (
                blocks[reference.anchor_knot]
                if anchor_override is None else anchor_override
            )
            anchor_rotation = self.kinematics.evaluate(
                anchor_q,
                include_manipulability=False,
                include_link_points=False,
            ).rotation
        else:
            anchor_rotation = anchor_rotation_override
        return stage_reference_rotation(
            anchor_rotation,
            reference.anchor_nominal_rotation,
            target.rotation,
        )

    def _set_spatial_targets(self, blocks: np.ndarray) -> None:
        for index, objective in enumerate(self.spatial):
            objective.target.rotation = self._reference_rotation(blocks, index)

    def _dependent_spatial_cost(
        self,
        blocks: np.ndarray,
        anchor_knot: int,
        anchor_rotation: np.ndarray,
    ) -> float:
        total = 0.0
        for index, (objective, reference) in enumerate(zip(
            self.spatial, self.stage_references, strict=True,
        )):
            if reference is None or reference.anchor_knot != anchor_knot:
                continue
            objective.target.rotation = self._reference_rotation(
                blocks,
                index,
                anchor_rotation_override=anchor_rotation,
            )
            total += float(objective.evaluate(blocks[index]))
        return total

    def evaluate(
        self,
        flat: np.ndarray,
        include_breakdown: bool = False,
    ) -> float | tuple[float, dict[str, float]]:
        blocks = self._blocks(flat)
        self._set_spatial_targets(blocks)
        parts: dict[str, float] = {}
        for objective, q in zip(self.spatial, blocks, strict=True):
            _, knot_parts = objective.evaluate(q, include_breakdown=True)
            for name, value in knot_parts.items():
                parts[name] = parts.get(name, 0.0) + float(value)

        motion_terms = (
            ("velocity", self.settings.velocity_weight, (1.0, -1.0)),
            ("jerk", self.settings.jerk_weight, (1.0, -3.0, 3.0, -1.0)),
        )
        for name, weight, coefficients in motion_terms:
            if weight == 0.0:
                continue
            parts[name] = sum(
                weight
                * (
                    float(groove(
                        np.linalg.norm(self._difference(blocks, knot, coefficients)),
                        0.0,
                        self.groove_loss,
                    ))
                    + 1.0
                )
                for knot in range(1, self.horizon + 1)
            )
        for name, weight, coefficients in (
            ("delta_q_squared", self.settings.squared_delta_q_weight, (1.0, -1.0)),
            ("acceleration_squared", self.settings.squared_acceleration_weight, (1.0, -2.0, 1.0)),
        ):
            if weight != 0.0:
                parts[name] = sum(
                    weight * float(difference @ difference)
                    for knot in range(1, self.horizon + 1)
                    for difference in (self._difference(blocks, knot, coefficients),)
                )
        total = float(sum(parts.values()))
        return (total, parts) if include_breakdown else total

    def gradient(self, flat: np.ndarray) -> np.ndarray:
        blocks = self._blocks(flat)
        self._set_spatial_targets(blocks)
        gradient = np.zeros_like(blocks)
        for index, (objective, q) in enumerate(
            zip(self.spatial, blocks, strict=True),
        ):
            gradient[index] += objective.gradient(
                q,
                self.derivative_epsilon,
                self.lower,
                self.upper,
                self.expensive_refresh_distance,
            )

        for anchor in sorted({
            reference.anchor_knot
            for reference in self.stage_references
            if reference is not None
        }):
            anchor_q = blocks[anchor]
            for joint in range(anchor_q.size):
                step = self.derivative_epsilon * max(
                    1.0, abs(float(anchor_q[joint])),
                )
                plus = min(float(anchor_q[joint]) + step, float(self.upper[joint]))
                minus = max(float(anchor_q[joint]) - step, float(self.lower[joint]))
                if plus <= minus:
                    continue
                q_plus = anchor_q.copy()
                q_minus = anchor_q.copy()
                q_plus[joint] = plus
                q_minus[joint] = minus
                rotation_plus = self.kinematics.evaluate(
                    q_plus,
                    include_manipulability=False,
                    include_link_points=False,
                ).rotation
                rotation_minus = self.kinematics.evaluate(
                    q_minus,
                    include_manipulability=False,
                    include_link_points=False,
                ).rotation
                value_plus = self._dependent_spatial_cost(
                    blocks, anchor, rotation_plus,
                )
                value_minus = self._dependent_spatial_cost(
                    blocks, anchor, rotation_minus,
                )
                gradient[anchor, joint] += (
                    value_plus - value_minus
                ) / (plus - minus)
        self._set_spatial_targets(blocks)

        motion_terms = (
            (self.settings.velocity_weight, (1.0, -1.0)),
            (self.settings.jerk_weight, (1.0, -3.0, 3.0, -1.0)),
        )
        for weight, coefficients in motion_terms:
            if weight == 0.0:
                continue
            for knot in range(1, self.horizon + 1):
                difference = self._difference(blocks, knot, coefficients)
                norm = float(np.linalg.norm(difference))
                if norm <= 1e-14:
                    continue
                scale = (
                    weight
                    * float(groove_derivative(
                        norm, 0.0, self.groove_loss,
                    ))
                    / norm
                )
                for offset, coefficient in enumerate(coefficients):
                    state_index = knot - offset
                    if state_index >= 1:
                        gradient[state_index - 1] += (
                            scale * coefficient * difference
                        )
        for weight, coefficients in (
            (self.settings.squared_delta_q_weight, (1.0, -1.0)),
            (self.settings.squared_acceleration_weight, (1.0, -2.0, 1.0)),
        ):
            if weight == 0.0:
                continue
            for knot in range(1, self.horizon + 1):
                difference = self._difference(blocks, knot, coefficients)
                for offset, coefficient in enumerate(coefficients):
                    state_index = knot - offset
                    if state_index >= 1:
                        gradient[state_index - 1] += 2.0 * weight * coefficient * difference
        return gradient.reshape(-1)
