"""Stage-relative SO(3) rotation release state and frozen cycle snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable

import numpy as np

from ..geometry import (
    rotation_tolerance_coordinates,
    stage_relative_rotation_coordinates,
    stage_reference_rotation,
)
from .projection import constraint_consistent_release_loss_reference
from .boundary_governor import govern_coordinate_box
from .so3_preprocessor import SO3PreprocessorSettings, SO3BoxResult, preprocess_so3_box


@dataclass(frozen=True)
class RotationReleaseStep:
    """Immutable rotational constraint data for one outer control cycle."""

    nominal_rotation: np.ndarray
    stage_handoff_rotation: np.ndarray
    stage_handoff_nominal_rotation: np.ndarray
    tolerance_frame: np.ndarray
    mask: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    effective_box_baseline_error: np.ndarray
    effective_lower_limits: np.ndarray
    effective_upper_limits: np.ndarray
    carried_rotation: np.ndarray
    release_loss_reference_rotation: np.ndarray
    maximum_release_coordinate_step_rad: float | None = None
    governor_mask: np.ndarray | None = None
    so3_result: SO3BoxResult | None = None
    previous_accepted_rotation: np.ndarray | None = None
    nominal_reference_step_rad: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "nominal_rotation",
            "stage_handoff_rotation",
            "stage_handoff_nominal_rotation",
            "tolerance_frame",
            "mask",
            "lower_limits",
            "upper_limits",
            "effective_box_baseline_error",
            "effective_lower_limits",
            "effective_upper_limits",
            "carried_rotation",
            "release_loss_reference_rotation",
        ):
            value = np.asarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.governor_mask is not None:
            mask = np.asarray(self.governor_mask, dtype=bool).copy()
            mask.setflags(write=False)
            object.__setattr__(self, "governor_mask", mask)
        if self.previous_accepted_rotation is not None:
            rotation = np.array(self.previous_accepted_rotation, copy=True)
            rotation.setflags(write=False)
            object.__setattr__(self, 'previous_accepted_rotation', rotation)

    def stage_reference_rotation(self) -> np.ndarray:
        return stage_reference_rotation(
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            self.nominal_rotation,
        )

    def stage_relative_coordinates(self, rotation: np.ndarray) -> np.ndarray:
        """Return total stage release used only for hard admissibility."""
        return stage_relative_rotation_coordinates(
            rotation,
            self.nominal_rotation,
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            self.tolerance_frame,
        )

    def incremental_release_coordinates(
        self, rotation: np.ndarray,
    ) -> np.ndarray:
        """Return this cycle's additional release relative to frozen R_loss."""
        return rotation_tolerance_coordinates(
            rotation,
            self.release_loss_reference_rotation,
            self.tolerance_frame,
        )

    def predicted_violation(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        return np.where(
            self.mask,
            np.maximum.reduce((
                self.lower_limits - values,
                np.zeros(3),
                values - self.upper_limits,
            )),
            0.0,
        )


@dataclass(frozen=True)
class RotationReleaseCandidate:
    """Diagnostics for one candidate under a frozen release step."""

    rotation: np.ndarray
    stage_relative_error_fixed_xyz_rad: np.ndarray
    stage_relative_tolerance_release_rad: np.ndarray
    predicted_tolerance_violation_rad: np.ndarray
    incremental_release_rad: np.ndarray


@dataclass
class RotationReleaseState:
    """Persistent physical state for one active manipulation stage."""

    stage_key: Hashable | None = None
    stage_handoff_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3),
    )
    stage_handoff_nominal_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3),
    )
    accepted_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    accepted_nominal_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3),
    )
    rotation_for_joint: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    # Opt-in single-step prediction experiment; None preserves authored boxes.
    maximum_release_coordinate_step_rad: float | None = None
    so3_preprocessor: SO3PreprocessorSettings | None = None
    _coordinate_governor_mask: np.ndarray | None = field(default=None, init=False, repr=False)
    _accepted_step: RotationReleaseStep | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    def set_stage_tolerant_mask(self, mask: np.ndarray) -> None:
        """Set known semantic eligibility before the first predicted box.

        Call after begin_stage/rebase_subtask. Numeric zero predictions cannot
        supply this information, so an upstream stage protocol may provide it.
        Once fixed, a different mask is rejected until the next stage reset.
        """
        value = np.asarray(mask, dtype=bool)
        if value.shape != (3,):
            raise ValueError('Stage tolerant mask must have three components')
        if self._coordinate_governor_mask is not None and not np.array_equal(self._coordinate_governor_mask, value):
            raise ValueError('Cannot change tolerant eligibility within a stage')
        self._coordinate_governor_mask = value.copy()

    def reset(self) -> None:
        self.stage_key = None
        self.stage_handoff_rotation = np.eye(3)
        self.stage_handoff_nominal_rotation = np.eye(3)
        self.accepted_rotation = np.eye(3)
        self.accepted_nominal_rotation = np.eye(3)
        self._accepted_step = None
        self._coordinate_governor_mask = None

    def begin_stage(
        self,
        stage_key: Hashable,
        *,
        optimized_rotation: np.ndarray,
        nominal_rotation: np.ndarray,
        rotation_for_joint: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Capture one physical/nominal handoff within a subtask.

        Within a subtask, the physical handoff anchors the next stage so a
        Pre-grasp realization is transported through Post-grasp. Subtask
        boundaries must use :meth:`rebase_subtask` so the physical and nominal
        origins are changed atomically instead of restoring an old nominal
        orientation.
        """
        optimized = np.asarray(optimized_rotation, dtype=float)
        nominal = np.asarray(nominal_rotation, dtype=float)
        if optimized.shape != (3, 3) or nominal.shape != (3, 3):
            raise ValueError("Handoff rotations must have shape (3, 3)")
        self.stage_key = stage_key
        self.stage_handoff_rotation = optimized.copy()
        self.stage_handoff_nominal_rotation = nominal.copy()
        self.accepted_rotation = optimized.copy()
        self.accepted_nominal_rotation = nominal.copy()
        self._accepted_step = None
        self._coordinate_governor_mask = None
        if rotation_for_joint is not None:
            self.rotation_for_joint = rotation_for_joint

    def rebase_subtask(
        self,
        stage_key: Hashable,
        rotation: np.ndarray,
    ) -> None:
        """Start a subtask with zero accumulated rotational release."""
        anchor = np.asarray(rotation, dtype=float)
        if anchor.shape != (3, 3) or not np.all(np.isfinite(anchor)):
            raise ValueError("Subtask rotation must be a finite 3x3 matrix")
        self.stage_key = stage_key
        self.stage_handoff_rotation = anchor.copy()
        self.stage_handoff_nominal_rotation = anchor.copy()
        self.accepted_rotation = anchor.copy()
        self.accepted_nominal_rotation = anchor.copy()
        self._accepted_step = None
        self._coordinate_governor_mask = None

    def prepare_cycle(
        self,
        nominal_rotation: np.ndarray,
        tolerance_frame: np.ndarray,
        mask: np.ndarray,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        *,
        allow_boundary_governor: bool = True,
    ) -> RotationReleaseStep:
        """Freeze the current spatial release and effective hard box."""
        nominal = np.asarray(nominal_rotation, dtype=float)
        frame = np.asarray(tolerance_frame, dtype=float)
        ranged = np.asarray(mask, dtype=bool)
        lower = np.asarray(lower_limits, dtype=float)
        upper = np.asarray(upper_limits, dtype=float)
        if nominal.shape != (3, 3) or frame.shape != (3, 3):
            raise ValueError("Release rotations and frame must have shape (3, 3)")
        if ranged.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("Release mask and limits must have shape (3,)")
        if np.any(lower > 0.0) or np.any(upper < 0.0) or np.any(lower > upper):
            raise ValueError("Release limits must satisfy lower <= 0 <= upper")
        accepted_reference = stage_reference_rotation(
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            self.accepted_nominal_rotation,
        )
        current_reference = stage_reference_rotation(
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            nominal,
        )
        # The project defines release as the left/spatial residual
        # D = R_actual @ R_reference.T.  Carry D unchanged while nominal
        # reference motion advances; the factors do not generally commute.
        accepted_spatial_release = (
            self.accepted_rotation @ accepted_reference.T
        )
        carried_rotation = accepted_spatial_release @ current_reference
        loss_reference, _ = constraint_consistent_release_loss_reference(
            current_reference,
            carried_rotation,
            frame,
            ranged,
        )
        # Re-express the transported physical release in this cycle's frozen
        # tolerance chart; fixed-XYZ coordinates are never carried as state.
        effective_box_baseline_error = stage_relative_rotation_coordinates(
            carried_rotation,
            nominal,
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            frame,
        )
        # Without the opt-in prediction governor, a rotating chart can mix an
        # accepted large Ry release into a tightly bounded Rx. That changes
        # the required correction, not the hard box.
        # Only an explicit contraction of the authored limits may temporarily
        # admit the carried pose. A recovery envelope may shrink, never grow
        # merely because the chart moved again on the next cycle.
        effective_lower = lower.copy()
        effective_upper = upper.copy()
        previous = self._accepted_step
        if previous is not None:
            epsilon = 1.0e-12
            relax_lower = (lower > previous.lower_limits + epsilon) | (
                previous.effective_lower_limits < previous.lower_limits - epsilon
            )
            relax_upper = (upper < previous.upper_limits - epsilon) | (
                previous.effective_upper_limits > previous.upper_limits + epsilon
            )
            effective_lower = np.where(
                relax_lower,
                np.minimum(
                    lower,
                    np.maximum(effective_box_baseline_error, previous.effective_lower_limits),
                ),
                lower,
            )
            effective_upper = np.where(
                relax_upper,
                np.maximum(
                    upper,
                    np.minimum(effective_box_baseline_error, previous.effective_upper_limits),
                ),
                upper,
            )
        eligible = None
        if self.so3_preprocessor is not None and self.maximum_release_coordinate_step_rad is not None:
            raise ValueError('Select SO(3) preprocessing or the historical coordinate governor, not both')
        step_budget = self.maximum_release_coordinate_step_rad if allow_boundary_governor else None
        if step_budget is not None:
            # Freeze stage eligibility independently of a predicted numeric zero.
            # begin_stage/rebase_subtask reset the mask; accepted history is the
            # source when enabling the experiment after an initialized release.
            if self._coordinate_governor_mask is None:
                self._coordinate_governor_mask = (
                    previous.mask if previous is not None else ranged
                ).copy()
            eligible = self._coordinate_governor_mask.copy()
            lower = np.where(eligible, lower, 0.)
            upper = np.where(eligible, upper, 0.)
            governed = govern_coordinate_box(
                effective_box_baseline_error, lower, upper, eligible, step_budget,
            )
            effective_lower, effective_upper = governed.lower, governed.upper
            # Eligibility is stable; an exactly recovered zero-width interval
            # becomes a Specific equality to avoid dependent inequalities.
            ranged = eligible & (effective_lower < effective_upper)
            loss_reference, _ = constraint_consistent_release_loss_reference(
                current_reference, carried_rotation, frame, ranged,
            )
        so3_result = None
        nominal_step = None
        if self.so3_preprocessor is not None and allow_boundary_governor:
            from .so3_preprocessor import rotation_distance
            if self._coordinate_governor_mask is None:
                self._coordinate_governor_mask = (previous.mask if previous is not None else ranged).copy()
            eligible = self._coordinate_governor_mask.copy()
            lower, upper = np.where(eligible, lower, 0.), np.where(eligible, upper, 0.)
            so3_result = preprocess_so3_box(
                self.accepted_rotation, current_reference, frame, effective_box_baseline_error,
                lower, upper, eligible, self.so3_preprocessor,
            )
            # Replace, rather than compose with, the legacy contraction envelope.
            effective_lower, effective_upper = so3_result.lower.copy(), so3_result.upper.copy()
            ranged = eligible & (effective_lower < effective_upper)
            # Keep the release-loss soft reference defined by the fixed stage
            # eligibility, even if a numeric prediction temporarily becomes zero.
            loss_reference, _ = constraint_consistent_release_loss_reference(
                current_reference, carried_rotation, frame, eligible,
            )
            nominal_step = rotation_distance(current_reference, accepted_reference)
        return RotationReleaseStep(
            nominal_rotation=nominal.copy(),
            stage_handoff_rotation=self.stage_handoff_rotation.copy(),
            stage_handoff_nominal_rotation=(
                self.stage_handoff_nominal_rotation.copy()
            ),
            tolerance_frame=frame.copy(),
            mask=ranged.copy(),
            lower_limits=lower.copy(),
            upper_limits=upper.copy(),
            effective_box_baseline_error=effective_box_baseline_error.copy(),
            effective_lower_limits=effective_lower,
            effective_upper_limits=effective_upper,
            carried_rotation=carried_rotation.copy(),
            release_loss_reference_rotation=loss_reference.copy(),
            maximum_release_coordinate_step_rad=step_budget,
            governor_mask=eligible,
            so3_result=so3_result,
            previous_accepted_rotation=self.accepted_rotation.copy() if so3_result is not None else None,
            nominal_reference_step_rad=nominal_step,
        )

    def candidate(
        self,
        q: np.ndarray,
        step: RotationReleaseStep,
    ) -> RotationReleaseCandidate:
        """Evaluate a physical or joint-space candidate in a frozen cycle."""
        command = np.asarray(q, dtype=float)
        if command.shape == (3, 3):
            rotation = command
        elif self.rotation_for_joint is not None:
            rotation = np.asarray(self.rotation_for_joint(command), dtype=float)
        else:
            raise ValueError("A joint-to-rotation evaluator is required")
        error = step.stage_relative_coordinates(rotation)
        incremental_release = step.incremental_release_coordinates(rotation)
        return RotationReleaseCandidate(
            rotation=rotation.copy(),
            stage_relative_error_fixed_xyz_rad=error.copy(),
            stage_relative_tolerance_release_rad=np.where(
                step.mask, error, 0.0,
            ),
            predicted_tolerance_violation_rad=step.predicted_violation(error),
            incremental_release_rad=incremental_release.copy(),
        )

    def commit(
        self, q_solution: np.ndarray, step: RotationReleaseStep,
    ) -> None:
        """Atomically accept a publishable physical/nominal snapshot pair."""
        command = np.asarray(q_solution, dtype=float)
        if command.shape == (3, 3):
            accepted = command.copy()
        elif self.rotation_for_joint is not None:
            accepted = np.asarray(
                self.rotation_for_joint(command), dtype=float,
            ).copy()
        else:
            raise ValueError("A joint-to-rotation evaluator is required")
        if accepted.shape != (3, 3):
            raise ValueError("Accepted rotation must have shape (3, 3)")
        self.accepted_rotation = accepted
        self.accepted_nominal_rotation = step.nominal_rotation.copy()
        self._accepted_step = step


__all__ = (
    "RotationReleaseCandidate",
    "RotationReleaseState",
    "RotationReleaseStep",
)
