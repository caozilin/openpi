"""Decoupled joint-reference tracking for kinematic controller outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class JointReferenceTrackerSettings:
    """Outer-loop gains around MuJoCo's existing joint position servos.

    The tracker never changes the planner reference.  It only biases the
    actuator command enough to remove the physical joint tracking residual.
    """

    proportional_gain: float = 0.18
    integral_gain_s: float = 0.30
    velocity_gain_s: float = 0.04
    maximum_correction_rad: float = math.radians(3.0)
    integration_error_limit_rad: float = math.radians(4.0)
    integral_time_constant_s: float = 1.0
    stationary_integral_time_constant_s: float = 0.25
    stationary_velocity_threshold_rad_s: float = 0.02

    def __post_init__(self) -> None:
        values = np.asarray((
            self.proportional_gain,
            self.integral_gain_s,
            self.velocity_gain_s,
            self.maximum_correction_rad,
            self.integration_error_limit_rad,
            self.integral_time_constant_s,
            self.stationary_integral_time_constant_s,
            self.stationary_velocity_threshold_rad_s,
        ), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Joint tracker settings must be finite and non-negative")
        if self.maximum_correction_rad <= 0.0:
            raise ValueError("maximum_correction_rad must be positive")
        if (
            self.integral_time_constant_s <= 0.0
            or self.stationary_integral_time_constant_s <= 0.0
        ):
            raise ValueError("Integral time constants must be positive")


@dataclass(frozen=True)
class JointTrackingDiagnostics:
    maximum_error_rad: float
    maximum_correction_rad: float
    saturated: bool


class JointReferenceTracker:
    """Track a joint reference with bounded leaky PID position correction."""

    def __init__(
        self,
        joint_count: int,
        settings: JointReferenceTrackerSettings = (
            JointReferenceTrackerSettings()
        ),
    ) -> None:
        if not isinstance(joint_count, int) or joint_count <= 0:
            raise ValueError("joint_count must be a positive integer")
        self.joint_count = joint_count
        self.settings = settings
        self._integral_error = np.zeros(self.joint_count, dtype=float)
        self._previous_reference: np.ndarray | None = None
        self._previous_actual: np.ndarray | None = None

    def reset(self) -> None:
        self._integral_error[:] = 0.0
        self._previous_reference = None
        self._previous_actual = None

    @property
    def integral_error(self) -> np.ndarray:
        return self._integral_error.copy()

    def update(
        self,
        reference_q: np.ndarray,
        actual_q: np.ndarray,
        dt: float,
        actual_velocity_q: np.ndarray | None = None,
    ) -> tuple[np.ndarray, JointTrackingDiagnostics]:
        reference = np.asarray(reference_q, dtype=float)
        actual = np.asarray(actual_q, dtype=float)
        actual_velocity = (
            None
            if actual_velocity_q is None
            else np.asarray(actual_velocity_q, dtype=float)
        )
        if (
            reference.shape != (self.joint_count,)
            or actual.shape != (self.joint_count,)
            or not np.all(np.isfinite(reference))
            or not np.all(np.isfinite(actual))
            or not np.isfinite(dt)
            or dt <= 0.0
            or (
                actual_velocity is not None
                and (
                    actual_velocity.shape != (self.joint_count,)
                    or not np.all(np.isfinite(actual_velocity))
                )
            )
        ):
            raise ValueError("Joint tracker inputs must be finite and correctly shaped")

        error = reference - actual
        reference_velocity = (
            np.zeros_like(reference)
            if self._previous_reference is None
            else (reference - self._previous_reference) / dt
        )
        if actual_velocity is None:
            actual_velocity = (
                np.zeros_like(actual)
                if self._previous_actual is None
                else (actual - self._previous_actual) / dt
            )
        velocity_error = reference_velocity - actual_velocity
        stationary = bool(
            np.max(np.abs(reference_velocity), initial=0.0)
            <= self.settings.stationary_velocity_threshold_rad_s
        )
        integral_time_constant = (
            self.settings.stationary_integral_time_constant_s
            if stationary else self.settings.integral_time_constant_s
        )
        integral_decay = math.exp(-dt / integral_time_constant)
        integrable_error = np.clip(
            error,
            -self.settings.integration_error_limit_rad,
            self.settings.integration_error_limit_rad,
        )
        proposed_integral = (
            integral_decay * self._integral_error
            + integrable_error * dt
        )
        if self.settings.integral_gain_s > 0.0:
            integral_limit = (
                self.settings.maximum_correction_rad
                / self.settings.integral_gain_s
            )
            proposed_integral = np.clip(
                proposed_integral, -integral_limit, integral_limit,
            )

        raw_correction = (
            self.settings.proportional_gain * error
            + self.settings.integral_gain_s * proposed_integral
            + self.settings.velocity_gain_s * velocity_error
        )
        correction = np.clip(
            raw_correction,
            -self.settings.maximum_correction_rad,
            self.settings.maximum_correction_rad,
        )
        command = reference + correction
        saturated_axes = np.abs(raw_correction - correction) > 1.0e-12

        # Conditional integration prevents windup when the requested
        # correction or the actuator command is already saturated and the
        # present error would push farther into that saturation.
        pushing_outward = saturated_axes & (
            np.sign(error) == np.sign(raw_correction)
        )
        self._integral_error[~pushing_outward] = proposed_integral[
            ~pushing_outward
        ]
        self._previous_reference = reference.copy()
        self._previous_actual = actual.copy()

        diagnostics = JointTrackingDiagnostics(
            maximum_error_rad=float(np.max(np.abs(error), initial=0.0)),
            maximum_correction_rad=float(
                np.max(np.abs(command - reference), initial=0.0)
            ),
            saturated=bool(np.any(saturated_axes)),
        )
        return command, diagnostics
