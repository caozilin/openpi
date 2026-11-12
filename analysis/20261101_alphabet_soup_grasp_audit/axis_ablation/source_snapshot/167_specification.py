"""Immutable TaskTol-VLA tolerance annotations.

These types describe labels, not optimizer state. A task tolerance is an
ordered fixed-axis XYZ/RPY orientation offset from a VLA nominal target,
expressed in an independently annotated tolerance frame. Bound magnitudes use
the flat axis-pair order
``[Rx negative, Rx positive, Ry negative, Ry positive, Rz negative, Rz positive]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import canonical_rotation_vector, rotation_matrix
from ..tasks.phases import ManipulationPhase


ROTATION_TOLERANCE_COORDINATE_CONVENTION = "fixed_axis_xyz_rpy"
ROTATION_TOLERANCE_COMPOSITION = "Rz(yaw) @ Ry(pitch) @ Rx(roll)"
ROTATION_TOLERANCE_MAXIMUM_RAD = math.radians(30.0)
ROTATION_TOLERANCE_PROFILE_ORDER = (
    "rx_negative", "rx_positive",
    "ry_negative", "ry_positive",
    "rz_negative", "rz_positive",
)


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    result.setflags(write=False)
    return result


def rotation_tolerance_bounds(value: object, name: str = "rotation tolerance") -> np.ndarray:
    """Validate fixed-axis XYZ/RPY negative/positive bound magnitudes."""
    result = np.asarray(value, dtype=float).copy()
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain six finite values")
    if np.any(result < 0.0):
        raise ValueError(
            f"{name} must use six non-negative [Rx-, Rx+, Ry-, Ry+, Rz-, Rz+] magnitudes"
        )
    if np.any(result > ROTATION_TOLERANCE_MAXIMUM_RAD + 1e-12):
        raise ValueError(f"{name} values must not exceed 30 degrees")
    result.setflags(write=False)
    return result


def symmetric_rotation_tolerance_bounds(half_widths: object) -> np.ndarray:
    """Expand three non-negative half-widths to six signed bounds."""
    widths = _vector3(np.asarray(half_widths, dtype=float), "tolerance half-widths")
    if np.any(widths < 0.0):
        raise ValueError("Tolerance half-widths cannot be negative")
    return rotation_tolerance_bounds(np.repeat(widths, 2), "tolerance half-widths")


@dataclass(frozen=True)
class StageTolerance:
    """One stable phase's target, frame, and ordered XYZ/RPY bounds."""

    phase: ManipulationPhase
    target_rotation_vector_base_rad: np.ndarray
    tolerance_frame_rotation_vector_base_rad: np.ndarray
    rotation_tolerance_bounds_rad: np.ndarray

    def __post_init__(self) -> None:
        if self.phase in (ManipulationPhase.GRASP, ManipulationPhase.RELEASE):
            raise ValueError("Grasp and Release are strict zero-tolerance phases")
        target = _vector3(self.target_rotation_vector_base_rad, "target rotation")
        frame = _vector3(self.tolerance_frame_rotation_vector_base_rad, "tolerance frame")
        frame = canonical_rotation_vector(rotation_matrix(frame))
        frame.setflags(write=False)
        bounds = rotation_tolerance_bounds(self.rotation_tolerance_bounds_rad)
        object.__setattr__(self, "target_rotation_vector_base_rad", target)
        object.__setattr__(self, "tolerance_frame_rotation_vector_base_rad", frame)
        object.__setattr__(self, "rotation_tolerance_bounds_rad", bounds)

    @property
    def negative_limits_rad(self) -> np.ndarray:
        return self.rotation_tolerance_bounds_rad[0::2]

    @property
    def positive_limits_rad(self) -> np.ndarray:
        return self.rotation_tolerance_bounds_rad[1::2]

    @property
    def maximum_limits_rad(self) -> np.ndarray:
        return np.maximum(self.negative_limits_rad, self.positive_limits_rad)

    @property
    def ranged_axes(self) -> np.ndarray:
        return (self.negative_limits_rad > 0.0) | (self.positive_limits_rad > 0.0)


@dataclass(frozen=True)
class TaskTolerance:
    """Pre-grasp/Post-grasp tolerance labels for one pick-place subtask."""

    pregrasp: StageTolerance
    postgrasp: StageTolerance

    def __post_init__(self) -> None:
        if self.pregrasp.phase is not ManipulationPhase.PREGRASP:
            raise ValueError("pregrasp annotation must use the Pre-grasp phase")
        if self.postgrasp.phase is not ManipulationPhase.POSTGRASP:
            raise ValueError("postgrasp annotation must use the Post-grasp phase")

    def stable_stage(self, phase: ManipulationPhase) -> StageTolerance:
        if phase is ManipulationPhase.PREGRASP:
            return self.pregrasp
        if phase is ManipulationPhase.POSTGRASP:
            return self.postgrasp
        raise ValueError("Grasp and Release are strict zero-tolerance phases")
