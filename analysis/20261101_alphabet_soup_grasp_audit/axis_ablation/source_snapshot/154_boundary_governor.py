"""State-aware correction of predicted rotational tolerance boundaries.

The execution path uses ``govern_coordinate_box`` and a separate optimizer
coordinate-step constraint. ``govern_rotation_box`` retains the historical
SO(3) witness experiment. Robot feasibility is checked by the shared optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import (
    rotation_from_tolerance_coordinates,
    rotation_matrix,
    rotation_tolerance_coordinates,
    rotation_vector,
)


@dataclass(frozen=True)
class GovernedRotationBox:
    lower: np.ndarray
    upper: np.ndarray
    witness_rotation: np.ndarray
    witness_coordinates: np.ndarray
    anchor_coordinates: np.ndarray
    projection_distance_rad: float
    witness_step_rad: float


def govern_rotation_box(
    anchor_rotation: np.ndarray,
    reference_rotation: np.ndarray,
    tolerance_frame: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    maximum_step_rad: float,
) -> GovernedRotationBox:
    """Accept the newest box or minimally extend it to contain a bounded witness.

    Clip current XYZ coordinates into the requested box. Walk toward that pose
    along the shortest SO(3) geodesic, at most ``maximum_step_rad``. Extend only
    the faces needed to include that intermediate pose. This is the smallest
    box containing the requested box and this witness, not a globally nearest
    SO(3) projection onto an Euler box. No history of old bounds is needed.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,)
            or np.any(np.isnan(lower)) or np.any(np.isnan(upper))
            or np.any(lower > 0.) or np.any(upper < 0.)):
        raise ValueError('Bounds must be three-vectors with lower <= 0 <= upper')
    if not np.isfinite(maximum_step_rad) or not 0. < maximum_step_rad < np.pi:
        raise ValueError('Rotation step budget must be finite and in (0, pi)')
    anchor = np.asarray(anchor_rotation, dtype=float)
    reference = np.asarray(reference_rotation, dtype=float)
    frame = np.asarray(tolerance_frame, dtype=float)
    if any(r.shape != (3, 3) or not np.all(np.isfinite(r))
           for r in (anchor, reference, frame)):
        raise ValueError('Rotations must be finite 3x3 matrices')
    current = rotation_tolerance_coordinates(anchor, reference, frame)
    clipped = np.clip(current, lower, upper)
    projected = rotation_from_tolerance_coordinates(reference, frame, clipped)
    direction = rotation_vector(projected @ anchor.T)
    distance = float(np.linalg.norm(direction))
    if distance <= maximum_step_rad:
        # This branch also covers a large boundary change around an idle pose.
        witness, coordinates = projected, clipped
        corrected_lower, corrected_upper = lower.copy(), upper.copy()
    else:
        witness = rotation_matrix(direction * (maximum_step_rad / distance)) @ anchor
        coordinates = rotation_tolerance_coordinates(witness, reference, frame)
        coordinates[np.abs(coordinates) < 1.e-12] = 0.
        corrected_lower = np.minimum(lower, coordinates)
        corrected_upper = np.maximum(upper, coordinates)
    return GovernedRotationBox(
        lower=corrected_lower, upper=corrected_upper,
        witness_rotation=witness, witness_coordinates=coordinates,
        anchor_coordinates=current, projection_distance_rad=distance,
        witness_step_rad=min(distance, maximum_step_rad),
    )


@dataclass(frozen=True)
class GovernedCoordinateBox:
    """Requested-box correction and a separate per-coordinate motion window."""
    lower: np.ndarray
    upper: np.ndarray
    step_lower: np.ndarray
    step_upper: np.ndarray
    witness_coordinates: np.ndarray
    strict_step_conflict: np.ndarray


def govern_coordinate_box(carried_coordinates, lower, upper, eligible_axes, maximum_step_rad):
    """Relax only the requested face unreachable from this cycle's carry.

    The step is a fixed-XYZ coordinate difference in ONE current frozen chart,
    after nominal reference transport. Ineligible axes remain exact zero.
    A strict-axis conflict is reported, never repaired by unlocking that axis.
    """
    carry, lower, upper = (np.asarray(v, dtype=float) for v in (carried_coordinates, lower, upper))
    eligible = np.asarray(eligible_axes, dtype=bool)
    if any(v.shape != (3,) for v in (carry, lower, upper, eligible)):
        raise ValueError('Coordinate boxes and masks require three components')
    if (not np.all(np.isfinite(carry)) or np.any(np.isnan(lower)) or np.any(np.isnan(upper))
            or np.any(lower > 0.) or np.any(upper < 0.)):
        raise ValueError('Finite carry and lower <= 0 <= upper are required')
    if not np.isfinite(maximum_step_rad) or not 0. < maximum_step_rad < np.pi:
        raise ValueError('Coordinate step budget must be finite and in (0, pi)')
    step_lower, step_upper = carry - maximum_step_rad, carry + maximum_step_rad
    corrected_lower = np.where(eligible, np.minimum(lower, step_upper), 0.)
    corrected_upper = np.where(eligible, np.maximum(upper, step_lower), 0.)
    witness = np.where(eligible, np.clip(np.clip(carry, lower, upper), step_lower, step_upper), 0.)
    return GovernedCoordinateBox(corrected_lower, corrected_upper, step_lower, step_upper,
                                witness, (~eligible) & (np.abs(carry) > maximum_step_rad))
