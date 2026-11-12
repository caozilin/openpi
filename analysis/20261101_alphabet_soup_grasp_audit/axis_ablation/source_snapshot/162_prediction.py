"""Six-dimensional policy tolerance validation and benchmark ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tasks.phases import ManipulationPhase
from .specification import ROTATION_TOLERANCE_MAXIMUM_RAD


MODEL_TOLERANCE_MAXIMUM_RAD = np.full(6, ROTATION_TOLERANCE_MAXIMUM_RAD)


@dataclass(frozen=True)
class DecodedStagePrediction:
    tolerance_bounds_rad: np.ndarray
    raw_tolerance_bounds_rad: np.ndarray
    clipped_tolerance_bounds_rad: np.ndarray
    ranged_axes: np.ndarray


class StagePredictionDecoder:
    """Validate and clip six directional policy tolerance bounds."""

    def decode(
        self,
        phase: ManipulationPhase,
        tolerance_bounds_rad: np.ndarray,
    ) -> DecodedStagePrediction:
        if phase not in (ManipulationPhase.PREGRASP, ManipulationPhase.POSTGRASP):
            raise ValueError("Policy tolerances are decoded only in Pre/Post")
        raw_tolerance = np.asarray(tolerance_bounds_rad, dtype=float).copy()
        if raw_tolerance.shape != (6,) or not np.all(np.isfinite(raw_tolerance)):
            raise ValueError("Policy rotation_tolerance must contain six finite values")
        clipped = np.clip(raw_tolerance, 0.0, MODEL_TOLERANCE_MAXIMUM_RAD)
        return DecodedStagePrediction(
            tolerance_bounds_rad=clipped.copy(),
            raw_tolerance_bounds_rad=raw_tolerance,
            clipped_tolerance_bounds_rad=clipped,
            ranged_axes=(
                (clipped[0::2] > 0.0) | (clipped[1::2] > 0.0)
            ),
        )


# Extracted from datasets/v1_tolerance_summary_v2.csv. Values are directional
# degrees ordered as Pre [Rx-, Rx+, Ry-, Ry+, Rz-, Rz+] followed by Post in
# the same order. This table is intentionally runtime-independent of the CSV.
_GROUND_TRUTH_DEG: dict[tuple[str, str], tuple[float, ...]] = {
    ('panda', 'adjust_cylindrical_bottle'): (30,30,30,10,0,0, 0,0,0,0,30,30),
    ('panda', 'adjust_rectangular_bottle'): (0,0,30,10,0,0, 0,0,0,0,30,30),
    ('panda', 'click_bell'): (30,30,30,30,30,30, 30,30,30,30,30,30),
    ('panda', 'close_cylindrical_pot_lid'): (0,0,30,30,30,30, 0,0,0,0,30,30),
    ('panda', 'close_handle_pot_lid'): (30,30,30,30,0,0, 0,0,0,0,30,30),
    ('panda', 'banana_to_plate'): (10,10,0,0,0,0, 30,30,30,30,30,30),
    ('panda', 'banana_to_tray'): (10,10,0,0,0,0, 30,30,30,30,30,30),
    ('panda', 'pear_to_bowl'): (30,30,30,30,30,30, 30,30,30,30,30,30),
    ('panda', 'pear_to_plate'): (30,30,30,30,30,30, 30,30,30,30,30,30),
    ('panda', 'pear_to_tray'): (30,30,30,30,30,30, 30,30,30,30,30,30),
    ('panda', 'strawberry_to_tray'): (30,30,30,30,30,30, 30,30,30,30,30,30),
    ('panda', 'geometry_plate_ball'): (20,20,30,30,30,30, 20,20,30,30,30,30),
    ('panda', 'geometry_plate_box_lying'): (0,0,30,30,0,0, 30,30,30,30,30,30),
    ('panda', 'geometry_plate_box_upright'): (0,0,30,30,0,0, 0,0,0,0,30,30),
    ('panda', 'geometry_plate_cube'): (0,0,30,30,0,0, 30,30,30,30,30,30),
    ('panda', 'geometry_plate_cylinder_lying'): (5,5,0,0,0,0, 20,20,30,30,30,30),
    ('panda', 'geometry_plate_cylinder_upright'): (0,0,30,30,30,30, 0,0,0,0,30,30),
    ('panda', 'geometry_region_box_lying'): (0,0,30,30,0,0, 0,0,0,0,0,0),
    ('panda', 'geometry_region_box_upright'): (0,0,30,30,0,0, 0,0,0,0,0,0),
    ('panda', 'geometry_region_cube'): (0,0,30,30,0,0, 0,0,0,0,0,0),
    ('panda', 'geometry_region_cylinder_lying'): (10,10,30,30,0,0, 30,30,0,0,0,0),
    ('panda', 'geometry_region_cylinder_upright'): (0,0,30,30,30,30, 0,0,0,0,30,30),
    ('panda', 'open_cylindrical_pot_lid'): (0,0,30,30,30,30, 0,0,0,0,30,30),
    ('panda', 'open_handle_pot_lid'): (30,30,30,30,0,0, 0,0,0,0,30,30),
    ('panda', 'press_power_strip'): (30,30,30,30,30,30, 30,30,30,30,30,30),
    ('panda', 'rotate_knob'): (0,0,30,30,0,0, 0,0,0,0,0,0),
    ('ur5e', 'geometry_plate_ball'): (30,30,30,30,30,30, 30,30,30,30,30,30),
    ('ur5e', 'geometry_plate_box_lying'): (0,0,30,30,0,0, 30,30,30,30,30,30),
    ('ur5e', 'geometry_plate_box_upright'): (0,0,30,30,0,0, 0,0,0,0,30,30),
    ('ur5e', 'geometry_plate_cube'): (0,0,30,30,0,0, 30,30,30,30,30,30),
    ('ur5e', 'geometry_plate_cylinder_lying'): (30,30,30,30,0,0, 30,30,30,30,30,30),
    ('ur5e', 'geometry_plate_cylinder_upright'): (0,0,30,30,30,30, 0,0,0,0,30,30),
    ('ur5e', 'geometry_region_box_lying'): (0,0,30,30,0,0, 0,0,0,0,0,0),
    ('ur5e', 'geometry_region_box_upright'): (0,0,30,30,0,0, 0,0,0,0,0,0),
    ('ur5e', 'geometry_region_cube'): (0,0,10,30,0,0, 0,0,0,0,0,0),
    ('ur5e', 'geometry_region_cylinder_lying'): (30,30,30,30,0,0, 30,30,0,0,0,0),
    ('ur5e', 'geometry_region_cylinder_upright'): (0,0,30,30,30,30, 0,0,0,0,30,30),
    ('xarm7', 'geometry_plate_ball'): (30,20,30,30,30,30, 30,30,30,30,30,30),
    ('xarm7', 'geometry_plate_box_lying'): (0,0,30,30,0,0, 30,30,30,30,30,30),
    ('xarm7', 'geometry_plate_box_upright'): (0,0,30,30,0,0, 0,0,0,0,30,30),
    ('xarm7', 'geometry_plate_cube'): (0,0,30,30,0,0, 30,30,30,30,30,30),
    ('xarm7', 'geometry_plate_cylinder_lying'): (30,30,30,30,0,0, 30,30,30,30,30,30),
    ('xarm7', 'geometry_plate_cylinder_upright'): (0,0,30,30,30,30, 0,0,0,0,30,30),
    ('xarm7', 'geometry_region_box_lying'): (0,0,30,30,0,0, 0,0,0,0,0,0),
    ('xarm7', 'geometry_region_box_upright'): (0,0,30,30,0,0, 0,0,0,0,0,0),
    ('xarm7', 'geometry_region_cube'): (0,0,30,20,0,0, 0,0,0,0,0,0),
    ('xarm7', 'geometry_region_cylinder_lying'): (30,30,30,30,0,0, 30,30,0,0,0,0),
    ('xarm7', 'geometry_region_cylinder_upright'): (0,0,30,30,30,30, 0,0,0,0,30,30),
}


def ground_truth_tolerance_bounds_rad(
    robot_uid: str,
    dataset_task_id: str,
    phase: ManipulationPhase,
) -> np.ndarray:
    """Return the hard-coded v2 directional tolerance for Pre or Post."""
    try:
        values = _GROUND_TRUTH_DEG[(str(robot_uid), str(dataset_task_id))]
    except KeyError as error:
        raise KeyError(
            "No v1_tolerance_summary_v2 ground truth for "
            f"robot={robot_uid!r}, task={dataset_task_id!r}"
        ) from error
    if phase is ManipulationPhase.PREGRASP:
        selected = values[:6]
    elif phase is ManipulationPhase.POSTGRASP:
        selected = values[6:]
    else:
        return np.zeros(6)
    return np.radians(np.asarray(selected, dtype=float))


__all__ = (
    "DecodedStagePrediction",
    "MODEL_TOLERANCE_MAXIMUM_RAD",
    "StagePredictionDecoder",
    "ground_truth_tolerance_bounds_rad",
)
