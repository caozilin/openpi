"""Full-trajectory kinematic degradation for executable tolerances."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


KINEMATIC_METRIC_NAMES = (
    "mean_joint_velocity_rad_s",
    "mean_joint_acceleration_rad_s2",
    "mean_joint_jerk_rad_s3",
    "mean_manipulability",
)

DEFAULT_KINEMATIC_WEIGHTS = {
    "mean_joint_velocity_rad_s": 0.15,
    "mean_joint_acceleration_rad_s2": 0.30,
    "mean_joint_jerk_rad_s3": 0.3,
    "mean_manipulability": 0.25,
}

# Floors only prevent a nearly-zero nominal term from making an otherwise
# harmless numerical difference infinite. They use the units of each metric.
_RATIO_FLOORS = {
    "mean_joint_velocity_rad_s": 1.0e-3,
    "mean_joint_acceleration_rad_s2": 1.0e-2,
    "mean_joint_jerk_rad_s3": 1.0e-1,
    "mean_manipulability": 1.0e-4,
}


@dataclass(frozen=True)
class TrajectoryKinematicMetrics:
    """Controller-independent statistics over one complete closed-loop rollout."""

    samples: int = 0
    mean_joint_velocity_rad_s: float = 0.0
    mean_joint_acceleration_rad_s2: float = 0.0
    mean_joint_jerk_rad_s3: float = 0.0
    mean_manipulability: float = 0.0
    # Sum of sampled joint-space L2 increments, including the initial pose
    # when supplied by the execution backend. No duration normalization.
    total_joint_travel_rad: float = 0.0
    # Full raw 6-D geometric Jacobian, minimum over recorded states.
    # None distinguishes historical/missing observations from a singular pose.
    minimum_sigma_min_full: float | None = None
    # Dx^-1 Jg Dq, velocity_capability_v1; historical raw-only data stays None.
    minimum_sigma_min_velocity_normalized: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object] | None,
    ) -> "TrajectoryKinematicMetrics":
        if not payload:
            return cls()
        return cls(**{
            field: payload.get(field, getattr(cls(), field))
            for field in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class KinematicDegradation:
    """Relative soft loss of a perturbed trajectory against its nominal pair."""

    weighted_loss_ratio: float
    metric_ratios: dict[str, float]
    weighted_terms: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_trajectory_kinematics(
    nominal: TrajectoryKinematicMetrics,
    perturbed: TrajectoryKinematicMetrics,
    weights: Mapping[str, float] = DEFAULT_KINEMATIC_WEIGHTS,
) -> KinematicDegradation:
    """Return a dimensionless weighted degradation ratio (nominal is 1).

    Every error/motion term is ``perturbed / nominal``. Manipulability is a
    benefit, so its ratio is inverted. This makes unlike physical units
    composable while retaining explicit per-metric diagnostics.
    """
    checked_weights = {
        name: float(weights.get(name, 0.0))
        for name in KINEMATIC_METRIC_NAMES
    }
    if any(not np.isfinite(value) or value < 0.0 for value in checked_weights.values()):
        raise ValueError("Kinematic weights must be finite and non-negative")
    weight_sum = sum(checked_weights.values())
    if weight_sum <= 0.0:
        raise ValueError("At least one kinematic weight must be positive")

    # Old cached/fake trials may not contain trajectory statistics. Treat a
    # missing pair as neutral so cache migration remains possible.
    if nominal.samples <= 0 or perturbed.samples <= 0:
        ratios = {name: 1.0 for name in KINEMATIC_METRIC_NAMES}
    else:
        ratios: dict[str, float] = {}
        for name in KINEMATIC_METRIC_NAMES:
            nominal_value = float(getattr(nominal, name))
            perturbed_value = float(getattr(perturbed, name))
            floor = _RATIO_FLOORS[name]
            if name == "mean_manipulability":
                ratio = max(nominal_value, floor) / max(perturbed_value, floor)
            else:
                ratio = max(perturbed_value, floor) / max(nominal_value, floor)
            ratios[name] = float(ratio) if np.isfinite(ratio) else 1.0e9

    terms = {
        name: checked_weights[name] * ratios[name]
        for name in KINEMATIC_METRIC_NAMES
    }
    return KinematicDegradation(
        weighted_loss_ratio=float(sum(terms.values()) / weight_sum),
        metric_ratios=ratios,
        weighted_terms=terms,
    )
