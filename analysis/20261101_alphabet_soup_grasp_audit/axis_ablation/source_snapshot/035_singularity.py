"""Scalar soft losses on velocity-capability-normalized sigma_min."""
from __future__ import annotations

import numpy as np


SIGMA_MIN_LOSS_KINDS = ("negative", "squared_hinge", "reciprocal_squared_hinge")
SIGMA_MIN_PENALTY_PROFILES = {
    "delta_q_squared_sigma_min_squared_hinge": "squared_hinge",
    "delta_q_squared_sigma_min_reciprocal_squared_hinge": "reciprocal_squared_hinge",
}


def validate_sigma_min_loss(kind: str, warning_threshold: float, reciprocal_epsilon: float) -> None:
    if kind not in SIGMA_MIN_LOSS_KINDS:
        raise ValueError(f"Unknown sigma_min loss kind: {kind}")
    if not np.isfinite(warning_threshold) or warning_threshold <= 0:
        raise ValueError("sigma_min_warning_threshold must be finite and positive")
    if not np.isfinite(reciprocal_epsilon) or reciprocal_epsilon <= 0:
        raise ValueError("sigma_min_reciprocal_epsilon must be finite and positive")


def sigma_min_loss_and_derivative(
    sigma: float, *, kind: str = "negative", warning_threshold: float = 1.0,
    reciprocal_epsilon: float = 0.001,
) -> tuple[float, float]:
    """Return unweighted phi(s), dphi/ds; every mode remains a soft objective.

    A: max(0, 1-r)^2, r=s/sw.
    B: [max(0, (1+eps)/(r+eps)-1)]^2. Additive reciprocal
       regularization preserves the zero wall at r=1 and a finite restoring
       derivative at r=0; it never clips sigma to a flat floor.
    eps is dimensionless relative to sw. As eps -> 0, B tends to (1/r-1)^2.
    """
    validate_sigma_min_loss(kind, warning_threshold, reciprocal_epsilon)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma_min must be finite and non-negative")
    if kind == "negative":
        return -float(sigma), -1.0
    if sigma >= warning_threshold:
        return 0.0, 0.0
    r = float(sigma / warning_threshold)
    gap = 1.0-r
    if kind == "squared_hinge":
        return gap*gap, -2.0*gap/warning_threshold
    denominator = r+reciprocal_epsilon
    residual = gap/denominator
    derivative = -2.0*residual*(1.0+reciprocal_epsilon)/(warning_threshold*denominator**2)
    return residual*residual, derivative
