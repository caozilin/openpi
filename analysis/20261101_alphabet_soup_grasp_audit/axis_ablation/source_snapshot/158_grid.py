"""Shared discretization helpers for directional tolerance searches."""

from __future__ import annotations

import math


def upward_aligned_maxima(
    peaks_deg: tuple[float, ...] | list[float],
    step_deg: float = 5.0,
    *,
    numerical_zero_deg: float = 1.0e-5,
) -> tuple[float, ...]:
    """Ceil measured directional peaks and enforce the hard 30° ceiling.

    Solver residuals within ``numerical_zero_deg`` of zero or a grid multiple
    are snapped first. Keep raw measured peaks separately from these caps.
    """
    step, epsilon = float(step_deg), float(numerical_zero_deg)
    if not math.isfinite(step) or step <= 0 or not math.isclose(30 / step, round(30 / step)):
        raise ValueError("step_deg must be finite, positive and divide 30 degrees")
    if not math.isfinite(epsilon) or not 0 <= epsilon < step / 2:
        raise ValueError("numerical_zero_deg must be finite and below half a step")
    result = []
    for value in peaks_deg:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Release peaks must be finite and nonnegative")
        nearest = round(value / step) * step
        if abs(value - nearest) <= epsilon:
            value = nearest
        result.append(min(30.0, round(math.ceil(value / step) * step, 12)))
    return tuple(result)


def box_corner_initial_caps(
    maxima_deg: tuple[float, ...] | list[float],
) -> tuple[tuple[float, ...], ...]:
    """Zero, full box, each single-max axis, then each single-zero axis.

    Dimensions are SIGNED directions. Stable first-occurrence deduplication
    gives 1/2/4/8/10/12/14 distinct boxes for k=0/1/2/3/4/5/6.
    """
    maximum = tuple(float(v) for v in maxima_deg)
    if not maximum or any(not math.isfinite(v) or v < 0 for v in maximum):
        raise ValueError("Require finite nonnegative maxima")
    zero = (0.0,) * len(maximum)
    active = [i for i, value in enumerate(maximum) if value > 0]
    points = [zero, maximum]
    for i in active:
        point = list(zero)
        point[i] = maximum[i]
        points.append(tuple(point))
    for i in active:
        point = list(maximum)
        point[i] = 0.0
        points.append(tuple(point))
    return tuple(dict.fromkeys(points))


def descending_aligned_caps(
    start_deg: float,
    step_deg: float,
) -> tuple[float, ...]:
    """Align the first cap downward, then return multiples to exact zero.

    For example, ``26.14`` with a ``2`` degree step produces
    ``(26, 24, ..., 2, 0)``.  The separately recorded untrimmed baseline is
    not duplicated in the grid.
    """
    start = float(start_deg)
    step = float(step_deg)
    if not math.isfinite(start) or start < 0.0:
        raise ValueError("start_deg must be finite and non-negative")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step_deg must be finite and positive")
    if start == 0.0:
        return (0.0,)

    aligned = math.floor((start + 1.0e-12) / step) * step
    aligned = min(start, round(aligned, 12))
    values = [aligned]
    while values[-1] > step + 1.0e-12:
        values.append(round(values[-1] - step, 12))
    if not math.isclose(values[-1], 0.0, abs_tol=1.0e-12):
        values.append(0.0)
    else:
        values[-1] = 0.0
    return tuple(values)


def aligned_reduction_values(
    maximum_deg: float,
    step_deg: float,
) -> tuple[float, ...]:
    """Return reductions corresponding to :func:`descending_aligned_caps`."""
    maximum = float(maximum_deg)
    return tuple(
        round(maximum - cap, 12)
        for cap in descending_aligned_caps(maximum, step_deg)
    )
