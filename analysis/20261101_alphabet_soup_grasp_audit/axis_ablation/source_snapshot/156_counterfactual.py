"""Contracts for simulation-based counterfactual tolerance calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tasks.phases import ManipulationPhase


@dataclass(frozen=True)
class PerturbationTrial:
    scene: str
    task: str
    seed: int
    phase: ManipulationPhase
    axis: int
    offset_rad: float
    success: bool
    collision: bool
    termination_reason: str


def conservative_symmetric_limit(
    trials: list[PerturbationTrial],
    *,
    phase: ManipulationPhase,
    axis: int,
) -> float:
    """Largest tested magnitude successful in both perturbation directions.

    This intentionally performs no interpolation and assumes neither monotonic
    success nor untested feasibility. Statistical confidence and adaptive
    sampling can be layered on top without changing stored trial records.
    """
    if phase in (ManipulationPhase.GRASP, ManipulationPhase.RELEASE):
        raise ValueError("Grasp and Release are strict and are not calibrated")
    selected = [
        trial for trial in trials
        if trial.phase is phase and trial.axis == axis and not trial.collision
    ]
    positive = {round(abs(trial.offset_rad), 12) for trial in selected if trial.offset_rad > 0 and trial.success}
    negative = {round(abs(trial.offset_rad), 12) for trial in selected if trial.offset_rad < 0 and trial.success}
    shared = positive & negative
    return 0.0 if not shared else float(max(shared))
