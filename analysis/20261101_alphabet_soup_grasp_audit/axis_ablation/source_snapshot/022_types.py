"""Settings and results for whole-stage Block optimization through IPOPT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ipopt.adapter import IpoptSettings


@dataclass(frozen=True)
class BlockStageReference:
    """Decision-dependent rotational reference for one trajectory knot.

    ``anchor_knot`` identifies the earlier joint configuration whose FK
    rotation is the physical stage handoff. ``anchor_nominal_rotation`` is
    the nominal planner rotation at that handoff.  The knot's nominal target
    is transported from the physical handoff with the exact stage-relative
    rule used by the online runtime.
    """

    anchor_knot: int
    anchor_nominal_rotation: np.ndarray

    def __post_init__(self) -> None:
        if self.anchor_knot < 0:
            raise ValueError("Block stage anchor knot must be non-negative")
        rotation = np.asarray(self.anchor_nominal_rotation, dtype=float).copy()
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(
                "Block stage anchor nominal rotation must be a finite 3x3 matrix",
            )
        rotation.setflags(write=False)
        object.__setattr__(self, "anchor_nominal_rotation", rotation)


@dataclass(frozen=True)
class BlockSQPSettings:
    """Whole-stage Block settings backed by IPOPT.

    One Block epoch is one complete IPOPT solve. The caller applies its
    trajectory-metric guard and best-result selection between epochs; IPOPT's
    internal barrier iterations must not be split across solver restarts.
    """

    horizon: int
    ipopt: IpoptSettings = IpoptSettings(
        max_iterations=100,
        max_cpu_time_s=1.0e6,
        position_tolerance=1e-5,
        rotation_tolerance=1e-4,
    )

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("BlockSQP horizon must be positive")


@dataclass(frozen=True)
class BlockSQPDiagnostics:
    cost: float
    seed_cost: float
    seed_feasible: bool
    seed_position_residual: float
    seed_strict_rotation_residual: float
    seed_tolerance_violation: float
    iterations: int
    optimality: float
    elapsed_ms: float
    feasible: bool
    # True for an accepted IPOPT termination, including iteration/time limits;
    # ``step_succeeded`` additionally requires independently checked feasibility.
    iteration_succeeded: bool
    step_succeeded: bool
    converged: bool
    position_residual: float
    strict_rotation_residual: float
    tolerance_violation: float
    status: str
    horizon: int
    task_costs: dict[str, float]
    seed_task_costs: dict[str, float]
