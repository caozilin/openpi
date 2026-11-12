"""Shared data passed between task skills and the teleoperation executor."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# Shared stand-off before the final grasp or press approach.
PREGRASP_CLEARANCE_M = 0.050


class PlanningInfeasibleError(RuntimeError):
    """The planner exhausted its candidate branches, rather than crashing."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.context: dict = {}


@dataclass(frozen=True)
class MotionLimits:
    linear_speed_m_s: float = 0.200
    angular_speed_rad_s: float = math.radians(45.0)
    command_rate_hz: float = 10.0


@dataclass(frozen=True)
class TaskFrame:
    position: np.ndarray
    rotation: np.ndarray
    gripper_closed: bool
    phase: str


@dataclass(frozen=True)
class PreparedTaskPlan:
    """Frozen task references prepared against an unchanged UI snapshot."""

    scene: str
    task: str
    robot_uid: str
    initial_qpos: np.ndarray
    frames: tuple[TaskFrame, ...]
    perturbations: dict[str, float]
    planning_stats: dict
    subtask_index: int = 0


@dataclass(frozen=True)
class TaskCommand:
    position: np.ndarray
    rotation: np.ndarray
    gripper_closed: bool
    status: str


@dataclass(frozen=True)
class TaskAction:
    """One base-frame 6-D Cartesian increment plus one gripper command.

    Rotational increments are always expressed in the robot base frame and
    applied by left multiplication.  Input adapters must convert tool-frame
    commands before constructing a ``TaskAction``.
    """

    value: np.ndarray
    phase: str
    status: str

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=float)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("TaskAction must contain seven finite values")
        object.__setattr__(self, "value", value.copy())

    @property
    def gripper_closed(self) -> bool:
        return bool(self.value[6] > 0.0)
