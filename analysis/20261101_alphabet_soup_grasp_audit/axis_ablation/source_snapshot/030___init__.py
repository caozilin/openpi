"""Shared task-space kinematics, tolerance tasks, and objective functions.

This package contains solver-independent building blocks used by the IPOPT
single-step and Block backends.  It intentionally has no standalone IK
solver; all optimization is owned by the active backend.
"""

from .conditioning import (
    ConditioningMetrics,
    ConditioningSafeguardConfig,
    conditioning_activation,
    conditioning_metrics,
    conditioning_penalty,
    conditioning_region,
)
from .objective import (
    MotionState,
    ObjectiveSettings,
    TaskObjective,
    joint_jerk_step,
    make_controller_objective,
)
from .kinematics import KinematicState, PandaKinematics, RobotKinematics
from .types import (
    AxisTask,
    LossParameters,
    TaskKind,
    resolve_axis_tasks,
    validate_axis_task,
)

__all__ = (
    "AxisTask",
    "ConditioningMetrics",
    "ConditioningSafeguardConfig",
    "KinematicState",
    "LossParameters",
    "MotionState",
    "ObjectiveSettings",
    "PandaKinematics",
    "TaskObjective",
    "RobotKinematics",
    "TaskKind",
    "conditioning_activation",
    "conditioning_metrics",
    "conditioning_penalty",
    "conditioning_region",
    "joint_jerk_step",
    "make_controller_objective",
    "resolve_axis_tasks",
    "validate_axis_task",
)
