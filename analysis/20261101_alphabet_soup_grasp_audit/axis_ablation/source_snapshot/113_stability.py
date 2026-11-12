"""Velocity-based stability predicates for free task objects."""

from __future__ import annotations

import mujoco
import numpy as np


def body_free_joint_speeds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[float, float]:
    """Return linear and angular speed norms for a free-joint body."""
    dof_address = int(model.body_dofadr[body_id])
    if dof_address < 0 or int(model.body_dofnum[body_id]) < 6:
        raise ValueError("Task object must have a six-DoF free joint")
    velocity = data.qvel[dof_address:dof_address + 6]
    return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))


def speeds_within_limits(
    linear_speed_m_s: float,
    angular_speed_rad_s: float,
    linear_limit_m_s: float,
    angular_limit_rad_s: float,
) -> bool:
    return (
        float(linear_speed_m_s) < float(linear_limit_m_s)
        and float(angular_speed_rad_s) < float(angular_limit_rad_s)
    )

