from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def scaled_solver_time_budget(
    base_time_s: float,
    speed_scale: float,
) -> float:
    """Keep iterative-controller wall time proportional to motion period."""
    base = float(base_time_s)
    speed = float(speed_scale)
    if not base > 0.0:
        raise ValueError("Base solver time budget must be positive.")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("Controller speed scale must be finite and positive.")
    return base / speed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "scene.xml"

# The robot pedestal remains mounted 0.50 m above the original model datum.
WORKCELL_ELEVATION_M = 0.50
# The table is intentionally 10 cm lower than that original shared elevation.
# Scene objects use this value so nominal placement and table-domain Z offsets
# stay in one coordinate frame.
TABLE_ELEVATION_M = WORKCELL_ELEVATION_M - 0.10

FRANKA_HOME_Q = np.array(
    [
        0.0,
        -math.pi / 4.0,
        0.0,
        -3.0 * math.pi / 4.0,
        0.0,
        math.pi / 2.0,
        math.pi / 4.0,
    ],
    dtype=float,
)

DOF_LABELS = ("X", "Y", "Z", "Rx", "Ry", "Rz")
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"
GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0

LINEAR_SPEED = 0.10  # m/s
ANGULAR_SPEED = math.radians(45.0)  # rad/s
TELEOP_HZ = 10.0
TELEOP_DT = 1.0 / TELEOP_HZ
VIEWER_REFRESH_HZ = 60.0

GRIPPER_TRAVEL_TIME = 1.0
DEFAULT_GRIPPER_FORCE_SCALE = 4.0
MAX_SAFE_GRIPPER_FORCE_SCALE = 4.0
DEFAULT_MAX_GRIPPER_FORCE = 30.0
DEFAULT_FINGER_SLIDE_FRICTION = 2.0
DEFAULT_FINGER_TORSIONAL_FRICTION = 0.01
DEFAULT_FINGER_ROLLING_FRICTION = 0.001
DEFAULT_MAX_PHYSICS_TIMESTEP = 0.001
