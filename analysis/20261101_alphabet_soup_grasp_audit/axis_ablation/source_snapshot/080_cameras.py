"""Shared fixed wrist-camera calibration used by UI and offline collection."""

from __future__ import annotations

import math

import mujoco
import numpy as np


WRIST_CAMERA_BASE_POS = np.array((0.058, 0.0, 0.010))
WRIST_CAMERA_BASE_QUAT = np.array((0.110616, 0.698401, 0.698401, 0.110616))


def set_wrist_camera_calibration(
    model: mujoco.MjModel,
    camera_id: int,
    x_offset: float,
    z_offset: float,
    pitch_deg: float,
    base_pos: np.ndarray | None = None,
    base_quat: np.ndarray | None = None,
) -> None:
    """Apply trims relative to the fixed XML D435 wrist calibration."""
    base_pos = WRIST_CAMERA_BASE_POS if base_pos is None else np.asarray(base_pos, dtype=float)
    base_quat = WRIST_CAMERA_BASE_QUAT if base_quat is None else np.asarray(base_quat, dtype=float)
    half_angle = math.radians(pitch_deg) * 0.5
    pitch_quat = np.array((math.cos(half_angle), 0.0, math.sin(half_angle), 0.0))
    w1, x1, y1, z1 = base_quat
    w2, x2, y2, z2 = pitch_quat
    model.cam_pos[camera_id] = base_pos + np.array((x_offset, 0.0, z_offset))
    model.cam_quat[camera_id] = np.array((
        w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
        w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
        w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
        w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
    ))
