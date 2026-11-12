"""Shared Cartesian target types for constrained controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetPose:
    position: np.ndarray
    rotation: np.ndarray
