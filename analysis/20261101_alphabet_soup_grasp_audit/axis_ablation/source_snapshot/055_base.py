"""Policy boundary used by teleoperation, collection, and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PolicyStep:
    """One time-aligned policy command and optional TaskTol prediction."""

    action: np.ndarray
    rotation_tolerance_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=float).copy()
        object.__setattr__(self, "action", action)
        if self.rotation_tolerance_rad is not None:
            bounds = np.asarray(self.rotation_tolerance_rad, dtype=float).copy()
            if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
                raise ValueError("Predicted rotation tolerance must be finite 6D")
            object.__setattr__(self, "rotation_tolerance_rad", bounds)

    @property
    def has_predicted_tolerance(self) -> bool:
        return self.rotation_tolerance_rad is not None


class ActionPolicy(Protocol):
    """Minimal VLA policy interface; implementations may be local or remote."""

    def infer(self, observation: dict[str, Any]) -> np.ndarray | dict[str, Any]: ...

    def reset(self) -> None: ...
