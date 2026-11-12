"""Validated source-scene inputs for the existing FK annotation backend.

These records contain no execution loop. Presets and imported datasets both
execute through counterfactual.runner.run_counterfactual_trial.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StoredStageReference:
    target_rotation: np.ndarray
    tolerance_frame: np.ndarray
    ranged_axes: tuple[int, int, int]


@dataclass(frozen=True)
class StoredSubtaskReference:
    label: str
    pregrasp: StoredStageReference
    postgrasp: StoredStageReference


@dataclass(frozen=True)
class StoredKinematicContext:
    handles: Any
    control_dt_s: float
    table_body_names: tuple[str, ...]
    initial_position: np.ndarray
    initial_rotation: np.ndarray
    subtasks: tuple[StoredSubtaskReference, ...]

    def __post_init__(self):
        if not np.isfinite(self.control_dt_s) or self.control_dt_s <= 0:
            raise ValueError("Stored control interval must be positive and finite")
        if not self.table_body_names or not self.subtasks:
            raise ValueError("Stored FK context requires table geometry and subtasks")
        if np.asarray(self.initial_position).shape != (3,) or np.asarray(self.initial_rotation).shape != (3, 3):
            raise ValueError("Invalid stored nominal anchor")
        if not np.isfinite(self.initial_position).all() or not np.isfinite(self.initial_rotation).all():
            raise ValueError("Stored nominal anchor must be finite")
        for subtask in self.subtasks:
            for stage in (subtask.pregrasp, subtask.postgrasp):
                if len(stage.ranged_axes) != 3 or any(v not in (0, 1) for v in stage.ranged_axes):
                    raise ValueError("Stored authored masks must contain three binary axes")
