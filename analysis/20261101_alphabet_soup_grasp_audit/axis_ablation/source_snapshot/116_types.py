"""Shared result types for task evaluation and benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import mujoco


@dataclass(frozen=True)
class CollisionEvent:
    """First observed prohibited contact or safety-clearance violation."""

    frame_index: int
    time_s: float
    body_a: str
    body_b: str
    geom_a: str
    geom_b: str
    category: str
    penetration_m: float = 0.0
    clearance_m: float | None = None
    clearance_threshold_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskEvaluator(Protocol):
    """Interface shared by collection and future benchmark runners."""

    @property
    def success(self) -> bool: ...

    @property
    def should_abort(self) -> bool: ...

    def observe_contacts(
        self, data: mujoco.MjData, frame_index: int, time_s: float,
    ) -> None: ...

    def observe_frame(
        self,
        data: mujoco.MjData,
        gripper_closed: bool,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> None: ...

    def result(self, executor_finished: bool, timed_out: bool = False) -> dict[str, Any]: ...
