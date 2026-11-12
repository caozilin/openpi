"""Automatic pick-place subtask segmentation for multi-cycle task plans."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .planning.types import TaskFrame


@dataclass(frozen=True)
class PickPlaceSubtaskSpan:
    """One gripper close/hold/release cycle in a scripted task plan."""

    index: int
    label: str
    pregrasp_index: int
    postgrasp_index: int


_LABEL_PATTERNS = (
    re.compile(r"^(?:raise|lift)\s+(.+)$", re.IGNORECASE),
    re.compile(
        r"^(?:move|lower|hold)\s+(.+?)\s+(?:above|onto|on)\b",
        re.IGNORECASE,
    ),
)


def _infer_label(frames: Sequence[TaskFrame], first: int, last: int, index: int) -> str:
    for frame in frames[first:last + 1]:
        for pattern in _LABEL_PATTERNS:
            match = pattern.search(frame.phase.strip())
            if match is not None:
                value = match.group(1).strip()
                if value and value.lower() not in {"block", "object"}:
                    return value.capitalize()
    return f"Subtask {index + 1}"


def identify_pick_place_subtasks(
    frames: Sequence[TaskFrame],
) -> tuple[PickPlaceSubtaskSpan, ...]:
    """Split a plan at every open-to-closed gripper cycle.

    Pre-grasp is the final open frame before closure. Post-grasp is the final
    closed frame before release. This depends only on the plan contract and
    therefore works for newly added long pick-place tasks without task names.
    """
    if not frames:
        return ()
    closed_runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, frame in enumerate(frames):
        if frame.gripper_closed and start is None:
            start = index
        if start is not None and (
            not frame.gripper_closed or index == len(frames) - 1
        ):
            end = index if frame.gripper_closed else index - 1
            closed_runs.append((start, end))
            start = None
    return tuple(
        PickPlaceSubtaskSpan(
            index,
            _infer_label(frames, first_closed, last_closed, index),
            max(0, first_closed - 1),
            last_closed,
        )
        for index, (first_closed, last_closed) in enumerate(closed_runs)
    )


__all__ = ("PickPlaceSubtaskSpan", "identify_pick_place_subtasks")
