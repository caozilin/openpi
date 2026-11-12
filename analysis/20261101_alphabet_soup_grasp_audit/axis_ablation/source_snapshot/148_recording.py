"""Interactive Cartesian-action recording and playback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IDLE_JOINT_MOVEMENT_THRESHOLD_RAD = 0.002


@dataclass(frozen=True)
class TrajectoryStep:
    action: np.ndarray
    gripper_closed: bool


class TrajectoryRecorder:
    """Records complete raw actions; playback applies the current DoF mask."""

    def __init__(self) -> None:
        self.recording = False
        self.playing = False
        self.steps: list[TrajectoryStep] = []
        self._index = 0

    def toggle_recording(self) -> bool:
        self.recording = not self.recording
        if self.recording:
            self.playing = False
            self.steps.clear()
            self._index = 0
        return self.recording

    def append(self, action: np.ndarray, gripper_closed: bool, *, include: bool = True) -> None:
        if self.recording and include:
            self.steps.append(TrajectoryStep(action.copy(), gripper_closed))

    def start_playback(self) -> bool:
        if not self.steps:
            return False
        self.playing = True
        self.recording = False
        self._index = 0
        return True

    def next_step(self) -> TrajectoryStep | None:
        if not self.playing:
            return None
        if self._index >= len(self.steps):
            self.playing = False
            return None
        step = self.steps[self._index]
        self._index += 1
        return step

    @property
    def progress(self) -> tuple[int, int]:
        return self._index, len(self.steps)
