"""Four-stage gripper phase recognition from 10 Hz finger observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import mujoco
import numpy as np

from ..simulation.robot import RobotHandles


class ManipulationPhase(IntEnum):
    """Canonical TaskTol-VLA phase labels stored in datasets and benchmarks."""

    PREGRASP = 0
    GRASP = 1
    POSTGRASP = 2
    RELEASE = 3

    @property
    def key(self) -> str:
        """Unhyphenated token used in JSON and internal interfaces."""
        return ("pregrasp", "grasp", "postgrasp", "release")[int(self)]

    @property
    def label(self) -> str:
        """Human-readable spelling used by UI and reports."""
        return ("Pre-grasp", "Grasp", "Post-grasp", "Release")[int(self)]

    @classmethod
    def from_key(cls, value: str) -> ManipulationPhase:
        keys = {phase.key: phase for phase in cls}
        try:
            return keys[str(value).lower()]
        except KeyError as error:
            raise ValueError(f"Unknown manipulation phase: {value!r}") from error


def scripted_manipulation_phase(
    label: str,
    gripper_closed: bool,
    previous: ManipulationPhase,
    *,
    has_future_grasp: bool,
) -> ManipulationPhase:
    """Map a planned action plus lookahead onto the canonical four stages.

    The lookahead is what separates a terminal open hold from the open
    approach of another subtask. Finger aperture alone cannot make that
    semantic distinction.
    """
    normalized = str(label).casefold()
    if gripper_closed:
        if "close gripper" in normalized:
            return ManipulationPhase.GRASP
        return ManipulationPhase.POSTGRASP
    if previous in (ManipulationPhase.GRASP, ManipulationPhase.POSTGRASP):
        return ManipulationPhase.RELEASE
    if previous is ManipulationPhase.RELEASE:
        if "release" in normalized or not has_future_grasp:
            return ManipulationPhase.RELEASE
        return ManipulationPhase.PREGRASP
    return ManipulationPhase.PREGRASP


@dataclass(frozen=True)
class PhaseObservation:
    phase: ManipulationPhase
    aperture_m: float
    motion_m: float
    commanded_closed: bool
    subtask_index: int = 0
    subtask_changed: bool = False


class PhaseClassifier:
    """Classify high-level manipulation phase from symmetric finger slides.

    Stability uses the change in total finger opening, with a tolerance for
    MuJoCo contact jitter. Stable phases latch until the command changes.
    Simulation entry points sample through PresetSimulationBackend before
    each 10 Hz solve, without planned-label overrides or future commands.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles | None = None,
        motion_threshold_m: float = 0.0002,
        endpoint_threshold_m: float | None = None,
        held_aperture_threshold_m: float = 0.001,
        symmetry_threshold_m: float = 0.0005,
        stable_frames: int = 3,
    ) -> None:
        if handles is None:
            joint_ids = np.array([
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
            ])
            self._open_at_upper = True
            self._maximum_aperture_m = 0.080
            default_endpoint_threshold = 0.0005
        else:
            joint_ids = handles.finger_joint_ids.copy()
            self._open_at_upper = handles.spec.gripper_open_at_upper
            self._maximum_aperture_m = handles.spec.gripper_max_aperture_m
            default_endpoint_threshold = handles.spec.gripper_endpoint_threshold_m
        if np.any(joint_ids < 0):
            raise RuntimeError("The robot model is missing its two observed gripper joints.")
        self.qpos_ids = model.jnt_qposadr[joint_ids].astype(int)
        self._joint_lower = model.jnt_range[joint_ids, 0].copy()
        self._joint_upper = model.jnt_range[joint_ids, 1].copy()
        self.maximum_positions = np.full(2, 0.5 * self._maximum_aperture_m)
        self.motion_threshold_m = float(motion_threshold_m)
        self.endpoint_threshold_m = float(
            default_endpoint_threshold
            if endpoint_threshold_m is None else endpoint_threshold_m
        )
        self.held_aperture_threshold_m = float(held_aperture_threshold_m)
        self.symmetry_threshold_m = float(symmetry_threshold_m)
        self.stable_frames = int(stable_frames)
        if min(
            self.motion_threshold_m,
            self.endpoint_threshold_m,
            self.held_aperture_threshold_m,
            self.symmetry_threshold_m,
        ) <= 0.0:
            raise ValueError("Gripper phase thresholds must be positive.")
        if self.stable_frames <= 0:
            raise ValueError("stable_frames must be positive")
        self._previous_positions: np.ndarray | None = None
        self._previous_commanded_closed: bool | None = None
        self._previous_phase: ManipulationPhase | None = None
        self._subtask_index = 0
        self._subtask_had_postgrasp = False
        self._stable_width_frames = 0

    @property
    def subtask_index(self) -> int:
        return self._subtask_index

    def reset(self) -> None:
        self._previous_positions = None
        self._previous_commanded_closed = None
        self._previous_phase = None
        self._subtask_index = 0
        self._subtask_had_postgrasp = False
        self._stable_width_frames = 0

    def update(
        self,
        finger_positions: np.ndarray,
        commanded_closed: bool,
        *,
        phase_override: ManipulationPhase | None = None,
    ) -> PhaseObservation:
        positions = np.asarray(finger_positions, dtype=float)
        if positions.shape != (2,) or not np.all(np.isfinite(positions)):
            raise ValueError("Finger positions must contain two finite values.")

        aperture = float(np.sum(positions))
        if self._previous_positions is None:
            motion = 0.0
            command_changed = False
        else:
            # Phase stability follows total gripper width. An object may be
            # held asymmetrically without preventing Grasp -> Post-grasp.
            motion = abs(aperture - float(np.sum(self._previous_positions)))
            command_changed = bool(commanded_closed) != self._previous_commanded_closed
        moving = motion > self.motion_threshold_m

        if command_changed:
            self._stable_width_frames = 0
        elif moving:
            # The new width is the first sample of a possible stable run.
            self._stable_width_frames = 1
        else:
            self._stable_width_frames += 1

        if phase_override is not None:
            phase = ManipulationPhase(phase_override)
            if phase in (ManipulationPhase.GRASP, ManipulationPhase.POSTGRASP) and not commanded_closed:
                raise ValueError("closed manipulation phases require a closed command")
            if phase in (ManipulationPhase.PREGRASP, ManipulationPhase.RELEASE) and commanded_closed:
                raise ValueError("open manipulation phases require an open command")
        elif self._previous_positions is None and not commanded_closed:
            # A rollout starts with an already-open gripper in Pre-grasp.
            phase = ManipulationPhase.PREGRASP
        elif commanded_closed:
            if self._previous_phase is ManipulationPhase.POSTGRASP:
                phase = ManipulationPhase.POSTGRASP
            else:
                phase = (
                    ManipulationPhase.POSTGRASP
                    if self._stable_width_frames >= self.stable_frames
                    else ManipulationPhase.GRASP
                )
        else:
            if self._previous_phase is ManipulationPhase.PREGRASP:
                phase = ManipulationPhase.PREGRASP
            else:
                phase = (
                    ManipulationPhase.PREGRASP
                    if self._stable_width_frames >= self.stable_frames
                    else ManipulationPhase.RELEASE
                )

        if phase is ManipulationPhase.POSTGRASP:
            self._subtask_had_postgrasp = True
        subtask_changed = bool(
            self._previous_phase is ManipulationPhase.RELEASE
            and phase is ManipulationPhase.PREGRASP
            and self._subtask_had_postgrasp
        )
        if subtask_changed:
            self._subtask_index += 1
            self._subtask_had_postgrasp = False
        self._previous_positions = positions.copy()
        self._previous_commanded_closed = bool(commanded_closed)
        self._previous_phase = phase
        return PhaseObservation(
            phase,
            aperture,
            motion,
            bool(commanded_closed),
            self._subtask_index,
            subtask_changed,
        )

    def update_from_data(
        self,
        data: mujoco.MjData,
        commanded_closed: bool,
        *,
        phase_override: ManipulationPhase | None = None,
    ) -> PhaseObservation:
        return self.update(
            self.finger_positions(data), commanded_closed,
            phase_override=phase_override,
        )

    def finger_positions(self, data: mujoco.MjData) -> np.ndarray:
        """Return two normalized physical finger openings in metres."""
        raw = data.qpos[self.qpos_ids]
        span = np.maximum(self._joint_upper - self._joint_lower, 1e-9)
        fraction = (
            (raw - self._joint_lower) / span
            if self._open_at_upper
            else (self._joint_upper - raw) / span
        )
        return np.clip(fraction, 0.0, 1.0) * (0.5 * self._maximum_aperture_m)
