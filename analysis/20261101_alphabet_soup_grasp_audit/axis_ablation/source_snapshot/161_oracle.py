"""Privileged tolerance parameters derived from the scripted dataset plan.

This module is deliberately separate from both the policy and the benchmark
runner.  It supplies the first benchmark's oracle condition: the same nominal
stable-phase endpoints and hand-authored tolerance rule used when dataset
episodes are annotated, but computed before a VLA rollout begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mujoco
import numpy as np

from ..geometry import (
    canonical_rotation_vector,
    continuous_rotation_vector,
    rotation_vector,
)
from ..simulation.robot import RobotHandles, ee_pose
from ..tasks.phases import ManipulationPhase
from ..tasks.planning.presets import PresetTaskExecutor
from ..tasks.planning.types import TaskFrame
from ..tasks.subtasks import identify_pick_place_subtasks
from .annotations import box_tolerance_frame
from .specification import StageTolerance, TaskTolerance, rotation_tolerance_bounds


TOLERANCE_MODE_OFF = "off"
TOLERANCE_MODE_DATASET_ORACLE = "dataset-oracle"
TOLERANCE_MODE_POLICY_PREDICTED = "policy-predicted"
TOLERANCE_MODES = (
    TOLERANCE_MODE_OFF,
    TOLERANCE_MODE_DATASET_ORACLE,
    TOLERANCE_MODE_POLICY_PREDICTED,
)
def _vector3(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


@dataclass(frozen=True)
class PlannedStageTolerance:
    """One stable phase's oracle position plus rotational annotation."""

    target_position_m: np.ndarray
    rotation: StageTolerance

    def __post_init__(self) -> None:
        position = _vector3(self.target_position_m, "stage target position").copy()
        position.setflags(write=False)
        object.__setattr__(self, "target_position_m", position)

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.rotation.phase.key,
            "stage_target_pose": {
                "position_m": self.target_position_m.tolist(),
                "rotation_vector_base_rad": (
                    self.rotation.target_rotation_vector_base_rad.tolist()
                ),
            },
            "tolerance_frame_rotation_vector_base_rad": (
                self.rotation.tolerance_frame_rotation_vector_base_rad.tolist()
            ),
            "rotation_tolerance_rad": (
                self.rotation.rotation_tolerance_bounds_rad.tolist()
            ),
        }


@dataclass(frozen=True)
class PlannedSubtaskTolerance:
    index: int
    label: str
    pregrasp: PlannedStageTolerance
    postgrasp: PlannedStageTolerance


@dataclass(frozen=True)
class PlannedTaskTolerance:
    """Precomputed Pre-grasp/Post-grasp parameters for one rollout."""

    pregrasp: PlannedStageTolerance
    postgrasp: PlannedStageTolerance
    subtasks: tuple[PlannedSubtaskTolerance, ...] = ()

    @property
    def task_tolerance(self) -> TaskTolerance:
        return TaskTolerance(self.pregrasp.rotation, self.postgrasp.rotation)

    def stable_stage(
        self,
        phase: ManipulationPhase,
        subtask_index: int = 0,
    ) -> PlannedStageTolerance:
        if self.subtasks:
            subtask = self.subtasks[
                min(max(int(subtask_index), 0), len(self.subtasks) - 1)
            ]
            if phase is ManipulationPhase.PREGRASP:
                return subtask.pregrasp
            if phase is ManipulationPhase.POSTGRASP:
                return subtask.postgrasp
        if phase is ManipulationPhase.PREGRASP:
            return self.pregrasp
        if phase is ManipulationPhase.POSTGRASP:
            return self.postgrasp
        raise ValueError("Grasp and Release are intentionally Specific")

    def as_dict(self) -> dict[str, object]:
        return {
            "source": "dataset scripted-plan oracle",
            "subtasks": [
                {
                    "index": subtask.index,
                    "label": subtask.label,
                    "pregrasp": subtask.pregrasp.as_dict(),
                    "postgrasp": subtask.postgrasp.as_dict(),
                }
                for subtask in self.subtasks
            ],
            "pregrasp": self.pregrasp.as_dict(),
            "grasp": {
                "phase": ManipulationPhase.GRASP.key,
                "rotation_tolerance_rad": [0.0] * 6,
            },
            "postgrasp": self.postgrasp.as_dict(),
            "release": {
                "phase": ManipulationPhase.RELEASE.key,
                "rotation_tolerance_rad": [0.0] * 6,
            },
        }


def _stage(
    phase: ManipulationPhase,
    frame: TaskFrame,
    limits: object,
    target_reference: np.ndarray,
    tolerance_reference: np.ndarray,
) -> tuple[PlannedStageTolerance, np.ndarray, np.ndarray]:
    target_rotvec = continuous_rotation_vector(frame.rotation, target_reference)
    tolerance_rotation = box_tolerance_frame(frame.rotation)
    tolerance_rotvec = canonical_rotation_vector(tolerance_rotation)
    specification = StageTolerance(
        phase=phase,
        target_rotation_vector_base_rad=target_rotvec,
        tolerance_frame_rotation_vector_base_rad=tolerance_rotvec,
        rotation_tolerance_bounds_rad=rotation_tolerance_bounds(
            limits, f"{phase.label} tolerance",
        ),
    )
    return (
        PlannedStageTolerance(frame.position, specification),
        target_rotvec,
        tolerance_rotvec,
    )


def plan_dataset_oracle_tolerance_from_frames(
    model: mujoco.MjModel,
    frames: tuple[TaskFrame, ...] | list[TaskFrame],
    *,
    scene: str,
    task: str,
    profiles_rad: Mapping[str, object],
    initial_rotation_vector: np.ndarray,
    stage_tolerance_limits_rad: tuple[float, float] | None = None,
    subtask_index_offset: int = 0,
) -> PlannedTaskTolerance:
    """Resolve stable-stage oracle targets for one planned task segment."""
    if not frames:
        raise RuntimeError("Scripted task produced no reference frames")
    spans = identify_pick_place_subtasks(frames)
    semantic_grasp = bool(spans)
    if spans and spans[0].pregrasp_index == spans[0].postgrasp_index:
        raise RuntimeError("Scripted task has no valid Pre-grasp/Post-grasp transition")
    stage_indices = (
        tuple(
            (
                subtask_index_offset + span.index,
                span.label,
                span.pregrasp_index,
                span.postgrasp_index,
            )
            for span in spans
        )
        if spans else ((subtask_index_offset, "Subtask 1", len(frames) - 1, len(frames) - 1),)
    )
    limits: tuple[float, float] | None = None
    if stage_tolerance_limits_rad is not None:
        limits = tuple(float(value) for value in stage_tolerance_limits_rad)
        if (
            len(limits) != 2
            or any(not np.isfinite(value) or value < 0.0 for value in limits)
        ):
            raise ValueError(
                "stage_tolerance_limits_rad must contain two finite "
                "non-negative values",
            )
        from .context import declared_stage_rotation_masks

    target_reference = _vector3(
        initial_rotation_vector, "initial rotation-vector reference",
    )
    tolerance_reference = target_reference
    resolved_subtasks: list[PlannedSubtaskTolerance] = []
    for index, label, pregrasp_index, postgrasp_index in stage_indices:
        pre_limits = profiles_rad["pregrasp"]
        post_limits = profiles_rad["postgrasp"]
        if limits is not None:
            pre_mask, post_mask = declared_stage_rotation_masks(
                model,
                scene,
                task,
                has_grasp=semantic_grasp,
                subtask_label=label,
                subtask_index=index,
            )
            pre_limits = np.repeat(np.asarray(pre_mask, dtype=float), 2) * limits[0]
            post_limits = np.repeat(np.asarray(post_mask, dtype=float), 2) * limits[1]
        pregrasp, target_reference, tolerance_reference = _stage(
            ManipulationPhase.PREGRASP,
            frames[pregrasp_index],
            pre_limits,
            target_reference,
            tolerance_reference,
        )
        postgrasp, target_reference, tolerance_reference = _stage(
            ManipulationPhase.POSTGRASP,
            frames[postgrasp_index],
            post_limits,
            target_reference,
            tolerance_reference,
        )
        resolved_subtasks.append(PlannedSubtaskTolerance(
            index, label, pregrasp, postgrasp,
        ))
    first = resolved_subtasks[0]
    return PlannedTaskTolerance(
        first.pregrasp, first.postgrasp, tuple(resolved_subtasks),
    )


def precompute_dataset_oracle_tolerance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    *,
    scene: str,
    task: str,
    profiles_rad: Mapping[str, object],
    initial_rotation_vector: np.ndarray | None = None,
    stage_tolerance_limits_rad: tuple[float, float] | None = None,
    episode_seed: int = -1,
) -> PlannedTaskTolerance:
    """Recreate dataset endpoint annotations before a policy rollout.

    The planner is read-only with respect to the randomized objects.  The last
    open command before the first close is the Pre-grasp endpoint; the last
    closed command before release is the Post-grasp endpoint. These are the
    nominal endpoints used by the completed-demonstration annotator.
    """
    if "pregrasp" not in profiles_rad or "postgrasp" not in profiles_rad:
        raise ValueError(
            "Dataset task metadata must define pregrasp and postgrasp tolerances"
        )
    position, rotation = ee_pose(data, handles)
    home_rotation_vector = (
        rotation_vector(rotation)
        if initial_rotation_vector is None
        else _vector3(initial_rotation_vector, "initial rotation-vector reference")
    )
    planner = PresetTaskExecutor(model, handles)
    planner.start(task, scene, data, position, rotation, episode_seed=episode_seed)
    frames = planner.frames
    return plan_dataset_oracle_tolerance_from_frames(
        model,
        frames,
        scene=scene,
        task=task,
        profiles_rad=profiles_rad,
        initial_rotation_vector=home_rotation_vector,
        stage_tolerance_limits_rad=stage_tolerance_limits_rad,
    )
