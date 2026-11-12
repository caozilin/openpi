"""Offline stage-goal interventions for scripted pick-place trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..geometry import axis_angle_matrix, rotation_matrix, rotation_vector
from ..tasks.planning.types import TaskAction, TaskFrame
from ..tasks.subtasks import identify_pick_place_subtasks
from ..tolerance.annotations import box_tolerance_frame


COUNTERFACTUAL_STAGES = ("pregrasp", "postgrasp")
ROTATION_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class CounterfactualPlan:
    """A complete precomputed trajectory with one modified stage goal."""

    frames: tuple[TaskFrame, ...]
    stage: str
    axis: str
    signed_angle_rad: float
    nominal_target_rotation: np.ndarray
    counterfactual_target_rotation: np.ndarray
    tolerance_frame_rotation: np.ndarray
    pregrasp_endpoint_index: int
    postgrasp_start_index: int
    postgrasp_endpoint_index: int
    verification_index: int
    interventions: tuple["StageRotationIntervention", ...] = ()


@dataclass(frozen=True)
class StageRotationIntervention:
    """One stage's joint rotation-vector intervention in its tolerance frame."""

    stage: str
    rotation_vector_tolerance_frame_rad: np.ndarray
    nominal_target_rotation: np.ndarray
    counterfactual_target_rotation: np.ndarray
    tolerance_frame_rotation: np.ndarray
    start_index: int
    endpoint_index: int
    verification_index: int


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _phase_boundaries(
    frames: tuple[TaskFrame, ...],
    subtask_index: int = 0,
) -> tuple[int, int, int]:
    spans = identify_pick_place_subtasks(frames)
    if not spans:
        raise ValueError("Counterfactual plan has no grasped trajectory segment")
    if not 0 <= int(subtask_index) < len(spans):
        raise ValueError(
            f"Counterfactual subtask index {subtask_index} is outside "
            f"[0, {len(spans) - 1}]",
        )
    span = spans[int(subtask_index)]
    first_closed = span.pregrasp_index + 1
    last_closed = span.postgrasp_index
    postgrasp_start = next(
        (
            index for index in range(first_closed, last_closed + 1)
            if frames[index].gripper_closed
            if frames[index].phase != "Close gripper"
        ),
        first_closed,
    )
    return first_closed - 1, postgrasp_start, last_closed


def build_counterfactual_plan(
    frames: tuple[TaskFrame, ...] | list[TaskFrame],
    stage: str,
    axis: str,
    signed_angle_rad: float,
    subtask_index: int = 0,
) -> CounterfactualPlan:
    """Describe one stage-local intervention without changing planned frames.

    The complete nominal trajectory remains the immutable source of base-frame
    increments.  The intervention is injected into only this stage's compiled
    actions later, matching the UI execution path.
    """
    source = tuple(frames)
    if not source:
        raise ValueError("Cannot perturb an empty trajectory")
    if stage not in COUNTERFACTUAL_STAGES:
        raise ValueError(f"Unknown counterfactual stage: {stage}")
    if axis not in ROTATION_AXES:
        raise ValueError(f"Unknown rotation axis: {axis}")
    if not np.isfinite(signed_angle_rad):
        raise ValueError("Counterfactual angle must be finite")

    pregrasp_endpoint, postgrasp_start, postgrasp_endpoint = (
        _phase_boundaries(source, subtask_index)
    )
    endpoint = (
        pregrasp_endpoint if stage == "pregrasp" else postgrasp_endpoint
    )
    spans = identify_pick_place_subtasks(source)
    start = (
        0
        if stage == "pregrasp" and subtask_index == 0
        else (
            spans[subtask_index - 1].postgrasp_index + 1
            if stage == "pregrasp"
            else postgrasp_start
        )
    )
    nominal_target = source[endpoint].rotation.copy()
    tolerance_frame = box_tolerance_frame(nominal_target)
    axis_index = ROTATION_AXES.index(axis)
    world_axis = tolerance_frame[:, axis_index]
    full_delta = axis_angle_matrix(world_axis, signed_angle_rad)
    counterfactual_target = full_delta @ nominal_target

    # A stage tolerance is measured exactly when that stable stage ends.
    # Later grasp/release/subtask motion must not alter this stage's result.
    verification_index = endpoint
    return CounterfactualPlan(
        frames=source,
        stage=stage,
        axis=axis,
        signed_angle_rad=float(signed_angle_rad),
        nominal_target_rotation=nominal_target,
        counterfactual_target_rotation=counterfactual_target,
        tolerance_frame_rotation=tolerance_frame,
        pregrasp_endpoint_index=pregrasp_endpoint,
        postgrasp_start_index=postgrasp_start,
        postgrasp_endpoint_index=postgrasp_endpoint,
        verification_index=verification_index,
        interventions=(StageRotationIntervention(
            stage=stage,
            rotation_vector_tolerance_frame_rad=(
                np.eye(3)[axis_index] * signed_angle_rad
            ),
            nominal_target_rotation=nominal_target,
            counterfactual_target_rotation=counterfactual_target,
            tolerance_frame_rotation=tolerance_frame,
            start_index=start,
            endpoint_index=endpoint,
            verification_index=verification_index,
        ),),
    )


def build_joint_counterfactual_plan(
    frames: tuple[TaskFrame, ...] | list[TaskFrame],
    stage_rotation_vectors_rad: dict[str, np.ndarray],
) -> CounterfactualPlan:
    """Describe independent stage vectors without modifying nominal frames."""
    source = tuple(frames)
    if not source:
        raise ValueError("Cannot perturb an empty trajectory")
    unknown = set(stage_rotation_vectors_rad).difference(COUNTERFACTUAL_STAGES)
    if unknown:
        raise ValueError(f"Unknown counterfactual stages: {sorted(unknown)}")
    pregrasp_endpoint, postgrasp_start, postgrasp_endpoint = (
        _phase_boundaries(source)
    )
    boundaries = {
        "pregrasp": (0, pregrasp_endpoint),
        "postgrasp": (postgrasp_start, postgrasp_endpoint),
    }
    interventions: list[StageRotationIntervention] = []
    for stage in COUNTERFACTUAL_STAGES:
        if stage not in stage_rotation_vectors_rad:
            continue
        vector = np.asarray(stage_rotation_vectors_rad[stage], dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{stage} intervention must contain three finite values")
        if np.linalg.norm(vector) <= 1e-15:
            continue
        start, endpoint = boundaries[stage]
        nominal = source[endpoint].rotation.copy()
        tolerance_frame = box_tolerance_frame(nominal)
        full_delta = rotation_matrix(tolerance_frame @ vector)
        verification = endpoint
        interventions.append(StageRotationIntervention(
            stage=stage,
            rotation_vector_tolerance_frame_rad=vector.copy(),
            nominal_target_rotation=nominal,
            counterfactual_target_rotation=full_delta @ nominal,
            tolerance_frame_rotation=tolerance_frame,
            start_index=start,
            endpoint_index=endpoint,
            verification_index=verification,
        ))
    if not interventions:
        raise ValueError("Joint counterfactual plan requires a non-zero intervention")

    primary = interventions[-1]
    return CounterfactualPlan(
        frames=source,
        stage="joint" if len(interventions) > 1 else primary.stage,
        axis="vector",
        signed_angle_rad=float(
            np.linalg.norm(primary.rotation_vector_tolerance_frame_rad),
        ),
        nominal_target_rotation=primary.nominal_target_rotation,
        counterfactual_target_rotation=primary.counterfactual_target_rotation,
        tolerance_frame_rotation=primary.tolerance_frame_rotation,
        pregrasp_endpoint_index=pregrasp_endpoint,
        postgrasp_start_index=postgrasp_start,
        postgrasp_endpoint_index=postgrasp_endpoint,
        verification_index=primary.verification_index,
        interventions=tuple(interventions),
    )


def apply_counterfactual_action_interventions(
    actions: tuple[TaskAction, ...] | list[TaskAction],
    initial_rotation: np.ndarray,
    interventions: tuple[StageRotationIntervention, ...],
) -> tuple[list[TaskAction], tuple[StageRotationIntervention, ...]]:
    """Inject forced bias only inside each stage's nominal action interval.

    Outside an intervention interval every base-frame increment is bit-for-bit
    copied from the precompiled nominal trajectory.  Within the interval the
    smoothly growing world-frame bias ``E_k`` changes only that action via
    ``E_k Delta_k E_(k-1)^T``.  Once the stage ends, subsequent nominal
    increments resume unchanged.
    """
    result = [
        TaskAction(action.value.copy(), action.phase, action.status)
        for action in actions
    ]
    for intervention in interventions:
        start = int(intervention.start_index)
        endpoint = int(intervention.endpoint_index)
        if not 0 <= start <= endpoint < len(result):
            raise ValueError("Counterfactual intervention is outside action sequence")
        denominator = max(1, endpoint - start)
        previous_bias = np.eye(3)
        for index in range(start, endpoint + 1):
            progress = (
                1.0
                if endpoint == start
                else _smoothstep((index - start) / denominator)
            )
            bias = rotation_matrix(
                intervention.tolerance_frame_rotation
                @ (
                    progress
                    * intervention.rotation_vector_tolerance_frame_rad
                )
            )
            nominal_delta = rotation_matrix(result[index].value[3:6])
            modified_delta = bias @ nominal_delta @ previous_bias.T
            value = result[index].value.copy()
            value[3:6] = rotation_vector(modified_delta)
            result[index] = TaskAction(
                value, result[index].phase, result[index].status,
            )
            previous_bias = bias

    endpoint_indices: dict[int, list[int]] = {}
    for intervention_index, intervention in enumerate(interventions):
        endpoint_indices.setdefault(intervention.endpoint_index, []).append(
            intervention_index,
        )
    updated = list(interventions)
    rotation = np.asarray(initial_rotation, dtype=float).copy()
    for index, action in enumerate(result):
        rotation = rotation_matrix(action.value[3:6]) @ rotation
        for intervention_index in endpoint_indices.get(index, ()):
            updated[intervention_index] = replace(
                updated[intervention_index],
                counterfactual_target_rotation=rotation.copy(),
            )
    return result, tuple(updated)
