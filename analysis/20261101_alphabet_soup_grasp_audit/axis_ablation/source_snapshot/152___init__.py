"""Task-conditioned orientation-tolerance representation and runtime state."""

from .specification import ROTATION_TOLERANCE_MAXIMUM_RAD, StageTolerance, TaskTolerance
from .state import (
    ORIENTATION_REFERENCE_MODES,
    ORIENTATION_REFERENCE_STAGE_RELATIVE,
    RotationalToleranceState,
)
from .projection import constraint_consistent_release_loss_reference
from .oracle import (
    PlannedSubtaskTolerance,
    PlannedTaskTolerance,
    TOLERANCE_MODE_DATASET_ORACLE,
    TOLERANCE_MODE_OFF,
    TOLERANCE_MODE_POLICY_PREDICTED,
)
from .annotations import (
    POT_LID_STAGE_ROTATION_MASKS,
    LYING_CYLINDER_STAGE_ROTATION_MASKS,
    active_pot_lid_stage_rotation_masks,
    lying_cylinder_stage_rotation_masks,
    pot_lid_stage_rotation_masks,
)
from .context import (
    ResolvedStageTolerance,
    ResolvedSubtaskTolerance,
    ResolvedTaskTolerance,
    declared_stage_rotation_masks,
    effective_rotation_specific_axes,
    resolve_task_tolerance_context,
)
from .release_state import (
    RotationReleaseCandidate,
    RotationReleaseState,
    RotationReleaseStep,
)
from .frame import (
    CANONICAL_HEADING_CODE,
    HEADING_CODE_EPSILON,
    TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
    TOLERANCE_FRAME_MODE_STAGE_ENDPOINT,
    TOLERANCE_FRAME_MODES,
    ToleranceFrameEMA,
    ToleranceLimitsEMA,
    box_tolerance_frame,
    cycle_tolerance_frame,
    frame_from_heading,
    heading_code,
)

__all__ = (
    "PlannedTaskTolerance",
    "PlannedSubtaskTolerance",
    "RotationalToleranceState",
    "constraint_consistent_release_loss_reference",
    "ORIENTATION_REFERENCE_MODES",
    "ORIENTATION_REFERENCE_STAGE_RELATIVE",
    "ROTATION_TOLERANCE_MAXIMUM_RAD",
    "StageTolerance",
    "TaskTolerance",
    "TOLERANCE_MODE_DATASET_ORACLE",
    "TOLERANCE_MODE_OFF",
    "TOLERANCE_MODE_POLICY_PREDICTED",
    "POT_LID_STAGE_ROTATION_MASKS",
    "LYING_CYLINDER_STAGE_ROTATION_MASKS",
    "active_pot_lid_stage_rotation_masks",
    "lying_cylinder_stage_rotation_masks",
    "pot_lid_stage_rotation_masks",
    "ResolvedStageTolerance",
    "ResolvedSubtaskTolerance",
    "ResolvedTaskTolerance",
    "declared_stage_rotation_masks",
    "effective_rotation_specific_axes",
    "resolve_task_tolerance_context",
    "RotationReleaseCandidate",
    "RotationReleaseState",
    "RotationReleaseStep",
    "ToleranceFrameEMA",
    "ToleranceLimitsEMA",
    "CANONICAL_HEADING_CODE",
    "HEADING_CODE_EPSILON",
    "TOLERANCE_FRAME_MODE_REALTIME_NOMINAL",
    "TOLERANCE_FRAME_MODE_STAGE_ENDPOINT",
    "TOLERANCE_FRAME_MODES",
    "box_tolerance_frame",
    "cycle_tolerance_frame",
    "frame_from_heading",
    "heading_code",
)
