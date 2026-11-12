"""Headless execution of one stage-goal counterfactual trial."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import copy
import math
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from ..config import (
    DEFAULT_FINGER_SLIDE_FRICTION,
    DEFAULT_GRIPPER_FORCE_SCALE,
    DEFAULT_MAX_GRIPPER_FORCE,
    DEFAULT_MAX_PHYSICS_TIMESTEP,
    GRIPPER_TRAVEL_TIME,
    TELEOP_DT,
)
from ..control.pose import TargetPose
from ..control.joint_tracker import JointReferenceTracker
from ..control.ipopt import (
    MAX_CONSECUTIVE_SOLVER_FAILURES,
    IpoptIK,
    IpoptSettings,
)
from ..control.task_space.objective import DEFAULT_EMA_RELEASE_LOSS_WEIGHT, ObjectiveSettings
from ..control.task_space.singularity import SIGMA_MIN_PENALTY_PROFILES, validate_sigma_min_loss
from ..control.task_space.conditioning import (
    velocity_normalized_sigma_min, sigma_min_normalization_metadata,
)
from ..geometry import (
    canonical_rotation_vector,
    continuous_rotation_vector,
    rotation_error_in_frame,
    rotation_matrix,
    rotation_vector,
)
from ..simulation.loader import load_scene_model
from ..simulation.contact_materials import model_contact_revision
from ..simulation.layouts import apply_custom_layout, load_custom_layout
from ..simulation.randomization.camera import (
    FrontCameraRandomizer,
    tabletop_corners,
    focus_on_world_x,
)
from ..simulation.randomization.light import LightRandomizer
from ..simulation.randomization.objects import (
    ObjectLayoutLimits,
    ObjectLayoutRandomizer,
    body_planar_footprint,
)
from ..simulation.randomization.table import TableRandomizer
from ..reachability.layout import apply_training_workspace_boundary
from ..simulation.robot import (
    configure_contact_solver,
    configure_gripper,
    ee_pose,
    find_handles,
    set_gripper_fully_open,
    set_home,
)
from ..simulation.robots import dataset_joint_state, get_robot_spec
from ..simulation.scenes import SceneManager
from ..tasks import create_task_evaluator, registered_tasks
from ..tasks.evaluation.predicates import RobotTableClearanceRecorder
from ..tasks.catalog import (
    BLOCK_TO_PLATE_TASK,
    pot_lid_task_variant_index,
)
from ..tasks.phases import ManipulationPhase
from ..tasks.planning.presets import (
    compile_task_actions,
    PresetTaskExecutor,
)
from ..tasks.planning.types import PlanningInfeasibleError, TaskAction, TaskFrame
from ..simulation.preset_runtime import PresetSimulationBackend
from ..tolerance.annotations import box_tolerance_frame
from ..tolerance.context import (
    declared_stage_rotation_masks,
    resolve_task_tolerance_context,
)
from ..tolerance.oracle import plan_dataset_oracle_tolerance_from_frames
from ..tolerance.runtime import (
    advance_reference_pose,
    rebase_rotation_subtask,
    solve_stage_relative_target,
)
from ..tolerance.state import RotationalToleranceState
from ..tolerance.release_state import RotationReleaseState
from ..tolerance.metrics import StageRotationReleaseRecorder
from .intervention import (
    apply_counterfactual_action_interventions,
    COUNTERFACTUAL_STAGES,
    CounterfactualPlan,
    build_counterfactual_plan,
    build_joint_counterfactual_plan,
)
from .kinematic import TrajectoryKinematicMetrics
from .stored import StoredKinematicContext
from .video import CounterfactualDatasetVideoRecorder, CounterfactualVideoRecorder


_SETTLE_TIME_S = 1.0
_FINAL_EVALUATION_HOLD_FRAMES = 2
_OFFLINE_EXECUTION_SPEED_SCALE = 1.0
_STAGNATION_WINDOW_FRAMES = 15
_STAGNATION_JOINT_MOVEMENT_RAD = 1.0e-4
_STAGNATION_POSITION_ERROR_M = 0.01
_STAGNATION_ROTATION_ERROR_RAD = 0.10
_IK_FAILURE_WINDOW_FRAMES = 20
_IK_FAILURE_GRADIENT_NORM = 1.0e-3
# Offline annotation must be independent of worker count and host load.  The
# UI keeps its real-time deadline, while counterfactual SQP always runs its
# fixed iteration budget to completion.
_COUNTERFACTUAL_SQP_SETTINGS = IpoptSettings(
    max_iterations=20,
    max_cpu_time_s=1.0e6,
)

# Reference transport and multi-subtask planning are trajectory semantics;
# records produced by the former fixed-offset/deferred-replan path must not be
# reused by the current controller.
# EMA release now uses ordinary squared L2; former Groove-loss trajectories
# must not be reused as results from the current objective.
COUNTERFACTUAL_CACHE_VERSION = 83


@dataclass(frozen=True)
class CounterfactualTrialSpec:
    scene: str = "Blocks"
    task: str = BLOCK_TO_PLATE_TASK
    robot_uid: str = "panda"
    seed: int = 0
    # Environment interference follows the source dataset's per-episode
    # routing.  Keeping this separate from ``seed`` preserves object-layout
    # randomization for every episode while allowing nominal environments to
    # remain cache-compatible with older trials.
    environment_seed: int = -1
    environment_randomization_probability: float = 0.0
    workspace_boundary: bool = False
    workspace_boundary_distance_min: float = 0.9
    workspace_boundary_distance_max: float = 2.0
    workspace_boundary_map_directory: str | None = None
    # Lock a source dataset's single allowed object variant during replay.
    # None preserves the existing seed-selected behavior.
    scene_variant_index: int | None = None
    custom_layout_path: str | None = None
    trajectory_revision: int = 0
    stage: str = "pregrasp"
    subtask_index: int = 0
    axis: str = "x"
    direction: int = 1
    angle_deg: int = 1
    maximum_frames: int = 600
    object_clearance_m: float = 0.05
    target_rotation_tolerance_deg: float = 3.0
    pregrasp_rotation_vector_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    postgrasp_rotation_vector_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controller_mode: str = "sqp_specific"
    # ``simulation`` is the historical closed-loop task rollout.  ``kinematic``
    # runs the same stage-local counterfactual over FK/IK only: no physics
    # stepping and no object/environment task evaluator.
    execution_mode: str = "simulation"
    # FK experiments may omit table geometry; robot self-collision is always
    # checked with the model's contact filters and exclusions.
    kinematic_check_table_collisions: bool = True
    # Optional geometric annotation observation on solved FK endpoints. This
    # does not impose a clearance threshold or introduce any physics steps.
    kinematic_record_robot_table_clearance: bool = False
    # Annotation-only FK observation: table contacts remain task failures but
    # do not truncate the solve/metric trajectory. Never enabled for physics.
    kinematic_continue_after_table_contact: bool = False
    # Coarse nominal candidate screening only. Production replay keeps 1.
    # Targets still advance at the original rate; only IK/FK observations
    # are thinned, retaining endpoints and phase/gripper transitions.
    kinematic_sampling_stride: int = 1
    # Release every enabled stable-stage rotation axis without a finite wall.
    # This is an experiment-only mode; normal UI/dataset tolerances remain
    # bounded by their configured directional limits.
    sqp_unbounded_rotation: bool = False
    sqp_symmetric_limits_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sqp_stage_rotation_masks: tuple[
        tuple[int, int, int], tuple[int, int, int]
    ] = ((0, 0, 0), (0, 0, 0))
    sqp_stage_negative_limits_deg: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    sqp_stage_positive_limits_deg: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    sqp_subtask_stage_rotation_masks: tuple[
        tuple[tuple[int, int, int], tuple[int, int, int]], ...
    ] = ()
    sqp_subtask_stage_negative_limits_deg: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ] = ()
    sqp_subtask_stage_positive_limits_deg: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ] = ()
    sqp_loss_profile: str = "delta_q_squared_conditioning_safeguard"
    # (delta-q squared, secondary objective, nominal-action EMA release).
    sqp_loss_weights: tuple[float, float, float] = (1.0, 0.0, DEFAULT_EMA_RELEASE_LOSS_WEIGHT)
    sqp_squared_acceleration_weight: float = 0.0
    # A/B soft protection on normalized sigma. These are configuration defaults,
    # not calibrated safety thresholds; secondary weight remains zero by default.
    sqp_sigma_min_warning_threshold: float = 1.0
    sqp_sigma_min_reciprocal_epsilon: float = 0.001
    sqp_conditioning_eta_low: float = 0.05
    sqp_conditioning_eta_high: float = 0.10
    sqp_conditioning_sigma_epsilon: float = 1.0e-12
    sqp_conditioning_fd_relative_step: float = 1.0e-5
    # Zero disables the wall-clock deadline, matching offline dataset runs.
    # A positive value reproduces the real-time UI budget.
    sqp_max_time_s: float = 0.0
    ipopt_max_iterations: int = 20
    sqp_diagnose_conditioning_gradient: bool = False
    # Explicit legacy/reference switch. Default UI and offline selection share
    # the same nominal candidate table; selection never depends on tolerance.
    nominal_plan_selection_enabled: bool = True


@dataclass(frozen=True)
class PreparedNominalTrial:
    """An already initialized scene and frozen plan for the SAME public backend.

    Used by nominal candidate validation, not a second rollout implementation.
    The backend clones both model and data; callers' live UI state is untouched.
    """

    model: Any
    data: Any
    robot_uid: str
    frames: tuple[TaskFrame, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    stored_context: StoredKinematicContext | None = None


@dataclass(frozen=True)
class CounterfactualTrialResult:
    spec: CounterfactualTrialSpec
    success: bool
    task_success: bool
    target_reached: bool
    termination_reason: str
    target_rotation_error_deg: float
    nominal_target_rotation_vector_base_rad: tuple[float, float, float]
    counterfactual_target_rotation_vector_base_rad: tuple[float, float, float]
    tolerance_frame_rotation_vector_base_rad: tuple[float, float, float]
    evaluation: dict[str, Any]
    domain_randomization: dict[str, Any] = field(default_factory=dict)
    hard_failure_reasons: tuple[str, ...] = ()
    trajectory_kinematics: dict[str, Any] = field(default_factory=dict)
    dataset_trajectory: tuple[dict[str, Any], ...] = ()
    dataset_source_frame_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = asdict(self.spec)
        if not self.dataset_trajectory:
            payload.pop("dataset_trajectory", None)
        payload["counterfactual_cache_version"] = COUNTERFACTUAL_CACHE_VERSION
        payload["sigma_min_normalization"] = sigma_min_normalization_metadata(
            get_robot_spec(self.spec.robot_uid).joint_velocity_limits,
        )
        return payload


@dataclass(frozen=True)
class KinematicStepContext:
    """Read-only live inputs just before a shared backend FK control step.

    Observers must fork the controller and copy data/tolerance/release state
    before trying other commands. They must never alter the carrier rollout.
    """

    frame_index: int
    phase: str
    subtask_index: int
    dt: float
    model: Any
    data: Any
    controller: IpoptIK
    target: TargetPose
    tolerance: RotationalToleranceState
    release_state: RotationReleaseState
    active_dofs: np.ndarray
    # Available only at a subtask's first step. Observers can freeze this
    # already planned segment and its entry state without replanning it.
    subtask_nominal_frames: tuple[TaskFrame, ...] = ()


class _TrajectoryRecorder:
    """Accumulate complete closed-loop kinematics at the task frame rate."""

    def __init__(self, initial_q: np.ndarray | None = None, *, control_dt_s: float = TELEOP_DT) -> None:
        if not np.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be positive and finite")
        self.control_dt_s = float(control_dt_s)
        self._samples = 0
        self._sums = np.zeros(4)
        self._counts = np.zeros(4, dtype=int)
        self._previous_q: np.ndarray | None = None
        self._previous_velocity: np.ndarray | None = None
        self._previous_acceleration: np.ndarray | None = None
        self._previous_velocity_dt: float | None = None
        self._previous_acceleration_dt: float | None = None
        self._minimum_q: np.ndarray | None = None
        self._maximum_q: np.ndarray | None = None
        self._travel_previous_q = None if initial_q is None else np.asarray(initial_q).copy()
        self._total_joint_travel_rad = 0.0
        self._minimum_sigma_min_full: float | None = None
        self._minimum_sigma_min_velocity_normalized: float | None = None

    def sample(
        self,
        target: TargetPose,
        actual_position: np.ndarray,
        actual_rotation: np.ndarray,
        q: np.ndarray,
        manipulability: float | None,
        rotation_error_rad: float | None = None,
        *,
        elapsed_frames: int = 1,
        sigma_min_full: float | None = None,
        sigma_min_velocity_normalized: float | None = None,
    ) -> None:
        if sigma_min_full is not None:
            sigma_min_full = float(sigma_min_full)
            if not np.isfinite(sigma_min_full) or sigma_min_full < 0.0:
                raise ValueError("sigma_min_full must be finite and non-negative")
            self._minimum_sigma_min_full = (
                sigma_min_full if self._minimum_sigma_min_full is None
                else min(self._minimum_sigma_min_full, sigma_min_full)
            )
        if sigma_min_velocity_normalized is not None:
            value = float(sigma_min_velocity_normalized)
            if not np.isfinite(value) or value < 0:
                raise ValueError("sigma_min_velocity_normalized must be finite and non-negative")
            self._minimum_sigma_min_velocity_normalized = (
                value if self._minimum_sigma_min_velocity_normalized is None
                else min(self._minimum_sigma_min_velocity_normalized, value)
            )
        # Cartesian tracking errors are controller diagnostics, not
        # counterfactual annotation quality metrics.
        if self._travel_previous_q is not None:
            self._total_joint_travel_rad += float(np.linalg.norm(q - self._travel_previous_q))
        self._travel_previous_q = q.copy()
        dt = elapsed_frames * self.control_dt_s
        acceleration_dt = (
            dt if self._previous_velocity_dt is None
            else 0.5 * (dt + self._previous_velocity_dt)
        )
        jerk_dt = (
            acceleration_dt if self._previous_acceleration_dt is None
            else 0.5 * (acceleration_dt + self._previous_acceleration_dt)
        )
        velocity = (
            None
            if self._previous_q is None
            else (q - self._previous_q) / dt
        )
        acceleration = (
            None
            if velocity is None or self._previous_velocity is None
            else (velocity - self._previous_velocity) / acceleration_dt
        )
        jerk = (
            None
            if acceleration is None or self._previous_acceleration is None
            else (acceleration - self._previous_acceleration) / jerk_dt
        )
        self._previous_q = q.copy()
        self._minimum_q = (
            q.copy() if self._minimum_q is None else np.minimum(self._minimum_q, q)
        )
        self._maximum_q = (
            q.copy() if self._maximum_q is None else np.maximum(self._maximum_q, q)
        )
        if velocity is not None:
            self._previous_velocity = velocity.copy()
            self._previous_velocity_dt = dt
        if acceleration is not None:
            self._previous_acceleration = acceleration.copy()
            self._previous_acceleration_dt = acceleration_dt
        self._samples += 1
        contributions = (
            None if velocity is None else float(np.linalg.norm(velocity)),
            None if acceleration is None else float(np.linalg.norm(acceleration)),
            None if jerk is None else float(np.linalg.norm(jerk)),
            0.0 if manipulability is None else float(manipulability),
        )
        for index, value in enumerate(contributions):
            if value is not None:
                self._sums[index] += value * elapsed_frames
                self._counts[index] += elapsed_frames

    def result(self) -> TrajectoryKinematicMetrics:
        means = np.divide(
            self._sums,
            self._counts,
            out=np.zeros(4),
            where=self._counts > 0,
        )
        return TrajectoryKinematicMetrics(
            self._samples,
            *(float(value) for value in means),
            total_joint_travel_rad=self._total_joint_travel_rad,
            minimum_sigma_min_full=self._minimum_sigma_min_full,
            minimum_sigma_min_velocity_normalized=self._minimum_sigma_min_velocity_normalized,
        )

    def joint_position_extrema_deg(self) -> dict[str, list[float]]:
        if self._minimum_q is None or self._maximum_q is None:
            return {"minimum": [], "maximum": []}
        return {
            "minimum": [float(value) for value in np.degrees(self._minimum_q)],
            "maximum": [float(value) for value in np.degrees(self._maximum_q)],
        }


def _kinematic_sample_indices(actions, stride):
    """Deterministic sparse samples, retaining semantic transition neighbors."""
    if not actions:
        return set()
    indices = set(range(0, len(actions), stride))
    indices.add(len(actions) - 1)
    for index in range(1, len(actions)):
        previous, current = actions[index - 1], actions[index]
        if (current.phase != previous.phase
                or current.gripper_closed != previous.gripper_closed):
            # The phase classifier applies a command's stage on the next
            # control interval. Keep both sides AND that following interval.
            indices.update(range(index - 1, min(index + 2, len(actions))))
    return indices


def _declared_subtask_stage_masks(
    model: mujoco.MjModel,
    spec: CounterfactualTrialSpec,
    resolved: Any,
    subtask_count: int,
) -> list[list[list[int]]]:
    """Return the task-authored Pre/Post mask, independent of trial bounds."""
    records: list[list[list[int]]] = []
    for index in range(subtask_count):
        descriptor = (
            resolved.subtask(index)
            if hasattr(resolved, "subtask")
            else resolved.subtasks[index]
        )
        if (
            hasattr(descriptor.pregrasp, "ranged_axes")
            and hasattr(descriptor.postgrasp, "ranged_axes")
        ):
            pre = np.asarray(descriptor.pregrasp.ranged_axes, dtype=int)
            post = np.asarray(descriptor.postgrasp.ranged_axes, dtype=int)
        else:
            pre_tuple, post_tuple = declared_stage_rotation_masks(
                model,
                spec.scene,
                spec.task,
                has_grasp=True,
                subtask_label=str(descriptor.label),
                subtask_index=index,
            )
            pre = np.asarray(pre_tuple, dtype=int)
            post = np.asarray(post_tuple, dtype=int)
        records.append([pre.tolist(), post.tolist()])
    return records


def _body_subtree_ids(model: mujoco.MjModel, root: int) -> set[int]:
    if root < 0:
        return set()
    bodies: set[int] = set()
    for body in range(model.nbody):
        current = body
        while current > root:
            current = int(model.body_parentid[current])
        if current == root:
            bodies.add(body)
    return bodies


def _robot_body_ids(model: mujoco.MjModel, root_name: str) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    return _body_subtree_ids(model, root)


def _kinematic_collision_failure(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_bodies: set[int],
    table_bodies: set[int],
    gripper_bodies: set[int],
    *,
    recorded_table_contacts: list[dict[str, Any]] | None = None,
    check_table_collisions: bool = True,
) -> dict[str, Any] | None:
    """Return the first prohibited FK-only collision at the current frame.

    MuJoCo's collision pipeline is used only as a geometry query here; the
    kinematic backend never advances physics.  Existing MJCF contact filters
    therefore remain authoritative for adjacent-link and other intentionally
    excluded robot pairs.
    """
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        contact_distance = float(contact.dist)
        if contact_distance > 0.0:
            continue
        penetration = max(0.0, -contact_distance)
        geom_a = int(contact.geom1)
        geom_b = int(contact.geom2)
        body_a = int(model.geom_bodyid[geom_a])
        body_b = int(model.geom_bodyid[geom_b])
        category = None
        if (
            body_a != body_b
            and body_a in robot_bodies
            and body_b in robot_bodies
        ):
            if body_a in gripper_bodies and body_b in gripper_bodies:
                continue
            category = "robot_self_collision"
        elif check_table_collisions and ((
            body_a in robot_bodies and body_b in table_bodies
        ) or (
            body_b in robot_bodies and body_a in table_bodies
        )):
            category = "robot_table_collision"
        if category is None:
            continue
        event = {
            "category": category,
            "contact_index": contact_index,
            "penetration_m": penetration,
            "distance_m": contact_distance,
            "geom_a": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_a,
            ),
            "geom_b": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_b,
            ),
            "body_a": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_a,
            ),
            "body_b": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_b,
            ),
        }
        if category == "robot_table_collision" and recorded_table_contacts is not None:
            recorded_table_contacts.append(event)
            continue  # A soft table contact must not hide self-contact later.
        return event
    return None


def _setup_trial(
    spec: CounterfactualTrialSpec,
    *,
    kinematic_only: bool = False,
    frozen_nominal_frames: tuple[TaskFrame, ...] | None = None,
) -> tuple[
    mujoco.MjModel,
    mujoco.MjData,
    Any,
    CounterfactualPlan,
    PresetTaskExecutor,
    dict[str, Any],
]:
    robot = get_robot_spec(spec.robot_uid)
    model = load_scene_model(spec.scene, spec.robot_uid)
    configure_contact_solver(model, DEFAULT_MAX_PHYSICS_TIMESTEP)
    handles = find_handles(model, robot)
    configure_gripper(
        model,
        handles,
        DEFAULT_GRIPPER_FORCE_SCALE,
        DEFAULT_MAX_GRIPPER_FORCE,
        DEFAULT_FINGER_SLIDE_FRICTION,
    )
    data = mujoco.MjData(model)
    set_home(model, data, handles)
    scene_manager = SceneManager(model, loaded_scene=spec.scene)
    scene_manager.activate(data, spec.scene)
    custom_layout = (
        load_custom_layout(spec.custom_layout_path)
        if spec.custom_layout_path is not None else None
    )

    work_surface_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
    )
    if work_surface_id < 0:
        raise RuntimeError("Counterfactual scene is missing work_surface")
    nominal_table_x = float(model.body_pos[work_surface_id, 0])
    nominal_tabletop = tabletop_corners(model)
    table_pose = TableRandomizer(model.body_pos[work_surface_id]).sample(
        spec.environment_seed,
    )
    model.body_pos[work_surface_id] = table_pose.position
    scene_manager.set_workspace_offset(data, table_pose.offset)

    layout = ObjectLayoutRandomizer(
        model,
        scene_manager,
        limits=ObjectLayoutLimits(
            minimum_clearance_m=spec.object_clearance_m,
        ),
    ).apply(
        data,
        spec.seed,
        variant_index=(
            spec.scene_variant_index
            if spec.scene_variant_index is not None
            else custom_layout.variant_index if custom_layout is not None
            else pot_lid_task_variant_index(spec.task)
        ),
    )
    boundary_layout = None
    if spec.workspace_boundary:
        boundary_layout = apply_training_workspace_boundary(
            model,
            data,
            scene_manager,
            handles,
            spec.task,
            "pre_boundary",
            spec.seed,
            distance_factor_lower=spec.workspace_boundary_distance_min,
            distance_factor_upper=spec.workspace_boundary_distance_max,
            map_directory=(
                None
                if spec.workspace_boundary_map_directory is None
                else Path(spec.workspace_boundary_map_directory)
            ),
        )

    if custom_layout is not None:
        apply_custom_layout(
            scene_manager, data, custom_layout, robot=spec.robot_uid,
            restore_variant=False,
        )

    front_camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "front_view",
    )
    camera_pose = None
    if front_camera_id >= 0:
        front_base_position = model.cam_pos[front_camera_id].copy()
        front_base_quaternion = model.cam_quat[front_camera_id].copy()
        front_focus = focus_on_world_x(
            front_base_position,
            front_base_quaternion,
            nominal_table_x,
        )
        camera_pose = FrontCameraRandomizer(
            front_base_position,
            front_base_quaternion,
            front_focus,
        ).sample(
            spec.environment_seed, tabletop=tabletop_corners(model), reference_tabletop=nominal_tabletop,
            vertical_fov_deg=float(model.cam_fovy[front_camera_id]),
        )
        model.cam_pos[front_camera_id] = camera_pose.position
        model.cam_quat[front_camera_id] = camera_pose.quaternion

    light_randomizer = LightRandomizer(model)
    lighting = light_randomizer.sample(spec.environment_seed)
    light_randomizer.apply(model, lighting)
    set_gripper_fully_open(model, data, handles)
    # Kinematic annotation deliberately has no dynamics/physics phase.  The
    # planner only needs a forward-kinematic scene snapshot to derive the same
    # scripted frames used by the normal counterfactual path.
    if not kinematic_only:
        PresetSimulationBackend(model, data, handles).settle(_SETTLE_TIME_S)

    position, rotation = ee_pose(data, handles)
    executor = PresetTaskExecutor(model, handles)
    environment = {
        "contact_material_revision": model_contact_revision(model),
        "seed_routing": {
            "episode_seed": int(spec.seed),
            "environment_randomization_enabled": spec.environment_seed != -1,
            "environment_randomization_probability": (
                spec.environment_randomization_probability
            ),
            "environment_seed": int(spec.environment_seed),
            "table_seed": int(spec.environment_seed),
            "front_camera_seed": int(spec.environment_seed),
            "lighting_seed": int(spec.environment_seed),
            "object_layout_seed": int(spec.seed),
            "semantic_goal_seed": int(spec.seed),
        },
        "semantic_goal_perturbations": [],
        "table_offset_m": table_pose.offset.tolist(),
        "front_camera_local_translation_m": (
            [0.0, 0.0, 0.0]
            if camera_pose is None else camera_pose.local_translation.tolist()
        ),
        "front_camera_local_angles_deg": (
            [0.0, 0.0, 0.0]
            if camera_pose is None else camera_pose.local_angles_deg.tolist()
        ),
        "front_camera_framing_angles_deg": (
            [0.0, 0.0, 0.0]
            if camera_pose is None else camera_pose.framing_angles_deg.tolist()
        ),
        "front_camera_tabletop_visible_fraction_before": (
            None if camera_pose is None else camera_pose.tabletop_visible_fraction_before
        ),
        "front_camera_tabletop_visible_fraction_after": (
            None if camera_pose is None else camera_pose.tabletop_visible_fraction_after
        ),
        "front_camera_tabletop_visible_fraction_reference": (
            None if camera_pose is None else camera_pose.tabletop_visible_fraction_reference
        ),
        "lighting_brightness": lighting.brightness.tolist(),
        "object_layout_seed": int(spec.seed),
        "object_layout_placement_count": len(layout.placements),
        "table_bounds_xy_m": layout.table_bounds_xy.tolist(),
        "table_edge_margin_m": ObjectLayoutLimits().table_edge_margin_m,
        "white_region_footprints_xy_m": [
            {
                "body_name": item.body_name,
                "footprint": body_planar_footprint(
                    model, data, model.body(item.body_name).id,
                    include_visual_geoms=True,
                ).tolist(),
            }
            for item in scene_manager.layout_objects(data)
            if item.body_name.startswith("paper_decal_")
            or item.body_name == "unseen_box_white_region"
        ],
        "active_scene_bodies": list(scene_manager.active_body_names),
        "active_geometry_color": scene_manager.active_geometry_color_name,
        "reachability_guided_layout": boundary_layout,
    }
    if custom_layout is not None:
        environment["custom_layout_path"] = str(Path(spec.custom_layout_path).resolve())
    if frozen_nominal_frames is not None:
        if not kinematic_only or not frozen_nominal_frames:
            raise ValueError("Frozen dataset preparation requires nonempty kinematic frames")
        # Stored expert references are authoritative. Do not choose a new
        # grasp branch or invoke a task's successor planner for this dataset.
        executor.frames = list(frozen_nominal_frames)
        executor.actions = compile_task_actions(executor.frames, position, rotation)
        executor.active = True
        plan = build_counterfactual_plan(executor.frames, spec.stage, spec.axis, 0.0)
        environment["nominal_source"] = "immutable recorded dataset"
        return model, data, handles, plan, executor, environment
    try:
        executor.start(
            spec.task, spec.scene, data, position, rotation, episode_seed=spec.seed,
            nominal_selection_enabled=spec.nominal_plan_selection_enabled,
            domain_randomization_enabled=spec.environment_seed != -1,
            nominal_validation_options=dict(
                sqp_loss_profile=spec.sqp_loss_profile, sqp_loss_weights=spec.sqp_loss_weights,
                sqp_conditioning_eta_low=spec.sqp_conditioning_eta_low,
                sqp_conditioning_eta_high=spec.sqp_conditioning_eta_high,
                sqp_conditioning_sigma_epsilon=spec.sqp_conditioning_sigma_epsilon,
                sqp_conditioning_fd_relative_step=spec.sqp_conditioning_fd_relative_step,
                ipopt_max_iterations=spec.ipopt_max_iterations,
            ),
        )
    except PlanningInfeasibleError as error:
        # Preserve the actual seeded scene even when no executable plan exists.
        # This typed failure currently comes from the single-subtask paper skill.
        environment["semantic_goal_perturbations"] = executor.perturbations.samples
        if error.context.get("nominal_plan_table"):
            table = error.context["nominal_plan_table"]
            environment["nominal_plan_table"] = {
                key:table[key] for key in ("path", "entry_key", "selected_candidate_id",
                                          "feasible_candidates", "candidate_count",
                                          "selection_metric", "selected_total_joint_travel_rad",
                                          "rotation_score_policy", "selected_rotation_deg", "selected_score",
                                          "selection_reason", "selected_nominal_feasible") if key in table
            }
        error.context = dict(
            domain_randomization=environment,
            declared_subtask_stage_rotation_masks=[[
                list(mask) for mask in declared_stage_rotation_masks(
                    model, spec.scene, spec.task, has_grasp=True,
                )
            ]],
        )
        raise
    environment["semantic_goal_perturbations"] = executor.perturbations.samples
    if executor.last_planning_stats.get("nominal_plan_table"):
        table = executor.last_planning_stats["nominal_plan_table"]
        environment["nominal_plan_table"] = {
            key:table[key] for key in ("path", "entry_key", "selected_candidate_id",
                                      "feasible_candidates", "candidate_count",
                                      "selection_metric", "selected_total_joint_travel_rad",
                                      "rotation_score_policy", "selected_rotation_deg", "selected_score",
                                          "selection_reason", "selected_nominal_feasible") if key in table
        }
    stage_vectors = {
        "pregrasp": np.radians(
            np.asarray(spec.pregrasp_rotation_vector_deg, dtype=float),
        ),
        "postgrasp": np.radians(
            np.asarray(spec.postgrasp_rotation_vector_deg, dtype=float),
        ),
    }
    nonzero_vectors = {
        stage: vector
        for stage, vector in stage_vectors.items()
        if np.linalg.norm(vector) > 1e-15
    }
    if nonzero_vectors:
        plan = build_joint_counterfactual_plan(executor.frames, nonzero_vectors)
    else:
        signed_angle = (0. if executor._pending_subtask_planners and spec.subtask_index > 0
                        else math.radians(spec.direction * spec.angle_deg))
        plan = build_counterfactual_plan(
            executor.frames, spec.stage, spec.axis, signed_angle,
            subtask_index=(0 if executor._pending_subtask_planners else spec.subtask_index),
        )
    return model, data, handles, plan, executor, environment


def prepare_dataset_nominal_trial(
    spec: CounterfactualTrialSpec,
    dataset_nominal_frames: tuple[dict[str, Any], ...],
) -> PreparedNominalTrial:
    """Prepare a seeded scene once for the existing stored-reference backend.

    This performs scene initialization only: no planner, optimizer, rendering,
    physics loop or alternative execution implementation. Actual searches call
    ``run_counterfactual_trial`` with this snapshot and the same source frames.
    """
    if (spec.execution_mode != "kinematic" or not dataset_nominal_frames
            or spec.angle_deg != 0 or any(spec.pregrasp_rotation_vector_deg)
            or any(spec.postgrasp_rotation_vector_deg)):
        raise ValueError("Dataset preparation requires unperturbed kinematic references")
    frames = tuple(TaskFrame(
        np.asarray(f["nominal_target_pose"]["position_m"], dtype=float),
        rotation_matrix(np.asarray(f["nominal_target_pose"]["rotation_vector_base_rad"], dtype=float)),
        bool(f["action"][6] > 0), str(f["phase"]),
    ) for f in dataset_nominal_frames)
    model, data, _, _, _, metadata = _setup_trial(
        spec, kinematic_only=True, frozen_nominal_frames=frames,
    )
    return PreparedNominalTrial(model, data, spec.robot_uid, frames, metadata)


def _plan_live_successor(spec, executor, data, target, accepted_rotation, offset, environment,
                         *, tolerance=None, release_state=None):
    """Task reference callback shared with UI boundary-time planning; no loop."""
    executor.index = len(executor.actions)
    try:
        executor.plan_next_subtask(data, target.position, accepted_rotation)
    except PlanningInfeasibleError as error:
        error.context['domain_randomization'] = dict(environment)
        error.context['failed_subtask_index'] = executor.subtask_index
        raise
    if tolerance is not None:
        rebase_rotation_subtask(
            target, tolerance, release_state,
            stage_key=('pregrasp', executor.subtask_index),
            accepted_rotation=accepted_rotation,
        )
    else:
        target.rotation = accepted_rotation.copy()
    frames = tuple(executor.frames)
    vectors = {stage: np.radians(np.asarray(vector)) for stage, vector in (
        ('pregrasp', spec.pregrasp_rotation_vector_deg),
        ('postgrasp', spec.postgrasp_rotation_vector_deg)) if any(vector)}
    local_plan = (build_joint_counterfactual_plan(frames, vectors) if vectors
        else build_counterfactual_plan(frames, spec.stage, spec.axis,
            math.radians(spec.direction * spec.angle_deg)
            if executor.subtask_index == spec.subtask_index else 0.))
    actions, local_interventions = apply_counterfactual_action_interventions(
        compile_task_actions(frames, target.position, target.rotation),
        target.rotation, local_plan.interventions)
    interventions = tuple(replace(i, start_index=i.start_index + offset,
        endpoint_index=i.endpoint_index + offset, verification_index=i.verification_index + offset)
        for i in local_interventions)
    environment['semantic_goal_perturbations'] = dict(executor.perturbations.samples)
    # Exclude lookup timing/cache-hit flags so cold/warm records agree.
    keys = ('path', 'entry_key', 'selected_candidate_id', 'feasible_candidates', 'candidate_count',
            'selection_metric', 'selected_total_joint_travel_rad', 'rotation_score_policy',
            'selected_rotation_deg', 'selected_score', 'selection_reason', 'selected_nominal_feasible', 'subtask_index')
    environment['nominal_plan_tables'] = [
        {k: v for k, v in table.items() if k in keys}
        for table in executor.last_planning_stats.get('nominal_plan_tables', [])]
    return frames, actions, interventions


def _run_kinematic_only_trial(
    spec: CounterfactualTrialSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: Any,
    plan: CounterfactualPlan,
    executor: PresetTaskExecutor,
    environment: dict[str, Any],
    dataset_nominal_frames: tuple[dict[str, Any], ...] | None = None,
    step_observer: Callable[[KinematicStepContext], None] | None = None,
    stored_context: StoredKinematicContext | None = None,
) -> CounterfactualTrialResult:
    """Run the production tolerance controller over FK-only 10 Hz frames.

    This branch shares the nominal action integration, stage-relative release
    state and IPOPT solve used by the physics rollout.  It deliberately omits
    task dynamics and the task evaluator.  Success means that every trajectory
    frame is optimizer-feasible and free of robot self-collision. Table
    geometry rejection is configurable for robot-only metric experiments.
    """
    control_dt = TELEOP_DT if stored_context is None else stored_context.control_dt_s
    source_frames: list[TaskFrame] | None = None
    if dataset_nominal_frames:
        source_frames = []
        for index, frame in enumerate(dataset_nominal_frames):
            try:
                nominal = frame["nominal_target_pose"]
                action = np.asarray(frame["action"], dtype=float)
                phase = str(frame["phase"])
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Dataset trajectory frame {index} is incomplete"
                ) from error
            source_frames.append(TaskFrame(
                np.asarray(nominal["position_m"], dtype=float),
                rotation_matrix(np.asarray(
                    nominal["rotation_vector_base_rad"], dtype=float,
                )),
                bool(action[6] > 0.0),
                phase,
            ))
        first = dataset_nominal_frames[0]
        source_q = np.asarray(
            first["joint_state"]["position_rad"], dtype=float,
        )[:len(handles.qpos_ids)]
        if source_q.shape != handles.qpos_ids.shape:
            raise ValueError("Dataset initial joint state has the wrong size")
        data.qpos[handles.qpos_ids] = source_q
        mujoco.mj_forward(model, data)
    target = TargetPose(*ee_pose(data, handles))
    if stored_context is not None:
        target = TargetPose(stored_context.initial_position.copy(), stored_context.initial_rotation.copy())
    nominal_rotation = target.rotation.copy()
    runtime_frames = list(plan.frames if source_frames is None else source_frames)
    if dataset_nominal_frames:
        actions = [
            TaskAction(
                np.asarray(frame["action"], dtype=float),
                str(frame.get("phase", "")),
                f"dataset frame {index}",
            )
            for index, frame in enumerate(dataset_nominal_frames)
        ]
        interventions = ()
    else:
        actions = compile_task_actions(
            runtime_frames, target.position, target.rotation,
        )
        actions, interventions = apply_counterfactual_action_interventions(
            actions, target.rotation, plan.interventions,
        )
    if interventions:
        plan = replace(
            plan,
            counterfactual_target_rotation=(
                interventions[-1].counterfactual_target_rotation.copy()
            ),
            interventions=interventions,
        )

    loss_weights = np.asarray(spec.sqp_loss_weights, dtype=float)
    if spec.sqp_loss_profile == "delta_q_squared_only":
        objective = replace(
            ObjectiveSettings.delta_q_squared_only(),
            squared_delta_q_weight=float(loss_weights[0]),
            ema_release_loss_weight=float(loss_weights[2]),
        )
    elif spec.sqp_loss_profile in SIGMA_MIN_PENALTY_PROFILES:
        objective = ObjectiveSettings.delta_q_squared_sigma_min_penalty(
            kind=SIGMA_MIN_PENALTY_PROFILES[spec.sqp_loss_profile],
            warning_threshold=spec.sqp_sigma_min_warning_threshold,
            reciprocal_epsilon=spec.sqp_sigma_min_reciprocal_epsilon,
            motion_weight=float(loss_weights[0]), sigma_min_weight=float(loss_weights[1]),
            ema_release_loss_weight=float(loss_weights[2]),
        )
    elif spec.sqp_loss_profile == "delta_q_squared_sigma_min_legacy":
        objective = ObjectiveSettings.delta_q_squared_sigma_min_legacy(
            motion_weight=float(loss_weights[0]),
            sigma_min_weight=float(loss_weights[1]),
            ema_release_loss_weight=float(loss_weights[2]),
        )
    else:
        objective = ObjectiveSettings.delta_q_squared_conditioning_safeguard(
            motion_weight=float(loss_weights[0]),
            conditioning_weight=0.0,
            ema_release_loss_weight=float(loss_weights[2]),
            eta_low=float(spec.sqp_conditioning_eta_low),
            eta_high=float(spec.sqp_conditioning_eta_high),
            sigma_epsilon=float(spec.sqp_conditioning_sigma_epsilon),
            finite_difference_relative_step=float(
                spec.sqp_conditioning_fd_relative_step
            ),
        )
    objective = replace(
        objective, squared_acceleration_weight=spec.sqp_squared_acceleration_weight,
    )
    controller = IpoptIK(
        model,
        handles,
        replace(
            _COUNTERFACTUAL_SQP_SETTINGS,
            max_iterations=int(spec.ipopt_max_iterations),
            max_cpu_time_s=(
                1.0e6
                if spec.sqp_max_time_s <= 0.0
                else float(spec.sqp_max_time_s)
            ),
        ),
        objective,
    )

    masks = np.asarray(spec.sqp_subtask_stage_rotation_masks, dtype=bool)
    negative_deg = np.asarray(
        spec.sqp_subtask_stage_negative_limits_deg, dtype=float,
    )
    positive_deg = np.asarray(
        spec.sqp_subtask_stage_positive_limits_deg, dtype=float,
    )
    if masks.size == 0:
        masks = np.asarray(spec.sqp_stage_rotation_masks, dtype=bool)[None, ...]
        negative_deg = np.asarray(
            spec.sqp_stage_negative_limits_deg, dtype=float,
        )[None, ...]
        positive_deg = np.asarray(
            spec.sqp_stage_positive_limits_deg, dtype=float,
        )[None, ...]
        if not np.any(negative_deg) and not np.any(positive_deg):
            symmetric = np.asarray(spec.sqp_symmetric_limits_deg, dtype=float)
            negative_deg = np.tile(symmetric, (1, 2, 1))
            positive_deg = np.tile(symmetric, (1, 2, 1))
    negative_rad = np.radians(negative_deg)
    positive_rad = np.radians(positive_deg)

    resolved = stored_context if stored_context is not None else (
        resolve_task_tolerance_context(
            model,
            data,
            handles,
            spec.scene,
            spec.task,
            target.position,
            target.rotation,
            planned_frames=runtime_frames,
        )
        if source_frames is None
        else plan_dataset_oracle_tolerance_from_frames(
            model,
            source_frames,
            scene=spec.scene,
            task=spec.task,
            profiles_rad={
                "pregrasp": np.zeros(6),
                "postgrasp": np.zeros(6),
            },
            initial_rotation_vector=rotation_vector(target.rotation),
        )
    )
    deferred_count = len(executor._pending_subtask_planners) if source_frames is None else 0
    if deferred_count and not spec.sqp_subtask_stage_rotation_masks:
        masks = np.repeat(masks, deferred_count + 1, axis=0)
        negative_deg = np.repeat(negative_deg, deferred_count + 1, axis=0)
        positive_deg = np.repeat(positive_deg, deferred_count + 1, axis=0)
        negative_rad, positive_rad = np.radians(negative_deg), np.radians(positive_deg)
    if len(masks) != len(resolved.subtasks) + deferred_count:
        raise ValueError(
            "Kinematic tolerance bounds must contain exactly one entry per "
            f"subtask: got {len(masks)}, expected {len(resolved.subtasks)}"
        )

    def stage_values(
        subtask_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        subtask_index = min(max(subtask_index, 0), len(masks) - 1)
        return (
            masks[subtask_index],
            negative_rad[subtask_index],
            positive_rad[subtask_index],
        )

    def resolved_subtask(subtask_index: int) -> Any:
        """Bridge live-scene and stored-trajectory tolerance descriptors."""
        if hasattr(resolved, "subtask"):
            return resolved.subtask(subtask_index)
        return resolved.subtasks[min(max(subtask_index, 0), len(resolved.subtasks) - 1)]

    def stage_rotations(descriptor: Any) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(descriptor, "target_rotation"):
            return descriptor.target_rotation, descriptor.tolerance_frame
        return (
            rotation_matrix(
                descriptor.rotation.target_rotation_vector_base_rad,
            ),
            rotation_matrix(
                descriptor.rotation.tolerance_frame_rotation_vector_base_rad,
            ),
        )

    initial_masks, initial_negative, initial_positive = stage_values(0)
    first = resolved_subtask(0).pregrasp
    first_target_rotation, first_tolerance_frame = stage_rotations(first)
    tolerance = RotationalToleranceState(
        target=first_target_rotation,
        frame=first_tolerance_frame,
        limits=0.5 * (initial_negative[0] + initial_positive[0]),
        negative_limits=initial_negative[0],
        positive_limits=initial_positive[0],
    )
    tolerance.ranged = initial_masks[0].copy()
    release_state = RotationReleaseState(
        rotation_for_joint=lambda q: controller.kinematics.evaluate(
            q, include_manipulability=False, include_link_points=False,
        ).rotation,
    )
    q = data.qpos[handles.qpos_ids].copy()
    release_state.begin_stage(
        ("pregrasp", 0),
        optimized_rotation=release_state.rotation_for_joint(q),
        nominal_rotation=target.rotation,
    )
    active_dofs = np.ones(6, dtype=bool)
    simulation_backend = PresetSimulationBackend(model, data, handles)
    phase_classifier = simulation_backend.phase_classifier
    active_phase = ManipulationPhase.PREGRASP
    previous_phase = active_phase
    previous_subtask_index = 0
    robot_bodies = _robot_body_ids(model, handles.spec.base_body)
    gripper_bodies = set().union(*(
        _body_subtree_ids(model, int(model.jnt_bodyid[joint_id]))
        for joint_id in handles.finger_joint_ids
    ))
    table_names = ("work_surface",) if stored_context is None else stored_context.table_body_names
    table_bodies = set()
    for name in table_names:
        table_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if table_root < 0:
            raise RuntimeError(f"Kinematic collision model is missing table body {name}")
        table_bodies.update(_body_subtree_ids(model, table_root))
    extrema = [
        {
            stage: {
                "negative": np.zeros(3),
                "positive": np.zeros(3),
            }
            for stage in COUNTERFACTUAL_STAGES
        }
        for _ in range(len(masks))
    ]
    sample_counts = [
        {stage: 0 for stage in COUNTERFACTUAL_STAGES}
        for _ in range(len(masks))
    ]
    rotation_release_recorder = StageRotationReleaseRecorder(len(masks))
    recorder = _TrajectoryRecorder(initial_q=q, control_dt_s=control_dt)
    clearance_recorder = (
        RobotTableClearanceRecorder(model, robot_root_name=handles.spec.base_body,
                                    table_body_name=table_names[0])
        if spec.kinematic_record_robot_table_clearance else None
    )
    hard_failures: list[str] = []
    failure_details: list[dict[str, Any]] = []
    solver_statuses: list[str] = []
    recorded_table_contacts: list[dict[str, Any]] = []
    max_position_error = 0.0
    max_rotation_error = 0.0
    executed = 0
    sampled_frame_indices = []
    previous_sample_index = None
    subtask_frame_offset = 0
    sample_indices = _kinematic_sample_indices(
        actions[:spec.maximum_frames], spec.kinematic_sampling_stride,
    )

    def set_kinematic_gripper(frame_index: int, closed: bool) -> None:
        """Place the gripper geometry at this 10 Hz trajectory sample."""
        lower = model.jnt_range[handles.finger_joint_ids, 0]
        upper = model.jnt_range[handles.finger_joint_ids, 1]
        if dataset_nominal_frames:
            physical = np.asarray(
                dataset_nominal_frames[frame_index]["gripper_state"][
                    "finger_positions_m"
                ],
                dtype=float,
            )
            if physical.shape != handles.finger_qpos_ids.shape:
                raise ValueError(
                    "Dataset finger positions have the wrong size at frame "
                    f"{frame_index}"
                )
            if stored_context is not None:
                raw = np.asarray(dataset_nominal_frames[frame_index]["gripper_state"]["raw_joint_positions_m"], dtype=float)
                if raw.shape != handles.finger_qpos_ids.shape or not np.isfinite(raw).all():
                    raise ValueError("Invalid stored native finger joints")
                data.qpos[handles.finger_qpos_ids] = raw
                return
            half_aperture = 0.5 * handles.spec.gripper_max_aperture_m
            fraction = np.clip(physical / half_aperture, 0.0, 1.0)
            raw = (
                lower + fraction * (upper - lower)
                if handles.spec.gripper_open_at_upper
                else upper - fraction * (upper - lower)
            )
        else:
            open_endpoint = upper if handles.spec.gripper_open_at_upper else lower
            closed_endpoint = lower if handles.spec.gripper_open_at_upper else upper
            raw = closed_endpoint if closed else open_endpoint
        data.qpos[handles.finger_qpos_ids] = raw

    for frame_index in range(spec.maximum_frames):
        if frame_index >= len(actions):
            if source_frames is not None or not executor._pending_subtask_planners:
                break
            new_frames, new_actions, new_interventions = _plan_live_successor(
                spec, executor, data, target, release_state.rotation_for_joint(q), len(actions), environment,
                tolerance=tolerance, release_state=release_state)
            subtask_frame_offset = len(actions)
            runtime_frames.extend(new_frames)
            actions.extend(new_actions)
            resolved = resolve_task_tolerance_context(model, data, handles, spec.scene, spec.task,
                target.position, target.rotation, planned_frames=runtime_frames)
            sample_indices = _kinematic_sample_indices(actions[:spec.maximum_frames], spec.kinematic_sampling_stride)
        action = actions[frame_index]
        commanded_closed = action.gripper_closed
        if dataset_nominal_frames:
            set_kinematic_gripper(frame_index, commanded_closed)
        semantic_index = (
            min(int(dataset_nominal_frames[frame_index].get('subtask_index', 0))
                - int(dataset_nominal_frames[0].get('subtask_index', 0)), len(masks) - 1)
            if dataset_nominal_frames else executor.subtask_index
        )
        if stored_context is None:
            phase_observation = simulation_backend.observe_manipulation_phase(
                commanded_closed, preset_subtask_index=semantic_index,
            )
        else:
            # Imported observations were classified causally at the source rate.
            # Do not reclassify counterfactual geometry or advance its phase clock.
            from ..tasks.phases import PhaseObservation
            phase = {p.key:p for p in ManipulationPhase}[dataset_nominal_frames[frame_index]["phase"]]
            aperture = float(sum(dataset_nominal_frames[frame_index]["gripper_state"]["finger_positions_m"]))
            phase_observation = PhaseObservation(phase, aperture, 0., commanded_closed, semantic_index)
        active_phase = phase_observation.phase
        if (active_phase is not previous_phase
                or phase_observation.subtask_index != previous_subtask_index):
            subtask_index = int(phase_observation.subtask_index)
            subtask_changed = subtask_index != previous_subtask_index
            stage_masks, stage_negative, stage_positive = stage_values(
                subtask_index,
            )
            if active_phase in (
                ManipulationPhase.PREGRASP,
                ManipulationPhase.POSTGRASP,
            ):
                stage_index = int(
                    active_phase is ManipulationPhase.POSTGRASP
                )
                descriptor = (
                    resolved_subtask(subtask_index).pregrasp
                    if stage_index == 0
                    else resolved_subtask(subtask_index).postgrasp
                )
                descriptor_target, descriptor_frame = stage_rotations(
                    descriptor,
                )
                tolerance.configure_stage(
                    int(active_phase),
                    descriptor_target,
                    descriptor_frame,
                    0.5 * (
                        stage_negative[stage_index]
                        + stage_positive[stage_index]
                    ),
                    negative_limits=stage_negative[stage_index],
                    positive_limits=stage_positive[stage_index],
                    stage_key=(active_phase.key, subtask_index),
                )
                tolerance.ranged = stage_masks[stage_index].copy()
            else:
                tolerance.active_stage = int(active_phase)
                tolerance.ranged[:] = False
            handoff = controller.kinematics.evaluate(
                q, include_manipulability=False, include_link_points=False,
            ).rotation
            stage_key = (active_phase.key, subtask_index)
            if subtask_changed:
                rebase_rotation_subtask(
                    target,
                    tolerance,
                    release_state,
                    stage_key=stage_key,
                    accepted_rotation=handoff,
                )
            else:
                release_state.begin_stage(
                    stage_key,
                    optimized_rotation=handoff,
                    nominal_rotation=target.rotation,
                )
            previous_phase = active_phase
            previous_subtask_index = subtask_index

        target.position, target.rotation = advance_reference_pose(
            target.position,
            target.rotation,
            action.value,
            tolerance=tolerance,
            dt=control_dt,
        )
        # Advance phase observations at every original 10 Hz command, even
        # when expensive kinematic solves are sampled more sparsely.
        if not dataset_nominal_frames:
            set_kinematic_gripper(frame_index, commanded_closed)
        if frame_index not in sample_indices:
            # Integrate EVERY original command, including noncommuting
            # rotations. Do not compress the reference or its timestamps.
            continue
        data.qpos[handles.qpos_ids] = q
        set_kinematic_gripper(frame_index, commanded_closed)
        mujoco.mj_forward(model, data)
        if step_observer is not None:
            step_observer(KinematicStepContext(
                frame_index=frame_index, phase=active_phase.key,
                subtask_index=int(phase_observation.subtask_index), dt=control_dt,
                model=model, data=data, controller=controller, target=target,
                tolerance=tolerance, release_state=release_state,
                active_dofs=active_dofs,
                subtask_nominal_frames=(tuple(runtime_frames[subtask_frame_offset:])
                    if frame_index == subtask_frame_offset else ()),
            ))
        try:
            q_next, diagnostics, _, _, release_step = (
                solve_stage_relative_target(
                    controller,
                    data,
                    target,
                    tolerance,
                    release_state,
                    active_dofs,
                )
            )
        except (FloatingPointError, ValueError, RuntimeError) as error:
            hard_failures.append("optimizer_exception")
            failure_details.append({
                "frame": frame_index,
                "phase": active_phase.key,
                "subtask_index": int(phase_observation.subtask_index),
                "exception": type(error).__name__,
                "message": str(error),
            })
            break
        solver_statuses.append(str(diagnostics.status))
        if not diagnostics.feasible:
            hard_failures.append("optimizer_infeasible")
            failure_details.append({
                "frame": frame_index,
                "phase": active_phase.key,
                "subtask_index": int(phase_observation.subtask_index),
                "status": str(diagnostics.status),
                "position_residual": float(diagnostics.position_residual),
                "rotation_residual": float(
                    diagnostics.strict_rotation_residual
                ),
                "constraint_violation": float(
                    diagnostics.effective_constraint_violation
                ),
            })
            break
        q_next = np.asarray(q_next, dtype=float)
        if q_next.shape != q.shape or not np.all(np.isfinite(q_next)):
            hard_failures.append("optimizer_non_finite")
            failure_details.append({"frame": frame_index})
            break
        q = q_next
        data.qpos[handles.qpos_ids] = q
        set_kinematic_gripper(frame_index, commanded_closed)
        mujoco.mj_forward(model, data)
        if clearance_recorder is not None:
            clearance_recorder.sample(
                data, phase=active_phase.key, frame_index=frame_index,
            )
        if not spec.kinematic_check_table_collisions:
            collision = _kinematic_collision_failure(
                model, data, robot_bodies, table_bodies, gripper_bodies,
                check_table_collisions=False,
            )
        elif spec.kinematic_continue_after_table_contact:
            frame_contacts: list[dict[str, Any]] = []
            collision = _kinematic_collision_failure(
                model, data, robot_bodies, table_bodies, gripper_bodies,
                recorded_table_contacts=frame_contacts,
            )
            recorded_table_contacts.extend(
                {"frame": frame_index, "phase": active_phase.key, **event}
                for event in frame_contacts
            )
        else:
            collision = _kinematic_collision_failure(
                model, data, robot_bodies, table_bodies, gripper_bodies,
            )
        if collision is not None:
            hard_failures.append(str(collision["category"]))
            failure_details.append({"frame": frame_index, **collision})
            break

        state = controller.kinematics.evaluate(
            q, include_manipulability=True, include_link_points=False,
        )
        position_error = float(np.linalg.norm(target.position - state.position))
        rotation_error = float(np.linalg.norm(
            rotation_vector(target.rotation.T @ state.rotation),
        ))
        max_position_error = max(max_position_error, position_error)
        max_rotation_error = max(max_rotation_error, rotation_error)
        recorder.sample(
            target,
            state.position,
            state.rotation,
            q,
            state.manipulability,
            rotation_error_rad=rotation_error,
            # Tool-frame angular rows are an orthogonal row transform of the
            # full geometric Jacobian: its singular values are unchanged.
            sigma_min_full=float(np.linalg.svd(state.jacobian, compute_uv=False)[-1]),
            sigma_min_velocity_normalized=velocity_normalized_sigma_min(state.jacobian, handles.spec.joint_velocity_limits),
            elapsed_frames=(
                1 if previous_sample_index is None
                else frame_index - previous_sample_index
            ),
        )
        previous_sample_index = frame_index
        sampled_frame_indices.append(frame_index)

        stage = (
            active_phase.key
            if active_phase in (
                ManipulationPhase.PREGRASP,
                ManipulationPhase.POSTGRASP,
            )
            else None
        )
        if stage is not None:
            subtask_index = min(int(phase_observation.subtask_index), len(extrema) - 1)
            release = release_state.candidate(
                release_state.accepted_rotation,
                release_step,
            ).stage_relative_tolerance_release_rad
            stage_extrema = extrema[subtask_index][stage]
            stage_extrema["negative"] = np.maximum(
                stage_extrema["negative"], np.maximum(-release, 0.0),
            )
            stage_extrema["positive"] = np.maximum(
                stage_extrema["positive"], np.maximum(release, 0.0),
            )
            sample_counts[subtask_index][stage] += 1
            rotation_release_recorder.sample(
                subtask_index,stage,
                release_state.accepted_rotation @ release_step.stage_reference_rotation().T,
            )

        executed = frame_index + 1

    expected_frames = min(len(actions), spec.maximum_frames)
    success = not hard_failures and not recorded_table_contacts and executed == expected_frames
    reported_failures = hard_failures + (["robot_table_collision"] if recorded_table_contacts else [])
    metrics = recorder.result().as_dict()
    release_extrema = [
        {
            "subtask_index": subtask_index,
            **{
                stage: {
                    side: np.degrees(values).tolist()
                    for side, values in sides.items()
                }
                for stage, sides in stages.items()
            },
        }
        for subtask_index, stages in enumerate(extrema)
    ]
    evaluation = {
        "backend": "kinematic_collision_checked",
        "sigma_min_normalization": sigma_min_normalization_metadata(handles.spec.joint_velocity_limits),
        "physics_steps": 0,
        "control_rate_hz": int(round(1.0 / control_dt)),
        "task_interaction": False,
        "objects_considered": False,
        "collision_engine": "mujoco_mj_forward",
        "collision_scope": ["robot_self", "robot_table"] if spec.kinematic_check_table_collisions else ["robot_self"],
        "collision_condition": "actual_geometry_contact_distance_m <= 0",
        "executed_frames": executed,
        "expected_frames": expected_frames,
        "planned_frames": len(actions),
        "trajectory_complete": not hard_failures and executed == len(actions) and not executor._pending_subtask_planners,
        "planned_subtasks": len(resolved.subtasks),
        "pending_subtasks": len(executor._pending_subtask_planners),
        "max_position_error_m": max_position_error,
        "max_rotation_error_rad": max_rotation_error,
        "optimizer_feasible": not any(
            reason.startswith("optimizer") for reason in hard_failures
        ),
        "optimizer_statuses": solver_statuses,
        "hard_failure_reasons": reported_failures,
        "hard_failure_details": failure_details,
        "sqp_subtask_stage_release_extrema_deg": release_extrema,
        "sqp_subtask_stage_release_sample_counts": [
            {"subtask_index": index, **counts}
            for index, counts in enumerate(sample_counts)
        ],
        "sqp_subtask_stage_rotation_masks": masks.astype(int).tolist(),
        "declared_subtask_stage_rotation_masks": (
            _declared_subtask_stage_masks(
                model, spec, resolved, len(masks),
            )
        ),
        "sqp_subtask_stage_negative_limits_deg": negative_deg.tolist(),
        "sqp_subtask_stage_positive_limits_deg": positive_deg.tolist(),
        "sqp_release_measurement": (
            "running positive and negative maxima of optimizer-owned "
            "stage-relative release at every source control frame"
        ),
        "trajectory_kinematics": metrics,
    }
    evaluation["sqp_rotation_release"] = rotation_release_recorder.summary(
        trajectory_complete=evaluation["trajectory_complete"],
        sampling_stride=spec.kinematic_sampling_stride,
    )
    if spec.kinematic_continue_after_table_contact:
        evaluation.update(
            table_contact_policy="record_and_continue_fk_only",
            kinematic_feasible_without_table=(not hard_failures and executed == len(actions)),
            motion_failure_reasons=list(hard_failures),
            table_contact_count=len(recorded_table_contacts),
            table_contact_frames=sorted({c["frame"] for c in recorded_table_contacts}),
            first_table_contact=recorded_table_contacts[0] if recorded_table_contacts else None,
        )
    if clearance_recorder is not None:
        evaluation["robot_table_clearance"] = clearance_recorder.summary()
        evaluation["clearance_domain"] = f"solved_joint_positions_{1 / control_dt:g}hz_fk; no physics/interpolation"
    if spec.kinematic_sampling_stride > 1:
        evaluation.update(
            kinematic_sampling_stride=spec.kinematic_sampling_stride,
            sampled_frame_indices=sampled_frame_indices,
            solved_samples=len(sampled_frame_indices),
            validation_domain="sparse_nominal_screening; unchecked intervening poses",
            kinematic_metric_domain="sparse_solved_samples; original elapsed time weighted",
            sqp_release_measurement="sampled FK endpoints only; zero nominal tolerance",
        )
        if clearance_recorder is not None:
            evaluation["clearance_domain"] = "sparse_solved_joint_positions_fk; no physics/interpolation"
    return CounterfactualTrialResult(
        spec=spec,
        success=success,
        task_success=success,
        target_reached=success,
        termination_reason=(
            "kinematic_success" if success
            else hard_failures[0] if hard_failures
            else "robot_table_collision" if recorded_table_contacts
            else "trajectory_incomplete"
        ),
        target_rotation_error_deg=float(np.degrees(max_rotation_error)),
        nominal_target_rotation_vector_base_rad=tuple(
            rotation_vector(nominal_rotation),
        ),
        counterfactual_target_rotation_vector_base_rad=tuple(
            rotation_vector(plan.counterfactual_target_rotation),
        ),
        tolerance_frame_rotation_vector_base_rad=tuple(
            rotation_vector(plan.tolerance_frame_rotation),
        ),
        evaluation=evaluation,
        domain_randomization={
            **environment,
            "execution_mode": "kinematic",
            "task_interaction": False,
        },
        hard_failure_reasons=tuple(reported_failures),
        trajectory_kinematics=metrics,
    )


def run_counterfactual_trial(
    spec: CounterfactualTrialSpec,
    *,
    video_path: Path | None = None,
    video_width: int = 640,
    video_height: int = 480,
    video_view: str = "combined",
    dataset_video_dir: Path | None = None,
    dataset_video_frame_count: int | None = None,
    dataset_video_source_indices: tuple[int, ...] | None = None,
    dataset_nominal_frames: tuple[dict[str, Any], ...] | None = None,
    prepared_nominal: PreparedNominalTrial | None = None,
    kinematic_step_observer: Callable[[KinematicStepContext], None] | None = None,
) -> CounterfactualTrialResult:
    """Run one deterministic no-render counterfactual rollout."""
    stored = None if prepared_nominal is None else prepared_nominal.stored_context
    if stored is not None and (spec.execution_mode != "kinematic" or dataset_nominal_frames is None):
        raise ValueError("Stored scene context requires source frames and kinematic execution")
    if stored is None and (
        spec.task not in PresetTaskExecutor.task_names(spec.scene)
        or (spec.scene, spec.task) not in registered_tasks()
    ):
        raise ValueError(
            "Counterfactual runner requires a registered preset task: "
            f"scene={spec.scene!r}, task={spec.task!r}",
        )
    if spec.direction not in (-1, 1):
        raise ValueError("Counterfactual direction must be -1 or +1")
    if not 0 <= spec.angle_deg <= 180:
        raise ValueError("Counterfactual angle must be in [0, 180] degrees")
    for vector in (
        spec.pregrasp_rotation_vector_deg,
        spec.postgrasp_rotation_vector_deg,
    ):
        array = np.asarray(vector, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError("Joint intervention vectors must contain three finite values")
        if np.any(np.abs(array) > 180.0):
            raise ValueError("Joint intervention components must be within +/-180 degrees")
    if spec.maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    if not 0.0 <= spec.environment_randomization_probability <= 1.0:
        raise ValueError("environment randomization probability must be in [0, 1]")
    if not (
        0.0 <= spec.workspace_boundary_distance_min
        <= spec.workspace_boundary_distance_max <= 2.0
    ):
        raise ValueError(
            "workspace-boundary factors must satisfy 0 <= min <= max <= 2",
        )
    if dataset_video_source_indices is not None and dataset_video_dir is None:
        raise ValueError(
            "dataset_video_source_indices requires dataset_video_dir",
        )
    if (
        dataset_nominal_frames is not None
        and dataset_video_source_indices is not None
        and len(dataset_nominal_frames) != len(dataset_video_source_indices)
    ):
        raise ValueError(
            "dataset_nominal_frames must match supplied source indices",
        )
    if spec.execution_mode not in ("simulation", "kinematic"):
        raise ValueError("execution_mode must be 'simulation' or 'kinematic'")
    if kinematic_step_observer is not None and spec.execution_mode != "kinematic":
        raise ValueError("Kinematic step observers require kinematic execution")
    if not spec.kinematic_check_table_collisions and spec.execution_mode != "kinematic":
        raise ValueError("Disabling table collision rejection requires kinematic execution")
    if not np.isfinite(spec.sqp_squared_acceleration_weight) or spec.sqp_squared_acceleration_weight < 0.0:
        raise ValueError("sqp_squared_acceleration_weight must be finite and non-negative")
    validate_sigma_min_loss("negative", spec.sqp_sigma_min_warning_threshold, spec.sqp_sigma_min_reciprocal_epsilon)
    if (isinstance(spec.kinematic_sampling_stride, bool)
            or not isinstance(spec.kinematic_sampling_stride, int)
            or spec.kinematic_sampling_stride < 1):
        raise ValueError("kinematic_sampling_stride must be a positive integer")
    if spec.kinematic_sampling_stride > 1 and (
        spec.execution_mode != "kinematic" or prepared_nominal is None
        or dataset_nominal_frames is not None or spec.angle_deg != 0
        or any(spec.pregrasp_rotation_vector_deg)
        or any(spec.postgrasp_rotation_vector_deg)
        or spec.sqp_unbounded_rotation
        or np.any(spec.sqp_stage_rotation_masks)
        or np.any(spec.sqp_subtask_stage_rotation_masks)
    ):
        raise ValueError("Sparse sampling requires a prepared zero-tolerance nominal kinematic trial")
    if spec.kinematic_continue_after_table_contact and (
        not (spec.execution_mode == "kinematic" or spec.controller_mode == "kinematic")
        or not spec.kinematic_record_robot_table_clearance
    ):
        raise ValueError("Continuing table contacts requires pure kinematic execution and clearance recording")
    if spec.controller_mode not in (
        "kinematic", "sqp_specific", "sqp_release", "hybrid_stage_release",
    ):
        raise ValueError(
            "unsupported counterfactual controller_mode",
        )
    sqp_limits_deg = np.asarray(spec.sqp_symmetric_limits_deg, dtype=float)
    sqp_masks = np.asarray(spec.sqp_stage_rotation_masks, dtype=int)
    sqp_negative_deg = np.asarray(
        spec.sqp_stage_negative_limits_deg, dtype=float,
    )
    sqp_positive_deg = np.asarray(
        spec.sqp_stage_positive_limits_deg, dtype=float,
    )
    subtask_sqp_masks = np.asarray(
        spec.sqp_subtask_stage_rotation_masks, dtype=int,
    )
    subtask_sqp_negative_deg = np.asarray(
        spec.sqp_subtask_stage_negative_limits_deg, dtype=float,
    )
    subtask_sqp_positive_deg = np.asarray(
        spec.sqp_subtask_stage_positive_limits_deg, dtype=float,
    )
    sqp_loss_weights = np.asarray(spec.sqp_loss_weights, dtype=float)
    sqp_profiles = (
        "delta_q_squared_only",
        "delta_q_squared_sigma_min_legacy",
        *SIGMA_MIN_PENALTY_PROFILES,
        "delta_q_squared_conditioning_safeguard",
    )
    if (
        sqp_limits_deg.shape != (3,)
        or not np.all(np.isfinite(sqp_limits_deg))
        or np.any(sqp_limits_deg < 0.0)
        or sqp_masks.shape != (2, 3)
        or np.any((sqp_masks != 0) & (sqp_masks != 1))
        or sqp_negative_deg.shape != (2, 3)
        or sqp_positive_deg.shape != (2, 3)
        or not np.all(np.isfinite(sqp_negative_deg))
        or not np.all(np.isfinite(sqp_positive_deg))
        or np.any(sqp_negative_deg < 0.0)
        or np.any(sqp_positive_deg < 0.0)
        or spec.subtask_index < 0
        or spec.sqp_loss_profile not in sqp_profiles
        or sqp_loss_weights.shape != (3,)
        or not np.all(np.isfinite(sqp_loss_weights))
        or np.any(sqp_loss_weights < 0.0)
        or not np.isfinite(spec.sqp_conditioning_eta_low)
        or not np.isfinite(spec.sqp_conditioning_eta_high)
        or not (
            0.0 <= spec.sqp_conditioning_eta_low
            < spec.sqp_conditioning_eta_high <= 1.0
        )
        or not np.isfinite(spec.sqp_conditioning_sigma_epsilon)
        or spec.sqp_conditioning_sigma_epsilon <= 0.0
        or not np.isfinite(spec.sqp_conditioning_fd_relative_step)
        or spec.sqp_conditioning_fd_relative_step <= 0.0
        or not np.isfinite(spec.sqp_max_time_s)
        or spec.sqp_max_time_s < 0.0
        or spec.ipopt_max_iterations <= 0
    ):
        raise ValueError("Invalid SQP limits or stage masks")
    if (
        spec.sqp_loss_profile == "delta_q_squared_conditioning_safeguard"
        and float(sqp_loss_weights[1]) != 0.0
    ):
        raise ValueError(
            "The conditioning (w_c) objective is suspended and cannot be enabled",
        )
    has_subtask_release = subtask_sqp_masks.size > 0
    if has_subtask_release:
        if (
            subtask_sqp_masks.ndim != 3
            or subtask_sqp_masks.shape[1:] != (2, 3)
            or np.any((subtask_sqp_masks != 0) & (subtask_sqp_masks != 1))
            or subtask_sqp_negative_deg.shape != subtask_sqp_masks.shape
            or subtask_sqp_positive_deg.shape != subtask_sqp_masks.shape
            or not np.all(np.isfinite(subtask_sqp_negative_deg))
            or not np.all(np.isfinite(subtask_sqp_positive_deg))
            or np.any(subtask_sqp_negative_deg < 0.0)
            or np.any(subtask_sqp_positive_deg < 0.0)
        ):
            raise ValueError("Invalid per-subtask SQP limits or stage masks")
    elif (
        subtask_sqp_negative_deg.size > 0
        or subtask_sqp_positive_deg.size > 0
    ):
        raise ValueError("Per-subtask SQP bounds require per-subtask masks")
    if (
        spec.controller_mode in (
            "sqp_release", "hybrid_stage_release",
        )
        and not (
            np.any(subtask_sqp_masks) if has_subtask_release
            else np.any(sqp_masks)
        )
    ):
        raise ValueError("SQP release mode requires at least one released axis")
    if (
        spec.sqp_unbounded_rotation
        and spec.controller_mode not in (
            "sqp_release", "hybrid_stage_release",
        )
    ):
        raise ValueError(
            "Unbounded rotation requires a release controller mode",
        )

    try:
        if prepared_nominal is None:
            model, data, handles, plan, executor, environment = _setup_trial(
                spec,
                kinematic_only=(
                    spec.execution_mode == "kinematic"
                    or spec.controller_mode == "kinematic"
                ),
            )
        else:
            if (spec.execution_mode != "kinematic"
                    or prepared_nominal.robot_uid != spec.robot_uid or not prepared_nominal.frames
                    or spec.angle_deg != 0 or any(spec.pregrasp_rotation_vector_deg)
                    or any(spec.postgrasp_rotation_vector_deg)):
                raise ValueError("Prepared nominal validation requires FK and frozen unperturbed-stage actions")
            model = copy.copy(prepared_nominal.model)
            data = mujoco.MjData(model)
            if hasattr(mujoco, "mj_copyData"):
                mujoco.mj_copyData(data, model, prepared_nominal.data)
            else:
                # MuJoCo 3.2 (native LIBERO runtime) exposes state transfer but
                # not mj_copyData. This preserves all integration inputs and
                # never advances dynamics.
                state_type = mujoco.mjtState.mjSTATE_INTEGRATION
                snapshot = np.empty(mujoco.mj_stateSize(prepared_nominal.model, state_type))
                mujoco.mj_getState(prepared_nominal.model, prepared_nominal.data, snapshot, state_type)
                mujoco.mj_setState(model, data, snapshot, state_type)
                mujoco.mj_forward(model, data)
            handles = find_handles(model, get_robot_spec(spec.robot_uid)) if stored is None else stored.handles
            if stored is None:
                executor = PresetTaskExecutor(model, handles)
                executor.frames = list(prepared_nominal.frames)
                executor.actions = compile_task_actions(executor.frames, *ee_pose(data, handles))
                executor.active = True
            else:
                # Stored input is already a complete plan; no preset task
                # constructor, planning callback or action recompilation runs.
                from types import SimpleNamespace
                executor = SimpleNamespace(frames=list(prepared_nominal.frames),
                                           _pending_subtask_planners=[], subtask_index=0)
            if stored is None:
                plan = build_counterfactual_plan(executor.frames, spec.stage, spec.axis, 0.0)
            else:
                first = executor.frames[0].rotation
                plan = CounterfactualPlan(tuple(executor.frames), spec.stage, spec.axis, 0., first, first, np.eye(3), 0, 0, len(executor.frames)-1, len(executor.frames)-1)
            environment = copy.deepcopy(prepared_nominal.metadata)
        if dataset_nominal_frames is not None:
            # Stored replay references already contain their complete task
            # sequence; never append newly planned successors to that record.
            executor._pending_subtask_planners.clear()
        if spec.execution_mode == "kinematic" or spec.controller_mode == "kinematic":
            return _run_kinematic_only_trial(
                spec, model, data, handles, plan, executor, environment,
                dataset_nominal_frames,
                step_observer=kinematic_step_observer,
                stored_context=stored,
            )
        runtime_frames = list(plan.frames)
        robot = handles.spec
        evaluator = create_task_evaluator(spec.scene, spec.task, model, data)
        recorder = (
            CounterfactualVideoRecorder(
                model,
                video_path,
                width=video_width,
                height=video_height,
                view=video_view,
            )
            if video_path is not None
            else None
        )
        dataset_recorder = (
            CounterfactualDatasetVideoRecorder(
                model,
                dataset_video_dir,
                width=video_width,
                height=video_height,
                frame_limit=dataset_video_frame_count,
                source_frame_indices=dataset_video_source_indices,
            )
            if dataset_video_dir is not None
            else None
        )
        simulation_backend = PresetSimulationBackend(model, data, handles)
        phase_classifier = simulation_backend.phase_classifier
        controller_handles = handles
        if spec.sqp_loss_profile == "delta_q_squared_only":
            sqp_settings = replace(
                ObjectiveSettings.delta_q_squared_only(),
                squared_delta_q_weight=float(sqp_loss_weights[0]),
                ema_release_loss_weight=float(sqp_loss_weights[2]),
            )
        elif spec.sqp_loss_profile in SIGMA_MIN_PENALTY_PROFILES:
            sqp_settings = ObjectiveSettings.delta_q_squared_sigma_min_penalty(
                kind=SIGMA_MIN_PENALTY_PROFILES[spec.sqp_loss_profile],
                warning_threshold=spec.sqp_sigma_min_warning_threshold,
                reciprocal_epsilon=spec.sqp_sigma_min_reciprocal_epsilon,
                motion_weight=float(sqp_loss_weights[0]), sigma_min_weight=float(sqp_loss_weights[1]),
                ema_release_loss_weight=float(sqp_loss_weights[2]),
            )
        elif spec.sqp_loss_profile == "delta_q_squared_sigma_min_legacy":
            sqp_settings = ObjectiveSettings.delta_q_squared_sigma_min_legacy(
                motion_weight=float(sqp_loss_weights[0]),
                sigma_min_weight=float(sqp_loss_weights[1]),
                ema_release_loss_weight=float(sqp_loss_weights[2]),
            )
        else:
            sqp_settings = ObjectiveSettings.delta_q_squared_conditioning_safeguard(
                motion_weight=float(sqp_loss_weights[0]),
                conditioning_weight=0.0,
                ema_release_loss_weight=float(sqp_loss_weights[2]),
                eta_low=float(spec.sqp_conditioning_eta_low),
                eta_high=float(spec.sqp_conditioning_eta_high),
                sigma_epsilon=float(spec.sqp_conditioning_sigma_epsilon),
                finite_difference_relative_step=float(
                    spec.sqp_conditioning_fd_relative_step
                ),
            )
        sqp_settings = replace(
            sqp_settings, squared_acceleration_weight=spec.sqp_squared_acceleration_weight,
        )
        sqp_ik = IpoptIK(
            model,
            controller_handles,
            replace(
                _COUNTERFACTUAL_SQP_SETTINGS,
                max_iterations=int(spec.ipopt_max_iterations),
                max_cpu_time_s=(
                    1.0e6
                    if spec.sqp_max_time_s <= 0.0
                    else float(spec.sqp_max_time_s)
                ),
            ),
            sqp_settings,
            diagnose_conditioning_gradient=(
                spec.sqp_diagnose_conditioning_gradient
            ),
        )
        target = TargetPose(*ee_pose(data, handles))
        # Keep the preset/action-integrated target independent from the
        # controller reference.  Rotational tolerance is allowed to change
        # only the latter; dataset supervision must remain the nominal pose
        # requested by the scripted action sequence.
        dataset_nominal_position = target.position.copy()
        dataset_nominal_rotation = target.rotation.copy()
        dataset_previous_nominal_rotvec = rotation_vector(target.rotation)
        dataset_previous_actual_rotvec = dataset_previous_nominal_rotvec.copy()
        actions = compile_task_actions(
            runtime_frames, target.position, target.rotation,
        )
        actions, updated_interventions = (
            apply_counterfactual_action_interventions(
                actions, target.rotation, plan.interventions,
            )
        )
        plan = replace(
            plan,
            counterfactual_target_rotation=(
                updated_interventions[-1].counterfactual_target_rotation.copy()
            ),
            interventions=updated_interventions,
        )
        result_plan = plan
        runtime_interventions = list(updated_interventions)
        joint_command = data.qpos[handles.qpos_ids].copy()
        joint_target = joint_command.copy()
        release_state = RotationReleaseState(
            rotation_for_joint=(
                None
                if sqp_ik is None
                else lambda q: sqp_ik.kinematics.evaluate(
                    q, include_manipulability=False, include_link_points=False,
                ).rotation
            ),
        )
        joint_tracker = JointReferenceTracker(
            controller_handles.spec.arm_dof,
        )
        maximum_joint_tracking_error_rad = 0.0
        maximum_joint_tracking_correction_rad = 0.0
        joint_tracking_saturation_steps = 0
        sqp_conditioning_samples: list[dict[str, object]] = []
        sqp_solver_statuses: list[str] = []
        sqp_solver_elapsed_ms: list[float] = []
        sqp_solver_rejected = 0
        sqp_solver_trace: list[dict[str, Any]] = []
        gripper_ctrl = robot.gripper_open_ctrl
        gripper_closed = False
        gripper_speed = abs(
            robot.gripper_open_ctrl - robot.gripper_closed_ctrl
        ) / GRIPPER_TRAVEL_TIME
        physics_steps = max(
            1, int(round(TELEOP_DT / float(model.opt.timestep))),
        )
        trajectory_recorder = _TrajectoryRecorder(initial_q=data.qpos[handles.qpos_ids])
        optimized_trajectory_recorder = _TrajectoryRecorder(initial_q=data.qpos[handles.qpos_ids])
        dataset_trajectory: list[dict[str, Any]] = []
        if dataset_nominal_frames is None:
            dataset_output_indices = (
                {index: index for index in range(spec.maximum_frames)}
                if dataset_video_dir is not None else {}
            )
        elif dataset_video_source_indices is None:
            dataset_output_indices = {
                index: index for index in range(len(dataset_nominal_frames))
            }
        else:
            dataset_output_indices = {
                source_index: output_index
                for output_index, source_index in enumerate(
                    dataset_video_source_indices
                )
            }
        hard_failure_reasons: list[str] = []
        hard_failure_details: list[dict[str, Any]] = []
        phase_upright_angles: dict[str, list[float]] = {}
        active_dofs = np.ones(6, dtype=bool)
        target_errors_deg = {
            intervention.stage: float("inf")
            for intervention in runtime_interventions
        }
        intervention_delta = (
            plan.counterfactual_target_rotation
            @ plan.nominal_target_rotation.T
        )
        postgrasp_nominal = (
            intervention_delta.T
            @ plan.frames[plan.postgrasp_endpoint_index].rotation
        )
        postgrasp_frame = box_tolerance_frame(postgrasp_nominal)
        if not np.any(sqp_negative_deg) and not np.any(sqp_positive_deg):
            sqp_negative_deg = np.tile(sqp_limits_deg, (2, 1))
            sqp_positive_deg = np.tile(sqp_limits_deg, (2, 1))
        sqp_negative_rad = np.radians(sqp_negative_deg)
        sqp_positive_rad = np.radians(sqp_positive_deg)
        sqp_masks_array = np.asarray(
            spec.sqp_stage_rotation_masks, dtype=bool,
        )
        if spec.sqp_unbounded_rotation:
            sqp_negative_rad = np.where(
                sqp_masks_array, np.inf, sqp_negative_rad,
            )
            sqp_positive_rad = np.where(
                sqp_masks_array, np.inf, sqp_positive_rad,
            )
        if has_subtask_release:
            sqp_subtask_masks_array = subtask_sqp_masks.astype(bool)
            sqp_subtask_negative_rad = np.radians(
                subtask_sqp_negative_deg,
            )
            sqp_subtask_positive_rad = np.radians(
                subtask_sqp_positive_deg,
            )
            if spec.sqp_unbounded_rotation:
                sqp_subtask_negative_rad = np.where(
                    sqp_subtask_masks_array,
                    np.inf,
                    sqp_subtask_negative_rad,
                )
                sqp_subtask_positive_rad = np.where(
                    sqp_subtask_masks_array,
                    np.inf,
                    sqp_subtask_positive_rad,
                )
        else:
            sqp_subtask_masks_array = np.empty((0, 2, 3), dtype=bool)
            sqp_subtask_negative_rad = np.empty((0, 2, 3), dtype=float)
            sqp_subtask_positive_rad = np.empty((0, 2, 3), dtype=float)

        def release_for_subtask(
            subtask_index: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not has_subtask_release:
                return sqp_masks_array, sqp_negative_rad, sqp_positive_rad
            index = min(
                max(int(subtask_index), 0),
                len(sqp_subtask_masks_array) - 1,
            )
            return (
                sqp_subtask_masks_array[index],
                sqp_subtask_negative_rad[index],
                sqp_subtask_positive_rad[index],
            )
        rotational_tolerance: RotationalToleranceState | None = None
        resolved_tolerance = None
        instantaneous_rotation_target = target.rotation.copy()
        active_phase = ManipulationPhase.PREGRASP
        previous_phase = active_phase
        previous_subtask_index = 0
        if spec.controller_mode in (
            "sqp_specific", "sqp_release", "hybrid_stage_release",
        ):
            resolved_tolerance = resolve_task_tolerance_context(
                model,
                data,
                handles,
                spec.scene,
                spec.task,
                target.position,
                target.rotation,
                planned_frames=runtime_frames,
            )
            initial_masks, initial_negative, initial_positive = (
                release_for_subtask(0)
            )
            pre_limits = 0.5 * (
                initial_negative[0] + initial_positive[0]
            )
            rotational_tolerance = RotationalToleranceState(
                target=resolved_tolerance.pregrasp.target_rotation,
                frame=resolved_tolerance.pregrasp.tolerance_frame,
                limits=pre_limits,
                negative_limits=initial_negative[0],
                positive_limits=initial_positive[0],
            )
            rotational_tolerance.configure_stage(
                int(ManipulationPhase.POSTGRASP),
                resolved_tolerance.postgrasp.target_rotation,
                resolved_tolerance.postgrasp.tolerance_frame,
                0.5 * (initial_negative[1] + initial_positive[1]),
                negative_limits=initial_negative[1],
                positive_limits=initial_positive[1],
                stage_key=("postgrasp", 0),
            )
            rotational_tolerance.configure_stage(
                int(ManipulationPhase.PREGRASP),
                resolved_tolerance.pregrasp.target_rotation,
                resolved_tolerance.pregrasp.tolerance_frame,
                pre_limits,
                negative_limits=initial_negative[0],
                positive_limits=initial_positive[0],
                stage_key=("pregrasp", 0),
            )
            rotational_tolerance.ranged = initial_masks[0].copy()
            if sqp_ik is not None:
                release_state.begin_stage(
                    ("pregrasp", 0),
                    optimized_rotation=release_state.rotation_for_joint(joint_target),
                    nominal_rotation=target.rotation,
                )
        sqp_endpoint_release = {
            "pregrasp": np.zeros(3),
            "postgrasp": np.zeros(3),
        }
        release_subtask_count = max(
            1,
            len(sqp_subtask_masks_array),
            1 + len(executor._pending_subtask_planners),
        )
        sqp_stage_release_extrema = [
            {
                stage: {
                    "negative": np.zeros(3),
                    "positive": np.zeros(3),
                }
                for stage in ("pregrasp", "postgrasp")
            }
            for _ in range(release_subtask_count)
        ]
        sqp_stage_final_release = [
            {
                stage: np.zeros(3)
                for stage in ("pregrasp", "postgrasp")
            }
            for _ in range(release_subtask_count)
        ]
        sqp_stage_release_sample_counts = [
            {
                stage: 0
                for stage in (
                    "pregrasp", "grasp", "postgrasp", "release",
                )
            }
            for _ in range(release_subtask_count)
        ]
        rotation_release_recorder = StageRotationReleaseRecorder(release_subtask_count)

        executed_frames = 0
        try:
            for frame_index in range(spec.maximum_frames):
                if frame_index == len(runtime_frames) and executor._pending_subtask_planners:
                    new_frames, new_actions, new_interventions = _plan_live_successor(
                        spec, executor, data, target, release_state.rotation_for_joint(joint_command),
                        len(runtime_frames), environment,
                        tolerance=rotational_tolerance, release_state=release_state)
                    runtime_frames.extend(new_frames)
                    actions.extend(new_actions)
                    runtime_interventions.extend(new_interventions)
                    for intervention in new_interventions:
                        target_errors_deg.setdefault(intervention.stage, float('inf'))
                    if resolved_tolerance is not None:
                        resolved_tolerance = resolve_task_tolerance_context(model, data, handles,
                            spec.scene, spec.task, target.position, target.rotation, planned_frames=runtime_frames)
                    dataset_nominal_rotation = target.rotation.copy()
                if frame_index >= (
                    len(runtime_frames) + _FINAL_EVALUATION_HOLD_FRAMES
                ):
                    break
                frame = (
                    runtime_frames[frame_index]
                    if frame_index < len(runtime_frames)
                    else runtime_frames[-1]
                )
                if frame_index < len(actions):
                    gripper_closed = actions[frame_index].gripper_closed
                phase_observation = simulation_backend.observe_manipulation_phase(
                    gripper_closed, preset_subtask_index=executor.subtask_index,
                )
                active_phase = phase_observation.phase
                dataset_output_index = dataset_output_indices.get(frame_index)
                dataset_observation = None
                if dataset_output_index is not None:
                    observed_position, observed_rotation = ee_pose(data, handles)
                    dataset_observation = {
                        "position": observed_position.copy(),
                        "rotation": observed_rotation.copy(),
                        "qpos": data.qpos[handles.qpos_ids].copy(),
                        "qvel": data.qvel[handles.dof_ids].copy(),
                        "fingers": phase_classifier.finger_positions(data).copy(),
                        "phase": active_phase.key,
                        "subtask_index": int(phase_classifier.subtask_index),
                    }
                # Dataset observations are captured before applying this
                # frame's action, matching the causal convention used by the
                # normal dataset collector.
                if dataset_recorder is not None:
                    dataset_recorder.capture(data)
                if (
                    rotational_tolerance is not None
                    and (active_phase is not previous_phase
                         or phase_observation.subtask_index != previous_subtask_index)
                ):
                    current_subtask_index = int(
                        phase_classifier.subtask_index
                    )
                    new_subtask = (
                        current_subtask_index != previous_subtask_index
                    )
                    resolved_subtask = resolved_tolerance.subtask(
                        current_subtask_index,
                    )
                    (
                        active_subtask_masks,
                        active_subtask_negative,
                        active_subtask_positive,
                    ) = release_for_subtask(
                        phase_classifier.subtask_index,
                    )
                    if active_phase in (
                        ManipulationPhase.PREGRASP,
                        ManipulationPhase.POSTGRASP,
                    ):
                        stage_index = (
                            0
                            if active_phase is ManipulationPhase.PREGRASP
                            else 1
                        )
                        resolved_stage = (
                            resolved_subtask.pregrasp
                            if stage_index == 0
                            else resolved_subtask.postgrasp
                        )
                        rotational_tolerance.configure_stage(
                            int(active_phase),
                            resolved_stage.target_rotation,
                            resolved_stage.tolerance_frame,
                            0.5 * (
                                active_subtask_negative[stage_index]
                                + active_subtask_positive[stage_index]
                            ),
                            negative_limits=(
                                active_subtask_negative[stage_index]
                            ),
                            positive_limits=(
                                active_subtask_positive[stage_index]
                            ),
                            stage_key=(
                                active_phase.key,
                                int(phase_classifier.subtask_index),
                            ),
                        )
                        rotational_tolerance.ranged = (
                            active_subtask_masks[stage_index].copy()
                        )
                    else:
                        # UI semantics: Grasp and Release retain the adjacent
                        # stable coordinate frame but force all rotations to
                        # Specific while the fingers are moving.
                        rotational_tolerance.active_stage = int(active_phase)
                        rotational_tolerance.ranged[:] = False
                    if sqp_ik is not None:
                        handoff_state = sqp_ik.kinematics.evaluate(
                            joint_target,
                            include_manipulability=False,
                            include_link_points=False,
                        )
                        stage_key = (
                            active_phase.key,
                            current_subtask_index,
                        )
                        if new_subtask:
                            rebase_rotation_subtask(
                                target,
                                rotational_tolerance,
                                release_state,
                                stage_key=stage_key,
                                accepted_rotation=handoff_state.rotation,
                            )
                        else:
                            release_state.begin_stage(
                                stage_key,
                                optimized_rotation=handoff_state.rotation,
                                nominal_rotation=target.rotation,
                            )
                        instantaneous_rotation_target = (
                            handoff_state.rotation.copy()
                        )
                    previous_phase = active_phase
                    previous_subtask_index = current_subtask_index

                action = None
                if frame_index < len(actions):
                    action = actions[frame_index]
                    (
                        dataset_nominal_position,
                        dataset_nominal_rotation,
                    ) = advance_reference_pose(
                        dataset_nominal_position,
                        dataset_nominal_rotation,
                        action.value,
                    )
                    target.position, target.rotation = advance_reference_pose(
                        target.position,
                        target.rotation,
                        action.value,
                        tolerance=rotational_tolerance,
                        dt=(
                            TELEOP_DT
                            if rotational_tolerance is not None
                            else None
                        ),
                    )
                    gripper_closed = action.gripper_closed
                sqp_stage = (
                    "pregrasp"
                    if active_phase is ManipulationPhase.PREGRASP
                    else "postgrasp"
                    if active_phase is ManipulationPhase.POSTGRASP
                    else None
                )
                sqp_stage_index = (
                    0 if sqp_stage == "pregrasp"
                    else 1 if sqp_stage == "postgrasp"
                    else None
                )
                use_rotation_runtime = rotational_tolerance is not None
                if use_rotation_runtime:
                    assert rotational_tolerance is not None
                    assert sqp_ik is not None
                    controller = sqp_ik
                    assert release_state.stage_key is not None
                    (
                        joint_target,
                        diagnostics,
                        _,
                        release_mask,
                        release_step,
                    ) = solve_stage_relative_target(
                        controller,
                        data,
                        target,
                        rotational_tolerance,
                        release_state,
                        active_dofs,
                    )
                    optimized_state = controller.kinematics.evaluate(
                        joint_target,
                        include_manipulability=True,
                        include_link_points=False,
                    )
                    instantaneous_rotation_target = (
                        release_state.accepted_rotation.copy()
                    )
                    sqp_solver_statuses.append(str(diagnostics.status))
                    sqp_solver_elapsed_ms.append(float(diagnostics.elapsed_ms))
                    if spec.sqp_diagnose_conditioning_gradient:
                        sqp_solver_trace.append({
                            "frame": int(frame_index),
                            "phase": int(active_phase),
                            "phase_name": active_phase.key,
                            "subtask_index": int(
                                phase_classifier.subtask_index
                            ),
                            "status": str(diagnostics.status),
                            "publishable": bool(diagnostics.feasible),
                            "consecutive_failures": int(
                                diagnostics.consecutive_failures
                            ),
                            "converged": bool(diagnostics.converged),
                            "iterations": int(diagnostics.iterations),
                            "optimality": float(diagnostics.optimality),
                            "elapsed_ms": float(diagnostics.elapsed_ms),
                            "position_residual": float(
                                diagnostics.position_residual
                            ),
                            "rotation_residual": float(
                                diagnostics.strict_rotation_residual
                            ),
                            "effective_constraint_violation": float(
                                diagnostics.effective_constraint_violation
                            ),
                            "inverse_condition_index": float(
                                diagnostics
                                .conditioning_diagnostic_inverse_condition_index
                                if diagnostics
                                .conditioning_diagnostic_inverse_condition_index
                                is not None
                                else diagnostics.inverse_condition_index
                            ),
                            "sigma_min_gap_normalized": float(
                                diagnostics
                                .conditioning_diagnostic_sigma_min_gap_normalized
                                if diagnostics
                                .conditioning_diagnostic_sigma_min_gap_normalized
                                is not None
                                else diagnostics.sigma_min_gap_normalized
                            ),
                            "conditioning_gradient_norm": (
                                diagnostics.conditioning_gradient_norm
                            ),
                            "gradient_relative_error_half_step": (
                                diagnostics
                                .conditioning_gradient_relative_error_half_step
                            ),
                            "gradient_relative_error_double_step": (
                                diagnostics
                                .conditioning_gradient_relative_error_double_step
                            ),
                            "gradient_cosine_half_step": (
                                diagnostics
                                .conditioning_gradient_cosine_half_step
                            ),
                            "gradient_cosine_double_step": (
                                diagnostics
                                .conditioning_gradient_cosine_double_step
                            ),
                        })
                    if not diagnostics.feasible:
                        sqp_solver_rejected += 1
                    else:
                        sqp_conditioning_samples.append({
                            "sigma_min_raw": float(diagnostics.sigma_min_raw),
                            "sigma_min_normalized": float(
                                diagnostics.sigma_min_normalized
                            ),
                            "sigma_max_normalized": float(
                                diagnostics.sigma_max_normalized
                            ),
                            "sigma_min_gap_normalized": float(
                                diagnostics.sigma_min_gap_normalized
                            ),
                            "inverse_condition_index": float(
                                diagnostics.inverse_condition_index
                            ),
                            "conditioning_penalty": float(
                                diagnostics.conditioning_penalty
                            ),
                            "conditioning_activation": float(
                                diagnostics.conditioning_activation
                            ),
                            "conditioning_region": (
                                diagnostics.conditioning_region
                            ),
                            "conditioning_active": bool(
                                diagnostics.conditioning_active
                            ),
                        })
                    if (
                        diagnostics.consecutive_failures
                        >= MAX_CONSECUTIVE_SOLVER_FAILURES
                    ):
                        hard_failure_reasons.append("optimizer_failure")
                        hard_failure_details.append({
                            "frame": int(frame_index),
                            "phase": active_phase.key,
                            "subtask_index": int(
                                phase_classifier.subtask_index
                            ),
                            "status": str(diagnostics.status),
                            "consecutive_failures": int(
                                diagnostics.consecutive_failures
                            ),
                        })
                        break
                else:
                    active_dofs[3:] = True
                    joint_target, diagnostics = sqp_ik.solve(
                        data, target, active_dofs,
                    )
                if (
                    not np.all(np.isfinite(joint_target))
                    or not np.isfinite(diagnostics.cost)
                ):
                    hard_failure_reasons.append("ik_non_finite")
                    break
                if not use_rotation_runtime:
                    optimized_state = sqp_ik.kinematics.evaluate(
                        joint_target,
                        include_manipulability=True,
                        include_link_points=False,
                    )
                optimized_rotation_error = float(np.linalg.norm(
                    rotation_vector(
                        target.rotation.T @ optimized_state.rotation,
                    ),
                ))
                optimized_trajectory_recorder.sample(
                    target,
                    optimized_state.position,
                    optimized_state.rotation,
                    joint_target,
                    optimized_state.manipulability,
                    rotation_error_rad=optimized_rotation_error,
                    sigma_min_full=float(np.linalg.svd(optimized_state.jacobian, compute_uv=False)[-1]),
                    sigma_min_velocity_normalized=velocity_normalized_sigma_min(optimized_state.jacobian, handles.spec.joint_velocity_limits),
                )

                if dataset_output_index is not None:
                    assert dataset_observation is not None
                    nominal_frame = (
                        None
                        if dataset_nominal_frames is None
                        else dataset_nominal_frames[dataset_output_index]
                    )
                    if nominal_frame is None:
                        nominal_rotvec = continuous_rotation_vector(
                            dataset_nominal_rotation,
                            dataset_previous_nominal_rotvec,
                        )
                        actual_rotvec = continuous_rotation_vector(
                            dataset_observation["rotation"],
                            dataset_previous_actual_rotvec,
                            reference=nominal_rotvec,
                            reference_weight=0.7,
                        )
                        nominal_target = {
                            "position_m": dataset_nominal_position.tolist(),
                            "rotation_vector_base_rad": nominal_rotvec.tolist(),
                        }
                        dataset_previous_nominal_rotvec = nominal_rotvec.copy()
                        dataset_previous_actual_rotvec = actual_rotvec.copy()
                    else:
                        nominal_target = nominal_frame.get(
                            "nominal_target_pose", {},
                        )
                        nominal_rotvec = np.asarray(
                            nominal_target.get(
                                "rotation_vector_base_rad",
                                canonical_rotation_vector(target.rotation),
                            ),
                            dtype=float,
                        )
                        actual_rotvec = continuous_rotation_vector(
                            dataset_observation["rotation"],
                            np.asarray(
                                nominal_frame["ee_state"][
                                    "rotation_vector_base_rad"
                                ],
                                dtype=float,
                            ),
                            reference=nominal_rotvec,
                            reference_weight=0.7,
                        )
                    controller_rotvec = continuous_rotation_vector(
                        target.rotation,
                        nominal_rotvec,
                        reference=nominal_rotvec,
                        reference_weight=1.0,
                    )
                    output_frame: dict[str, Any] = {
                        "index": dataset_output_index,
                        "time_s": dataset_output_index * TELEOP_DT,
                        "video_frame_index": dataset_output_index,
                        "joint_state": {
                            "position_rad": dataset_joint_state(
                                dataset_observation["qpos"], robot,
                            ).tolist(),
                            "velocity_rad_s": dataset_joint_state(
                                dataset_observation["qvel"], robot,
                            ).tolist(),
                        },
                        "ee_state": {
                            "position_m": dataset_observation[
                                "position"
                            ].tolist(),
                            "rotation_vector_base_rad": actual_rotvec.tolist(),
                        },
                        "gripper_state": {
                            "finger_positions_m": dataset_observation[
                                "fingers"
                            ].tolist(),
                            "aperture_m": float(np.sum(
                                dataset_observation["fingers"],
                            )),
                        },
                        # Record the preset Cartesian command before SQP. The
                        # optimizer may realize legal tolerance release in
                        # joint space, but that release never enters action.
                        "action": (
                            action.value.tolist()
                            if action is not None
                            else [
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                1.0 if gripper_closed else -1.0,
                            ]
                        ),
                        "phase": dataset_observation["phase"],
                        "subtask_index": dataset_observation["subtask_index"],
                        "nominal_target_pose": nominal_target,
                        "controller_target_pose": {
                            "position_m": target.position.tolist(),
                            "rotation_vector_base_rad": controller_rotvec.tolist(),
                        },
                        "optimizer": {
                            "iterations": int(diagnostics.iterations),
                            "cost": float(diagnostics.cost),
                        },
                        "observation_variant": (
                            "sqp_tolerance_release"
                            if spec.controller_mode == "sqp_release"
                            else "nominal_sqp_specific"
                        ),
                    }
                    if (
                        nominal_frame is not None
                        and "stage_annotation_id" in nominal_frame
                    ):
                        output_frame["stage_annotation_id"] = nominal_frame[
                            "stage_annotation_id"
                        ]
                    dataset_trajectory.append(output_frame)

                def observe_tracking(tracking_diagnostics: Any) -> None:
                    nonlocal maximum_joint_tracking_error_rad
                    nonlocal maximum_joint_tracking_correction_rad
                    nonlocal joint_tracking_saturation_steps
                    maximum_joint_tracking_error_rad = max(
                        maximum_joint_tracking_error_rad,
                        tracking_diagnostics.maximum_error_rad,
                    )
                    maximum_joint_tracking_correction_rad = max(
                        maximum_joint_tracking_correction_rad,
                        tracking_diagnostics.maximum_correction_rad,
                    )
                    joint_tracking_saturation_steps += int(
                        tracking_diagnostics.saturated
                    )

                interval = simulation_backend.run_control_interval(
                    tracker=joint_tracker,
                    joint_command=joint_command,
                    joint_target=joint_target,
                    gripper_ctrl=gripper_ctrl,
                    gripper_closed=gripper_closed,
                    gripper_speed=gripper_speed,
                    speed_scale=_OFFLINE_EXECUTION_SPEED_SCALE,
                    physics_steps=physics_steps,
                    evaluator=evaluator,
                    frame_index=frame_index,
                    tracking_observer=observe_tracking,
                )
                joint_command = interval.joint_command
                gripper_ctrl = interval.gripper_ctrl
                if interval.aborted:
                    collision = getattr(evaluator, "collision", None)
                    hard_failure_reasons.append(str(getattr(
                        collision, "category", "robot_safety_collision",
                    )))
                    if collision is not None:
                        hard_failure_details.append({
                            "frame": int(frame_index),
                            **collision.as_dict(),
                        })
                    break
                target_body = getattr(evaluator, "target_body", None)
                if target_body is not None:
                    bottle_axis = data.xmat[int(target_body)].reshape(3, 3)[:, 2]
                    phase_upright_angles.setdefault(active_phase.key, []).append(
                        float(np.arccos(np.clip(bottle_axis[2], -1.0, 1.0)))
                    )
                if recorder is not None:
                    recorder.capture(data)
                executed_frames = frame_index + 1
                actual_position, actual_rotation = ee_pose(data, handles)
                q = data.qpos[handles.qpos_ids].copy()
                actual_kinematic_state = sqp_ik.kinematics.evaluate(
                    q,
                    include_manipulability=True,
                    include_link_points=False,
                )
                tolerance_aware_rotation_error = None
                if use_rotation_runtime:
                    phase_sample_subtask_index = min(
                        max(int(phase_classifier.subtask_index), 0),
                        release_subtask_count - 1,
                    )
                    sqp_stage_release_sample_counts[
                        phase_sample_subtask_index
                    ][active_phase.key] += 1
                    # Physical FK lag and contact disturbance are not
                    # tolerance release; telemetry follows the last accepted
                    # optimizer rotation in this cycle's frozen chart.
                    tolerance_aware_rotation_error = float(np.linalg.norm(
                        rotation_vector(target.rotation.T @ actual_rotation),
                    ))
                    if sqp_stage is not None:
                        assert rotational_tolerance is not None
                        assert release_step is not None
                        accepted_candidate = release_state.candidate(
                            release_state.accepted_rotation,
                            release_step,
                        )
                        current_release = (
                            accepted_candidate.stage_relative_tolerance_release_rad
                        )
                        release_subtask_index = min(
                            max(int(phase_classifier.subtask_index), 0),
                            release_subtask_count - 1,
                        )
                        extrema = sqp_stage_release_extrema[
                            release_subtask_index
                        ][sqp_stage]
                        extrema["negative"] = np.maximum(
                            extrema["negative"],
                            np.maximum(-current_release, 0.0),
                        )
                        extrema["positive"] = np.maximum(
                            extrema["positive"],
                            np.maximum(current_release, 0.0),
                        )
                        rotation_release_recorder.sample(
                            release_subtask_index,sqp_stage,
                            release_state.accepted_rotation @ release_step.stage_reference_rotation().T,
                        )
                        sqp_stage_final_release[
                            release_subtask_index
                        ][sqp_stage] = current_release.copy()
                        endpoint_index = (
                            plan.pregrasp_endpoint_index
                            if sqp_stage == "pregrasp"
                            else plan.postgrasp_endpoint_index
                        )
                        if frame_index == endpoint_index:
                            sqp_endpoint_release[sqp_stage] = (
                                current_release.copy()
                            )
                trajectory_recorder.sample(
                    target,
                    actual_position,
                    actual_rotation,
                    q,
                    actual_kinematic_state.manipulability,
                    rotation_error_rad=tolerance_aware_rotation_error,
                    sigma_min_full=float(np.linalg.svd(actual_kinematic_state.jacobian, compute_uv=False)[-1]),
                    sigma_min_velocity_normalized=velocity_normalized_sigma_min(actual_kinematic_state.jacobian, handles.spec.joint_velocity_limits),
                )
                position_error = float(np.linalg.norm(
                    target.position - actual_position,
                ))
                rotation_error = float(np.linalg.norm(
                    rotation_vector(target.rotation.T @ actual_rotation),
                ))
                for intervention in runtime_interventions:
                    if frame_index == intervention.verification_index:
                        verification_rotation = (
                            intervention.counterfactual_target_rotation
                        )
                        error = rotation_vector(
                            verification_rotation.T
                            @ actual_rotation,
                        )
                        target_errors_deg[intervention.stage] = min(
                            target_errors_deg[intervention.stage],
                            float(np.degrees(np.linalg.norm(error))),
                        )

                if evaluator.should_abort:
                    break
        finally:
            if recorder is not None:
                recorder.close()
            if dataset_recorder is not None:
                dataset_recorder.close()

        executor_finished = executed_frames >= len(runtime_frames) and not executor._pending_subtask_planners
        timed_out = (
            not executor_finished
            and not evaluator.should_abort
        )
        evaluation = evaluator.result(
            executor_finished=executor_finished,
            timed_out=timed_out,
        )
        finite_target_errors = {
            stage: (value if np.isfinite(value) else 1.0e9)
            for stage, value in target_errors_deg.items()
        }
        target_error_deg = max(finite_target_errors.values(), default=1.0e9)
        target_reached = all(
            value <= spec.target_rotation_tolerance_deg
            for value in finite_target_errors.values()
        )
        evaluation["stage_target_rotation_errors_deg"] = finite_target_errors
        evaluation["orientation_reference"] = {
            "mode": (
                "stage_relative_release"
                if rotational_tolerance is not None
                else "not_applicable"
            ),
        }
        evaluation["counterfactual_intervention"] = {
            "mode": "stage_local_action_injection",
            "nominal_actions_preserved_outside_stage": True,
        }
        evaluation["executed_frames"] = int(executed_frames)
        evaluation["expected_frames"] = min(
            spec.maximum_frames,
            len(runtime_frames) + _FINAL_EVALUATION_HOLD_FRAMES,
        )
        evaluation["backend"] = "shared_preset_simulation"
        evaluation["sigma_min_normalization"] = sigma_min_normalization_metadata(handles.spec.joint_velocity_limits)
        evaluation["execution_speed_scale"] = (
            _OFFLINE_EXECUTION_SPEED_SCALE
        )
        if spec.controller_mode in (
            "kinematic", "sqp_specific", "sqp_release", "hybrid_stage_release",
        ):
            evaluation["sqp_loss_profile"] = spec.sqp_loss_profile
            evaluation["sqp_loss_weights"] = list(spec.sqp_loss_weights)
            status_counts = {
                status: sqp_solver_statuses.count(status)
                for status in sorted(set(sqp_solver_statuses))
            }
            evaluation["sqp_solver"] = {
                "cycles": len(sqp_solver_statuses),
                "accepted_cycles": (
                    len(sqp_solver_statuses) - sqp_solver_rejected
                ),
                "rejected_cycles": sqp_solver_rejected,
                "status_counts": status_counts,
                "maximum_elapsed_ms": (
                    max(sqp_solver_elapsed_ms)
                    if sqp_solver_elapsed_ms else 0.0
                ),
                "mean_elapsed_ms": (
                    float(np.mean(sqp_solver_elapsed_ms))
                    if sqp_solver_elapsed_ms else 0.0
                ),
                "time_budget_ms": (
                    None
                    if spec.sqp_max_time_s <= 0.0
                    else 1000.0 * spec.sqp_max_time_s
                ),
            }
            if sqp_solver_trace:
                evaluation["sqp_solver_trace"] = sqp_solver_trace
            if sqp_conditioning_samples:
                eta = np.asarray([
                    sample["inverse_condition_index"]
                    for sample in sqp_conditioning_samples
                ], dtype=float)
                raw = np.asarray([
                    sample["sigma_min_raw"]
                    for sample in sqp_conditioning_samples
                ], dtype=float)
                normalized = np.asarray([
                    sample["sigma_min_normalized"]
                    for sample in sqp_conditioning_samples
                ], dtype=float)
                penalties = np.asarray([
                    sample["conditioning_penalty"]
                    for sample in sqp_conditioning_samples
                ], dtype=float)
                active = np.asarray([
                    sample["conditioning_active"]
                    for sample in sqp_conditioning_samples
                ], dtype=bool)
                activation = np.asarray([
                    sample["conditioning_activation"]
                    for sample in sqp_conditioning_samples
                ], dtype=float)
                regions = np.asarray([
                    sample["conditioning_region"]
                    for sample in sqp_conditioning_samples
                ], dtype=object)
                evaluation["sqp_conditioning"] = {
                    "samples": int(eta.size),
                    "minimum_inverse_condition_index": float(np.min(eta)),
                    "mean_inverse_condition_index": float(np.mean(eta)),
                    "fraction_below_eta_high": float(np.mean(
                        eta < spec.sqp_conditioning_eta_high
                    )),
                    "minimum_sigma_min_raw": float(np.min(raw)),
                    "minimum_sigma_min_normalized": float(
                        np.min(normalized)
                    ),
                    "maximum_conditioning_penalty": float(
                        np.max(penalties)
                    ),
                    "active_fraction": float(np.mean(active)),
                    "mean_activation": float(np.mean(activation)),
                    "healthy_fraction": float(np.mean(
                        regions == "healthy"
                    )),
                    "transition_fraction": float(np.mean(
                        regions == "transition"
                    )),
                    "critical_fraction": float(np.mean(
                        regions == "critical"
                    )),
                    "eta_low": float(spec.sqp_conditioning_eta_low),
                    "eta_high": float(spec.sqp_conditioning_eta_high),
                    "eta_target": float(spec.sqp_conditioning_eta_high),
                }
            release_signs: dict[str, list[int]] = {}
            fallback_axes: dict[str, list[bool]] = {}
            for stage in ("pregrasp", "postgrasp"):
                values = sqp_endpoint_release[stage]
                fallback = np.abs(values) <= 1.0e-8
                signs = np.where(fallback, 1, np.sign(values)).astype(int)
                mask_index = 0 if stage == "pregrasp" else 1
                signs[~sqp_masks_array[mask_index]] = 0
                fallback[~sqp_masks_array[mask_index]] = False
                release_signs[stage] = signs.tolist()
                fallback_axes[stage] = fallback.tolist()
            evaluation["controller_mode"] = spec.controller_mode
            evaluation["optimizer_backend"] = (
                "ipopt"
            )
            evaluation["sqp_unbounded_rotation"] = bool(
                spec.sqp_unbounded_rotation,
            )
            evaluation["sqp_stage_rotation_masks"] = (
                sqp_masks_array.astype(int).tolist()
            )
            evaluation["sqp_symmetric_limits_deg"] = list(
                spec.sqp_symmetric_limits_deg,
            )
            evaluation["sqp_stage_negative_limits_deg"] = (
                [
                    [
                        None if not np.isfinite(value) else float(value)
                        for value in row
                    ]
                    for row in np.degrees(sqp_negative_rad)
                ]
            )
            evaluation["sqp_stage_positive_limits_deg"] = (
                [
                    [
                        None if not np.isfinite(value) else float(value)
                        for value in row
                    ]
                    for row in np.degrees(sqp_positive_rad)
                ]
            )
            evaluation["sqp_stage_endpoint_release_deg"] = {
                stage: np.degrees(values).tolist()
                for stage, values in sqp_endpoint_release.items()
            }
            evaluation["sqp_subtask_stage_release_extrema_deg"] = [
                {
                    "subtask_index": subtask_index,
                    **{
                        stage: {
                            side: np.degrees(values).tolist()
                            for side, values in sides.items()
                        }
                        for stage, sides in stages.items()
                    },
                }
                for subtask_index, stages in enumerate(
                    sqp_stage_release_extrema
                )
            ]
            evaluation["sqp_stage_release_extrema_deg"] = {
                stage: {
                    side: np.degrees(values).tolist()
                    for side, values in sides.items()
                }
                for stage, sides in sqp_stage_release_extrema[0].items()
            }
            evaluation["declared_subtask_stage_rotation_masks"] = (
                _declared_subtask_stage_masks(
                    model,
                    spec,
                    resolved_tolerance,
                    release_subtask_count,
                )
            )
            evaluation["sqp_subtask_stage_final_release_deg"] = [
                {
                    "subtask_index": subtask_index,
                    **{
                        stage: np.degrees(values).tolist()
                        for stage, values in stages.items()
                    },
                }
                for subtask_index, stages in enumerate(
                    sqp_stage_final_release
                )
            ]
            evaluation["sqp_subtask_stage_release_sample_counts"] = [
                {
                    "subtask_index": subtask_index,
                    **counts,
                }
                for subtask_index, counts in enumerate(
                    sqp_stage_release_sample_counts
                )
            ]
            evaluation["sqp_release_measurement"] = (
                "running positive and negative extrema of the optimizer-owned "
                "stage-relative SO(3) release on every released axis; "
                "physical FK lag and contact disturbance are excluded"
            )
            evaluation["sqp_release_signs"] = release_signs
            evaluation["sqp_zero_release_fallback_axes"] = fallback_axes
            evaluation["sqp_rotation_release"] = rotation_release_recorder.summary(
                trajectory_complete=bool(
                    executor_finished and sqp_solver_rejected == 0
                ),
            )
        trajectory = trajectory_recorder.result()
        optimized_trajectory = optimized_trajectory_recorder.result()
        evaluation["trajectory_kinematics"] = trajectory.as_dict()
        evaluation["optimized_10hz_trajectory_kinematics"] = (
            optimized_trajectory.as_dict()
        )
        evaluation["joint_position_extrema_deg"] = (
            trajectory_recorder.joint_position_extrema_deg()
        )
        evaluation["optimized_joint_position_extrema_deg"] = (
            optimized_trajectory_recorder.joint_position_extrema_deg()
        )
        evaluation["trajectory_metric_sources"] = {
            "trajectory_kinematics": (
                "actual MuJoCo qpos sampled at each 10 Hz control endpoint"
            ),
            "optimized_10hz_trajectory_kinematics": (
                "IPOPT joint_target and its FK at 10 Hz"
            ),
        }
        evaluation["joint_reference_tracking"] = {
            "controller": "decoupled_bounded_leaky_pid_outer_loop",
            "reference_source": (
                "unmodified IPOPT joint reference"
            ),
            "maximum_error_rad": maximum_joint_tracking_error_rad,
            "maximum_correction_rad": (
                maximum_joint_tracking_correction_rad
            ),
            "saturation_steps": joint_tracking_saturation_steps,
        }
        evaluation["trajectory_planning_policy"] = (
            "each subtask is planned once at its live boundary through the shared UI executor; "
            "no replanning within a subtask"
        )
        evaluation["rotation_error_metric"] = (
            "actual EE tracking residual to the UI-projected legal rotation "
            "target; legal fixed-stage tolerance release is excluded"
        )
        evaluation["collision_policy"] = {
            "prohibited": [
                "robot_table_collision",
                "robot_self_collision",
            ],
            "condition": "actual_geometry_contact_distance_m <= 0",
        }
        evaluation["hard_failure_reasons"] = list(hard_failure_reasons)
        evaluation["hard_failure_details"] = hard_failure_details
        evaluation["phase_upright_angles_deg"] = {
            phase: [float(np.degrees(value)) for value in values]
            for phase, values in phase_upright_angles.items()
        }
        evaluation["hard_failure_thresholds"] = {
            "robot_collision_maximum_contact_distance_m": 0.0,
        }
        task_success = bool(evaluation["success"]) and not hard_failure_reasons
        if spec.controller_mode in (
            "sqp_specific", "sqp_release",
        ):
            # These are zero-intervention controller-evaluation trajectories.
            # Their task outcome is authoritative; the counterfactual
            # verification probe is only meaningful during axis search.
            target_reached = True
        success = task_success and target_reached
        if hard_failure_reasons:
            reason = f"hard_failure:{hard_failure_reasons[0]}"
        elif not target_reached:
            reason = "counterfactual_target_not_reached"
        else:
            reason = str(evaluation["termination_reason"])
        return CounterfactualTrialResult(
            spec=spec,
            success=success,
            task_success=task_success,
            target_reached=target_reached,
            termination_reason=reason,
            target_rotation_error_deg=target_error_deg,
            nominal_target_rotation_vector_base_rad=tuple(
                float(x) for x in rotation_vector(
                    result_plan.nominal_target_rotation,
                )
            ),
            counterfactual_target_rotation_vector_base_rad=tuple(
                float(x) for x in rotation_vector(
                    result_plan.counterfactual_target_rotation,
                )
            ),
            tolerance_frame_rotation_vector_base_rad=tuple(
                float(x) for x in canonical_rotation_vector(
                    result_plan.tolerance_frame_rotation,
                )
            ),
            evaluation=evaluation,
            domain_randomization=environment,
            hard_failure_reasons=tuple(hard_failure_reasons),
            trajectory_kinematics=trajectory.as_dict(),
            dataset_trajectory=tuple(dataset_trajectory),
            dataset_source_frame_count=(
                dataset_recorder.source_frame_index
                if dataset_recorder is not None
                else 0
            ),
        )
    except PlanningInfeasibleError as error:
        kinematic = spec.execution_mode == "kinematic" or spec.controller_mode == "kinematic"
        declared = error.context.get("declared_subtask_stage_rotation_masks", [])
        metrics = dict(samples=0, mean_joint_velocity_rad_s=None,
                       mean_joint_acceleration_rad_s2=None,
                       mean_joint_jerk_rad_s3=None, mean_manipulability=None,
                       total_joint_travel_rad=None)
        evaluation = dict(
            backend="kinematic_collision_checked" if kinematic else "simulation_initial_planning",
            physics_steps=0 if kinematic else None, control_rate_hz=10.0,
            task_interaction=False if kinematic else None, objects_considered=False if kinematic else None,
            planning_feasible=False, optimizer_feasible=False, optimizer_statuses=[],
            trajectory_complete=False, executed_frames=0, expected_frames=0, planned_frames=0,
            error=str(error), failure_stage="initial_planning",
            hard_failure_reasons=["planning_infeasible"],
            declared_subtask_stage_rotation_masks=declared,
            sqp_subtask_stage_release_extrema_deg=[dict(subtask_index=i,
                pregrasp=dict(negative=[0.0]*3, positive=[0.0]*3),
                postgrasp=dict(negative=[0.0]*3, positive=[0.0]*3)) for i in range(len(declared))],
            sqp_subtask_stage_release_sample_counts=[dict(subtask_index=i, pregrasp=0, postgrasp=0)
                                                   for i in range(len(declared))],
            kinematic_feasible_without_table=False,
            robot_table_clearance=dict(distribution=dict(samples=0, minimum_m=None)),
            clearance_domain="solved_joint_positions_10hz_fk; no physics/interpolation" if kinematic else None,
            trajectory_kinematics=metrics,
        )
        return CounterfactualTrialResult(
            spec=spec, success=False, task_success=False, target_reached=False,
            termination_reason="planning_infeasible", target_rotation_error_deg=1.0e9,
            nominal_target_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            counterfactual_target_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            tolerance_frame_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            evaluation=evaluation, trajectory_kinematics=metrics,
            domain_randomization=error.context.get("domain_randomization", {}),
            hard_failure_reasons=("planning_infeasible",),
        )
    except Exception as error:
        return CounterfactualTrialResult(
            spec=spec,
            success=False,
            task_success=False,
            target_reached=False,
            termination_reason=f"runner_error:{type(error).__name__}",
            target_rotation_error_deg=1.0e9,
            nominal_target_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            counterfactual_target_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            tolerance_frame_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            evaluation={
                "error": str(error),
                "hard_failure_reasons": ["ik_or_runner_exception"],
            },
            domain_randomization={},
            hard_failure_reasons=("ik_or_runner_exception",),
            trajectory_kinematics={},
        )
