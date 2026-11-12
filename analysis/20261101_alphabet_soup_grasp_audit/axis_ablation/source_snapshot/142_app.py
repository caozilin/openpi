from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Hashable

import mujoco
import mujoco.viewer
import numpy as np

from ..config import (
    DEFAULT_FINGER_SLIDE_FRICTION,
    DEFAULT_GRIPPER_FORCE_SCALE,
    DEFAULT_MAX_GRIPPER_FORCE,
    DEFAULT_MAX_PHYSICS_TIMESTEP,
    DOF_LABELS,
    GRIPPER_TRAVEL_TIME,
    TELEOP_DT,
    VIEWER_REFRESH_HZ,
)
from ..control.ipopt import (
    IpoptIK,
    IpoptSettings,
)
from ..control.pose import TargetPose
from ..control.joint_tracker import JointReferenceTracker
from ..control.task_space.objective import DEFAULT_EMA_RELEASE_LOSS_WEIGHT, ObjectiveSettings
from ..control.task_space.conditioning import velocity_normalized_sigma_min
from ..tasks import create_task_evaluator
from ..tasks.catalog import WIPE_WHITEBOARD_TASK
from ..tasks.evaluation.types import TaskEvaluator
from ..tasks.phases import ManipulationPhase
from ..simulation.randomization.camera import FrontCameraRandomizer, focus_on_world_x, tabletop_corners
from ..simulation.cameras import set_wrist_camera_calibration
from ..simulation.randomization.light import LightRandomizer
from ..simulation.randomization.objects import ObjectLayoutLimits, ObjectLayoutRandomizer
from ..reachability.layout import (
    WORKSPACE_BOUNDARY_MODES,
    apply_blocks_difficulty,
    apply_training_workspace_boundary,
    default_boundary_surface_path,
)
from ..reachability.boundary_surface import WorkspaceBoundarySurface
from ..reachability.visualization import (
    clear_reachability_scene,
    populate_workspace_boundary_surface_scene,
)
from ..geometry import (
    action_in_base_frame,
    action_from_pressed,
    apply_action,
    continuous_rotation_vector,
    rotation_matrix,
    rotation_vector,
)
from ..simulation.robot import RobotHandles, configure_contact_solver, configure_gripper, ee_pose, find_handles, scaled_gripper_control_step, set_gripper_fully_open, set_home
from ..tasks.planning.presets import PresetTaskExecutor
from ..tasks.planning.nominal_candidates import supports_nominal_selection, nominal_candidate_definitions
from .ui import InputSnapshot, KeyboardPanel
from .playback import SimulationPlaybackClock
from .nominal_preparation import BackgroundNominalPreparation, PreparationPending
from ..benchmark.metrics import MultiRateEvaluationRecorder
from .recording import TrajectoryRecorder
from .recording import IDLE_JOINT_MOVEMENT_THRESHOLD_RAD
from .contact_markers import GripperTableContactMarkers
from ..simulation.preset_runtime import PresetSimulationBackend
from ..tolerance.state import RotationalToleranceState
from ..tolerance.release_state import RotationReleaseState
from ..tolerance.runtime import (
    advance_reference_pose,
    rebase_rotation_subtask,
    solve_stage_relative_target,
    preset_stage_lookahead,
)
from ..tolerance.frame import (
    TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
    TOLERANCE_FRAME_MODE_STAGE_ENDPOINT,
)
from ..tolerance.context import (
    ResolvedSubtaskTolerance,
    effective_rotation_specific_axes,
    long_task_subtask_labels,
    resolve_planned_subtask_tolerances,
    resolve_task_tolerance_context,
)
from ..simulation.scenes import GEOMETRY_PLATE_BOX_UPRIGHT_SCENE, GEOMETRY_REGION_CYLINDER_LYING_SCENE, SceneManager, WHITEBOARD_SCENE
from ..simulation.layouts import (
    CustomLayout,
    apply_custom_layout,
    capture_custom_layout,
    load_custom_layout,
    load_named_custom_layout,
    quaternion_with_yaw,
    save_named_custom_layout,
    yaw_from_quaternion,
)
from ..simulation.loader import load_scene_model
from ..simulation.robots import ROBOT_CHOICES, get_robot_spec
from ..data.task_sets import (
    TRAINING_TASK_GROUP,
    runtime_scene_from_choice,
    scene_choices_for_group,
    scene_groups,
)
from ..simulation.randomization.table import TableRandomizer
from .windows import schedule_mujoco_viewer_placement
from .layout_visualization import append_placement_region


def _restore_robot_home_for_scene_reset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
) -> None:
    """Restore the planning start before resolving reset-episode tolerances."""
    set_home(model, data, handles)
    set_gripper_fully_open(model, data, handles)
    data.qvel[handles.dof_ids] = 0.0
    data.qacc[handles.dof_ids] = 0.0
    data.qacc_warmstart[handles.dof_ids] = 0.0
    mujoco.mj_forward(model, data)


def _project_world_to_image(
    point: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    fovy_deg: float,
    aspect: float,
) -> np.ndarray | None:
    local = np.asarray(camera_rotation, dtype=float).reshape(3, 3).T @ (
        np.asarray(point, dtype=float) - np.asarray(camera_position, dtype=float)
    )
    depth = -float(local[2])
    if depth <= 1e-8:
        return None
    tan_half = math.tan(math.radians(float(fovy_deg)) * 0.5)
    return np.array((
        0.5 + float(local[0]) / (2.0 * depth * tan_half * aspect),
        0.5 - float(local[1]) / (2.0 * depth * tan_half),
    ))


def _image_ray_to_z_plane(
    image_xy: np.ndarray,
    plane_z: float,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    fovy_deg: float,
    aspect: float,
) -> np.ndarray | None:
    point = np.asarray(image_xy, dtype=float)
    tan_half = math.tan(math.radians(float(fovy_deg)) * 0.5)
    local_direction = np.array((
        (2.0 * point[0] - 1.0) * tan_half * aspect,
        (1.0 - 2.0 * point[1]) * tan_half,
        -1.0,
    ))
    direction = np.asarray(camera_rotation, dtype=float).reshape(3, 3) @ local_direction
    if abs(float(direction[2])) <= 1e-10:
        return None
    origin = np.asarray(camera_position, dtype=float)
    distance = (float(plane_z) - float(origin[2])) / float(direction[2])
    if distance <= 0.0:
        return None
    return origin + distance * direction


def _image_ray_direction(
    image_xy: np.ndarray,
    camera_rotation: np.ndarray,
    fovy_deg: float,
    aspect: float,
) -> np.ndarray:
    point = np.asarray(image_xy, dtype=float)
    tan_half = math.tan(math.radians(float(fovy_deg)) * 0.5)
    local = np.array((
        (2.0 * point[0] - 1.0) * tan_half * aspect,
        (1.0 - 2.0 * point[1]) * tan_half,
        -1.0,
    ))
    world = np.asarray(camera_rotation, dtype=float).reshape(3, 3) @ local
    return world / np.linalg.norm(world)


def _camera_local_pitch(base_quat: np.ndarray, pitch_deg: float) -> np.ndarray:
    """Offline helper for a fixed camera's local-horizontal pitch calibration."""
    half_angle = math.radians(pitch_deg) * 0.5
    local_pitch = np.array((math.cos(half_angle), math.sin(half_angle), 0.0, 0.0))
    result = np.empty(4)
    mujoco.mju_mulQuat(result, base_quat, local_pitch)
    return result


def _resolved_initial_scene_choice(
    scene_group: str,
    scene_name: str,
    requested_choice: str | None,
) -> str | None:
    """Resolve the UI's logical second-level choice before its first poll."""
    choices = scene_choices_for_group(scene_group)
    if requested_choice in choices:
        return requested_choice
    return next(
        (
            choice for choice in choices
            if runtime_scene_from_choice(scene_group, choice) == scene_name
        ),
        None,
    )


def _initial_profile_loss_weights(
    args: argparse.Namespace,
    objective_settings: ObjectiveSettings,
) -> np.ndarray:
    """Initialize the two maintained single-step loss weights.

    Conditioning is suspended project-wide and is intentionally absent from
    both the UI vector and recorded maintained-profile metadata.
    """
    del objective_settings
    return np.array((
        args.sqp_motion_weight,
        args.sqp_ema_release_loss_weight,
    ), dtype=float)


def _front_camera_center_pose(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    x_offset: float,
    z_offset: float,
    pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the optional calibration-adjusted centre of domain randomization."""
    position = base_pos + np.array((x_offset, 0.0, z_offset))
    return position, _camera_local_pitch(base_quat, pitch_deg)


def _has_enabled_command(pressed: set[str], active: np.ndarray) -> bool:
    key_groups = (("w", "s"), ("a", "d"), ("i", "k"), ("q", "e"), ("u", "o"), ("j", "l"))
    # Ranged rotational axes still accept commands and update the centres of
    # their valid intervals. Only released translations suppress keyboard input.
    return any((index >= 3 or active[index]) and any(key in pressed for key in keys) for index, keys in enumerate(key_groups))


def _reconcile_target_after_toggle(
    target: TargetPose,
    position: np.ndarray,
    toggled: tuple[int, ...],
    active_dofs: np.ndarray,
    rotational_tolerance: RotationalToleranceState,
) -> None:
    """Apply mode toggles without redefining the commanded rotation."""
    for index in toggled:
        if index < 3:
            target.position[index] = position[index]
    rotational_toggles = tuple(index - 3 for index in toggled if index >= 3)
    if rotational_toggles:
        target.rotation = rotational_tolerance.reconcile_modes(
            target.rotation, active_dofs[3:], rotational_toggles,
        )


def _apply_target_action(
    target: TargetPose,
    action: np.ndarray,
    active_dofs: np.ndarray,
    rotational_tolerance: RotationalToleranceState,
) -> None:
    """Apply one already-normalized base-frame command."""
    base_action = np.asarray(action, dtype=float)
    translation_only = base_action.copy()
    translation_only[3:] = 0.0
    target.position, _ = apply_action(
        target.position, target.rotation, translation_only, active_dofs,
    )
    rotation_only = np.zeros(6)
    rotation_only[3:] = base_action[3:]
    _, target.rotation = advance_reference_pose(
        np.zeros(3),
        target.rotation,
        rotation_only,
        tolerance=rotational_tolerance,
        dt=TELEOP_DT,
    )


def _activate_runtime_subtask_tolerance(
    rotational_tolerance: RotationalToleranceState,
    active_dofs: np.ndarray,
    resolved_subtask: ResolvedSubtaskTolerance,
    limits_rad: np.ndarray,
    *,
    tolerance_enabled: bool,
    stage_key: Hashable,
) -> None:
    """Atomically enter a newly planned subtask's Pre-grasp descriptor."""
    pregrasp = resolved_subtask.pregrasp
    active_dofs[3:] = effective_rotation_specific_axes(
        tolerance_enabled,
        int(ManipulationPhase.PREGRASP),
        pregrasp.ranged_axes,
    )
    rotational_tolerance.configure_stage(
        int(ManipulationPhase.PREGRASP),
        pregrasp.target_rotation,
        pregrasp.tolerance_frame,
        limits_rad,
        stage_key=stage_key,
        update_prediction=False,
    )
    rotational_tolerance.ranged = (
        pregrasp.ranged_axes.copy()
        if tolerance_enabled
        else np.zeros(3, dtype=bool)
    )


def _configure_ui_tolerance(
    tolerance: RotationalToleranceState,
    snapshot: InputSnapshot,
    stage_key: tuple,
) -> None:
    """Install the edited UI descriptor without substituting task defaults."""
    stage = snapshot.active_stage
    source_stage = 0 if stage == 1 else 2 if stage == 3 else stage
    tolerance.configure_stage(
        source_stage,
        snapshot.rotation_tolerance_target,
        snapshot.rotation_tolerance_frame,
        snapshot.rotation_tolerance_rad,
        stage_key=stage_key + (source_stage,),
        update_prediction=False,
    )
    tolerance.active_stage = stage
    tolerance.ranged = (
        ~snapshot.active_dofs[3:]
        if snapshot.tolerance_enabled and stage not in (1, 3)
        else np.zeros(3, dtype=bool)
    )


def run(args: argparse.Namespace) -> dict[str, str] | None:
    """Run one isolated scene and return the next scene requested by the UI."""
    robot_spec = get_robot_spec(args.robot)
    model = load_scene_model(args.scene, robot_spec.uid)
    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    front_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_view")
    if wrist_camera_id < 0:
        raise RuntimeError(f"The {robot_spec.label} model is missing wrist_camera.")
    if front_camera_id < 0:
        raise RuntimeError("The scene is missing front_view.")
    side_camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "table_side_view",
    )
    side_camera_name = "table_side_view"
    if side_camera_id < 0:
        side_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "side_view",
        )
        side_camera_name = "side_view"
    if side_camera_id < 0:
        raise RuntimeError("The scene is missing a side camera.")
    front_camera_base_pos = model.cam_pos[front_camera_id].copy()
    front_camera_base_quat = model.cam_quat[front_camera_id].copy()
    work_surface_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface")
    has_workspace = work_surface_id >= 0
    front_camera_focus = focus_on_world_x(
        front_camera_base_pos,
        front_camera_base_quat,
        float(model.body_pos[work_surface_id, 0]) if has_workspace else 0.0,
    )
    front_camera_reference_tabletop = tabletop_corners(model)
    front_camera_randomizer = FrontCameraRandomizer(
        front_camera_base_pos, front_camera_base_quat, front_camera_focus,
    )
    table_randomizer = (
        TableRandomizer(model.body_pos[work_surface_id])
        if has_workspace else None
    )
    light_randomizer = LightRandomizer(model)
    wrist_camera_base_pos = model.cam_pos[wrist_camera_id].copy()
    wrist_camera_base_quat = model.cam_quat[wrist_camera_id].copy()
    handles: RobotHandles = find_handles(model, robot_spec)
    controller_handles = handles
    configure_contact_solver(model, args.physics_timestep)
    configure_gripper(model, handles, args.gripper_force_scale, args.max_gripper_force, args.finger_friction)
    data = mujoco.MjData(model)
    set_home(model, data, handles)
    set_gripper_fully_open(model, data, handles)
    scene_manager = SceneManager(model, loaded_scene=args.scene)
    scene_manager.activate(data, args.scene)
    initial_custom_layout_path = getattr(args, "custom_layout", None)
    initial_custom_layout: CustomLayout | None = None
    if initial_custom_layout_path is not None:
        initial_custom_layout = load_custom_layout(initial_custom_layout_path)
        apply_custom_layout(
            scene_manager,
            data,
            initial_custom_layout,
            robot=robot_spec.uid,
        )
    object_layout_randomizer = (
        ObjectLayoutRandomizer(
            model, scene_manager,
            limits=ObjectLayoutLimits(minimum_clearance_m=args.object_clearance),
        )
        if has_workspace else None
    )
    preset_task = PresetTaskExecutor(model, handles)

    position, rotation = ee_pose(data, handles)
    target = TargetPose(position, rotation)
    home_position, home_rotation = position.copy(), rotation.copy()
    home_rotation_vector = rotation_vector(home_rotation)
    # Each stage owns an independent rotational target and tolerance frame.
    # Both begin at the official Home EE orientation and can be edited or
    # captured independently in the UI.
    rotational_tolerance = RotationalToleranceState(
        target=home_rotation, frame=home_rotation,
    )
    ipopt_time_budget = (
        1.0e6 if args.ipopt_max_time <= 0.0 else args.ipopt_max_time
    )
    if args.sqp_loss == "delta-q-squared-only":
        objective_settings = replace(
            ObjectiveSettings.delta_q_squared_only(),
            squared_delta_q_weight=args.sqp_motion_weight,
            ema_release_loss_weight=args.sqp_ema_release_loss_weight,
        )
    elif args.sqp_loss in (
        "delta-q-squared-sigma-min",
        "delta-q-squared-sigma-min-legacy",
    ):
        objective_settings = ObjectiveSettings.delta_q_squared_sigma_min_legacy(
            motion_weight=args.sqp_motion_weight,
            sigma_min_weight=args.sqp_sigma_min_weight,
            length_scale_m=args.sqp_sigma_min_length_scale,
            ema_release_loss_weight=args.sqp_ema_release_loss_weight,
        )
    else:
        objective_settings = (
            ObjectiveSettings.delta_q_squared_conditioning_safeguard(
                motion_weight=args.sqp_motion_weight,
                conditioning_weight=0.0,
                ema_release_loss_weight=args.sqp_ema_release_loss_weight,
            )
        )
    ipopt_ik = IpoptIK(
        model,
        controller_handles,
        IpoptSettings(
            max_iterations=args.ipopt_max_iterations,
            max_cpu_time_s=ipopt_time_budget,
        ),
        objective_settings,
        diagnose_conditioning_gradient=False,
        block_warm_start=not args.no_block_warm_start,
    )
    objective_settings = ipopt_ik.objective_settings
    joint_target = robot_spec.home_q.copy()
    release_state = RotationReleaseState(
        rotation_for_joint=lambda q: ipopt_ik.kinematics.evaluate(
            q, include_manipulability=False, include_link_points=False,
        ).rotation,
    )
    # A 10 Hz planner produces joint-space waypoints.  The physics loop
    # linearly interpolates the actuator command at every 1 kHz step.
    joint_command = robot_spec.home_q.copy()
    joint_tracker = JointReferenceTracker(robot_spec.arm_dof)
    simulation_backend = PresetSimulationBackend(model, data, handles)
    interpolation_start = joint_command.copy()
    interpolation_goal = joint_command.copy()
    interpolation_elapsed = TELEOP_DT
    initial_loss_profile = {
        "delta-q-squared-only": "delta_q_squared_only",
        "delta-q-squared-sigma-min": "delta_q_squared_sigma_min_legacy",
        "delta-q-squared-sigma-min-legacy": (
            "delta_q_squared_sigma_min_legacy"
        ),
        "delta-q-squared-ema-release": "delta_q_squared_ema_release",
    }[args.sqp_loss]
    legacy_objective_settings = ObjectiveSettings.reference_defaults()
    legacy_weights = np.array((
        legacy_objective_settings.position_weight,
        legacy_objective_settings.rotation_weight,
        legacy_objective_settings.velocity_weight,
        legacy_objective_settings.jerk_weight,
        legacy_objective_settings.manipulability_weight,
        legacy_objective_settings.self_collision_weight,
        legacy_objective_settings.release_loss_weight,
    ))
    initial_weights = (
        legacy_weights
    )
    initial_sqp_loss_weights = _initial_profile_loss_weights(
        args,
        objective_settings,
    )
    initial_scene_choice = _resolved_initial_scene_choice(
        args.scene_group,
        args.scene,
        args.scene_choice,
    )
    panel = None if args.no_gui else KeyboardPanel(
        initial_weights, camera_adjust_mode=args.camera_adjust,
        initial_scene=args.scene,
        initial_scene_group=args.scene_group,
        initial_scene_choice=initial_scene_choice,
        initial_robot=robot_spec.uid,
        initial_tolerance_frame_rotvec_deg=np.degrees(continuous_rotation_vector(
            home_rotation, home_rotation_vector,
        )),
        initial_target_rotvec_deg=np.degrees(continuous_rotation_vector(
            home_rotation, home_rotation_vector,
        )),
        initial_target_position=home_position,
        initial_loss_profile=initial_loss_profile,
        initial_sqp_loss_weights=initial_sqp_loss_weights,
        initial_playback_scale=args.teleop_speed_scale,
    )
    if panel is not None and args.auto_execute_preset:
        panel.configure_automated_run(
            domain_seed=args.domain_seed,
            speed_scale=args.teleop_speed_scale,
        )
    # The main preview uses the same fixed front camera as recorded observations.
    # Expand this model instance's offscreen buffer before constructing the
    # three UI renderers; scene XML and offline collection remain unchanged.
    if not args.no_viewer and panel is not None:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1200)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 900)
    # Wrist and tabletop-side previews keep their existing window and layout.
    main_renderer = None if args.no_viewer or panel is None else mujoco.Renderer(model, height=900, width=1200)
    wrist_renderer = None if args.no_viewer or panel is None else mujoco.Renderer(model, height=300, width=400)
    side_renderer = None if args.no_viewer or panel is None else mujoco.Renderer(model, height=300, width=400)
    contact_markers = (
        None
        if args.no_viewer or not has_workspace
        else GripperTableContactMarkers(model, handles)
    )
    home_hold = False
    active_dofs = np.ones(6, dtype=bool)
    rotation_in_base = False
    gripper_ctrl = robot_spec.gripper_open_ctrl
    gripper_speed = abs(robot_spec.gripper_open_ctrl - robot_spec.gripper_closed_ctrl) / GRIPPER_TRAVEL_TIME
    physics_dt = float(model.opt.timestep)
    teleop_accumulator = TELEOP_DT
    gripper_phase_accumulator = TELEOP_DT
    playback_speed_scale = float(args.teleop_speed_scale)
    playback_clock = SimulationPlaybackClock(physics_dt, VIEWER_REFRESH_HZ)
    last_ui_render_time = -math.inf
    cached_snapshot = None
    started = time.monotonic()
    diagnostics = None
    live_release_loss_multipliers = np.full(3, np.nan)
    live_nominal_action_ema = np.full(3, np.nan)
    live_release_mask = np.zeros(3, dtype=bool)
    evaluation_recorder = MultiRateEvaluationRecorder(
        model, handles, TELEOP_DT,
    )
    preset_evaluation_active = False
    preset_evaluation_remaining_s: float | None = None
    preset_evaluator: TaskEvaluator | None = None
    preset_evaluator_frame = 0
    preset_result_status: str | None = None
    active_tolerance_frame_mode = TOLERANCE_FRAME_MODE_REALTIME_NOMINAL
    trajectory_recorder = TrajectoryRecorder()
    trajectory_previous_q = data.qpos[handles.qpos_ids].copy()
    active_scene_name = scene_manager.current_name
    active_scene_choice = initial_scene_choice or active_scene_name
    scene_evaluator: TaskEvaluator | None = None

    def reset_scene_evaluator() -> None:
        nonlocal scene_evaluator
        scene_evaluator = (
            create_task_evaluator(active_scene_name, WIPE_WHITEBOARD_TASK, model, data)
            if active_scene_name == WHITEBOARD_SCENE else None
        )

    reset_scene_evaluator()
    evaluation_commanded = False
    task_gripper_closed = False
    active_phase = ManipulationPhase.PREGRASP
    active_subtask_index = 0
    previous_front_camera_config: tuple[object, ...] | None = None
    previous_domain_config: tuple[object, ...] | None = None
    previous_tolerance_context_key: tuple[object, ...] | None = None
    nominal_preparer = (
        BackgroundNominalPreparation()
        if panel is not None and supports_nominal_selection(active_scene_name)
        else None
    )
    prepared_nominal = None
    subtask_preparer = BackgroundNominalPreparation() if nominal_preparer is not None else None
    prepared_subtask = None
    nominal_busy = False
    nominal_prepare_error = None
    pending_nominal_execute = False
    nominal_layout_revision = 0
    requested_scene: str | None = None
    requested_scene_group: str | None = None
    requested_scene_choice: str | None = None
    requested_robot: str | None = None
    last_rotation_task_error = np.zeros(3)
    instantaneous_rotation_target = target.rotation.copy()
    evaluation_reference_start_position = target.position.copy()
    evaluation_reference_goal_position = target.position.copy()
    evaluation_reference_start_rotation = target.rotation.copy()
    evaluation_reference_goal_rotation = target.rotation.copy()
    evaluation_summary = evaluation_recorder.summary
    diagnostic_trace: list[dict[str, object]] = []
    creation_baseline: CustomLayout | None = None
    creation_layout: CustomLayout | None = None
    retained_custom_layout: CustomLayout | None = initial_custom_layout
    creation_selected_name: str | None = None
    creation_drag_offset = np.zeros(2)
    creation_mode_previous = False
    boundary_surface: WorkspaceBoundarySurface | None = None
    boundary_surface_error: str | None = None

    def load_boundary_surface() -> WorkspaceBoundarySurface:
        nonlocal boundary_surface, boundary_surface_error
        if boundary_surface is not None:
            return boundary_surface
        if boundary_surface_error is not None:
            raise RuntimeError(boundary_surface_error)
        path = (
            Path(args.reachability_map_dir)
            / f"{robot_spec.uid}_workspace_boundary_v2.npz"
            if args.reachability_map_dir is not None
            else default_boundary_surface_path(robot_spec.uid)
        )
        try:
            boundary_surface = WorkspaceBoundarySurface.load(path)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            boundary_surface_error = str(error)
            raise
        return boundary_surface

    def refresh_creation_selection() -> None:
        if panel is None:
            return
        selected = next((
            item for item in scene_manager.layout_objects(data)
            if item.body_name == creation_selected_name
        ), None)
        if selected is None:
            panel.set_layout_selection(None)
        else:
            panel.set_layout_selection(
                selected.body_name,
                float(selected.position[0]), float(selected.position[1]),
                math.degrees(yaw_from_quaternion(selected.quaternion)),
            )

    def capture_creation_layout(seed: int) -> CustomLayout:
        return capture_custom_layout(
            scene_manager, data, robot=robot_spec.uid, source_seed=seed,
        )

    def process_layout_commands(snapshot: object) -> None:
        nonlocal creation_baseline, creation_layout, retained_custom_layout
        nonlocal creation_selected_name, creation_drag_offset
        nonlocal previous_domain_config
        commands = snapshot.layout_edit_commands
        if not commands:
            return
        camera_position = data.cam_xpos[front_camera_id]
        camera_rotation = data.cam_xmat[front_camera_id]
        aspect = 1200.0 / 900.0
        for command in commands:
            try:
                if (preset_task.active or preset_evaluation_active) and command.kind not in {
                    "export", "exit",
                }:
                    panel.preset_task_status("creation editing locked during task execution")
                    continue
                if command.kind == "enter":
                    creation_baseline = capture_creation_layout(snapshot.domain_seed)
                    creation_layout = creation_baseline
                    retained_custom_layout = creation_baseline
                    creation_selected_name = None
                elif command.kind == "exit":
                    retained_custom_layout = capture_creation_layout(
                        snapshot.domain_seed,
                    )
                    creation_layout = None
                    creation_selected_name = None
                    previous_domain_config = (
                        snapshot.domain_seed,
                        snapshot.domain_randomization_enabled,
                        snapshot.boundary_unreachable_placement,
                        (
                            snapshot.preset_task_name
                            if snapshot.boundary_unreachable_placement else None
                        ),
                        snapshot.workspace_boundary_distance_min,
                        snapshot.workspace_boundary_distance_max,
                    )
                elif command.kind == "reset" and creation_baseline is not None:
                    apply_custom_layout(
                        scene_manager, data, creation_baseline,
                        robot=robot_spec.uid, restore_variant=False,
                    )
                    reset_scene_evaluator()
                    creation_layout = capture_creation_layout(snapshot.domain_seed)
                    retained_custom_layout = creation_layout
                elif command.kind == "export" and command.name:
                    path = save_named_custom_layout(
                        command.name,
                        (
                            capture_creation_layout(snapshot.domain_seed)
                            if snapshot.creation_mode
                            else retained_custom_layout
                            or capture_creation_layout(snapshot.domain_seed)
                        ),
                    )
                    panel.refresh_layout_names(command.name)
                    panel.preset_task_status(f"layout exported: {path}")
                elif command.kind == "import" and command.name:
                    imported = load_named_custom_layout(
                        active_scene_name, command.name,
                    )
                    apply_custom_layout(
                        scene_manager, data, imported, robot=robot_spec.uid,
                    )
                    reset_scene_evaluator()
                    creation_baseline = capture_creation_layout(imported.source_seed)
                    retained_custom_layout = creation_baseline
                    creation_layout = creation_baseline if snapshot.creation_mode else None
                    creation_selected_name = None
                    previous_domain_config = (
                        snapshot.domain_seed,
                        snapshot.domain_randomization_enabled,
                        snapshot.boundary_unreachable_placement,
                        (
                            snapshot.preset_task_name
                            if snapshot.boundary_unreachable_placement else None
                        ),
                        snapshot.workspace_boundary_distance_min,
                        snapshot.workspace_boundary_distance_max,
                    )
                    panel.preset_task_status(
                        f"layout imported: {command.name}",
                    )
                elif command.kind == "select" and snapshot.creation_mode:
                    click = np.array((command.x, command.y), dtype=float)
                    ray_geom = np.array((-1,), dtype=np.int32)
                    ray_direction = _image_ray_direction(
                        click, camera_rotation,
                        model.cam_fovy[front_camera_id], aspect,
                    )
                    mujoco.mj_ray(
                        model, data, camera_position, ray_direction,
                        None, True, -1, ray_geom,
                    )
                    exact = next((
                        item for item in scene_manager.layout_objects(data)
                        if int(ray_geom[0]) in set(int(value) for value in item.geom_ids)
                    ), None)
                    candidates: list[tuple[float, object]] = []
                    for item in scene_manager.layout_objects(data):
                        points = [item.position]
                        points.extend(data.geom_xpos[int(index)] for index in item.geom_ids)
                        projected = [
                            _project_world_to_image(
                                point, camera_position, camera_rotation,
                                model.cam_fovy[front_camera_id], aspect,
                            )
                            for point in points
                        ]
                        distances = [
                            float(np.linalg.norm(pixel - click))
                            for pixel in projected if pixel is not None
                        ]
                        if distances:
                            candidates.append((min(distances), item))
                    if exact is not None or candidates:
                        distance, selected = (
                            (0.0, exact)
                            if exact is not None
                            else min(candidates, key=lambda pair: pair[0])
                        )
                        if selected is not None and distance <= 0.09:
                            creation_selected_name = selected.body_name
                            world = _image_ray_to_z_plane(
                                click, float(selected.position[2]), camera_position,
                                camera_rotation, model.cam_fovy[front_camera_id], aspect,
                            )
                            if world is not None:
                                creation_drag_offset = selected.position[:2] - world[:2]
                        else:
                            creation_selected_name = None
                elif command.kind in {"drag", "rotate", "set_pose"}:
                    selected = next((
                        item for item in scene_manager.layout_objects(data)
                        if item.body_name == creation_selected_name
                    ), None)
                    if selected is None or not snapshot.creation_mode:
                        continue
                    position = selected.position.copy()
                    quaternion = selected.quaternion.copy()
                    if command.kind == "drag":
                        world = _image_ray_to_z_plane(
                            np.array((command.x, command.y)), float(position[2]),
                            camera_position, camera_rotation,
                            model.cam_fovy[front_camera_id], aspect,
                        )
                        if world is None:
                            continue
                        position[:2] = world[:2] + creation_drag_offset
                    elif command.kind == "rotate":
                        quaternion = quaternion_with_yaw(
                            quaternion,
                            yaw_from_quaternion(quaternion)
                            + math.radians(float(command.yaw_deg)),
                        )
                    else:
                        position[:2] = (float(command.x), float(command.y))
                        quaternion = quaternion_with_yaw(
                            quaternion, math.radians(float(command.yaw_deg)),
                        )
                    scene_manager.set_layout_pose(
                        data, selected.body_name, position, quaternion,
                    )
                    mujoco.mj_forward(model, data)
                    creation_layout = capture_creation_layout(snapshot.domain_seed)
                    retained_custom_layout = creation_layout
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                panel.preset_task_status(f"layout operation failed: {error}")
        refresh_creation_selection()

    def active_profile_loss_weights() -> list[float]:
        settings = ipopt_ik.objective_settings
        return [
            float(settings.squared_delta_q_weight),
            float(settings.ema_release_loss_weight),
        ]

    def write_diagnostic_trace() -> None:
        if args.diagnostics_output is None:
            return
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.write_text(json.dumps({
            "scene": args.scene,
            "scene_choice": active_scene_choice,
            "robot": robot_spec.uid,
            "seed": args.domain_seed,
            "speed_scale": playback_speed_scale,
            "speed_scale_semantics": "wall_clock_playback_only",
            "simulation_time_s": float(data.time),
            "physics_timestep_s": physics_dt,
            "sqp_loss_argument": args.sqp_loss,
            "sqp_loss_profile": ipopt_ik.loss_profile,
            "sqp_loss_weights": active_profile_loss_weights(),
            "optimizer_backend": "ipopt",
            "preset_status": displayed_preset_status(),
            "trace": diagnostic_trace,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def rebase_evaluation_reference(
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> None:
        nonlocal evaluation_reference_start_position
        nonlocal evaluation_reference_goal_position
        nonlocal evaluation_reference_start_rotation
        nonlocal evaluation_reference_goal_rotation
        evaluation_reference_start_position = position.copy()
        evaluation_reference_goal_position = position.copy()
        evaluation_reference_start_rotation = rotation.copy()
        evaluation_reference_goal_rotation = rotation.copy()

    def stop_preset_evaluation(*, finalize_task: bool = False) -> None:
        nonlocal preset_evaluation_active
        nonlocal preset_evaluation_remaining_s
        nonlocal preset_evaluator
        nonlocal preset_result_status
        if preset_evaluation_active:
            evaluation_recorder.stop()
        if finalize_task and preset_evaluator is not None:
            result = preset_evaluator.result(True, timed_out=False)
            preset_result_status = (
                "success"
                if bool(result.get("success"))
                else f"failed: {result.get('failure_reason') or 'task_failed'}"
            )
        elif not finalize_task:
            preset_result_status = None
        preset_evaluator = None
        preset_evaluation_active = False
        preset_evaluation_remaining_s = None

    def displayed_preset_status() -> str:
        if nominal_busy:
            return (
                "Preparing nominal plan in background; execution queued"
                if pending_nominal_execute else "Preparing nominal plan in background"
            )
        if nominal_prepare_error is not None and not preset_task.active:
            return f"Nominal plan unavailable: {nominal_prepare_error}"
        if not preset_task.active and preset_result_status is not None:
            return preset_result_status
        return preset_task.status

    def render_cameras(snapshot) -> None:
        if main_renderer is None:
            return
        main_renderer.update_scene(data, camera="front_view")
        if snapshot.show_reachability_map:
            try:
                line_count = populate_workspace_boundary_surface_scene(
                    main_renderer.scene, load_boundary_surface(),
                    factor_lower=snapshot.workspace_boundary_distance_min,
                    factor_upper=snapshot.workspace_boundary_distance_max,
                    maximum_lines=2600, append=True)
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                panel.reachability_status(f"3D boundary unavailable: {error}")
            else:
                panel.reachability_status(f"3D boundary visible ({line_count} segments)")
        else:
            panel.reachability_status("3D boundary hidden")
        if snapshot.creation_mode and object_layout_randomizer is not None:
            append_placement_region(main_renderer.scene, object_layout_randomizer.placement_region(data))
        panel.main_camera_frame(main_renderer.render())
        assert wrist_renderer is not None and side_renderer is not None
        wrist_renderer.update_scene(data, camera="wrist_camera")
        side_renderer.update_scene(data, camera=side_camera_name)
        panel.camera_frames(wrist_renderer.render(), np.flip(side_renderer.render(), axis=1))

    def start_evaluation() -> None:
        evaluation_recorder.start()
        evaluation_recorder.begin_control_interval(
            evaluation_reference_start_position,
            evaluation_reference_start_rotation,
            evaluation_reference_goal_position,
            evaluation_reference_goal_rotation,
            joint_target,
        )

    def toggle_evaluation() -> None:
        evaluation_recorder.toggle()

    def cancel_preset(reason: str) -> None:
        nonlocal active_subtask_index, active_phase
        nonlocal live_release_loss_multipliers, live_nominal_action_ema
        nonlocal prepared_subtask
        if subtask_preparer is not None:
            subtask_preparer.cancel()
        prepared_subtask = None
        preset_task.cancel(reason)
        ipopt_ik.discard_block_plan()
        active_subtask_index = 0
        active_phase = ManipulationPhase.PREGRASP
        # A cancelled preset is a new manual-control context. Do not carry
        # its stage-local release state into the next command stream.
        release_state.reset()
        live_release_loss_multipliers.fill(np.nan)
        live_nominal_action_ema.fill(np.nan)
        stop_preset_evaluation()

    def step(*, poll_ui: bool = False, render_ui: bool = False) -> bool:
        nonlocal active_dofs, rotation_in_base, home_hold, joint_target, joint_command, gripper_ctrl
        nonlocal interpolation_start, interpolation_goal, interpolation_elapsed
        nonlocal teleop_accumulator, gripper_phase_accumulator, cached_snapshot, diagnostics, trajectory_previous_q, evaluation_commanded
        nonlocal active_scene_name, active_scene_choice, task_gripper_closed
        nonlocal active_phase
        nonlocal previous_front_camera_config, previous_domain_config, requested_scene, requested_scene_group, requested_scene_choice, requested_robot
        nonlocal previous_tolerance_context_key
        nonlocal prepared_nominal, nominal_busy, nominal_prepare_error, pending_nominal_execute
        nonlocal prepared_subtask
        nonlocal nominal_layout_revision
        nonlocal last_rotation_task_error, instantaneous_rotation_target
        nonlocal evaluation_reference_start_position, evaluation_reference_goal_position
        nonlocal evaluation_reference_start_rotation, evaluation_reference_goal_rotation
        nonlocal evaluation_summary
        nonlocal preset_evaluation_active, preset_evaluation_remaining_s
        nonlocal preset_evaluator, preset_evaluator_frame
        nonlocal active_tolerance_frame_mode
        nonlocal playback_speed_scale, last_ui_render_time
        nonlocal preset_result_status
        nonlocal active_subtask_index
        nonlocal live_release_loss_multipliers
        nonlocal live_nominal_action_ema
        nonlocal live_release_mask
        nonlocal creation_baseline, creation_layout, creation_mode_previous
        nonlocal retained_custom_layout
        pressed: set[str] = set()
        gripper_closed = False
        reset = False
        reset_scene = False
        toggled: tuple[int, ...] = ()
        if panel is not None:
            ui_refreshed = cached_snapshot is None or poll_ui
            if ui_refreshed:
                snapshot = panel.snapshot()
                # Cache continuous controls only. One-shot UI events must be
                # consumed exactly once rather than replayed at 1 kHz.
                cached_snapshot = replace(
                    snapshot,
                    home_requested=False,
                    reset_scene_requested=False,
                    toggled_dofs=(),
                    capture_tolerance_target_stage=None,
                    capture_tolerance_frame_stage=None,
                    toggle_evaluation=False,
                    toggle_trajectory=False,
                    play_trajectory=False,
                    execute_preset_task=False,
                    layout_edit_commands=(),
                )
            else:
                assert cached_snapshot is not None
                snapshot = cached_snapshot
            if not panel.running:
                return False
            task_execution_locked = (
                preset_task.active or preset_evaluation_active
            )
            if not task_execution_locked:
                # The operator may select the branch only before execution.
                # Once EXECUTE starts, this latched value owns every subtask
                # and every optimizer cycle through final evaluation.
                active_tolerance_frame_mode = snapshot.tolerance_frame_mode
            pressed = snapshot.pressed
            gripper_closed = snapshot.gripper_closed
            reset = snapshot.home_requested
            reset_scene = snapshot.reset_scene_requested
            toggled = snapshot.toggled_dofs
            active_dofs = snapshot.active_dofs
            release_context_prefix = (
                active_scene_choice,
                snapshot.preset_task_name,
                int(active_subtask_index),
            )
            previous_active_stage = rotational_tolerance.active_stage
            _configure_ui_tolerance(rotational_tolerance, snapshot, release_context_prefix)
            if snapshot.capture_tolerance_target_stage is not None:
                captured_position, captured_rotation = ee_pose(data, handles)
                panel.set_stage_target(
                    snapshot.capture_tolerance_target_stage,
                    captured_rotation,
                    captured_position,
                )
                if snapshot.capture_tolerance_target_stage == snapshot.active_stage:
                    rotational_tolerance.configure_stage(
                        snapshot.active_stage, captured_rotation,
                        snapshot.rotation_tolerance_frame,
                        snapshot.rotation_tolerance_rad,
                        stage_key=release_context_prefix + (snapshot.active_stage,),
                        update_prediction=False,
                    )
            if snapshot.capture_tolerance_frame_stage is not None:
                _, captured_rotation = ee_pose(data, handles)
                panel.set_stage_tolerance_frame(snapshot.capture_tolerance_frame_stage, captured_rotation)
                if snapshot.capture_tolerance_frame_stage == snapshot.active_stage:
                    rotational_tolerance.configure_stage(
                        snapshot.active_stage,
                        snapshot.rotation_tolerance_target,
                        captured_rotation,
                        snapshot.rotation_tolerance_rad,
                        stage_key=release_context_prefix + (snapshot.active_stage,),
                        update_prediction=False,
                    )
            rotation_in_base = snapshot.rotation_in_base
            playback_speed_scale = snapshot.teleop_speed_scale
            # Playback changes only wall-clock scheduling. The optimizer keeps
            # the same iteration and CPU budget at every presentation speed.
            if ui_refreshed:
                set_wrist_camera_calibration(
                    model,
                    wrist_camera_id,
                    snapshot.camera_x_offset,
                    snapshot.camera_z_offset,
                    snapshot.camera_pitch_offset_deg,
                    wrist_camera_base_pos,
                    wrist_camera_base_quat,
                )
            effective_domain_seed = (
                snapshot.domain_seed
                if snapshot.domain_randomization_enabled else -1
            )
            front_camera_config = (
                effective_domain_seed,
                snapshot.main_camera_x_offset,
                snapshot.main_camera_z_offset,
                snapshot.main_camera_pitch_offset_deg,
            )
            domain_config = (
                snapshot.domain_seed,
                snapshot.domain_randomization_enabled,
                snapshot.boundary_unreachable_placement,
                (
                    snapshot.preset_task_name
                    if snapshot.boundary_unreachable_placement else None
                ),
                snapshot.workspace_boundary_distance_min,
                snapshot.workspace_boundary_distance_max,
            )
            if reset_scene and retained_custom_layout is not None:
                cancel_preset("cancelled by scene reset")
                apply_custom_layout(
                    scene_manager, data, retained_custom_layout,
                    robot=robot_spec.uid, restore_variant=False,
                )
                reset_scene_evaluator()
                creation_layout = (
                    retained_custom_layout if snapshot.creation_mode else None
                )
                previous_domain_config = domain_config
                panel.preset_task_status("custom layout restored")
            elif initial_custom_layout_path is not None and previous_domain_config is None:
                # Preserve the explicitly requested reproducible layout on the
                # first UI poll. A later seed/control change resumes the usual
                # domain-layout path.
                previous_domain_config = domain_config
            elif (
                domain_config != previous_domain_config or reset_scene
            ) and not snapshot.creation_mode:
                if domain_config != previous_domain_config:
                    retained_custom_layout = None
                cancel_preset(
                    "cancelled by scene reset"
                    if reset_scene
                    else "cancelled by domain change",
                )
                if table_randomizer is not None and object_layout_randomizer is not None:
                    table_seed = (
                        -1
                        if snapshot.boundary_unreachable_placement
                        else effective_domain_seed
                    )
                    randomized_table = table_randomizer.sample(table_seed)
                    model.body_pos[work_surface_id] = randomized_table.position
                    scene_manager.set_workspace_offset(data, randomized_table.offset)
                    object_layout_randomizer.apply(data, snapshot.domain_seed)
                    if snapshot.boundary_unreachable_placement:
                        boundary_seed = (
                            0 if snapshot.domain_seed == -1
                            else snapshot.domain_seed
                        )
                        boundary_mode = WORKSPACE_BOUNDARY_MODES[
                            boundary_seed % len(WORKSPACE_BOUNDARY_MODES)
                        ]
                        try:
                            boundary_result = apply_training_workspace_boundary(
                                model, data, scene_manager, handles,
                                snapshot.preset_task_name,
                                boundary_mode,
                                snapshot.domain_seed,
                                distance_factor_lower=(
                                    snapshot.workspace_boundary_distance_min
                                ),
                                distance_factor_upper=(
                                    snapshot.workspace_boundary_distance_max
                                ),
                                map_directory=args.reachability_map_dir,
                            )
                        except (FileNotFoundError, RuntimeError, ValueError) as error:
                            object_layout_randomizer.apply(
                                data, snapshot.domain_seed,
                            )
                            if panel is not None:
                                panel.preset_task_status(
                                    f"boundary placement unavailable: {error}"
                                )
                        else:
                            if panel is not None:
                                panel.preset_task_status(
                                    "workspace boundary active: "
                                    f"{boundary_result['requested_strategy']}"
                                    " -> "
                                    f"{boundary_result['effective_strategy']}"
                                )
                    else:
                        apply_blocks_difficulty(
                            model, data, scene_manager, handles,
                            snapshot.preset_task_name,
                            args.placement_difficulty,
                            snapshot.domain_seed,
                            map_directory=args.reachability_map_dir,
                        )
                reset_scene_evaluator()
                light_randomizer.apply(
                    model, light_randomizer.sample(effective_domain_seed),
                )
                previous_domain_config = domain_config
            process_layout_commands(snapshot)
            front_camera_config += (
                tuple(model.body_pos[work_surface_id]) if has_workspace else (),
            )
            if front_camera_config != previous_front_camera_config:
                center_position, center_quaternion = _front_camera_center_pose(
                    front_camera_base_pos,
                    front_camera_base_quat,
                    snapshot.main_camera_x_offset,
                    snapshot.main_camera_z_offset,
                    snapshot.main_camera_pitch_offset_deg,
                )
                randomized_pose = front_camera_randomizer.sample(
                    effective_domain_seed, center_position, center_quaternion,
                    tabletop=tabletop_corners(model),
                    reference_tabletop=front_camera_reference_tabletop,
                    vertical_fov_deg=float(model.cam_fovy[front_camera_id]),
                )
                model.cam_pos[front_camera_id] = randomized_pose.position
                model.cam_quat[front_camera_id] = randomized_pose.quaternion
                previous_front_camera_config = front_camera_config
            if (
                snapshot.creation_mode
                and creation_layout is not None
                and not preset_task.active
                and not preset_evaluation_active
            ):
                # Creation mode is an editor: keep dynamic objects exactly at
                # their authored pose instead of letting gravity advance them
                # between mouse events. Exiting releases normal physics again.
                apply_custom_layout(
                    scene_manager, data, creation_layout,
                    robot=robot_spec.uid, restore_variant=False,
                )
                reset_scene_evaluator()
            if snapshot.creation_mode != creation_mode_previous:
                creation_mode_previous = snapshot.creation_mode
                refresh_creation_selection()
            panel.set_creation_locked(
                preset_task.active or preset_evaluation_active,
            )
            panel.set_tolerance_frame_mode_locked(task_execution_locked)
            if (
                snapshot.scene_name != active_scene_name
                or snapshot.scene_choice != active_scene_choice
            ):
                cancel_preset("cancelled by scene change")
                requested_scene = snapshot.scene_name
                requested_scene_group = snapshot.scene_group
                requested_scene_choice = snapshot.scene_choice
                return False
            if snapshot.robot_uid != robot_spec.uid:
                cancel_preset("cancelled by robot change")
                requested_robot = snapshot.robot_uid
                return False
            if reset_scene:
                # Resolve endpoints from the reset robot pose, not the last
                # episode's released pose. Restore the UI phase before using
                # its limits so Post/Release cannot configure the new PRE.
                _restore_robot_home_for_scene_reset(model, data, handles)
                active_phase = ManipulationPhase.PREGRASP
                active_subtask_index = 0
                pregrasp_limits = panel.reset_episode_tolerance()
                snapshot = replace(
                    snapshot,
                    active_stage=0,
                    gripper_closed=False,
                    rotation_tolerance_rad=pregrasp_limits,
                )
                release_context_prefix = (
                    active_scene_choice,
                    snapshot.preset_task_name,
                    0,
                )
                cached_snapshot = None
            if nominal_preparer is not None and snapshot.layout_edit_commands:
                nominal_layout_revision += 1
            tolerance_context_key = (
                active_scene_name,
                snapshot.preset_task_name,
                snapshot.domain_seed,
                scene_manager.active_body_names,
                snapshot.tolerance_enabled,
                snapshot.boundary_unreachable_placement,
                args.ipopt_max_iterations,
                domain_config,
                nominal_layout_revision,
            )
            if (
                reset_scene
                or tolerance_context_key != previous_tolerance_context_key
            ):
                if (
                    previous_tolerance_context_key is not None
                    and snapshot.preset_task_name
                    != previous_tolerance_context_key[1]
                ):
                    cancel_preset("cancelled by task change")
                try:
                    current_position, current_rotation = ee_pose(data, handles)
                    if nominal_preparer is not None:
                        if nominal_preparer.key != tolerance_context_key or reset_scene:
                            nominal_preparer.request(tolerance_context_key, model, data,
                                robot_uid=robot_spec.uid, scene=active_scene_name,
                                task=snapshot.preset_task_name, seed=snapshot.domain_seed,
                                domain_randomization_enabled=snapshot.domain_randomization_enabled,
                                iterations=args.ipopt_max_iterations)
                            prepared_nominal = None
                            nominal_prepare_error = None
                            pending_nominal_execute = False
                        nominal_busy = True
                        prepared_nominal, resolved_context = nominal_preparer.poll()
                        if not np.array_equal(prepared_nominal.initial_qpos, data.qpos):
                            # Editing the live scene while a worker was running
                            # must not publish stale targets or execute its plan.
                            nominal_preparer.request(tolerance_context_key, model, data,
                                robot_uid=robot_spec.uid, scene=active_scene_name,
                                task=snapshot.preset_task_name, seed=snapshot.domain_seed,
                                domain_randomization_enabled=snapshot.domain_randomization_enabled,
                                iterations=args.ipopt_max_iterations)
                            prepared_nominal = None
                            raise PreparationPending()
                        nominal_busy = False
                    else:
                        resolved_context = resolve_task_tolerance_context(
                            model, data, handles, active_scene_name,
                            snapshot.preset_task_name, current_position, current_rotation,
                            episode_seed=snapshot.domain_seed,
                            nominal_validation_options=dict(ipopt_max_iterations=args.ipopt_max_iterations))
                except PreparationPending:
                    pass
                except (RuntimeError, ValueError, KeyError) as error:
                    nominal_busy = False
                    pending_nominal_execute = False
                    nominal_prepare_error = str(error) if nominal_preparer is not None else None
                    panel.preset_task_status(
                        f"tolerance auto-config unavailable: {error}"
                    )
                    previous_tolerance_context_key = tolerance_context_key
                else:
                    panel.apply_task_tolerance_context(resolved_context)
                    cached_snapshot = None
                    snapshot = panel.tolerance_snapshot(snapshot)
                    active_dofs = snapshot.active_dofs
                    _configure_ui_tolerance(rotational_tolerance, snapshot, release_context_prefix)
                    previous_tolerance_context_key = tolerance_context_key
            if ui_refreshed:
                ipopt_ik.set_loss_profile(
                    snapshot.loss_profile,
                    snapshot.sqp_loss_weights,
                )
            if snapshot.toggle_evaluation:
                if preset_task.active or preset_evaluation_active:
                    cancel_preset("stopped early by key 8")
                else:
                    toggle_evaluation()
            if snapshot.toggle_trajectory:
                trajectory_recorder.toggle_recording()
                trajectory_previous_q = data.qpos[handles.qpos_ids].copy()
            if snapshot.play_trajectory:
                position, rotation = ee_pose(data, handles)
                target.position = position
                target.rotation = rotation
                instantaneous_rotation_target = rotation.copy()
                rebase_evaluation_reference(position, rotation)
                trajectory_recorder.start_playback()
            execute_requested = snapshot.execute_preset_task
            if nominal_preparer is not None:
                if snapshot.home_requested or any(key in pressed for key in "wasdikqejuol"):
                    nominal_preparer.cancel()
                    prepared_nominal = None
                    nominal_busy = False
                    pending_nominal_execute = False
                    previous_tolerance_context_key = tolerance_context_key
                if execute_requested:
                    pending_nominal_execute = True
                    if nominal_prepare_error is None and not nominal_busy and (
                        prepared_nominal is None
                        or not np.array_equal(prepared_nominal.initial_qpos, data.qpos)
                    ):
                        nominal_preparer.request(tolerance_context_key, model, data,
                            robot_uid=robot_spec.uid, scene=active_scene_name,
                            task=snapshot.preset_task_name, seed=snapshot.domain_seed,
                            domain_randomization_enabled=snapshot.domain_randomization_enabled,
                            iterations=args.ipopt_max_iterations)
                        previous_tolerance_context_key = None
                        prepared_nominal = None
                        nominal_busy = True
                execute_requested = bool(pending_nominal_execute and not nominal_busy
                                         and prepared_nominal is not None)
                if nominal_prepare_error is not None:
                    pending_nominal_execute = False
                    execute_requested = False
            if execute_requested:
                pending_nominal_execute = False
                current_position, current_rotation = ee_pose(data, handles)
                try:
                    preset_task.start(
                        snapshot.preset_task_name, active_scene_name, data,
                        current_position, current_rotation,
                        episode_seed=snapshot.domain_seed,
                        nominal_validation_options=dict(
                            ipopt_max_iterations=args.ipopt_max_iterations,
                        ),
                        domain_randomization_enabled=snapshot.domain_randomization_enabled,
                        prepared_plan=prepared_nominal if nominal_preparer is not None else None,
                    )
                    preset_evaluator = (
                        scene_evaluator
                        if active_scene_name == WHITEBOARD_SCENE
                        else create_task_evaluator(
                            active_scene_name, snapshot.preset_task_name,
                            model, data,
                        )
                    )
                    preset_evaluator_frame = 0
                    preset_result_status = None
                    simulation_backend.reset_manipulation_phase()
                    gripper_phase_accumulator = TELEOP_DT
                    panel.reset_tolerance_subtask()
                    snapshot = panel.tolerance_snapshot(snapshot)
                    active_dofs = snapshot.active_dofs
                    release_context_prefix = (active_scene_choice, snapshot.preset_task_name, 0)
                    _configure_ui_tolerance(rotational_tolerance, snapshot, release_context_prefix)
                    cached_snapshot = None
                    target.position = current_position.copy()
                    target.rotation = current_rotation.copy()
                    instantaneous_rotation_target = current_rotation.copy()
                    rebase_evaluation_reference(
                        current_position, current_rotation,
                    )
                    task_gripper_closed = False
                    active_phase = ManipulationPhase.PREGRASP
                    active_subtask_index = 0
                    home_hold = False
                    trajectory_recorder.playing = False
                    current_q = data.qpos[handles.qpos_ids].copy()
                    joint_target = joint_command = current_q.copy()
                    interpolation_start = interpolation_goal = current_q.copy()
                    interpolation_elapsed = TELEOP_DT
                    ipopt_ik.reset()
                    joint_tracker.reset()
                    release_state.reset()
                    teleop_accumulator = TELEOP_DT
                    start_evaluation()
                    preset_evaluation_active = True
                    preset_evaluation_remaining_s = None
                    panel.set_tolerance_frame_mode_locked(True)
                except (RuntimeError, ValueError) as error:
                    cancel_preset(str(error))
            if preset_task.active and any(key in pressed for key in "wasdikqejuol"):
                cancel_preset("cancelled by manual motion")
            if preset_task.active:
                gripper_closed = task_gripper_closed
            if ui_refreshed:
                panel.preset_task_status(displayed_preset_status())
                panel.recording_state(
                    evaluation_recorder.recording,
                    trajectory_recorder.recording,
                    trajectory_recorder.playing,
                    trajectory_recorder.progress,
                )
            if toggled:
                current_position, _ = ee_pose(data, handles)
                _reconcile_target_after_toggle(
                    target, current_position, toggled,
                    active_dofs, rotational_tolerance,
                )
            stage_changed = (
                snapshot.active_stage != previous_active_stage
            )
            if stage_changed:
                # Keep the action-integrated nominal branch unchanged.  The
                # next outer cycle captures its nominal/physical handoff pair
                # and transports the physical pose by later base-frame
                # nominal motion.  Telemetry may follow the currently
                # executed pose, but that observation must not replace R_nom.
                handoff_rotation = release_state.rotation_for_joint(
                    joint_command,
                )
                last_rotation_task_error[:] = 0.0
                instantaneous_rotation_target = handoff_rotation.copy()

        if reset_scene:
            # Episode-style reset restores either the retained custom layout
            # or the current seeded scene, then places the robot and gripper
            # at Home with no stale controller/playback/classifier state.
            cancel_preset("scene reset")
            trajectory_recorder.playing = False
            trajectory_recorder.recording = False
            evaluation_recorder.stop()
            rotational_tolerance.reset_intent()
            target.position, target.rotation = (
                home_position.copy(),
                home_rotation.copy(),
            )
            instantaneous_rotation_target = target.rotation.copy()
            rebase_evaluation_reference(target.position, target.rotation)
            joint_target = robot_spec.home_q.copy()
            joint_command = robot_spec.home_q.copy()
            interpolation_start = robot_spec.home_q.copy()
            interpolation_goal = robot_spec.home_q.copy()
            interpolation_elapsed = TELEOP_DT
            gripper_ctrl = robot_spec.gripper_open_ctrl
            gripper_closed = False
            task_gripper_closed = False
            home_hold = True
            trajectory_previous_q = robot_spec.home_q.copy()
            ipopt_ik.reset()
            release_state.reset()
            joint_tracker.reset()
            simulation_backend.reset_manipulation_phase()
            if contact_markers is not None:
                contact_markers.clear()
            teleop_accumulator = TELEOP_DT
            gripper_phase_accumulator = TELEOP_DT
            reset = False
        if reset:
            cancel_preset("cancelled by Home")
            rotational_tolerance.reset_intent()
            home_hold = True
            target.position, target.rotation = home_position.copy(), home_rotation.copy()
            instantaneous_rotation_target = target.rotation.copy()
            evaluation_reference_start_position = (
                evaluation_reference_goal_position.copy()
            )
            evaluation_reference_start_rotation = (
                evaluation_reference_goal_rotation.copy()
            )
            evaluation_reference_goal_position = target.position.copy()
            evaluation_reference_goal_rotation = target.rotation.copy()
            joint_target = robot_spec.home_q.copy()
            interpolation_start = joint_command.copy()
            interpolation_goal = robot_spec.home_q.copy()
            interpolation_elapsed = 0.0
            ipopt_ik.reset()
            release_state.reset()
            joint_tracker.reset()
            teleop_accumulator = 0.0
        if home_hold and _has_enabled_command(pressed, active_dofs):
            home_hold = False
            joint_target = data.qpos[handles.qpos_ids].copy()
            joint_command = joint_target.copy()
            interpolation_start = joint_command.copy()
            interpolation_goal = joint_command.copy()
            interpolation_elapsed = TELEOP_DT
            position, rotation = ee_pose(data, handles)
            target.position = position
            target.rotation = rotation
            instantaneous_rotation_target = rotation.copy()
            rebase_evaluation_reference(position, rotation)
            joint_tracker.reset()
            teleop_accumulator = TELEOP_DT

        if (subtask_preparer is not None and preset_task.needs_subtask_plan
                and teleop_accumulator + physics_dt + 1e-12 >= TELEOP_DT):
            configuration = preset_task._nominal_configuration or {}
            next_index = preset_task.subtask_index + 1
            if configuration.get('enabled') and nominal_candidate_definitions(
                    model, data, configuration['scene'], configuration['task'], next_index):
                key = (configuration['scene'], configuration['task'], configuration['episode_seed'],
                       configuration['domain_randomization_enabled'], next_index, nominal_layout_revision)
                if subtask_preparer.key != key:
                    subtask_preparer.request(key, model, data, robot_uid=robot_spec.uid,
                        scene=configuration['scene'], task=configuration['task'],
                        seed=configuration['episode_seed'], iterations=args.ipopt_max_iterations,
                        domain_randomization_enabled=configuration['domain_randomization_enabled'],
                        subtask_index=next_index, initial_position=target.position,
                        initial_rotation=release_state.rotation_for_joint(joint_command),
                        perturbations=preset_task.perturbations.samples)
                try:
                    prepared_subtask, _ = subtask_preparer.poll()
                except PreparationPending:
                    preset_task.status = f"Preset task: preparing subtask {next_index + 1} in background"
                    if render_ui and panel is not None:
                        render_cameras(snapshot)
                    return True
                except Exception as error:
                    cancel_preset(f"subtask planning failed: {error}")
                    preset_result_status = f"Preset task failed: {error}"

        if nominal_busy:
            # Keep the requested initial state unchanged until its background
            # result is adopted. UI polling, cameras, reset and scene changes
            # remain live; no simulation time or control ticks are advanced.
            if render_ui and panel is not None:
                render_cameras(snapshot)
            return True

        teleop_accumulator += physics_dt
        if home_hold:
            joint_target = robot_spec.home_q.copy()
            interpolation_goal = joint_target.copy()
            last_rotation_task_error[:] = 0.0
            instantaneous_rotation_target = target.rotation.copy()
            teleop_accumulator = 0.0
        else:
            while teleop_accumulator + 1e-12 >= TELEOP_DT:
                rotation_action = np.zeros(3, dtype=float)
                evaluation_reference_start_position = (
                    evaluation_reference_goal_position.copy()
                )
                evaluation_reference_start_rotation = (
                    evaluation_reference_goal_rotation.copy()
                )
                # The stored target is the nominal expectation.  A stage
                # handoff snapshots the currently executing optimized pose
                # against this value before the next input action advances it.
                task_action = None
                preset_cycle = preset_task.active
                if preset_cycle:
                    if preset_task.needs_subtask_plan:
                        # A subtask boundary adopts the accepted optimized
                        # orientation as the new nominal zero.  This clears
                        # all carried release, tolerance-filter, and intent
                        # history before the next segment is generated from
                        # the live scene and current nominal EE target.
                        next_subtask_index = preset_task.subtask_index + 1
                        accepted_rotation = release_state.rotation_for_joint(
                            joint_command,
                        )
                        active_subtask_index = next_subtask_index
                        boundary_key = (
                            active_scene_choice,
                            (
                                snapshot.preset_task_name
                                if panel is not None
                                else "(no preset task)"
                            ),
                            int(active_subtask_index),
                            int(ManipulationPhase.PREGRASP),
                        )
                        rebase_rotation_subtask(
                            target,
                            rotational_tolerance,
                            release_state,
                            stage_key=boundary_key,
                            accepted_rotation=accepted_rotation,
                        )
                        instantaneous_rotation_target = accepted_rotation.copy()
                        if prepared_subtask is None:
                            preset_task.plan_next_subtask(data, target.position, target.rotation)
                        else:
                            preset_task.adopt_next_subtask_plan(prepared_subtask, data, target.position, target.rotation)
                            prepared_subtask = None
                            subtask_preparer.cancel()
                        task_name = snapshot.preset_task_name
                        planned_tolerances = resolve_planned_subtask_tolerances(
                            model,
                            preset_task.frames,
                            scene=active_scene_name,
                            task=task_name,
                            subtask_index_offset=active_subtask_index,
                        )
                        if (
                            len(planned_tolerances) != 1
                            or planned_tolerances[0].index
                            != active_subtask_index
                        ):
                            raise RuntimeError(
                                "Deferred preset segment must resolve exactly "
                                "one task-wide tolerance subtask"
                            )
                        resolved_subtask = planned_tolerances[0]
                        pregrasp_limits = snapshot.rotation_tolerance_rad
                        if panel is not None:
                            pregrasp_limits = (
                                panel.apply_runtime_tolerance_subtask(
                                    resolved_subtask,
                                    total_subtasks=max(
                                        active_subtask_index + 1,
                                        len(long_task_subtask_labels(task_name)),
                                    ),
                                )
                            )
                            snapshot = panel.tolerance_snapshot(snapshot)
                            resolved_subtask = replace(
                                resolved_subtask,
                                pregrasp=replace(
                                    resolved_subtask.pregrasp,
                                    target_position=snapshot.tolerance_target_position,
                                    target_rotation=snapshot.rotation_tolerance_target,
                                    tolerance_frame=snapshot.rotation_tolerance_frame,
                                    ranged_axes=~snapshot.active_dofs[3:],
                                ),
                            )
                            # The cached snapshot still describes the previous
                            # Release stage. Force the next physics step to
                            # read the newly activated Pre-grasp descriptor.
                            cached_snapshot = None
                        _activate_runtime_subtask_tolerance(
                            rotational_tolerance,
                            active_dofs,
                            resolved_subtask,
                            pregrasp_limits,
                            tolerance_enabled=snapshot.tolerance_enabled,
                            stage_key=boundary_key,
                        )
                    # Scripted plans are sampled at 10 Hz already. Stream one
                    # frame per controller tick; feedback-gating each frame
                    # can repeat it for up to 0.6 s under grasp load and turn
                    # a nominal 0.1 s step into a visible pause.
                    task_action = preset_task.trajectory_action()
                    if (
                        preset_evaluation_active
                        and task_action is not None
                        and not preset_task.active
                    ):
                        # Keep measuring through the complete 1 kHz execution
                        # of the last 10 Hz waypoint before freezing results.
                        preset_evaluation_remaining_s = TELEOP_DT
                replay_step = None if task_action is not None else trajectory_recorder.next_step()
                if task_action is not None:
                    gripper_closed = task_action.gripper_closed
                elif replay_step is not None:
                    gripper_closed = replay_step.gripper_closed
                phase_observation = simulation_backend.observe_manipulation_phase(gripper_closed)
                active_phase = phase_observation.phase
                active_subtask_index = phase_observation.subtask_index
                if panel is not None:
                    panel.set_gripper_phase(phase_observation)
                    snapshot = panel.tolerance_snapshot(snapshot)
                    active_dofs = snapshot.active_dofs.copy()
                    _configure_ui_tolerance(
                        rotational_tolerance, snapshot,
                        (active_scene_choice, snapshot.preset_task_name, active_subtask_index),
                    )
                    cached_snapshot = None
                else:
                    rotational_tolerance.active_stage = int(active_phase)
                release_key = (
                    active_scene_choice,
                    # Task selection is part of the semantic stage context.
                    # A task change can keep the same scene/stage numbers but
                    # must capture a fresh handoff snapshot.
                    (
                        snapshot.preset_task_name
                        if panel is not None
                        else "(no preset task)"
                    ),
                    # Subtasks are planned before execution, but each has an
                    # independent handoff-offset lifetime.
                    int(active_subtask_index),
                    int(rotational_tolerance.active_stage),
                )
                if release_state.stage_key != release_key:
                    previous_release_key = release_state.stage_key
                    new_subtask = bool(
                        previous_release_key is not None
                        and previous_release_key[:-1] != release_key[:-1]
                    )
                    accepted_rotation = release_state.rotation_for_joint(
                        joint_command,
                    )
                    if new_subtask:
                        rebase_rotation_subtask(
                            target,
                            rotational_tolerance,
                            release_state,
                            stage_key=release_key,
                            accepted_rotation=accepted_rotation,
                        )
                    else:
                        release_state.begin_stage(
                            release_key,
                            optimized_rotation=accepted_rotation,
                            nominal_rotation=target.rotation,
                        )
                if task_action is not None:
                    # Preset plans are compiled offline into base-frame
                    # increments. They advance the previous nominal
                    # expectation before this frame's solve.
                    target.position, target.rotation = advance_reference_pose(
                        target.position,
                        target.rotation,
                        task_action.value,
                        tolerance=rotational_tolerance,
                        dt=TELEOP_DT,
                    )
                    rotation_action = np.asarray(task_action.value[3:6], dtype=float)
                    task_gripper_closed = task_action.gripper_closed
                    gripper_closed = task_gripper_closed
                    # Keep the persistent UI command aligned with the last
                    # scripted state. Otherwise the first cycle after a preset
                    # finishes falls back to the stale pre-task UI value and
                    # can reopen a gripper even though no release action exists.
                    if (
                        panel is not None
                        and panel.closed != task_gripper_closed
                    ):
                        panel.set_gripper(task_gripper_closed)
                        cached_snapshot = None
                    evaluation_commanded = True
                elif replay_step is not None:
                    _apply_target_action(
                        target, replay_step.action, active_dofs,
                        rotational_tolerance,
                    )
                    rotation_action = np.asarray(replay_step.action[3:6], dtype=float)
                    gripper_closed = replay_step.gripper_closed
                    evaluation_commanded = True
                else:
                    raw_action = action_from_pressed(pressed, TELEOP_DT)
                    raw_action = action_in_base_frame(
                        raw_action, target.rotation, rotation_in_base,
                    )
                    _apply_target_action(
                        target, raw_action, active_dofs,
                        rotational_tolerance,
                    )
                    rotation_action = raw_action[3:6].copy()
                    evaluation_commanded = bool(np.any(raw_action))
                    current_q = data.qpos[handles.qpos_ids]
                    movement = float(np.linalg.norm(current_q - trajectory_previous_q))
                    trajectory_previous_q = current_q.copy()
                    trajectory_recorder.append(raw_action, gripper_closed, include=evaluation_commanded or movement >= IDLE_JOINT_MOVEMENT_THRESHOLD_RAD)
                # `target.rotation` was advanced from the action above. The
                # nominal release reference must not be overwritten by the
                # optimizer result.
                # The preset supplies only the Cartesian increment.
                # Keep the UI-selected Specific/Ranged axes in force during
                # execution so the same script supports the strict baseline
                # and the tolerance-release condition.
                cycle_position_target = target.position.copy()
                cycle_rotation_target = target.rotation.copy()
                if (
                    active_tolerance_frame_mode
                    == TOLERANCE_FRAME_MODE_STAGE_ENDPOINT
                ):
                    rotational_tolerance.update_tolerance_frame(
                        rotational_tolerance.raw_tolerance_frame,
                        stage_key=release_key,
                    )
                rotational_tolerance.update_tolerance_limits(
                    rotational_tolerance.raw_negative_limits,
                    rotational_tolerance.raw_positive_limits,
                    stage_key=release_key,
                )
                (
                    joint_target,
                    diagnostics,
                    rotation_tasks,
                    release_mask,
                    release_step,
                ) = solve_stage_relative_target(
                    ipopt_ik,
                    data,
                    target,
                    rotational_tolerance,
                    release_state,
                    active_dofs,
                    tolerance_frame_mode=active_tolerance_frame_mode,
                    future_actions=(
                        preset_stage_lookahead(
                            preset_task, 1.0, active_phase, args.preset_horizon,
                        ) if task_action is not None else ()
                    ),
                )
                if args.diagnostics_output is not None:
                    diagnostic_trace.append({
                        "cycle": len(diagnostic_trace),
                        "loss_profile": ipopt_ik.loss_profile,
                        "optimizer_backend": "ipopt",
                        "loss_weights": active_profile_loss_weights(),
                        "phase": int(active_phase),
                        "phase_name": active_phase.key,
                        "preset_active": bool(preset_task.active),
                        "tolerance_frame_mode": active_tolerance_frame_mode,
                        "status": str(diagnostics.status),
                        "publishable": bool(diagnostics.feasible),
                        "consecutive_failures": int(
                            diagnostics.consecutive_failures
                        ),
                        "converged": bool(diagnostics.converged),
                        "iterations": int(diagnostics.iterations),
                        "elapsed_ms": float(diagnostics.elapsed_ms),
                        "position_residual": float(
                            diagnostics.position_residual
                        ),
                        "rotation_residual": float(
                            diagnostics.strict_rotation_residual
                        ),
                        "inverse_condition_index": float(
                            diagnostics.inverse_condition_index
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
                        "conditioning_activation": float(
                            diagnostics.conditioning_activation
                        ),
                        "conditioning_region": (
                            diagnostics.conditioning_region
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
                            diagnostics.conditioning_gradient_cosine_half_step
                        ),
                        "gradient_cosine_double_step": (
                            diagnostics
                            .conditioning_gradient_cosine_double_step
                        ),
                        "joint_target": joint_target.tolist(),
                        "cartesian_target": target.position.tolist(),
                    })
                if (
                    preset_cycle
                    and simulation_backend.optimizer_should_abort(diagnostics)
                ):
                    cancel_preset("optimizer_failure")
                    preset_result_status = "failed: optimizer_failure"
                live_release_mask = release_mask.copy()
                live_release_loss_multipliers = ipopt_ik.release_loss_multipliers(
                    rotation_tasks,
                )
                live_nominal_action_ema = (
                    rotational_tolerance.nominal_action_ema
                )
                # The release controller owns the rotational command in every
                # phase.  A zero mask means all axes are strict for this
                # stage, but still uses the same stage-relative SO(3) path,
                # so grasp/release cannot fall back to target-relative
                # correction.
                optimized_pose = ipopt_ik.kinematics.evaluate(
                    joint_target,
                    include_manipulability=False,
                    include_link_points=False,
                )
                last_rotation_task_error[:] = 0.0
                # The current nominal rotation is the only release reference
                # in this controller.  Do not project the optimized pose
                # against the newly selected stage target:
                # a fixed stage-target chart creates a discontinuity when
                # entering Post-grasp (often an apparent +/-30 degree jump).
                # Telemetry error is the measured EE rotation relative to the
                # pose actually selected by this SQP cycle.  The nominal VLA
                # target remains in ``target.rotation`` for the next action
                # integration, but it is not the optimizer target and must
                # not be used as the live error reference.
                instantaneous_rotation_target = optimized_pose.rotation.copy()
                # Position never accepts FK error. Rotation accepts only the
                # legal optimized coordinates on explicitly ranged axes.
                target.position = cycle_position_target
                target.rotation = cycle_rotation_target.copy()
                evaluation_reference_goal_position = target.position.copy()
                evaluation_reference_goal_rotation = (
                    instantaneous_rotation_target.copy()
                )
                # Start each segment at the last commanded point, so a new
                # waypoint never introduces a discontinuity at 10 Hz.
                interpolation_start = joint_command.copy()
                interpolation_goal = joint_target.copy()
                interpolation_elapsed = 0.0
                evaluation_recorder.begin_control_interval(
                    evaluation_reference_start_position,
                    evaluation_reference_start_rotation,
                    evaluation_reference_goal_position,
                    evaluation_reference_goal_rotation,
                    joint_target,
                )
                teleop_accumulator -= TELEOP_DT

        active_evaluator = (
            preset_evaluator if preset_evaluator is not None else scene_evaluator
        )
        physics_step = simulation_backend.step_physics(
            tracker=joint_tracker,
            joint_command=joint_command,
            interpolation_start=interpolation_start,
            joint_target=interpolation_goal,
            interpolation_elapsed=interpolation_elapsed,
            gripper_ctrl=gripper_ctrl,
            gripper_closed=gripper_closed,
            gripper_speed=gripper_speed,
            speed_scale=1.0,
            evaluator=active_evaluator,
            frame_index=preset_evaluator_frame,
        )
        joint_command = physics_step.joint_command
        gripper_ctrl = physics_step.gripper_ctrl
        interpolation_elapsed = physics_step.interpolation_elapsed
        alpha = physics_step.interpolation_alpha
        if contact_markers is not None:
            contact_markers.observe(data)
        if preset_evaluator is not None and physics_step.aborted:
            collision = getattr(preset_evaluator, "collision", None)
            reason = (
                collision.category
                if collision is not None
                else "robot_safety_collision"
            )
            preset_task.cancel(reason)
            preset_result_status = f"failed: {reason}"
            stop_preset_evaluation(finalize_task=True)
        evaluation_summary = evaluation_recorder.sample_physics(
            data,
            alpha,
            commanded=evaluation_commanded,
        )
        gripper_phase_accumulator += physics_dt
        if gripper_phase_accumulator + 1e-12 >= TELEOP_DT:
            gripper_phase_accumulator %= TELEOP_DT
            if active_evaluator is not None:
                simulation_backend.observe_frame(
                    active_evaluator, gripper_closed, preset_evaluator_frame,
                )
                preset_evaluator_frame += 1

        if preset_evaluation_remaining_s is not None:
            preset_evaluation_remaining_s -= physics_dt
            if preset_evaluation_remaining_s <= 1e-12:
                stop_preset_evaluation(finalize_task=True)

        if (
            panel is not None and render_ui
            and time.monotonic() - last_ui_render_time >= TELEOP_DT
        ):
            last_ui_render_time = time.monotonic()
            position, rotation = ee_pose(data, handles)
            # Goal error bars track the current optimized reference target,
            # not the fixed stage target, and the rotational part is shown as
            # a target-minus-actual axis-angle vector.
            rotation_goal_error = rotation_vector(
                instantaneous_rotation_target @ rotation.T,
            )
            # ``tolerance_deg`` is optimizer-owned release in tolerance
            # coordinates and is intentionally masked to released axes.  It
            # must not also be used as the displayed pose error: strict axes
            # can have a real FK error even when their release amount is
            # exactly zero.
            if release_state.stage_key is None:
                tolerance_release = np.zeros(3)
                stage_release_error = np.zeros(3)
            else:
                telemetry_step = release_state.prepare_cycle(
                    target.rotation,
                    rotational_tolerance.frame,
                    live_release_mask,
                    -rotational_tolerance.negative_limits,
                    rotational_tolerance.positive_limits,
                )
                release_candidate = release_state.candidate(
                    joint_target,
                    telemetry_step,
                )
                tolerance_release = (
                    release_candidate.stage_relative_tolerance_release_rad.copy()
                )
                stage_release_error = (
                    release_candidate.stage_relative_error_fixed_xyz_rad.copy()
                )
            tolerance_deg = np.degrees(tolerance_release)
            goal_error = np.concatenate((target.position - position, rotation_goal_error))
            # Only released translations are omitted.  Ranged rotations remain
            # real tasks and their bounded error must stay visible.
            goal_error[:3][~active_dofs[:3]] = 0.0
            # Use the measured configuration, not the last optimized command.
            # Match the full, unscaled Jacobian used by rollout evaluation.
            actual_state = ipopt_ik.kinematics.evaluate(
                data.qpos[handles.qpos_ids],
                include_manipulability=False,
                include_link_points=False,
            )
            panel.telemetry(
                np.degrees(data.qpos[handles.qpos_ids]), position,
                rotation, goal_error, diagnostics,
                home_hold, tolerance_deg, active_dofs[3:], rotational_tolerance.intent_preference,
                # Total stage-relative release in the current tolerance chart.
                np.degrees(stage_release_error),
                live_release_loss_multipliers,
                live_nominal_action_ema,
                current_sigma_min=velocity_normalized_sigma_min(
                    actual_state.jacobian, handles.spec.joint_velocity_limits,
                ),
            )
            panel.evaluation(
                evaluation_summary,
                evaluation_recorder.recording,
            )
            panel.recording_state(evaluation_recorder.recording, trajectory_recorder.recording, trajectory_recorder.playing, trajectory_recorder.progress)
            panel.preset_task_status(displayed_preset_status())
            render_cameras(snapshot)
        return True

    if args.no_viewer:
        next_ui_poll = started
        while args.duration is None or time.monotonic() - started < args.duration:
            now = time.monotonic()
            refresh_ui = now >= next_ui_poll
            if refresh_ui:
                next_ui_poll = now + 1.0 / VIEWER_REFRESH_HZ
            if not step(poll_ui=refresh_ui, render_ui=refresh_ui):
                break
        if nominal_preparer is not None:
            nominal_preparer.close()
        if subtask_preparer is not None:
            subtask_preparer.close()
        if panel is not None:
            panel.close()
        if main_renderer is not None:
            main_renderer.close()
        if wrist_renderer is not None:
            wrist_renderer.close()
        if side_renderer is not None:
            side_renderer.close()
        write_diagnostic_trace()
        if requested_scene is not None or requested_robot is not None:
            return {
                "scene": requested_scene or active_scene_name,
                "scene_group": requested_scene_group or args.scene_group,
                "scene_choice": requested_scene_choice or args.scene_choice,
                "robot": requested_robot or robot_spec.uid,
            }
        return None

    if panel is not None:
        stop_requested = False
        while panel.running:
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            frame_started = time.monotonic()
            steps_per_frame, wall_frame_dt = playback_clock.frame(playback_speed_scale)
            for index in range(steps_per_frame):
                if not step(poll_ui=index == 0, render_ui=index == steps_per_frame - 1):
                    stop_requested = True
                    break
            if stop_requested:
                break
            remaining = wall_frame_dt - (time.monotonic() - frame_started)
            if remaining > 0.0:
                time.sleep(remaining)
        if nominal_preparer is not None:
            nominal_preparer.close()
        if subtask_preparer is not None:
            subtask_preparer.close()
        write_diagnostic_trace()
        panel.close()
        if main_renderer is not None:
            main_renderer.close()
        if wrist_renderer is not None:
            wrist_renderer.close()
        if side_renderer is not None:
            side_renderer.close()
        if requested_scene is not None or requested_robot is not None:
            return {
                "scene": requested_scene or active_scene_name,
                "scene_group": requested_scene_group or args.scene_group,
                "scene_choice": requested_scene_choice or args.scene_choice,
                "robot": requested_robot or robot_spec.uid,
            }
        return None

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        schedule_mujoco_viewer_placement()
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = front_camera_id
        reachability_view_key: tuple[bool, float, float] | None = None
        base_user_geom_count = 0

        def update_reachability_view() -> None:
            nonlocal reachability_view_key
            nonlocal boundary_surface
            nonlocal base_user_geom_count
            snapshot = cached_snapshot
            show = bool(
                snapshot is not None
                and snapshot.show_reachability_map
            )
            view_key = (
                show,
                1.0 if snapshot is None else snapshot.workspace_boundary_distance_min,
                1.0 if snapshot is None else snapshot.workspace_boundary_distance_max,
            )
            if view_key == reachability_view_key:
                return
            reachability_view_key = view_key
            if viewer.user_scn is None:
                base_user_geom_count = 0
                return
            if not show:
                with viewer.lock():
                    clear_reachability_scene(viewer.user_scn)
                    base_user_geom_count = 0
                viewer.clear_texts()
                if panel is not None:
                    panel.reachability_status("3D boundary hidden")
                return
            try:
                surface = load_boundary_surface()
                with viewer.lock():
                    marker_capacity = (
                        0
                        if contact_markers is None
                        else contact_markers.maximum_points
                    )
                    line_count = populate_workspace_boundary_surface_scene(
                        viewer.user_scn,
                        surface,
                        factor_lower=view_key[1],
                        factor_upper=view_key[2],
                        maximum_lines=min(
                            3000,
                            int(viewer.user_scn.maxgeom) - marker_capacity,
                        ),
                    )
                    base_user_geom_count = int(viewer.user_scn.ngeom)
                bounds = boundary_surface.search_bounds_xy_m
                viewer.set_texts((
                    None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"{robot_spec.label} workspace boundary ({line_count} lines)",
                    (
                        "cyan = IK boundary (factor 1) | green/orange = selected limits\n"
                        f"X {bounds[0, 0]:.2f}..{bounds[0, 1]:.2f}  "
                        f"Y {bounds[1, 0]:.2f}..{bounds[1, 1]:.2f}  "
                        f"factor {view_key[1]:.2f}..{view_key[2]:.2f}\n"
                        "each 0.1 factor = 5 cm signed normal offset"
                    ),
                ))
                if panel is not None:
                    panel.reachability_status(
                        f"2 mm IK boundary | factor {view_key[1]:.2f}..{view_key[2]:.2f}"
                    )
            except (FileNotFoundError, ValueError) as error:
                with viewer.lock():
                    clear_reachability_scene(viewer.user_scn)
                    base_user_geom_count = 0
                viewer.set_texts((
                    None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Workspace boundary unavailable", str(error),
                ))
                if panel is not None:
                    panel.reachability_status("3D boundary unavailable")

        stop_requested = False
        while viewer.is_running():
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            frame_started = time.monotonic()
            steps_per_frame, wall_frame_dt = playback_clock.frame(playback_speed_scale)
            for index in range(steps_per_frame):
                if not step(poll_ui=index == 0, render_ui=index == steps_per_frame - 1):
                    stop_requested = True
                    break
            if stop_requested:
                break
            update_reachability_view()
            if viewer.user_scn is not None and contact_markers is not None:
                with viewer.lock():
                    contact_markers.append_to_scene(
                        viewer.user_scn, base_user_geom_count,
                    )
            viewer.sync()
            remaining = wall_frame_dt - (time.monotonic() - frame_started)
            if remaining > 0.0:
                time.sleep(remaining)
    if nominal_preparer is not None:
        nominal_preparer.close()
    if subtask_preparer is not None:
        subtask_preparer.close()
    if panel is not None:
        panel.close()
    if main_renderer is not None:
        main_renderer.close()
    if wrist_renderer is not None:
        wrist_renderer.close()
    if side_renderer is not None:
        side_renderer.close()
    write_diagnostic_trace()
    if requested_scene is not None or requested_robot is not None:
        return {
            "scene": requested_scene or active_scene_name,
            "scene_group": requested_scene_group or args.scene_group,
            "scene_choice": requested_scene_choice or args.scene_choice,
            "robot": requested_robot or robot_spec.uid,
        }
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-robot MuJoCo teleoperation with stage-relative SQP.",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--auto-execute-preset", action="store_true",
        help="Queue EXECUTE on the first UI poll for reproducible diagnostics.",
    )
    parser.add_argument("--domain-seed", type=int, default=-1)
    parser.add_argument(
        "--custom-layout", type=Path, default=None,
        help=(
            "Apply a versioned creation-mode JSON layout instead of the "
            "initial seeded object layout. The scene and robot must match."
        ),
    )
    parser.add_argument(
        "--teleop-speed-scale", "--simulation-speed-scale",
        dest="teleop_speed_scale", type=float, default=5.0,
        help="Simulation playback relative to wall time; preserves motion and physics.",
    )
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--scene", choices=SceneManager.UI_NAMES, default=GEOMETRY_PLATE_BOX_UPRIGHT_SCENE, help="Load exactly one UI-visible object scene into the isolated worker.")
    parser.add_argument("--scene-group", choices=scene_groups(), default=TRAINING_TASK_GROUP, help="Initial first-level scene collection shown in the UI.")
    parser.add_argument("--scene-choice", default=None, help="Initial second-level logical scene selection.")
    parser.add_argument("--robot", choices=ROBOT_CHOICES, default="panda", help="Robot embodiment loaded by the isolated worker.")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--camera-adjust", choices=("none", "main", "wrist"), default="none", help="Enable keyboard camera calibration for the selected view; default: none.")
    parser.add_argument("--gripper-force-scale", type=float, default=DEFAULT_GRIPPER_FORCE_SCALE)
    parser.add_argument("--max-gripper-force", type=float, default=DEFAULT_MAX_GRIPPER_FORCE)
    parser.add_argument("--finger-friction", type=float, default=DEFAULT_FINGER_SLIDE_FRICTION)
    parser.add_argument("--physics-timestep", type=float, default=DEFAULT_MAX_PHYSICS_TIMESTEP)
    parser.add_argument("--object-clearance", type=float, default=0.05, help="Minimum XY surface gap between independently placed object groups, in metres.")
    parser.add_argument("--placement-difficulty", type=float, default=0.0, help="Home-orientation tabletop reachability difficulty in [0, 1].")
    parser.add_argument(
        "--reachability-map-dir", type=Path, default=None,
        help=(
            "Directory containing <robot>_table_home_v1.npz and "
            "<robot>_workspace_boundary_v2.npz."
        ),
    )
    parser.add_argument(
        "--ipopt-max-iterations",
        type=int,
        default=20,
        help="IPOPT iteration ceiling",
    )
    parser.add_argument(
        "--ipopt-max-time", type=float, default=0.095,
        help="Per-update IPOPT CPU-time budget in seconds; set 0 to disable.",
    )
    parser.add_argument(
        "--preset-horizon", type=int, choices=(1, 10), default=1,
        help="Preset IPOPT horizon: 1 for single step, 10 for one solve followed by execution of all Block knots.",
    )
    parser.add_argument(
        "--no-block-warm-start", action="store_true",
        help="Initialize every Block window from the last accepted joint command.",
    )
    parser.add_argument(
        "--sqp-loss",
        choices=("delta-q-squared-ema-release",),
        default="delta-q-squared-ema-release",
        help=(
            "Maintained single-step loss: Δq² + EMA release."
        ),
    )
    parser.add_argument("--sqp-motion-weight", type=float, default=1.0)
    parser.add_argument("--sqp-sigma-min-weight", type=float, default=1.0)
    parser.add_argument("--sqp-sigma-min-length-scale", type=float, default=1.0, choices=(1.0,), help="Deprecated compatibility option; sigma_min uses fixed velocity-capability normalization")
    parser.add_argument(
        "--sqp-ema-release-loss-weight", type=float, default=DEFAULT_EMA_RELEASE_LOSS_WEIGHT,
        help=(
            "Weight of the per-axis current-cycle extra-release cost scaled "
            "by nominal-action EMA; zero disables it."
        ),
    )
    return parser.parse_args()
