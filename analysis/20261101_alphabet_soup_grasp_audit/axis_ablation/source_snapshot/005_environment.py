"""Offscreen MuJoCo environment for traditional seven-dimensional VLA actions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import imageio_ffmpeg
import mujoco
import numpy as np

from ..config import (
    DEFAULT_FINGER_SLIDE_FRICTION,
    DEFAULT_GRIPPER_FORCE_SCALE,
    DEFAULT_MAX_GRIPPER_FORCE,
    GRIPPER_TRAVEL_TIME,
    TELEOP_DT,
)
from ..control.pose import TargetPose
from ..control.joint_tracker import JointReferenceTracker
from ..control.ipopt import IpoptIK, IpoptSettings
from ..control.task_space.objective import ObjectiveSettings
from ..geometry import (
    canonical_rotation_vector,
    continuous_rotation_vector,
    rotation_error_in_frame,
    rotation_matrix,
    rotation_vector,
)
from ..policies.base import PolicyStep
from ..policies.openpi.client import franka_mujoco_observation
from ..simulation.cameras import set_wrist_camera_calibration
from ..simulation.preset_runtime import (
    PresetPhysicsStep,
    PresetSimulationBackend,
)
from ..simulation.loader import load_scene_model
from ..simulation.layouts import apply_custom_layout, load_custom_layout
from ..simulation.robots import dataset_joint_state, get_robot_spec
from ..simulation.randomization.camera import FrontCameraRandomizer, focus_on_world_x, tabletop_corners
from ..simulation.randomization.light import LightRandomizer
from ..simulation.randomization.objects import ObjectLayoutLimits, ObjectLayoutRandomizer
from ..reachability.layout import (
    WORKSPACE_BOUNDARY_DISTANCE_MAX,
    WORKSPACE_BOUNDARY_DISTANCE_MIN,
    WORKSPACE_BOUNDARY_MODES,
    apply_blocks_difficulty,
    apply_training_workspace_boundary,
)
from ..simulation.randomization.table import TableRandomizer
from ..simulation.randomization.routing import RandomizationSeedRouting
from ..simulation.robot import (
    configure_contact_solver,
    configure_gripper,
    ee_pose,
    find_handles,
    set_gripper_fully_open,
    set_home,
)
from ..simulation.scenes import SceneManager
from ..tasks import create_task_evaluator
from ..tasks.evaluation.predicates import RobotTableClearanceRecorder
from ..tasks.catalog import pot_lid_task_variant_index
from ..tasks.phases import ManipulationPhase
from ..tasks.planning.types import TaskFrame
from ..tolerance.oracle import (
    PlannedTaskTolerance,
    TOLERANCE_MODE_DATASET_ORACLE,
    TOLERANCE_MODE_OFF,
    TOLERANCE_MODE_POLICY_PREDICTED,
    TOLERANCE_MODES,
    plan_dataset_oracle_tolerance_from_frames,
    precompute_dataset_oracle_tolerance,
)
from ..tolerance.runtime import (
    advance_reference_pose,
    rebase_rotation_subtask,
    solve_stage_relative_target,
)
from ..tolerance.annotations import box_tolerance_frame
from ..tolerance.frame import (
    TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
    TOLERANCE_FRAME_MODE_STAGE_ENDPOINT,
    TOLERANCE_FRAME_MODES,
    heading_code,
)
from ..tolerance.prediction import (
    DecodedStagePrediction,
    StagePredictionDecoder,
    ground_truth_tolerance_bounds_rad,
)
from ..tolerance.state import (
    ORIENTATION_REFERENCE_STAGE_RELATIVE,
    RotationalToleranceState,
)
from ..tolerance.release_state import RotationReleaseState, RotationReleaseStep
from ..tolerance.so3_preprocessor import SO3PreprocessorSettings
from .metrics import (
    BenchmarkEpisodeResult,
    MultiRateEvaluationRecorder,
)
from .runner import BenchmarkCase


_SETTLE_TIME_S = 1.0
_RANDOMIZED_ENVIRONMENT_SEED_FOR_NOMINAL_EPISODE = 0x42584D4B5


def _array(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _orientation_semantic_key(
    phase: ManipulationPhase,
    subtask_index: int,
) -> tuple[int, int]:
    """Identify the task stage that owns one release snapshot."""
    return int(phase), int(subtask_index)


def _render(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return np.ascontiguousarray(renderer.render(), dtype=np.uint8)


def _layout_json(layout: Any) -> dict[str, Any] | None:
    if layout is None:
        return None
    minimum_clearance = float(layout.minimum_clearance_m)
    return {
        "minimum_clearance_m": (
            minimum_clearance if math.isfinite(minimum_clearance) else None
        ),
        "placements": [
            {
                "bodies": list(item.names),
                "anchor_xy_m": _array(item.anchor_xy),
                "yaw_rad": float(item.yaw_rad),
            }
            for item in layout.placements
        ],
    }


@dataclass(frozen=True)
class BenchmarkEnvironmentConfig:
    output_dir: Path
    robot_uid: str = "panda"
    width: int = 640
    height: int = 480
    object_clearance_m: float = 0.05
    tolerance_mode: str = TOLERANCE_MODE_OFF
    tolerance_frame_mode: str = TOLERANCE_FRAME_MODE_STAGE_ENDPOINT
    dataset_task_id: str | None = None
    scene_variant_index: int | None = None
    custom_layout_path: Path | None = None
    environment_randomization: bool = False
    orientation_reference_mode: str = ORIENTATION_REFERENCE_STAGE_RELATIVE
    placement_difficulty: float = 0.0
    reachability_map_dir: Path | None = None
    workspace_boundary_mode: str | None = None
    workspace_boundary_distance_min: float = WORKSPACE_BOUNDARY_DISTANCE_MIN
    workspace_boundary_distance_max: float = WORKSPACE_BOUNDARY_DISTANCE_MAX
    terminate_on_evaluation: bool = True
    record_media: bool = True
    write_rollout: bool = True
    record_robot_table_clearance: bool = False
    robot_table_clearance_failure_m: float = 0.0
    objective_settings: ObjectiveSettings | None = None
    oracle_tolerance_limit_rad: float | None = None
    oracle_stage_tolerance_limits_rad: tuple[float, float] | None = None
    oracle_stage_tolerance_bounds_rad: tuple[
        tuple[float, float, float, float, float, float],
        tuple[float, float, float, float, float, float],
    ] | None = None
    oracle_pre_rx_limit_rad: float | None = None
    # Isolate one semantic subtask; the remaining oracle stages stay strict.
    oracle_release_subtask_index: int | None = None
    preset_horizon: int = 1
    block_warm_start: bool = True
    ipopt_settings: IpoptSettings = IpoptSettings()
    maximum_release_coordinate_step_rad: float | None = None
    so3_preprocessor: SO3PreprocessorSettings | None = None
    evaluator_factory: Callable[[mujoco.MjModel, mujoco.MjData], Any] | None = None

    def __post_init__(self) -> None:
        if self.so3_preprocessor is not None:
            if self.maximum_release_coordinate_step_rad is not None:
                raise ValueError('Select only one tolerance preprocessor')
            if self.preset_horizon != 1:
                raise ValueError('SO(3) preprocessing requires single-step control')
        budget = self.maximum_release_coordinate_step_rad
        if budget is not None and (not np.isfinite(budget) or not 0. < budget < np.pi):
            raise ValueError("Release coordinate step budget must be in (0, pi)")
        if budget is not None and self.preset_horizon != 1:
            raise ValueError("Coordinate governor currently requires single-step control")
        if self.oracle_release_subtask_index is not None and (
            not isinstance(self.oracle_release_subtask_index, int)
            or isinstance(self.oracle_release_subtask_index, bool)
            or self.oracle_release_subtask_index < 0
        ):
            raise ValueError("oracle_release_subtask_index must be a nonnegative integer")
        if self.oracle_pre_rx_limit_rad is not None and not 0.0 <= self.oracle_pre_rx_limit_rad <= math.pi:
            raise ValueError("oracle_pre_rx_limit_rad must be in [0, pi]")
        if self.preset_horizon < 1:
            raise ValueError("preset_horizon must be positive")
        get_robot_spec(self.robot_uid)
        if self.width <= 0 or self.height <= 0 or self.width * 3 != self.height * 4:
            raise ValueError("Benchmark RGB images must use a positive 4:3 resolution")
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 yuv420p requires even image dimensions")
        if self.tolerance_mode not in TOLERANCE_MODES:
            raise ValueError(f"Unknown tolerance mode: {self.tolerance_mode}")
        if self.tolerance_frame_mode not in TOLERANCE_FRAME_MODES:
            raise ValueError(
                f"Unknown tolerance-frame mode: {self.tolerance_frame_mode}",
            )
        if self.tolerance_mode == TOLERANCE_MODE_POLICY_PREDICTED:
            # Policy output contains bounds only. The shared backend constructs
            # the frame from each action-integrated nominal pose, as in the UI.
            object.__setattr__(
                self, "tolerance_frame_mode", TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
            )
        if self.scene_variant_index is not None and self.scene_variant_index < 0:
            raise ValueError("scene_variant_index must be non-negative")
        if (
            not math.isfinite(self.robot_table_clearance_failure_m)
            or self.robot_table_clearance_failure_m < 0.0
        ):
            raise ValueError(
                "robot_table_clearance_failure_m must be finite and non-negative"
            )
        if self.orientation_reference_mode != ORIENTATION_REFERENCE_STAGE_RELATIVE:
            raise ValueError(
                "Stage-relative release is the sole maintained benchmark "
                "reference mode",
            )
        if not 0.0 <= self.placement_difficulty <= 1.0:
            raise ValueError("placement_difficulty must be in [0, 1]")
        if (
            self.workspace_boundary_mode is not None
            and self.workspace_boundary_mode not in WORKSPACE_BOUNDARY_MODES
        ):
            raise ValueError(
                f"Unknown workspace-boundary mode: "
                f"{self.workspace_boundary_mode!r}",
            )
        if not (
            0.0 <= self.workspace_boundary_distance_min
            <= self.workspace_boundary_distance_max <= 2.0
        ):
            raise ValueError(
                "workspace-boundary distance factors must satisfy "
                "0 <= min <= max <= 2",
            )
        if self.workspace_boundary_mode is not None and self.placement_difficulty:
            raise ValueError(
                "workspace-boundary mode and legacy placement difficulty "
                "cannot be enabled together",
            )
        if (
            self.tolerance_mode == TOLERANCE_MODE_DATASET_ORACLE
            and not self.dataset_task_id
        ):
            raise ValueError("ground-truth tolerance mode requires dataset_task_id")
        if (
            self.oracle_tolerance_limit_rad is not None
            and not 0.0 <= self.oracle_tolerance_limit_rad <= math.pi
        ):
            raise ValueError("oracle_tolerance_limit_rad must be in [0, pi]")
        explicitly_bounded = self.oracle_stage_tolerance_bounds_rad is not None
        if self.oracle_stage_tolerance_limits_rad is not None:
            if self.oracle_tolerance_limit_rad is not None or explicitly_bounded:
                raise ValueError(
                    "oracle scalar, stage-specific limits, and directional bounds "
                    "are mutually exclusive"
                )
            limits = tuple(float(value) for value in self.oracle_stage_tolerance_limits_rad)
            if len(limits) != 2 or any(
                not 0.0 <= value <= math.pi for value in limits
            ):
                raise ValueError(
                    "oracle_stage_tolerance_limits_rad must contain PRE/POST limits in [0, pi]"
                )
        if explicitly_bounded:
            if self.oracle_tolerance_limit_rad is not None:
                raise ValueError(
                    "oracle scalar and directional tolerance bounds are mutually exclusive"
                )
            bounds = np.asarray(self.oracle_stage_tolerance_bounds_rad, dtype=float)
            if bounds.shape != (2, 6) or not np.all(np.isfinite(bounds)) or np.any(
                (bounds < 0.0) | (bounds > math.pi)
            ):
                raise ValueError(
                    "oracle_stage_tolerance_bounds_rad must contain finite PRE/POST "
                    "directional bounds in [0, pi]"
                )


class MuJoCoVLABenchmarkEnvironment:
    """Execute VLA actions through the maintained constrained-IPOPT runtime."""

    def __init__(self, config: BenchmarkEnvironmentConfig) -> None:
        self.config = config
        self.case: BenchmarkCase | None = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._main_renderer: mujoco.Renderer | None = None
        self._wrist_renderer: mujoco.Renderer | None = None
        self._main_writer: Any = None
        self._wrist_writer: Any = None
        self._closed = False
        self._final_result: BenchmarkEpisodeResult | None = None
        self._timed_out = False
        self._override_reason: str | None = None
        self._policy_latencies_ms: list[float] = []
        self._frames: list[dict[str, Any]] = []
        self._domain_randomization: dict[str, Any] = {}
        self._cached_observation: dict[str, Any] | None = None
        self._cached_raw: dict[str, Any] | None = None
        self._rotational_tolerance: RotationalToleranceState | None = None
        self._active_phase: ManipulationPhase | None = None
        self._active_subtask_index = 0
        self._rotation_release_state: RotationReleaseState | None = None
        self._prediction_decoder = StagePredictionDecoder()
        self._decoded_prediction: DecodedStagePrediction | None = None
        self._ground_truth_stage_plan: PlannedTaskTolerance | None = None
        self._ground_truth_tolerance_bounds: dict[
            ManipulationPhase, np.ndarray
        ] = {}
        self._ground_truth_unavailable_reason: str | None = None
        self.robot_table_clearance: RobotTableClearanceRecorder | None = None

    def reset(self, case: BenchmarkCase) -> dict[str, Any]:
        if self.case is not None:
            raise RuntimeError("Create one benchmark environment per episode")
        self.case = case
        self.config.output_dir.mkdir(parents=True, exist_ok=False)

        spec = get_robot_spec(self.config.robot_uid)
        model = load_scene_model(case.scene, self.config.robot_uid)
        handles = find_handles(model, spec)
        configure_contact_solver(model, 0.001)
        configure_gripper(
            model, handles, DEFAULT_GRIPPER_FORCE_SCALE,
            DEFAULT_MAX_GRIPPER_FORCE, DEFAULT_FINGER_SLIDE_FRICTION,
        )
        data = mujoco.MjData(model)
        set_home(model, data, handles)
        home_rotation_vector = rotation_vector(ee_pose(data, handles)[1])
        scene_manager = SceneManager(model, loaded_scene=case.scene)
        custom_layout = (
            load_custom_layout(self.config.custom_layout_path)
            if self.config.custom_layout_path is not None else None
        )
        fixed_variant_index = (
            self.config.scene_variant_index
            if self.config.scene_variant_index is not None
            else (
                custom_layout.variant_index
                if custom_layout is not None
                else pot_lid_task_variant_index(case.task)
            )
        )
        if fixed_variant_index is None:
            scene_manager.select_seeded_variant(case.seed)
        else:
            scene_manager.select_variant_index(fixed_variant_index)
        scene_manager.activate(data, case.scene)

        table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface")
        front_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_view")
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
        empty_workspace = case.scene == SceneManager.EMPTY_SCENE
        if min(front_id, wrist_id) < 0 or (table_id < 0 and not empty_workspace):
            raise RuntimeError("Benchmark scene is missing table or RGB cameras")

        seed_routing = RandomizationSeedRouting(
            case.seed, self.config.environment_randomization,
        )
        environment_seed = seed_routing.environment_seed
        if self.config.environment_randomization and case.seed == -1:
            # Seed -1 keeps the task/object layout nominal, but a benchmark
            # ratio may still select this episode for visual randomization.
            environment_seed = _RANDOMIZED_ENVIRONMENT_SEED_FOR_NOMINAL_EPISODE
        seed_routing_metadata = seed_routing.as_dict()
        if environment_seed != seed_routing.environment_seed:
            seed_routing_metadata.update({
                "environment_randomization_enabled": True,
                "environment_seed": environment_seed,
                "table_seed": environment_seed,
                "front_camera_seed": environment_seed,
                "lighting_seed": environment_seed,
            })
        if empty_workspace:
            # The maintained empty model has no work surface or task objects.
            # Do not index body -1 or fabricate table geometry for diagnostics.
            nominal_table_x = .5
            nominal_tabletop = None
            table_pose = TableRandomizer(np.zeros(3)).sample(-1)
            layout = None
            difficult_layout = None
        else:
            nominal_table_x = float(model.body_pos[table_id, 0])
            nominal_tabletop = tabletop_corners(model)
            table_pose = TableRandomizer(model.body_pos[table_id]).sample(
                environment_seed,
            )
            model.body_pos[table_id] = table_pose.position
            scene_manager.set_workspace_offset(data, table_pose.offset)
            layout = ObjectLayoutRandomizer(
                model, scene_manager,
                limits=ObjectLayoutLimits(minimum_clearance_m=self.config.object_clearance_m),
            ).apply(
                data,
                seed_routing.object_layout_seed,
                variant_index=fixed_variant_index,
            )
            difficult_layout = (
                apply_training_workspace_boundary(
                    model, data, scene_manager, handles, case.task,
                    self.config.workspace_boundary_mode,
                    seed_routing.object_layout_seed,
                    distance_factor_lower=(
                        self.config.workspace_boundary_distance_min
                    ),
                    distance_factor_upper=(
                        self.config.workspace_boundary_distance_max
                    ),
                    map_directory=self.config.reachability_map_dir,
                )
                if self.config.workspace_boundary_mode is not None
                else apply_blocks_difficulty(
                    model, data, scene_manager, handles, case.task,
                    self.config.placement_difficulty,
                    seed_routing.object_layout_seed,
                    map_directory=self.config.reachability_map_dir,
                )
            )
        if custom_layout is not None:
            apply_custom_layout(
                scene_manager,
                data,
                custom_layout,
                robot=self.config.robot_uid,
                restore_variant=False,
            )

        front_position = model.cam_pos[front_id].copy()
        front_quaternion = model.cam_quat[front_id].copy()
        focus = focus_on_world_x(
            front_position, front_quaternion, nominal_table_x,
        )
        camera_pose = FrontCameraRandomizer(
            front_position, front_quaternion, focus,
        ).sample(
            environment_seed, tabletop=tabletop_corners(model), reference_tabletop=nominal_tabletop,
            vertical_fov_deg=float(model.cam_fovy[front_id]),
            aspect_ratio=self.config.width / self.config.height,
        )
        model.cam_pos[front_id] = camera_pose.position
        model.cam_quat[front_id] = camera_pose.quaternion
        set_wrist_camera_calibration(
            model, wrist_id,
            float(spec.wrist_camera_trim_pos[0]),
            float(spec.wrist_camera_trim_pos[2]),
            spec.wrist_camera_trim_pitch_deg,
            spec.wrist_camera_pos, spec.wrist_camera_quat,
        )

        light_randomizer = LightRandomizer(model)
        lighting = light_randomizer.sample(environment_seed)
        light_randomizer.apply(model, lighting)
        set_gripper_fully_open(model, data, handles)
        simulation_backend = PresetSimulationBackend(model, data, handles)
        simulation_backend.settle(_SETTLE_TIME_S)

        self.model, self.data, self.handles = model, data, handles
        self.simulation_backend = simulation_backend
        if self.config.evaluator_factory is not None:
            self.evaluator = self.config.evaluator_factory(model, data)
        else:
            self.evaluator = create_task_evaluator(
                case.scene,
                case.task,
                model,
                data,
                robot_table_clearance_m=(
                    self.config.robot_table_clearance_failure_m
                ),
            )
        if self.config.record_robot_table_clearance:
            self.robot_table_clearance = RobotTableClearanceRecorder(model)
        position, rotation = ee_pose(data, handles)
        self._prepare_task_tolerance_ground_truth(
            home_rotation_vector=home_rotation_vector,
        )
        if self.config.tolerance_mode in (
            TOLERANCE_MODE_DATASET_ORACLE,
            TOLERANCE_MODE_POLICY_PREDICTED,
        ):
            self._rotational_tolerance = RotationalToleranceState()
        self.target = TargetPose(position.copy(), rotation.copy())
        self._instantaneous_rotation_target = rotation.copy()
        controller_handles = handles
        self.ik = IpoptIK(
            model, controller_handles,
            self.config.ipopt_settings,
            self.config.objective_settings,
            block_warm_start=self.config.block_warm_start,
        )
        self._rotation_release_state = RotationReleaseState(
            maximum_release_coordinate_step_rad=self.config.maximum_release_coordinate_step_rad,
            so3_preprocessor=self.config.so3_preprocessor,
            rotation_for_joint=lambda q: self.ik.kinematics.evaluate(
                q,
                include_manipulability=False,
                include_link_points=False,
            ).rotation,
        )
        self.active_dofs = np.ones(6, dtype=bool)
        self.joint_command = data.qpos[handles.qpos_ids].copy()
        self.joint_tracker = JointReferenceTracker(
            controller_handles.spec.arm_dof,
        )
        self.gripper_ctrl = spec.gripper_open_ctrl
        self.gripper_closed = False
        self.gripper_speed = abs(spec.gripper_open_ctrl - spec.gripper_closed_ctrl) / GRIPPER_TRAVEL_TIME
        self.physics_steps = max(1, int(round(TELEOP_DT / float(model.opt.timestep))))
        self.previous_rotvec = home_rotation_vector.copy()
        self.metrics = MultiRateEvaluationRecorder(model, handles, TELEOP_DT)
        self.metrics.start()

        if self.config.record_media:
            self._main_renderer = mujoco.Renderer(
                model, height=self.config.height, width=self.config.width,
            )
            self._wrist_renderer = mujoco.Renderer(
                model, height=self.config.height, width=self.config.width,
            )
            self._main_writer = imageio_ffmpeg.write_frames(
                str(self.config.output_dir / "main_rgb.mp4"),
                (self.config.width, self.config.height), fps=10,
                codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                quality=8, macro_block_size=1,
            )
            self._wrist_writer = imageio_ffmpeg.write_frames(
                str(self.config.output_dir / "wrist_rgb.mp4"),
                (self.config.width, self.config.height), fps=10,
                codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                quality=8, macro_block_size=1,
            )
            self._main_writer.send(None)
            self._wrist_writer.send(None)
        self._domain_randomization = {
            "seed_routing": seed_routing_metadata,
            "scene_variant_index": fixed_variant_index,
            "active_scene_bodies": list(scene_manager.active_body_names),
            "table_offset_m": _array(table_pose.offset),
            "front_camera_local_translation_m": _array(camera_pose.local_translation),
            "front_camera_local_angles_deg": _array(camera_pose.local_angles_deg),
            "front_camera_framing_angles_deg": _array(camera_pose.framing_angles_deg),
            "front_camera_tabletop_visible_fraction_before": camera_pose.tabletop_visible_fraction_before,
            "front_camera_tabletop_visible_fraction_after": camera_pose.tabletop_visible_fraction_after,
            "front_camera_tabletop_visible_fraction_reference": camera_pose.tabletop_visible_fraction_reference,
            "lighting_brightness": _array(lighting.brightness),
            "object_layout": _layout_json(layout),
            "custom_layout_path": (
                None
                if self.config.custom_layout_path is None
                else str(self.config.custom_layout_path)
            ),
            "reachability_guided_layout": difficult_layout,
        }
        return self._make_observation()

    def _prepare_task_tolerance_ground_truth(
        self,
        *,
        home_rotation_vector: np.ndarray,
    ) -> None:
        """Recreate the seeded preset endpoints and attach hard-coded bounds."""
        assert self.case is not None
        assert self.model is not None and self.data is not None
        assert self.handles is not None
        self._ground_truth_stage_plan = None
        self._ground_truth_tolerance_bounds = {}
        self._ground_truth_unavailable_reason = None

        dataset_task_id = self.config.dataset_task_id
        if not dataset_task_id:
            self._ground_truth_unavailable_reason = "dataset_task_id is unavailable"
            return
        if self.config.oracle_stage_tolerance_bounds_rad is not None:
            bounds = np.asarray(
                self.config.oracle_stage_tolerance_bounds_rad, dtype=float,
            )
            self._ground_truth_tolerance_bounds = {
                ManipulationPhase.PREGRASP: bounds[0].copy(),
                ManipulationPhase.POSTGRASP: bounds[1].copy(),
            }
            limits = None
        elif (
            self.config.oracle_tolerance_limit_rad is not None
            or self.config.oracle_stage_tolerance_limits_rad is not None
        ):
            limits = (
                (float(self.config.oracle_tolerance_limit_rad),) * 2
                if self.config.oracle_stage_tolerance_limits_rad is None
                else tuple(self.config.oracle_stage_tolerance_limits_rad)
            )
        else:
            limits = None
            try:
                for phase in (
                    ManipulationPhase.PREGRASP,
                    ManipulationPhase.POSTGRASP,
                ):
                    self._ground_truth_tolerance_bounds[phase] = (
                        ground_truth_tolerance_bounds_rad(
                            self.config.robot_uid,
                            str(dataset_task_id),
                            phase,
                        )
                    )
            except KeyError as error:
                self._ground_truth_tolerance_bounds = {}
                self._ground_truth_unavailable_reason = str(error)

        # The tolerance values do not affect preset endpoint generation. Zero
        # placeholders let target-pose telemetry remain available for tasks
        # that do not have an entry in the hard-coded v2 tolerance table.
        profiles = {
            phase.key: self._ground_truth_tolerance_bounds.get(
                phase, np.zeros(6, dtype=float),
            )
            for phase in (
                ManipulationPhase.PREGRASP,
                ManipulationPhase.POSTGRASP,
            )
        }
        try:
            self._ground_truth_stage_plan = precompute_dataset_oracle_tolerance(
                self.model,
                self.data,
                self.handles,
                scene=self.case.scene,
                task=self.case.task,
                profiles_rad=profiles,
                initial_rotation_vector=home_rotation_vector,
                stage_tolerance_limits_rad=limits,
                episode_seed=self.case.seed,
            )
            first = self._ground_truth_stage_plan
            self._ground_truth_tolerance_bounds = {
                ManipulationPhase.PREGRASP: (
                    first.pregrasp.rotation.rotation_tolerance_bounds_rad.copy()
                ),
                ManipulationPhase.POSTGRASP: (
                    first.postgrasp.rotation.rotation_tolerance_bounds_rad.copy()
                ),
            }
        except (RuntimeError, ValueError) as error:
            reason = f"{type(error).__name__}: {error}"
            self._ground_truth_unavailable_reason = (
                reason
                if self._ground_truth_unavailable_reason is None
                else f"{self._ground_truth_unavailable_reason}; {reason}"
            )

    def _task_tolerance_ground_truth_record(
        self,
        phase: ManipulationPhase,
    ) -> dict[str, Any] | None:
        """Return truth aligned with one Pre/Post policy frame.

        Repeated gripper cycles are retries of the same short task, not semantic
        subtask transitions. Every Pre-grasp/Post-grasp cycle therefore reuses
        the task's first oracle Pre-grasp/Post-grasp target pair.
        """
        if phase not in (
            ManipulationPhase.PREGRASP,
            ManipulationPhase.POSTGRASP,
        ):
            return None
        plan = self._ground_truth_stage_plan
        if plan is None:
            return None
        runtime_subtask_index = int(self._active_subtask_index)
        oracle_subtask_index = (
            min(runtime_subtask_index, len(plan.subtasks) - 1)
            if len(plan.subtasks) > 1 else 0
        )
        stage = plan.stable_stage(phase, oracle_subtask_index)

        target_rotation = rotation_matrix(
            stage.rotation.target_rotation_vector_base_rad,
        )
        target_rotation_6d = np.concatenate((
            target_rotation[:, 0], target_rotation[:, 1],
        ))
        frame_code = heading_code(target_rotation)
        bounds = stage.rotation.rotation_tolerance_bounds_rad.copy()
        if (
            self.config.oracle_release_subtask_index is not None
            and runtime_subtask_index != self.config.oracle_release_subtask_index
        ):
            bounds[:] = 0.0
        if phase is ManipulationPhase.PREGRASP and self.config.oracle_pre_rx_limit_rad is not None:
            bounds[:2] = np.minimum(bounds[:2], self.config.oracle_pre_rx_limit_rad)
        return {
            "phase": phase.key,
            "subtask_index": runtime_subtask_index,
            "oracle_subtask_index": oracle_subtask_index,
            "stage_target_pose": {
                "position_m": _array(stage.target_position_m),
                "rotation_6d": _array(target_rotation_6d),
                "rotation_matrix": _array(target_rotation),
                "rotation_vector_base_rad": _array(
                    stage.rotation.target_rotation_vector_base_rad,
                ),
            },
            "tolerance_frame_code": _array(frame_code),
            "rotation_tolerance_rad": (
                None if bounds is None else _array(bounds)
            ),
            "sources": {
                "stage_target_pose": "seeded_preset_demonstration_endpoint",
                "tolerance_frame_code": "seeded_preset_demonstration_endpoint",
                "rotation_tolerance": (
                    "declared_training_mask_limit"
                    if (
                        self.config.oracle_tolerance_limit_rad is not None
                        or self.config.oracle_stage_tolerance_limits_rad is not None
                        or self.config.oracle_stage_tolerance_bounds_rad is not None
                    )
                    else "hard_coded_v1_tolerance_summary_v2"
                ),
            },
        }

    def refresh_dataset_oracle_segment(
        self,
        frames: Sequence[TaskFrame],
        *,
        subtask_index: int,
    ) -> None:
        """Refresh oracle targets after a dynamically planned task segment."""
        if self.config.tolerance_mode != TOLERANCE_MODE_DATASET_ORACLE:
            return
        assert self.case is not None and self.model is not None
        explicit_bounds = self.config.oracle_stage_tolerance_bounds_rad
        limits = None
        profiles = None
        if explicit_bounds is not None:
            profiles = {
                ManipulationPhase.PREGRASP.key: np.asarray(
                    explicit_bounds[0], dtype=float,
                ),
                ManipulationPhase.POSTGRASP.key: np.asarray(
                    explicit_bounds[1], dtype=float,
                ),
            }
        else:
            limits = (
                (float(self.config.oracle_tolerance_limit_rad),) * 2
                if self.config.oracle_stage_tolerance_limits_rad is None
                and self.config.oracle_tolerance_limit_rad is not None
                else self.config.oracle_stage_tolerance_limits_rad
            )
        if limits is None and profiles is None:
            raise RuntimeError(
                "Dynamic oracle refresh currently requires an explicit "
                "uniform, per-stage, or directional tolerance limit",
            )
        if profiles is None:
            profiles = {
                ManipulationPhase.PREGRASP.key: np.zeros(6, dtype=float),
                ManipulationPhase.POSTGRASP.key: np.zeros(6, dtype=float),
            }
        self._ground_truth_stage_plan = plan_dataset_oracle_tolerance_from_frames(
            self.model,
            list(frames),
            scene=self.case.scene,
            task=self.case.task,
            profiles_rad=profiles,
            initial_rotation_vector=rotation_vector(self.target.rotation),
            stage_tolerance_limits_rad=(
                None if limits is None else tuple(limits)
            ),
            subtask_index_offset=int(subtask_index),
        )

    def _decode_policy_prediction(
        self,
        phase: ManipulationPhase,
        policy_step: PolicyStep,
    ) -> DecodedStagePrediction | None:
        """Decode six policy tolerance bounds only during Pre/Post stages."""
        self._decoded_prediction = None
        if not policy_step.has_predicted_tolerance:
            return None
        if phase not in (ManipulationPhase.PREGRASP, ManipulationPhase.POSTGRASP):
            return None
        assert policy_step.rotation_tolerance_rad is not None
        self._decoded_prediction = self._prediction_decoder.decode(
            phase,
            policy_step.rotation_tolerance_rad,
        )
        return self._decoded_prediction

    def _activate_tolerance_phase(
        self,
        phase: ManipulationPhase,
        decoded_prediction: DecodedStagePrediction | None,
        *,
        semantic_changed: bool = False,
    ) -> None:
        """Mirror the four-stage state transition used by interactive main."""
        state = self._rotational_tolerance
        if state is None:
            return
        if self.config.tolerance_mode in (
            TOLERANCE_MODE_POLICY_PREDICTED,
            TOLERANCE_MODE_DATASET_ORACLE,
        ):
            if (
                self.config.tolerance_mode == TOLERANCE_MODE_POLICY_PREDICTED
                and decoded_prediction is None
                and phase in (
                    ManipulationPhase.PREGRASP,
                    ManipulationPhase.POSTGRASP,
                )
            ):
                raise RuntimeError(
                    "policy tolerance mode requires 6D rotation_tolerance "
                    "alongside 7D actions from the OpenPI output transform"
                )
            stage = int(phase)
            if phase in (ManipulationPhase.GRASP, ManipulationPhase.RELEASE):
                state.active_stage = stage
                state.ranged[:] = False
            else:
                if self.config.tolerance_mode == TOLERANCE_MODE_DATASET_ORACLE:
                    ground_truth = self._task_tolerance_ground_truth_record(phase)
                    if (
                        ground_truth is None
                        or ground_truth["rotation_tolerance_rad"] is None
                    ):
                        reason = self._ground_truth_unavailable_reason or (
                            "no ground-truth stage is available for the active subtask"
                        )
                        raise RuntimeError(
                            "dataset-oracle tolerance requires a ground-truth "
                            f"stage target and tolerance: {reason}"
                        )
                    target_rotation = np.asarray(
                        ground_truth["stage_target_pose"]["rotation_matrix"],
                        dtype=float,
                    )
                    bounds = np.asarray(
                        ground_truth["rotation_tolerance_rad"], dtype=float,
                    )
                    descriptor_mask = (
                        (bounds[0::2] > 0.0) | (bounds[1::2] > 0.0)
                    )
                else:
                    assert decoded_prediction is not None
                    target_rotation = self.target.rotation.copy()
                    bounds = decoded_prediction.tolerance_bounds_rad
                    descriptor_mask = decoded_prediction.ranged_axes
                stage_key = _orientation_semantic_key(
                    phase, self._active_subtask_index,
                )
                tolerance_frame = box_tolerance_frame(target_rotation)
                state.configure_stage(
                    stage,
                    target_rotation,
                    tolerance_frame,
                    np.maximum(bounds[0::2], bounds[1::2]),
                    negative_limits=bounds[0::2],
                    positive_limits=bounds[1::2],
                    stage_key=stage_key,
                )
                state.ranged = descriptor_mask.copy()
            if phase is not self._active_phase or semantic_changed:
                # A phase transition must inherit the last accepted Cartesian
                # command.  ``target`` is the nominal planner branch and may
                # still point at the pre-release orientation; using it as the
                # Specific goal would immediately restore a released offset.
                # ``joint_command`` is the command currently accepted by the
                # interpolation/tracking loop, so its FK is the authoritative
                # handoff pose.
                accepted_state = self.ik.kinematics.evaluate(
                    self.joint_command,
                    include_manipulability=False,
                    include_link_points=False,
                )
                handoff_rotation = accepted_state.rotation
                self._instantaneous_rotation_target = handoff_rotation.copy()
                assert self._rotation_release_state is not None
                stage_key = _orientation_semantic_key(
                    phase, self._active_subtask_index,
                )
                if semantic_changed:
                    rebase_rotation_subtask(
                        self.target,
                        state,
                        self._rotation_release_state,
                        stage_key=stage_key,
                        accepted_rotation=handoff_rotation,
                    )
                else:
                    self._rotation_release_state.begin_stage(
                        stage_key,
                        optimized_rotation=handoff_rotation,
                        nominal_rotation=self.target.rotation,
                    )
            self._active_phase = phase
            return
        raise AssertionError("Unexpected tolerance mode")

    def _tolerance_frame_record(
        self,
        phase: ManipulationPhase,
        measured_rotation: np.ndarray,
        release_xyz: np.ndarray,
        release_mask: np.ndarray,
        release_step: RotationReleaseStep | None = None,
        command_rotation: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if self._rotational_tolerance is None:
            return {
                "mode": TOLERANCE_MODE_OFF,
                "ranged_axes": [False, False, False],
            }
        state = self._rotational_tolerance
        governor_record = None
        if release_step is not None and command_rotation is not None:
            coordinates = release_step.stage_relative_coordinates(command_rotation)
            baseline = release_step.effective_box_baseline_error
            budget = release_step.maximum_release_coordinate_step_rad
            eligible = release_step.governor_mask
            governor_record = {
                "maximum_step_rad": budget,
                "eligible_axes": None if eligible is None else eligible.tolist(),
                "carried_coordinates_rad": baseline.tolist(),
                "command_coordinates_rad": coordinates.tolist(),
                "extra_coordinate_change_rad": (coordinates - baseline).tolist(),
                "reference_rotation_matrix": release_step.stage_reference_rotation().tolist(),
                "carried_rotation_matrix": release_step.carried_rotation.tolist(),
                "effective_lower_limits_rad": release_step.effective_lower_limits.tolist(),
                "effective_upper_limits_rad": release_step.effective_upper_limits.tolist(),
                "strict_step_conflict": None if budget is None else (
                    (~eligible) & (np.abs(baseline) > budget)
                ).tolist(),
            }
        return {
            "coordinate_governor": governor_record,
            "so3_preprocessor": None if release_step is None or release_step.so3_result is None else {
                **release_step.so3_result.as_dict(),
                'budget_rad': self.config.so3_preprocessor.maximum_step_rad,
                'acceptance_mode': self.config.so3_preprocessor.acceptance_mode,
                'publication_tolerance_rad': self.config.so3_preprocessor.publication_tolerance_rad,
                'nominal_reference_step_rad': release_step.nominal_reference_step_rad,
                'previous_accepted_rotation': release_step.previous_accepted_rotation.tolist(),
                'semantic_eligible_axes': release_step.governor_mask.tolist(),
            },
            "mode": self.config.tolerance_mode,
            "frame_mode": self.config.tolerance_frame_mode,
            "parameter_source": (
                "policy_rotation_tolerance_raw_so3_preprocessed"
                if release_step is not None and release_step.so3_result is not None else
                "policy_rotation_tolerance_ema"
                if self.config.tolerance_mode
                == TOLERANCE_MODE_POLICY_PREDICTED
                else "hard_coded_v1_tolerance_summary_v2"
            ),
            "active_stage": state.active_stage,
            "phase": phase.key,
            "rotation_coordinate_convention": "fixed_axis_xyz_rpy",
            "rotation_composition": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "ranged_axes": np.asarray(release_mask, dtype=bool).tolist(),
            "stage_target_rotation_matrix": state.target.tolist(),
            "raw_tolerance_frame": state.raw_tolerance_frame.tolist(),
            "tolerance_frame_rotation_matrix": state.frame.tolist(),
            "rotation_tolerance_rad_raw": np.column_stack((
                state.raw_negative_limits, state.raw_positive_limits,
            )).reshape(6).tolist(),
            "rotation_tolerance_rad": np.column_stack((
                state.negative_limits, state.positive_limits,
            )).reshape(6).tolist(),
            # Full matrix-derived error for every axis.  Do not mask strict
            # axes here: a [0,0,1] Post-grasp mask releases only Rz, while
            # Rx/Ry may still visibly differ from the stage target.
            "actual_task_error_rad": rotation_error_in_frame(
                measured_rotation,
                state.target,
                state.frame,
            ).tolist(),
            "optimizer_release_error_rad": np.asarray(
                release_xyz, dtype=float,
            ).tolist(),
            "controller_target_error_rad": state.target_error(
                measured_rotation, self._instantaneous_rotation_target,
            ).tolist(),
            "nominal_task_error_rad": rotation_error_in_frame(
                measured_rotation, self.target.rotation, state.frame,
            ).tolist(),
            "counterfactual_release_rad":
                np.asarray(release_xyz, dtype=float).tolist(),
            "release_axes": np.asarray(release_mask, dtype=bool).tolist(),
            "intent_preference": state.intent_preference.tolist(),
            "nominal_action_ema": state.nominal_action_ema.tolist(),
        }

    def _make_observation(self) -> dict[str, Any]:
        assert self.data is not None and self.case is not None
        position, rotation = ee_pose(self.data, self.handles)
        rotvec = continuous_rotation_vector(rotation, self.previous_rotvec)
        self.previous_rotvec = rotvec
        fingers = self.simulation_backend.phase_classifier.finger_positions(self.data)
        if self.config.record_media:
            assert self._main_renderer is not None
            assert self._wrist_renderer is not None
            main_rgb = _render(self._main_renderer, self.data, "front_view")
            wrist_rgb = _render(self._wrist_renderer, self.data, "wrist_camera")
        else:
            shape = (self.config.height, self.config.width, 3)
            main_rgb = np.zeros(shape, dtype=np.uint8)
            wrist_rgb = np.zeros(shape, dtype=np.uint8)
        self._cached_raw = {
            "position": position,
            "rotation": rotation,
            "rotation_vector": rotvec,
            "fingers": fingers,
            "main_rgb": main_rgb,
            "wrist_rgb": wrist_rgb,
        }
        self._cached_observation = franka_mujoco_observation(
            position, rotvec, fingers, main_rgb, wrist_rgb, self.case.instruction,
        )
        return self._cached_observation

    def record_policy_latency(self, latency_ms: float) -> None:
        self._policy_latencies_ms.append(float(latency_ms))

    def mark_timeout(self) -> None:
        self._timed_out = True

    def step(self, action: PolicyStep | np.ndarray, *, future_actions: tuple[np.ndarray, ...] = ()) -> tuple[dict[str, Any], bool]:
        control_started = time.perf_counter()
        if self._cached_observation is None or self._cached_raw is None:
            raise RuntimeError("Benchmark environment has not been reset")
        if self._final_result is not None:
            raise RuntimeError("Benchmark episode is already finalized")
        policy_step = action if isinstance(action, PolicyStep) else PolicyStep(
            np.asarray(action, dtype=float),
        )
        action = np.asarray(policy_step.action, dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            self._override_reason = "invalid_action"
            return self._cached_observation, True

        assert self.model is not None and self.data is not None
        raw = self._cached_raw
        frame_index = len(self._frames)
        evaluation_start_position = self.target.position.copy()
        evaluation_start_rotation = self.target.rotation.copy()
        next_gripper_closed = bool(action[6] >= 0.0)
        phase_observation = self.simulation_backend.observe_manipulation_phase(next_gripper_closed)
        if phase_observation.subtask_changed:
            self._active_subtask_index = phase_observation.subtask_index
        phase = phase_observation.phase
        measured_rotation = np.asarray(raw["rotation"], dtype=float)
        decoded_prediction = self._decode_policy_prediction(
            phase, policy_step,
        )
        self._activate_tolerance_phase(
            phase,
            decoded_prediction,
            semantic_changed=phase_observation.subtask_changed,
        )
        release_mask = np.zeros(3, dtype=bool)
        if self._rotational_tolerance is None:
            self.target.position, self.target.rotation = advance_reference_pose(
                self.target.position, self.target.rotation, action,
            )
            self.active_dofs[:] = True
        else:
            state = self._rotational_tolerance
            self.target.position, self.target.rotation = advance_reference_pose(
                self.target.position,
                self.target.rotation,
                action,
                tolerance=state,
                dt=TELEOP_DT,
            )
            if self.config.tolerance_mode == TOLERANCE_MODE_POLICY_PREDICTED:
                # The policy predicts only bounds. The integrated VLA action
                # supplies the nominal reference and the backend derives its frame.
                state.target = self.target.rotation.copy()
        self.gripper_closed = next_gripper_closed
        nominal_target_position = self.target.position.copy()
        nominal_target_rotation = self.target.rotation.copy()
        release_step = None
        if self._rotational_tolerance is None:
            joint_target, diagnostics = self.ik.solve(
                self.data,
                self.target,
                self.active_dofs,
            )
        else:
            assert self._rotation_release_state is not None
            (
                joint_target,
                diagnostics,
                _,
                release_mask,
                release_step,
            ) = solve_stage_relative_target(
                self.ik,
                self.data,
                self.target,
                self._rotational_tolerance,
                self._rotation_release_state,
                self.active_dofs,
                tolerance_frame_mode=self.config.tolerance_frame_mode,
                future_actions=future_actions[:self.config.preset_horizon - 1],
            )
        optimized_pose = self.ik.kinematics.evaluate(
            joint_target,
            include_manipulability=False,
            include_link_points=False,
        )
        if release_step is not None:
            assert self._rotation_release_state is not None
            accepted_candidate = self._rotation_release_state.candidate(
                self._rotation_release_state.accepted_rotation,
                release_step,
            )
            release_xyz = (
                accepted_candidate.stage_relative_tolerance_release_rad.copy()
            )
            self._instantaneous_rotation_target = (
                self._rotation_release_state.accepted_rotation.copy()
            )
        else:
            release_xyz = np.zeros(3)
            self._instantaneous_rotation_target = optimized_pose.rotation.copy()
        evaluation_goal_position = self.target.position.copy()
        evaluation_goal_rotation = self._instantaneous_rotation_target.copy()
        evaluation_commanded = bool(np.linalg.norm(action[:6]) > 1e-12)
        # The next action remains integrated from the independent nominal
        # expectation. FK position error and every non-tolerance rotation
        # residual are absent.

        if self._main_writer is not None:
            self._main_writer.send(raw["main_rgb"])
        if self._wrist_writer is not None:
            self._wrist_writer.send(raw["wrist_rgb"])
        self._frames.append({
            "index": frame_index,
            "time_s": frame_index * TELEOP_DT,
            "video_frame_index": frame_index,
            "joint_state": {
                "position_rad": _array(dataset_joint_state(
                    self.data.qpos[self.handles.qpos_ids], self.handles.spec,
                )),
                "velocity_rad_s": _array(dataset_joint_state(
                    self.data.qvel[self.handles.dof_ids], self.handles.spec,
                )),
            },
            "ee_state": {
                "position_m": _array(raw["position"]),
                "rotation_vector_base_rad": _array(raw["rotation_vector"]),
            },
            "gripper_state": {
                "finger_positions_m": _array(raw["fingers"]),
                "aperture_m": float(np.sum(raw["fingers"])),
            },
            "action": _array(action),
            "policy_task_tolerance_prediction": (
                None if not policy_step.has_predicted_tolerance else {
                    "rotation_tolerance_rad_raw": _array(
                        policy_step.rotation_tolerance_rad,
                    ),
                    "rotation_tolerance_rad_clipped": (
                        None
                        if decoded_prediction is None
                        else _array(
                            decoded_prediction.clipped_tolerance_bounds_rad,
                        )
                    ),
                    "descriptor_filter": (
                        None
                        if decoded_prediction is None or self._rotational_tolerance is None
                        else {
                            "alpha": (
                                self._rotational_tolerance
                                .tolerance_prediction_ema_alpha
                            ),
                            "rotation_tolerance_rad": _array(
                                np.column_stack((
                                    self._rotational_tolerance.negative_limits,
                                    self._rotational_tolerance.positive_limits,
                                )).reshape(6),
                            ),
                        }
                    ),
                }
            ),
            "task_tolerance_ground_truth": (
                self._task_tolerance_ground_truth_record(phase)
            ),
            "phase": phase.key,
            "subtask_index": int(self._active_subtask_index),
            "vla_nominal_target_pose": {
                "position_m": _array(nominal_target_position),
                "rotation_vector_base_rad": _array(
                    continuous_rotation_vector(
                        nominal_target_rotation, raw["rotation_vector"],
                    )
                ),
            },
            "controller_target_pose": {
                "position_m": _array(nominal_target_position),
                "rotation_vector_base_rad": _array(
                    continuous_rotation_vector(
                        self._instantaneous_rotation_target,
                        raw["rotation_vector"],
                    )
                ),
            },
            "optimizer": {
                "backend": "ipopt",
                "joint_target_rad": _array(joint_target),
                "iterations": int(diagnostics.iterations),
                "cost": float(diagnostics.cost),
                "elapsed_ms": float(diagnostics.elapsed_ms),
                'control_wall_ms': diagnostics.control_wall_ms,
                'prepublication_rejection_reason': diagnostics.prepublication_rejection_reason,
                'candidate_rotation_step_rad': diagnostics.candidate_rotation_step_rad,
                'so3_budget_exceeded': diagnostics.so3_budget_exceeded,
                "publishable": bool(diagnostics.feasible),
                "controller_mode": diagnostics.controller_mode,
                "block_window_feasible": diagnostics.block_window_feasible,
                "block_first_knot_feasible": diagnostics.block_first_knot_feasible,
                "block_warm_start_used": diagnostics.block_warm_start_used,
                "block_seed_source": diagnostics.block_seed_source,
                "horizon": diagnostics.block_horizon,
                "solver_called": diagnostics.solver_called,
                "block_knot_index": diagnostics.block_knot_index,
                "position_residual": float(diagnostics.position_residual),
                "strict_rotation_residual": float(
                    diagnostics.strict_rotation_residual,
                ),
                "manipulability": float(diagnostics.manipulability or 0.0),
                "sigma_min_raw": float(diagnostics.sigma_min_raw),
                "sigma_min_normalized": float(
                    diagnostics.sigma_min_normalized,
                ),
                "inverse_condition_index": float(
                    diagnostics.inverse_condition_index,
                ),
                "status": str(getattr(diagnostics, "status", "")),
                "effective_constraint_violation": float(
                    diagnostics.effective_constraint_violation,
                ),
                "task_costs": {
                    name: float(value)
                    for name, value in diagnostics.task_costs.items()
                },
                "conditioning": {
                    "inverse_condition_index": float(
                        diagnostics.inverse_condition_index,
                    ),
                    "activation": float(diagnostics.conditioning_activation),
                    "penalty": float(diagnostics.conditioning_penalty),
                    "region": str(diagnostics.conditioning_region),
                },
                "objective_gradient_norms": {
                    name: float(value)
                    for name, value in diagnostics.objective_gradient_norms.items()
                },
                "objective_tangent_gradient_norms": {
                    name: float(value)
                    for name, value in (
                        diagnostics.objective_tangent_gradient_norms.items()
                    )
                },
                "objective_gradient_cancellation_ratio": float(
                    diagnostics.objective_gradient_cancellation_ratio,
                ),
                "objective_tangent_gradient_cancellation_ratio": float(
                    diagnostics.objective_tangent_gradient_cancellation_ratio,
                ),
                "predicted_tolerance_violation_rad": (
                    None
                    if diagnostics.predicted_tolerance_violation_rad is None
                    else _array(
                        diagnostics.predicted_tolerance_violation_rad,
                    )
                ),
            },
            "tolerance": self._tolerance_frame_record(
                phase, measured_rotation, release_xyz, release_mask,
                release_step, optimized_pose.rotation,
            ),
            "orientation_reference": {
                "mode": self.config.orientation_reference_mode,
            },
        })

        self.metrics.begin_control_interval(
            evaluation_start_position,
            evaluation_start_rotation,
            evaluation_goal_position,
            evaluation_goal_rotation,
            joint_target,
            interval_index=frame_index,
        )
        def observe_physics(step: PresetPhysicsStep) -> None:
            self.metrics.sample_physics(
                self.data,
                step.interpolation_alpha,
                commanded=evaluation_commanded,
            )
            if self.robot_table_clearance is not None:
                self.robot_table_clearance.sample(
                    self.data,
                    phase=phase.key,
                    frame_index=frame_index,
                )

        interval = self.simulation_backend.run_control_interval(
            tracker=self.joint_tracker,
            joint_command=self.joint_command,
            joint_target=joint_target,
            gripper_ctrl=self.gripper_ctrl,
            gripper_closed=self.gripper_closed,
            gripper_speed=self.gripper_speed,
            speed_scale=1.0,
            physics_steps=self.physics_steps,
            evaluator=self.evaluator,
            frame_index=frame_index,
            physics_observer=observe_physics,
        )
        self.joint_command = interval.joint_command
        self.gripper_ctrl = interval.gripper_ctrl
        # Task evaluators own both stable success semantics and the existing
        # early-failure policy (for example prohibited robot collisions).
        # Stop as soon as either terminal condition becomes true so a VLA is
        # not asked for actions after the task has already finished.
        optimizer_aborted = self.simulation_backend.optimizer_should_abort(diagnostics)
        if optimizer_aborted:
            self._override_reason = "optimizer_failure"
        terminated = optimizer_aborted or bool(
            self.config.terminate_on_evaluation
            and (self.evaluator.success or self.evaluator.should_abort)
        )
        if terminated:
            self._frames[-1]['control_chain_wall_ms'] = (time.perf_counter()-control_started)*1000.
            return self._cached_observation, True
        observation = self._make_observation()
        self._frames[-1]['control_chain_wall_ms'] = (time.perf_counter()-control_started)*1000.
        return observation, False

    def result(self) -> BenchmarkEpisodeResult:
        if self._final_result is not None:
            return self._final_result
        if self.case is None:
            raise RuntimeError("Benchmark environment has not been reset")
        evaluation = self.evaluator.result(True, self._timed_out)
        if self._override_reason is not None:
            evaluation = {
                **evaluation,
                "success": False,
                "termination_reason": self._override_reason,
            }
        collision = evaluation.get("collision")
        latencies = np.asarray(self._policy_latencies_ms, dtype=float)
        mean_latency = float(np.mean(latencies)) if latencies.size else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if latencies.size else 0.0
        self._final_result = BenchmarkEpisodeResult(
            scene=self.case.scene,
            task=self.case.task,
            seed=self.case.seed,
            success=bool(evaluation["success"]),
            termination_reason=str(evaluation["termination_reason"]),
            collision_count=int(collision is not None),
            elapsed_s=len(self._frames) * TELEOP_DT,
            frames=len(self._frames),
            trajectory=self.metrics.summary.actual_1khz,
            optimized_10hz_trajectory=self.metrics.summary.optimized_10hz,
            instruction=self.case.instruction,
            policy_calls=len(self._policy_latencies_ms),
            mean_policy_latency_ms=mean_latency,
            p95_policy_latency_ms=p95_latency,
        )
        self._close_media()
        payload = {
            "episode": {
                "scene": self.case.scene,
                "robot": {
                    "uid": self.handles.spec.uid,
                    "name": self.handles.spec.label,
                    "arm_dof": self.handles.spec.arm_dof,
                },
                "task": self.case.task,
                "instruction": self.case.instruction,
                "seed": self.case.seed,
                "frequency_hz": 1.0 / TELEOP_DT,
                "evaluation": {
                    "feedback_frequency_hz":
                        1.0 / float(self.model.opt.timestep),
                    "joint_state_source": "MuJoCo qpos after each physics step",
                    "cartesian_reference":
                        "10 Hz controller target interpolated at physics rate",
                },
                "action_format": {
                    "translation": "base-frame delta XYZ in metres",
                    "rotation": "base-frame rotation-vector delta RxRyRz in radians; left-multiplied",
                    "gripper": "negative open, non-negative close",
                },
                "controller": "TaskTol stage-relative constrained IPOPT",
                "optimizer_backend": "ipopt",
                "tolerance_mode": self.config.tolerance_mode,
                "tolerance_frame_mode": self.config.tolerance_frame_mode,
                "rotation_tolerance_coordinate_convention": (
                    "fixed_axis_xyz_rpy"
                ),
                "rotation_tolerance_composition": (
                    "Rz(yaw) @ Ry(pitch) @ Rx(roll)"
                ),
                "ground_truth_tolerance_source": (
                    "hard_coded_v1_tolerance_summary_v2"
                    if (
                        self.config.tolerance_mode
                        == TOLERANCE_MODE_DATASET_ORACLE
                        and self._ground_truth_tolerance_bounds
                    )
                    else None
                ),
                "task_tolerance_ground_truth": {
                    "stage_target_pose_source": (
                        "seeded_preset_demonstration_endpoint"
                        if self._ground_truth_stage_plan is not None
                        else None
                    ),
                    "rotation_tolerance_source": (
                        "hard_coded_v1_tolerance_summary_v2"
                        if self._ground_truth_tolerance_bounds
                        else None
                    ),
                    "unavailable_reason": self._ground_truth_unavailable_reason,
                },
                "domain_randomization": self._domain_randomization,
                "result": evaluation,
                "policy_timing": {
                    "calls": len(self._policy_latencies_ms),
                    "mean_ms": mean_latency,
                    "p95_ms": p95_latency,
                },
            },
            # Keep the legacy field as raw physical feedback while exposing
            # all aligned domains for new analysis consumers.
            "trajectory_metrics": self.metrics.summary.actual_1khz.as_dict(),
            "trajectory_metric_domains": self.metrics.summary.as_dict(),
            "rollout": self._frames,
        }
        if self.config.write_rollout:
            (self.config.output_dir / "rollout.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        return self._final_result

    def _close_media(self) -> None:
        if self._closed:
            return
        self._closed = True
        for writer in (self._main_writer, self._wrist_writer):
            if writer is not None:
                writer.close()
        for renderer in (self._main_renderer, self._wrist_renderer):
            if renderer is not None:
                renderer.close()

    def close(self) -> None:
        self._close_media()
