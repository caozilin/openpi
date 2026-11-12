"""Headless MuJoCo RGB + JSON collection for preset manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

import mujoco
import numpy as np
import imageio_ffmpeg

from ..control.ipopt import IpoptIK, IpoptSettings
from ..control.task_space.objective import ObjectiveSettings
from ..tasks import create_task_evaluator, registered_tasks
from ..simulation.cameras import set_wrist_camera_calibration
from ..simulation.preset_runtime import PresetSimulationBackend
from ..simulation.randomization.camera import FrontCameraRandomizer, focus_on_world_x, tabletop_corners
from ..config import (
    DEFAULT_FINGER_SLIDE_FRICTION,
    DEFAULT_GRIPPER_FORCE_SCALE,
    DEFAULT_MAX_GRIPPER_FORCE,
    GRIPPER_TRAVEL_TIME,
    TELEOP_DT,
)
from ..control.pose import TargetPose
from ..control.joint_tracker import (
    JointReferenceTracker,
    JointReferenceTrackerSettings,
)
from ..policies.openpi.clientlib import image_tools as openpi_image_tools
from ..simulation.randomization.light import LightRandomizer
from ..geometry import (
    continuous_rotation_vector,
    rotation_matrix,
    rotation_vector,
)
from ..simulation.robot import configure_contact_solver, configure_gripper, ee_pose, find_handles, set_gripper_fully_open, set_home
from ..simulation.randomization.objects import ObjectLayoutLimits, ObjectLayoutRandomizer
from ..tasks.catalog import (
    BLOCK_TO_PLATE_TASK,
    CLOSE_CYLINDRICAL_POT_LID_TASK,
    CLOSE_HANDLE_POT_LID_TASK,
    CUBE_TO_PLATE_TASK,
    FRUIT_RECEPTACLE_TASKS,
    OPEN_CYLINDRICAL_POT_LID_TASK,
    OPEN_HANDLE_POT_LID_TASK,
    ROTATE_KNOB_TASK,
    STACK_COLORED_BLOCKS_TASK,
    pot_lid_task_variant_index,
)
from ..tasks.planning.presets import PresetTaskExecutor
from ..tolerance.runtime import advance_reference_pose
from ..simulation.loader import load_scene_model
from ..simulation.robots import (
    DATASET_JOINT_STATE_DIMENSION,
    dataset_joint_state,
    get_robot_spec,
)
from ..simulation.scenes import (
    CLOSE_CYLINDRICAL_POT_LID_SCENE,
    CLOSE_HANDLE_POT_LID_SCENE,
    OPEN_CYLINDRICAL_POT_LID_SCENE,
    OPEN_HANDLE_POT_LID_SCENE,
    SceneManager,
)
from ..simulation.randomization.table import TableRandomizer
from ..simulation.randomization.routing import RandomizationSeedRouting
from ..reachability.layout import apply_blocks_difficulty
from ..tolerance.annotations import (
    active_pot_lid_stage_rotation_masks,
)
from .tolerance_labels import DATASET_SCHEMA_VERSION, tolerance_label_metadata
from .instructions import (
    TASK_LANGUAGE,
    load_or_create_instructions,
    resolve_episode_instruction,
)
from .frame_filter import (
    frame_filter_metadata,
    retain_source_frame,
)
from .seeding import sample_batch_seeds


DATASET_RELEASE_VERSION = "5.0"
GENERATOR_VERSION = "4.0.0"
_PRE_RECORD_SETTLE_TIME_S = 1.0
_DATASET_SQP_SETTINGS = IpoptSettings(
    max_iterations=20,
    max_cpu_time_s=1.0e6,
)


def _initialize_collection_worker() -> None:
    """Parallelize episodes, not both episodes and nominal candidates."""
    os.environ["TASKTOL_NOMINAL_CANDIDATE_WORKERS"] = "1"


OPENPI_IMAGE_WIDTH = 224
OPENPI_IMAGE_HEIGHT = 224
_OPEN_POT_LID_SCENES = {
    "Open Pot Lid",
    OPEN_CYLINDRICAL_POT_LID_SCENE,
    OPEN_HANDLE_POT_LID_SCENE,
}
_CLOSE_POT_LID_SCENES = {
    "Close Pot Lid",
    CLOSE_CYLINDRICAL_POT_LID_SCENE,
    CLOSE_HANDLE_POT_LID_SCENE,
}
_POT_LID_SCENES = _OPEN_POT_LID_SCENES | _CLOSE_POT_LID_SCENES


def _progress_bar(current: int, total: int, width: int = 22) -> str:
    """Render a fixed-width terminal bar without adding a UI dependency."""
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    filled = int(width * current / total)
    return f"[{'=' * filled}{' ' * (width - filled)}] {current}/{total}"


class _CollectionProgress:
    """One-line batch and current-plan progress for PowerShell terminals."""

    def __init__(self, episode_total: int, *, enabled: bool = True) -> None:
        self.episode_total = episode_total
        self.enabled = enabled
        self._last_width = 0

    def update(
        self,
        episode_completed: int,
        episode_number: int,
        seed: int,
        frame_count: int,
        plan_current: int,
        plan_total: int,
    ) -> None:
        if not self.enabled:
            return
        line = (
            f"Batch {_progress_bar(episode_completed, self.episode_total)}  "
            f"Episode {episode_number}/{self.episode_total} seed={seed}  "
            f"Plan {_progress_bar(plan_current, plan_total)}  frames={frame_count}"
        )
        padding = " " * max(0, self._last_width - len(line))
        print(f"\r{line}{padding}", end="", file=sys.stderr, flush=True)
        self._last_width = len(line)

    def newline(self) -> None:
        if self._last_width:
            print(file=sys.stderr, flush=True)
            self._last_width = 0

@dataclass(frozen=True)
class CollectionConfig:
    robot_uid: str = "panda"
    scene: str = "Blocks"
    task: str = BLOCK_TO_PLATE_TASK
    batch_size: int = 5
    output_dir: Path | None = None
    width: int = 640
    height: int = 480
    maximum_frames: int = 600
    object_clearance_m: float = 0.05
    mode: str = "overwrite"
    maximum_attempts: int | None = None
    workers: int = 1
    environment_randomization_probability: float = 0.0
    placement_difficulty: float = 0.0
    reachability_map_dir: Path | None = None
    variant_indices: tuple[int, ...] | None = None
    # None preserves the historical -1 + random seed selection. An integer
    # selects consecutive nonnegative seeds, including retry candidates.
    sequential_seed_start: int | None = None

    def __post_init__(self) -> None:
        get_robot_spec(self.robot_uid)
        if self.output_dir is None:
            default_directory = {
                BLOCK_TO_PLATE_TASK: "blocks_blue_block_to_plate",
                CUBE_TO_PLATE_TASK: "blocks_cube_to_plate",
            }.get(
                self.task,
                "".join(
                    character.lower() if character.isalnum() else "_"
                    for character in self.task
                ).strip("_"),
            )
            object.__setattr__(
                self,
                "output_dir",
                Path("datasets") / self.robot_uid / default_directory,
            )
        else:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.width <= 0 or self.height <= 0 or self.width * 3 != self.height * 4:
            raise ValueError("RGB images must use a positive 4:3 resolution")
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 yuv420p requires even image dimensions")
        if self.maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")
        if (
            (self.scene, self.task) not in registered_tasks()
            or self.task not in PresetTaskExecutor.task_names(self.scene)
        ):
            raise ValueError(
                f"The collector requires a registered task, got "
                f"scene={self.scene!r}, task={self.task!r}"
            )
        if self.mode not in {"overwrite", "append"}:
            raise ValueError("mode must be 'overwrite' or 'append'")
        if self.maximum_attempts is not None and self.maximum_attempts < self.batch_size:
            raise ValueError("maximum_attempts must be at least batch_size")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.sequential_seed_start is not None and (
            type(self.sequential_seed_start) is not int or self.sequential_seed_start < 0
        ):
            raise ValueError("sequential_seed_start must be a nonnegative integer")
        if not 0.0 <= self.environment_randomization_probability <= 1.0:
            raise ValueError("environment_randomization_probability must be in [0, 1]")
        if not 0.0 <= self.placement_difficulty <= 1.0:
            raise ValueError("placement_difficulty must be in [0, 1]")
        if self.reachability_map_dir is not None:
            object.__setattr__(self, "reachability_map_dir", Path(self.reachability_map_dir))
        if self.variant_indices is not None:
            indices = tuple(int(index) for index in self.variant_indices)
            if not indices or any(index < 0 for index in indices):
                raise ValueError("variant_indices must contain non-negative indices")
            object.__setattr__(self, "variant_indices", indices)


def _json_array(value: np.ndarray) -> list[Any]:
    return np.asarray(value).tolist()


def _pose_json(position: np.ndarray, rotation_vector_base: np.ndarray) -> dict[str, Any]:
    return {
        "position_m": _json_array(position),
        "rotation_vector_base_rad": _json_array(rotation_vector_base),
    }


def _maximum_attempts(config: CollectionConfig) -> int:
    """Resolve the total-attempt cap, including a default retry reserve."""
    if config.maximum_attempts is not None:
        return int(config.maximum_attempts)
    retry_budget = max(10, math.ceil(0.30 * config.batch_size))
    return config.batch_size + retry_budget


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON file atomically so interruption cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n",
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as temporary:
        temporary.write(encoded)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _episode_summary_from_directory(directory: Path) -> dict[str, Any]:
    """Recover one successful manifest entry from an episode directory."""
    trajectory_path = directory / "trajectory.json"
    for required in (trajectory_path, directory / "main_rgb.mp4", directory / "wrist_rgb.mp4"):
        if not required.is_file():
            raise RuntimeError(f"Incomplete episode directory: missing {required}")
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    episode = payload["episode"]
    result = episode["result"]
    if not bool(result.get("success")):
        raise RuntimeError(f"Non-success episode found outside failed_episodes: {directory}")
    frames = int(episode["frame_count"])
    if frames != len(payload["trajectory"]):
        raise RuntimeError(f"Frame count mismatch while recovering {directory}")
    frame_filter = episode.get("frame_filter", {})
    source_frames = int(frame_filter.get("source_frame_count", frames))
    return {
        "episode_index": int(episode["episode_index"]),
        "seed": int(episode["seed"]),
        "path": directory.name,
        "frames": frames,
        "source_frames": source_frames,
        "removed_frames": source_frames - frames,
        "instruction_index": int(episode["instruction_index"]),
        "instruction": str(episode["instruction"]),
        "success": True,
        "termination_reason": str(result["termination_reason"]),
    }


def _recover_episode_manifest_entries(root: Path) -> list[dict[str, Any]]:
    """Rebuild successful entries after an interrupted pre-manifest batch."""
    entries = [
        _episode_summary_from_directory(directory)
        for directory in sorted(root.glob("episode_*_seed_*"))
        if directory.is_dir()
    ]
    indices = [int(entry["episode_index"]) for entry in entries]
    if len(indices) != len(set(indices)):
        raise RuntimeError("Cannot recover dataset with duplicate episode indices")
    return sorted(entries, key=lambda entry: int(entry["episode_index"]))


def _next_episode_indices(
    existing_episodes: list[dict[str, Any]], count: int,
) -> list[int]:
    """Fill index gaps first, then allocate new contiguous indices."""
    used = {int(episode["episode_index"]) for episode in existing_episodes}
    result: list[int] = []
    candidate = 0
    while len(result) < count:
        if candidate not in used:
            result.append(candidate)
        candidate += 1
    return result


def _recover_sequential_collection(config, episodes, failures):
    """Recover finished attempts; preserve partial files and retry their seed."""
    root = config.output_dir
    known = {item["path"] for item in episodes}
    failures = list(failures)
    known_failures = {item["path"] for item in failures}
    for receipt in sorted(root.glob("failed_episodes/attempt_*/attempt_summary.json")):
        item = json.loads(receipt.read_text(encoding="utf-8"))
        if item["path"] not in known_failures:
            failures.append(item)
            known_failures.add(item["path"])
    episodes = list(episodes)
    for directory in sorted(root.glob("episode_*_seed_*")):
        if not directory.is_dir():
            continue
        if directory.name in known:
            continue
        trajectory = directory / "trajectory.json"
        item = None
        try:
            if trajectory.is_file():
                payload = json.loads(trajectory.read_text(encoding="utf-8"))
                episode = payload["episode"]
                if episode["result"]["success"]:
                    episodes.append(_episode_summary_from_directory(directory))
                    continue
                item = dict(episode_index=episode["episode_index"], seed=episode["seed"],
                    path=directory.name, frames=episode["frame_count"], success=False,
                    termination_reason=episode["result"]["termination_reason"])
            elif (directory / "failure.json").is_file():
                item = json.loads((directory / "failure.json").read_text(encoding="utf-8"))
                item.update(path=directory.name, frames=0, success=False)
        except (json.JSONDecodeError, KeyError, ValueError, RuntimeError):
            # A kill may interrupt trajectory serialization or video closing.
            # Incomplete results must be retried rather than labelled failed.
            item = None
        if item is not None:
            archived = _archive_failed_episode(root, item, len(failures))
            failures.append(archived)
            continue
        # Incomplete recordings are not evidence of task failure. Keep all
        # partial artifacts, then allow the same seed to be submitted again.
        parent = root / "interrupted_episodes"
        parent.mkdir(exist_ok=True)
        destination = parent / directory.name
        suffix = 0
        while destination.exists():
            suffix += 1
            destination = parent / f"{directory.name}.interrupted_{suffix}"
        shutil.move(str(directory), str(destination))
    indices = [int(item["episode_index"]) for item in episodes]
    if len(indices) != len(set(indices)):
        raise RuntimeError("Cannot recover dataset with duplicate episode indices")
    return sorted(episodes, key=lambda e: e["episode_index"]), failures


def _layout_json(layout: Any) -> dict[str, Any]:
    minimum_clearance = float(layout.minimum_clearance_m)
    return {
        "minimum_clearance_m": (
            minimum_clearance if math.isfinite(minimum_clearance) else None
        ),
        "placements": [
            {
                "bodies": list(item.names),
                "anchor_xy_m": _json_array(item.anchor_xy),
                "yaw_rad": float(item.yaw_rad),
            }
            for item in layout.placements
        ],
    }


def _camera_calibration(config: CollectionConfig) -> dict[str, Any]:
    """Return task-invariant simulated RGB intrinsics and nominal extrinsics."""
    spec = get_robot_spec(config.robot_uid)
    model = load_scene_model(config.scene, config.robot_uid)
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    if wrist_id < 0:
        raise RuntimeError("Dataset scene is missing wrist_camera")
    set_wrist_camera_calibration(
        model, wrist_id,
        float(spec.wrist_camera_trim_pos[0]),
        float(spec.wrist_camera_trim_pos[2]),
        spec.wrist_camera_trim_pitch_deg,
        spec.wrist_camera_pos, spec.wrist_camera_quat,
    )

    def camera(name: str) -> dict[str, Any]:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise RuntimeError(f"Dataset scene is missing camera: {name}")
        fovy_deg = float(model.cam_fovy[camera_id])
        fy = 0.5 * config.height / math.tan(math.radians(fovy_deg) * 0.5)
        fx = fy
        body_id = int(model.cam_bodyid[camera_id])
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
        return {
            "resolution": [config.width, config.height],
            "intrinsics": {
                "fx": fx, "fy": fy,
                "cx": 0.5 * (config.width - 1),
                "cy": 0.5 * (config.height - 1),
                "fovy_deg": fovy_deg,
            },
            "distortion": {"model": "none", "coefficients": []},
            "nominal_pose_parent": {
                "parent_body": parent,
                "position_m": _json_array(model.cam_pos[camera_id]),
                "quaternion_wxyz": _json_array(model.cam_quat[camera_id]),
            },
        }

    return {
        "camera_frame_convention": "+X right, +Y up, -Z optical axis",
        "quaternion_order": "wxyz",
        "main_rgb": camera("front_view"),
        "wrist_rgb": camera("wrist_camera"),
    }


def _home_rotation_vector(config: CollectionConfig) -> np.ndarray:
    """Return this robot model's EE rotation-vector branch at Home."""
    spec = get_robot_spec(config.robot_uid)
    model = load_scene_model(config.scene, config.robot_uid)
    handles = find_handles(model, spec)
    data = mujoco.MjData(model)
    set_home(model, data, handles)
    return rotation_vector(ee_pose(data, handles)[1])


def _task_metadata(config: CollectionConfig, instruction_file: str) -> dict[str, Any]:
    """Return fields that are invariant across every episode of this task."""
    # Demonstrations use the maintained single-step IPOPT profile. All six
    # Cartesian axes are Specific during collection, so the EMA release term
    # is structurally inactive but remains part of the recorded configuration.
    weights = ObjectiveSettings.delta_q_squared_conditioning_safeguard()
    tracker = JointReferenceTrackerSettings()
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_release_version": DATASET_RELEASE_VERSION,
        "generator": {
            "name": "TaskTol-VLA MuJoCo collector",
            "version": GENERATOR_VERSION,
            "mujoco_version": mujoco.__version__,
        },
        "scene": config.scene,
        "robot": {
            "uid": (spec := get_robot_spec(config.robot_uid)).uid,
            "name": spec.label,
            "arm_dof": spec.arm_dof,
            "joint_names": list(spec.joint_names),
            "joint_state_dimension": DATASET_JOINT_STATE_DIMENSION,
            "joint_state_padding": (
                "none" if spec.arm_dof == DATASET_JOINT_STATE_DIMENSION
                else "append zero after the physical joints"
            ),
            "ee_site": spec.ee_site,
            "gripper_max_aperture_m": spec.gripper_max_aperture_m,
        },
        "task_id": config.task,
        "task_name": TASK_LANGUAGE[config.task],
        "instruction_file": instruction_file,
        "frequency_hz": 10.0,
        "action_format": {
            "dimension": 7,
            "translation": "base-frame delta XYZ in metres",
            "rotation": "base-frame rotation-vector delta RxRyRz in radians; left-multiplied",
            "gripper": "-1 open, +1 close",
        },
        "phase_labels": {
            "pregrasp": "Pre-grasp",
            "grasp": "Grasp",
            "postgrasp": "Post-grasp",
            "release": "Release",
        },
        "rotation_representation": {
            "type": "temporally continuous rotation vector",
            "frame": "base",
            "unit": "rad",
            "initial_branch_reference_rad": _json_array(
                _home_rotation_vector(config),
            ),
            "initial_branch_reference_source": "selected robot Home EE pose",
            "branch_rule": "nearest equivalent 2*pi branch to previous frame",
        },
        "video": {
            "resolution": [OPENPI_IMAGE_WIDTH, OPENPI_IMAGE_HEIGHT],
            "source_render_resolution": [config.width, config.height],
            "streams": {"main_rgb": "main_rgb.mp4", "wrist_rgb": "wrist_rgb.mp4"},
            "codec": "H.264",
            "pixel_format": "yuv420p",
            "fps": 10.0,
            "preprocessing": {
                "name": "openpi_resize_with_pad",
                "target_size": [OPENPI_IMAGE_WIDTH, OPENPI_IMAGE_HEIGHT],
                "preserve_aspect_ratio": True,
                "interpolation": "PIL bilinear",
                "padding": "centered black RGB zeros",
            },
        },
        "camera_calibration": _camera_calibration(config),
        "controller": {
            "optimizer_backend": "ipopt",
            "method": "IPOPT constrained NLP with all six Cartesian axes Specific",
            "max_iterations": _DATASET_SQP_SETTINGS.max_iterations,
            "max_time_s": None,
            "loss_profile": "delta_q_squared_conditioning_safeguard",
            "loss_weights": {
                "delta_q_squared": weights.squared_delta_q_weight,
                "conditioning": weights.conditioning.weight,
                "ema_release": weights.ema_release_loss_weight,
            },
            "objective_weights": {
                "position": weights.position_weight,
                "rotation": weights.rotation_weight,
                "velocity": weights.velocity_weight,
                "jerk": weights.jerk_weight,
                "manipulability": weights.manipulability_weight,
                "self_collision": weights.self_collision_weight,
                "release_loss": weights.release_loss_weight,
            },
            "joint_reference_tracker": {
                "method": "decoupled bounded leaky PID outer loop",
                "proportional_gain": tracker.proportional_gain,
                "integral_gain_s": tracker.integral_gain_s,
                "velocity_gain_s": tracker.velocity_gain_s,
                "maximum_correction_rad": tracker.maximum_correction_rad,
                "integration_error_limit_rad": (
                    tracker.integration_error_limit_rad
                ),
                "integral_time_constant_s": (
                    tracker.integral_time_constant_s
                ),
                "stationary_integral_time_constant_s": (
                    tracker.stationary_integral_time_constant_s
                ),
                "stationary_velocity_threshold_rad_s": (
                    tracker.stationary_velocity_threshold_rad_s
                ),
            },
        },
        "collection": {
            "maximum_frames_per_episode": config.maximum_frames,
            "minimum_object_clearance_m": config.object_clearance_m,
            "nominal_seed": -1 if config.sequential_seed_start is None else None,
            "random_seed_type": ("unique unsigned 32-bit integer"
                if config.sequential_seed_start is None else "consecutive nonnegative integers"),
            **({"sequential_seed_start": config.sequential_seed_start}
               if config.sequential_seed_start is not None else {}),
            "environment_randomization_probability": (
                config.environment_randomization_probability
            ),
            "placement_difficulty": config.placement_difficulty,
            "allowed_scene_variant_indices": (
                None if config.variant_indices is None
                else list(config.variant_indices)
            ),
            "reachability_map_dir": (
                None
                if config.reachability_map_dir is None
                else str(config.reachability_map_dir)
            ),
            "seed_groups": {
                "environment": ["table", "front_camera", "lighting"],
                "task": ["object_layout", "semantic_goal_perturbations"],
            },
        },
        "tolerance_labels": tolerance_label_metadata(),
        "success_evaluation": {
            "source": "MuJoCo privileged state and physics contacts",
            "required_stable_frames": 0,
            "intermediate_action_requirements": [],
            "prohibited_collisions": [
                "robot_table_collision",
                "robot_self_collision",
            ],
            "collision_condition": "actual_geometry_contact_distance_m <= 0",
            **(
                {
                    "placement": (
                        "released upright lid centred on and contacting the pot"
                        if config.scene in _CLOSE_POT_LID_SCENES
                        else "released upright lid on the table beside the pot"
                    ),
                    "maximum_closed_lid_center_error_m": (
                        0.045 if config.scene in _CLOSE_POT_LID_SCENES else None
                    ),
                    "minimum_open_lid_separation_m": (
                        0.19 if config.scene in _OPEN_POT_LID_SCENES else None
                    ),
                }
                if config.scene in _POT_LID_SCENES
                else {
                    "plate_edge_margin_m": 0.003,
                    "contact_policy": (
                        "contacts are recorded for diagnostics but do not invalidate success"
                    ),
                    "failure_disturbance": {
                        "type": "upright distractor knocked over",
                        "tilt_threshold_rad": float(np.deg2rad(60.0)),
                    },
                }
            ),
        },
    }


def _render_rgb(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return np.ascontiguousarray(renderer.render(), dtype=np.uint8)


def _openpi_video_frame(image: np.ndarray) -> np.ndarray:
    """Return the exact uint8 RGB image shape consumed by OpenPI."""
    frame = np.asarray(
        openpi_image_tools.resize_with_pad(
            np.asarray(image, dtype=np.uint8),
            OPENPI_IMAGE_HEIGHT,
            OPENPI_IMAGE_WIDTH,
        ),
        dtype=np.uint8,
    )
    expected_shape = (OPENPI_IMAGE_HEIGHT, OPENPI_IMAGE_WIDTH, 3)
    if frame.shape != expected_shape:
        raise ValueError(
            f"OpenPI video frame must have shape {expected_shape}, got {frame.shape}"
        )
    return np.ascontiguousarray(frame)


def _episode_directory(root: Path, episode_index: int, seed: int) -> Path:
    return root / f"episode_{episode_index:04d}_seed_{seed}"


def _archive_failed_episode(
    root: Path,
    episode: dict[str, Any],
    attempt_index: int,
) -> dict[str, Any]:
    """Move one complete failed attempt outside the trainable episode set."""
    failed_root = root / "failed_episodes"
    failed_root.mkdir(exist_ok=True)
    source = root / str(episode["path"])
    destination = failed_root / (
        f"attempt_{attempt_index:05d}_episode_{int(episode['episode_index']):04d}"
        f"_seed_{int(episode['seed'])}"
    )
    if destination.exists():
        raise FileExistsError(f"Failure archive already exists: {destination}")
    archived = dict(episode)
    archived["attempt_index"] = int(attempt_index)
    archived["path"] = destination.relative_to(root).as_posix()
    # Persist the receipt before the rename, so interruption cannot leave an
    # archived failure that has neither a manifest entry nor a receipt.
    _atomic_write_json(source / "attempt_summary.json", archived)
    shutil.move(str(source), str(destination))
    return archived


def _exception_episode(
    config: CollectionConfig,
    episode_index: int,
    seed: int,
    instruction_index: int,
    instruction: str,
    error: Exception,
) -> dict[str, Any]:
    """Serialize one worker exception as an ordinary retryable failure."""
    episode_dir = _episode_directory(config.output_dir, episode_index, seed)
    episode_dir.mkdir(parents=True, exist_ok=True)
    error_type = type(error).__name__
    reason = f"worker_exception:{error_type}"
    _atomic_write_json(episode_dir / "failure.json", {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "instruction_index": int(instruction_index),
        "instruction": instruction,
        "termination_reason": reason,
        "error_type": error_type,
        "error_message": str(error),
    })
    return {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "path": episode_dir.name,
        "frames": 0,
        "instruction_index": int(instruction_index),
        "instruction": instruction,
        "success": False,
        "termination_reason": reason,
        "error_type": error_type,
        "error_message": str(error),
    }


def _collect_episode(
    config: CollectionConfig,
    episode_index: int,
    seed: int,
    instruction_index: int,
    instruction: str,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, Any]:
    episode_dir = _episode_directory(config.output_dir, episode_index, seed)
    if episode_dir.exists():
        raise FileExistsError(f"Episode already exists: {episode_dir}")
    episode_dir.mkdir(parents=True)
    main_video_path = episode_dir / "main_rgb.mp4"
    wrist_video_path = episode_dir / "wrist_rgb.mp4"

    spec = get_robot_spec(config.robot_uid)
    model = load_scene_model(config.scene, config.robot_uid)
    handles = find_handles(model, spec)
    configure_contact_solver(model, 0.001)
    configure_gripper(
        model, handles, DEFAULT_GRIPPER_FORCE_SCALE,
        DEFAULT_MAX_GRIPPER_FORCE, DEFAULT_FINGER_SLIDE_FRICTION,
    )
    data = mujoco.MjData(model)
    set_home(model, data, handles)
    home_rotation_vector = rotation_vector(ee_pose(data, handles)[1])
    scene_manager = SceneManager(model, loaded_scene=config.scene)
    scene_manager.activate(data, config.scene)

    work_surface_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface")
    front_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_view")
    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    if min(work_surface_id, front_camera_id, wrist_camera_id) < 0:
        raise RuntimeError("Dataset scene is missing a required table or camera")

    seed_routing = RandomizationSeedRouting(
        seed, config.environment_randomization_probability,
    )
    nominal_table_x = float(model.body_pos[work_surface_id, 0])
    nominal_tabletop = tabletop_corners(model)
    table_randomizer = TableRandomizer(model.body_pos[work_surface_id])
    table_pose = table_randomizer.sample(seed_routing.environment_seed)
    model.body_pos[work_surface_id] = table_pose.position
    scene_manager.set_workspace_offset(data, table_pose.offset)
    object_randomizer = ObjectLayoutRandomizer(
        model, scene_manager,
        limits=ObjectLayoutLimits(minimum_clearance_m=config.object_clearance_m),
    )
    layout = object_randomizer.apply(
        data,
        seed_routing.object_layout_seed,
        variant_index=pot_lid_task_variant_index(config.task),
        variant_indices=(
            None
            if pot_lid_task_variant_index(config.task) is not None
            else config.variant_indices
        ),
    )
    instruction = resolve_episode_instruction(
        instruction,
        task_id=config.task,
        scene=config.scene,
        seed=seed,
        active_body_names=scene_manager.active_body_names,
        geometry_color=scene_manager.active_geometry_color_name,
    )
    difficult_layout = apply_blocks_difficulty(
        model, data, scene_manager, handles, config.task,
        config.placement_difficulty, seed_routing.object_layout_seed,
        map_directory=config.reachability_map_dir,
    )

    front_base_position = model.cam_pos[front_camera_id].copy()
    front_base_quaternion = model.cam_quat[front_camera_id].copy()
    front_focus = focus_on_world_x(
        front_base_position, front_base_quaternion,
        nominal_table_x,
    )
    camera_pose = FrontCameraRandomizer(
        front_base_position, front_base_quaternion, front_focus,
    ).sample(
        seed_routing.environment_seed, tabletop=tabletop_corners(model), reference_tabletop=nominal_tabletop,
        vertical_fov_deg=float(model.cam_fovy[front_camera_id]),
        aspect_ratio=config.width / config.height,
    )
    model.cam_pos[front_camera_id] = camera_pose.position
    model.cam_quat[front_camera_id] = camera_pose.quaternion
    set_wrist_camera_calibration(
        model, wrist_camera_id,
        float(spec.wrist_camera_trim_pos[0]),
        float(spec.wrist_camera_trim_pos[2]),
        spec.wrist_camera_trim_pitch_deg,
        spec.wrist_camera_pos, spec.wrist_camera_quat,
    )

    light_randomizer = LightRandomizer(model)
    lighting = light_randomizer.sample(seed_routing.environment_seed)
    light_randomizer.apply(model, lighting)
    # Dataset capture starts from a deterministic, symmetric maximum aperture.
    # This is intentionally scoped to collection; interactive key-9 recording
    # preserves whatever gripper state the operator currently commands.
    set_gripper_fully_open(model, data, handles)

    # Let randomized objects and the robot settle before any video or state is
    # recorded. The scripted plan is built afterwards from the settled poses.
    simulation_backend = PresetSimulationBackend(model, data, handles)
    simulation_backend.settle(_PRE_RECORD_SETTLE_TIME_S)

    executor = PresetTaskExecutor(model, handles)
    evaluator = create_task_evaluator(config.scene, config.task, model, data)
    position, rotation = ee_pose(data, handles)
    executor.start(config.task, config.scene, data, position, rotation, episode_seed=seed)
    # A long task may legitimately contain several complete pick-place cycles.
    # Never truncate a generated expert plan merely because the legacy
    # single-cycle default frame budget is shorter than that plan.
    episode_frame_limit = max(
        config.maximum_frames,
        len(executor.frames) + int(np.ceil(GRIPPER_TRAVEL_TIME / TELEOP_DT)) + 2,
    )
    target = TargetPose(position.copy(), rotation.copy())
    previous_nominal_rotvec = home_rotation_vector.copy()
    previous_actual_rotvec = home_rotation_vector.copy()
    # Every newly collected demonstration uses the maintained IPOPT backend.
    controller_handles = handles
    ik = IpoptIK(
        model,
        controller_handles,
        _DATASET_SQP_SETTINGS,
        ObjectiveSettings.delta_q_squared_conditioning_safeguard(),
    )
    active_dofs = np.ones(6, dtype=bool)
    joint_command = data.qpos[handles.qpos_ids].copy()
    joint_tracker = JointReferenceTracker(controller_handles.spec.arm_dof)
    gripper_ctrl = spec.gripper_open_ctrl
    gripper_closed = False
    gripper_speed = abs(spec.gripper_open_ctrl - spec.gripper_closed_ctrl) / GRIPPER_TRAVEL_TIME
    physics_steps = max(1, int(round(TELEOP_DT / float(model.opt.timestep))))
    frames: list[dict[str, Any]] = []
    previous_source_frame: dict[str, Any] | None = None
    source_frame_count = 0
    timed_out = False

    # Initial open-gripper dwell belongs to simulator preparation, not to the
    # demonstration. Start JSON and both videos at the first moving command.
    first_action = executor.first_moving_trajectory_action()
    if first_action is None:
        raise RuntimeError("The preset trajectory contains no Cartesian motion")

    # Do not start FFmpeg until scene setup and CPU trajectory planning have
    # succeeded. Recoverable planner rejection must not create empty MP4 files.
    main_renderer = mujoco.Renderer(model, height=config.height, width=config.width)
    wrist_renderer = mujoco.Renderer(model, height=config.height, width=config.width)
    main_writer = imageio_ffmpeg.write_frames(
        str(main_video_path), (OPENPI_IMAGE_WIDTH, OPENPI_IMAGE_HEIGHT), fps=10,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        quality=8, macro_block_size=1,
    )
    wrist_writer = imageio_ffmpeg.write_frames(
        str(wrist_video_path), (OPENPI_IMAGE_WIDTH, OPENPI_IMAGE_HEIGHT), fps=10,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        quality=8, macro_block_size=1,
    )
    main_writer.send(None)
    wrist_writer.send(None)

    try:
        for frame_index in range(episode_frame_limit):
            # Pair the current observation with the action that will be applied
            # over [time_s, time_s + TELEOP_DT). This is the causal convention
            # expected by behavior cloning and LeRobot-style datasets.
            actual_position, actual_rotation = ee_pose(data, handles)
            # Advance the scripted expert trajectory exactly once per recorded
            # 10 Hz frame.  Robot feedback is recorded as state, but must not
            # inject controller-wait no-ops into the demonstrated action.
            if frame_index == 0:
                planned_action = first_action
            else:
                if executor.needs_subtask_plan:
                    executor.plan_next_subtask(
                        data, target.position, target.rotation,
                    )
                planned_action = (
                    executor.trajectory_action()
                    if executor.active else None
                )
            if planned_action is not None:
                action = planned_action.value.copy()
                target.position, target.rotation = advance_reference_pose(
                    target.position,
                    target.rotation,
                    action,
                )
                gripper_closed = planned_action.gripper_closed
            else:
                action = np.array((
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0 if gripper_closed else -1.0,
                ))
            if frame_index == 0 and np.linalg.norm(action[:6]) <= 1e-12:
                raise RuntimeError("The first recorded frame must contain Cartesian action")

            phase_observation = simulation_backend.observe_manipulation_phase(gripper_closed)
            joint_target, diagnostics = ik.solve(
                data, target, active_dofs,
            )
            phase = phase_observation.phase.key
            nominal_rotvec = continuous_rotation_vector(
                target.rotation, previous_nominal_rotvec,
            )
            actual_rotvec = continuous_rotation_vector(
                actual_rotation, previous_actual_rotvec,
                reference=nominal_rotvec, reference_weight=0.7,
            )
            previous_nominal_rotvec = nominal_rotvec
            previous_actual_rotvec = actual_rotvec
            finger_positions = simulation_backend.phase_classifier.finger_positions(data)
            source_frame = {
                "index": frame_index,
                "time_s": frame_index * TELEOP_DT,
                "video_frame_index": frame_index,
                "joint_state": {
                    "position_rad": _json_array(dataset_joint_state(
                        data.qpos[handles.qpos_ids], spec,
                    )),
                    "velocity_rad_s": _json_array(dataset_joint_state(
                        data.qvel[handles.dof_ids], spec,
                    )),
                },
                "ee_state": _pose_json(actual_position, actual_rotvec),
                "gripper_state": {
                    "finger_positions_m": _json_array(finger_positions),
                    "aperture_m": float(np.sum(finger_positions)),
                },
                "action": _json_array(action),
                "phase": phase,
                "subtask_index": int(phase_observation.subtask_index),
                "nominal_target_pose": _pose_json(target.position, nominal_rotvec),
                "optimizer": {
                    "iterations": int(diagnostics.iterations),
                    "cost": float(diagnostics.cost),
                },
            }
            retained_frame = retain_source_frame(
                previous_source_frame,
                source_frame,
                output_index=len(frames),
                frequency_hz=1.0 / TELEOP_DT,
            )
            previous_source_frame = source_frame
            source_frame_count += 1
            if retained_frame is not None:
                # JSON and both videos share this compact zero-based index.
                main_writer.send(_openpi_video_frame(
                    _render_rgb(main_renderer, data, "front_view"),
                ))
                wrist_writer.send(_openpi_video_frame(
                    _render_rgb(wrist_renderer, data, "wrist_camera"),
                ))
                frames.append(retained_frame)

            evaluation_frame_index = max(0, len(frames) - 1)
            interval = simulation_backend.run_control_interval(
                tracker=joint_tracker,
                joint_command=joint_command,
                joint_target=joint_target,
                gripper_ctrl=gripper_ctrl,
                gripper_closed=gripper_closed,
                gripper_speed=gripper_speed,
                speed_scale=1.0,
                physics_steps=physics_steps,
                evaluator=evaluator,
                frame_index=evaluation_frame_index,
            )
            joint_command = interval.joint_command
            gripper_ctrl = interval.gripper_ctrl
            if progress is not None:
                plan_total = len(executor.frames)
                plan_current = plan_total if not executor.active else min(executor.index + 1, plan_total)
                progress(len(frames), plan_current, plan_total)

            if evaluator.should_abort:
                break
            # Task semantics belong to the planned trajectory. A task may end
            # open, closed, or without ever actuating the gripper; once every
            # planned frame has been applied, recording is complete.
            if not executor.active:
                break
        else:
            timed_out = True
    finally:
        main_writer.close()
        wrist_writer.close()
        main_renderer.close()
        wrist_renderer.close()

    video_frame_counts = {}
    for video_path in (main_video_path, wrist_video_path):
        count, _ = imageio_ffmpeg.count_frames_and_secs(str(video_path))
        if count != len(frames):
            raise RuntimeError(
                f"{video_path.name} has {count} frames; trajectory has {len(frames)}"
            )
        video_frame_counts[video_path.name] = int(count)

    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "teleop_tall_block")
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "teleop_plate")
    horizontal_error = (
        float(np.linalg.norm(data.xpos[block_id, :2] - data.xpos[plate_id, :2]))
        if block_id >= 0 and plate_id >= 0
        else None
    )
    evaluation = evaluator.result(not executor.active, timed_out)

    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "episode": {
            "robot_uid": config.robot_uid,
            "scene": config.scene,
            "task_id": config.task,
            "instruction_index": instruction_index,
            "instruction": instruction,
            "seed": seed,
            "episode_index": episode_index,
            "frame_count": len(frames),
            "video_frame_counts": video_frame_counts,
            "frame_filter": frame_filter_metadata(
                source_frame_count, len(frames),
            ),
            "subtask_count": (
                max((int(frame["subtask_index"]) for frame in frames), default=0)
                + 1
            ),
            "effective_maximum_frames": episode_frame_limit,
            "preset_planning": dict(executor.last_planning_stats),
            "domain_randomization": {
                "seed_routing": seed_routing.as_dict(),
                "semantic_goal_perturbations": dict(executor.perturbations.samples),
                "table_offset_m": _json_array(table_pose.offset),
                "front_camera_local_translation_m": _json_array(camera_pose.local_translation),
                "front_camera_local_angles_deg": _json_array(camera_pose.local_angles_deg),
                "front_camera_framing_angles_deg": _json_array(camera_pose.framing_angles_deg),
                "front_camera_tabletop_visible_fraction_before": camera_pose.tabletop_visible_fraction_before,
                "front_camera_tabletop_visible_fraction_after": camera_pose.tabletop_visible_fraction_after,
                "front_camera_tabletop_visible_fraction_reference": camera_pose.tabletop_visible_fraction_reference,
                "lighting_brightness": _json_array(lighting.brightness),
                "object_layout": _layout_json(layout),
                "active_scene_bodies": list(scene_manager.active_body_names),
                **(
                    {
                        "stage_rotation_masks": {
                            stage: list(mask)
                            for stage, mask in zip(
                                ("pregrasp", "postgrasp"),
                                active_pot_lid_stage_rotation_masks(model),
                                strict=True,
                            )
                        },
                    }
                    if config.scene in _POT_LID_SCENES
                    else {}
                ),
                "reachability_guided_layout": difficult_layout,
            },
            "result": {
                **evaluation,
                **(
                    {"final_block_plate_xy_error_m": horizontal_error}
                    if horizontal_error is not None
                    else {}
                ),
            },
        },
        "trajectory": frames,
    }
    metadata_path = episode_dir / "trajectory.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "episode_index": episode_index,
        "seed": seed,
        "path": episode_dir.name,
        "frames": len(frames),
        "source_frames": source_frame_count,
        "removed_frames": source_frame_count - len(frames),
        "instruction_index": instruction_index,
        "instruction": instruction,
        "success": bool(evaluation["success"]),
        "termination_reason": str(evaluation["termination_reason"]),
    }


def collect_batch(
    config: CollectionConfig, *, target_total: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Collect successful demonstrations, retrying failures.

    By default ``batch_size`` successes are added. ``target_total`` instead
    fills an absolute quota and makes repeated append calls idempotent.
    """
    manifest_path = config.output_dir / "manifest.json"
    task_metadata_path = config.output_dir / "task_metadata.json"
    failure_manifest_path = config.output_dir / "failed_episodes" / "manifest.json"
    if config.mode == "overwrite":
        # Keep the task directory itself. On Windows, Explorer or an editor may
        # hold a directory handle that prevents rmdir even though every dataset
        # file is removable.
        config.output_dir.mkdir(parents=True, exist_ok=True)
        for child in config.output_dir.iterdir():
            # Task-level language, tolerance values, and trial simulations are
            # independent sidecars explicitly retained across raw episode
            # recollection.
            if child.name in {
                "instructions.json",
                "tolerance_annotation.json",
                "counterfactual_trials.jsonl",
            }:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        existing_episodes: list[dict[str, Any]] = []
        existing_failures: list[dict[str, Any]] = []
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_episodes = list(existing_manifest.get("episodes", ()))
        else:
            existing_episodes = ([] if config.sequential_seed_start is not None
                                 else _recover_episode_manifest_entries(config.output_dir))
            unknown = [
                child.name for child in config.output_dir.iterdir()
                if child.name not in {
                    "instructions.json", "task_metadata.json", "failed_episodes",
                    "interrupted_episodes", "collection_registration.json",
                }
                and not child.name.startswith("episode_")
            ]
            if unknown:
                raise RuntimeError(
                    "Append target has no manifest and contains unknown entries: "
                    + ", ".join(sorted(unknown))
                )
            if existing_episodes:
                print(
                    f"Recovered {len(existing_episodes)} successful episodes "
                    "from trajectory.json files.",
                    file=sys.stderr, flush=True,
                )

        if failure_manifest_path.exists():
            failure_manifest = json.loads(failure_manifest_path.read_text(encoding="utf-8"))
            existing_failures = list(failure_manifest.get("failures", ()))
        else:
            existing_failures = []

    if config.mode == "append" and config.sequential_seed_start is not None:
        existing_episodes, existing_failures = _recover_sequential_collection(
            config, existing_episodes, existing_failures,
        )
        if existing_failures:
            _atomic_write_json(failure_manifest_path, {
                "schema_version": DATASET_SCHEMA_VERSION,
                "failure_count": len(existing_failures), "failures": existing_failures,
            })
    already_complete = False
    if target_total is not None:
        if type(target_total) is not int or target_total <= 0:
            raise ValueError("target_total must be a positive integer")
        remaining = max(0, target_total - len(existing_episodes))
        already_complete = remaining == 0
        config = replace(config, batch_size=max(1, remaining))

    existing_seeds = {
        int(episode["seed"])
        for episode in (*existing_episodes, *existing_failures)
    }
    instruction_path, instructions = load_or_create_instructions(config.output_dir, config.task)
    metadata = _task_metadata(config, instruction_path.name)
    if config.mode == "append":
        if manifest_path.exists() and not task_metadata_path.exists():
            raise RuntimeError("Append target has no task_metadata.json")
        if task_metadata_path.exists():
            existing_metadata = json.loads(task_metadata_path.read_text(encoding="utf-8"))
            if existing_metadata != metadata:
                raise ValueError("Append configuration differs from task_metadata.json")
    _atomic_write_json(task_metadata_path, metadata)
    episode_indices = _next_episode_indices(existing_episodes, config.batch_size)
    episode_number_by_index = {
        episode_index: offset + 1
        for offset, episode_index in enumerate(episode_indices)
    }
    maximum_attempts = _maximum_attempts(config)
    progress = _CollectionProgress(config.batch_size, enabled=progress_callback is None)
    new_episodes: list[dict[str, Any]] = []
    new_failures: list[dict[str, Any]] = []
    attempted_seeds = set(existing_seeds)
    attempts = 0

    def checkpoint(status: str) -> None:
        episodes = sorted(
            [*existing_episodes, *new_episodes],
            key=lambda episode: int(episode["episode_index"]),
        )
        _atomic_write_json(manifest_path, {
            "schema_version": DATASET_SCHEMA_VERSION,
            "task_metadata_file": task_metadata_path.name,
            "instruction_file": instruction_path.name,
            "episode_count": len(episodes),
            "last_operation": {
                "mode": config.mode,
                "status": status,
                "workers": min(config.workers, config.batch_size),
                "target_episodes_added": config.batch_size,
                "episodes_added": len(new_episodes),
                "failed_attempts": len(new_failures),
                "total_attempts": attempts,
                "maximum_attempts": maximum_attempts,
            },
            "episodes": episodes,
        })
        if progress_callback is not None:
            progress_callback(dict(status=status, episode_count=len(episodes),
                failed_attempts=len(existing_failures)+len(new_failures),
                attempts_this_run=attempts, target_total=target_total,
                seeds=[int(e["seed"]) for e in episodes]))

    # A valid manifest exists before expensive workers start. If collection is
    # interrupted, append mode can continue from every completed success.
    checkpoint("in_progress")
    if already_complete:
        checkpoint("complete")
        return manifest_path

    def next_seed() -> int:
        if config.sequential_seed_start is not None:
            seed = config.sequential_seed_start
            while seed in attempted_seeds:
                seed += 1
        else:
            seed = -1 if -1 not in attempted_seeds else sample_batch_seeds(
                1, include_nominal=False, excluded=attempted_seeds,
            )[0]
        attempted_seeds.add(seed)
        return seed

    def archive_failure(episode: dict[str, Any], attempt_number: int) -> None:
        archived = _archive_failed_episode(
            config.output_dir, episode, len(existing_failures) + len(new_failures),
        )
        archived["attempt_number"] = int(attempt_number)
        new_failures.append(archived)
        progress.newline()
        print(
            f"Attempt {attempt_number} seed={episode['seed']} failed: "
            f"{episode['termination_reason']}; archived at {archived['path']}",
            file=sys.stderr,
            flush=True,
        )
        failures = [*existing_failures, *new_failures]
        _atomic_write_json(failure_manifest_path, {
            "schema_version": DATASET_SCHEMA_VERSION,
            "failure_count": len(failures),
            "failures": failures,
        })
        checkpoint("in_progress")

    def episode_args(episode_index: int, seed: int) -> tuple[Any, ...]:
        instruction_index = episode_index % len(instructions)
        return (
            config, episode_index, seed, instruction_index,
            instructions[instruction_index], None,
        )

    try:
        if config.workers == 1:
            while len(new_episodes) < config.batch_size:
                if attempts >= maximum_attempts:
                    break
                seed = next_seed()
                attempts += 1
                success_offset = len(new_episodes)
                episode_number = success_offset + 1
                progress.update(success_offset, episode_number, seed, 0, 0, 1)
                episode_index = episode_indices[success_offset]
                instruction_index = episode_index % len(instructions)
                try:
                    episode = _collect_episode(
                        config, episode_index, seed, instruction_index,
                        instructions[instruction_index],
                        progress=lambda frames, plan_current, plan_total, offset=success_offset,
                        episode_number=episode_number, seed=seed: progress.update(
                            offset, episode_number, seed, frames, plan_current, plan_total,
                        ),
                    )
                except Exception as error:
                    episode = _exception_episode(
                        config, episode_index, seed, instruction_index,
                        instructions[instruction_index], error,
                    )
                if episode["success"]:
                    new_episodes.append(episode)
                    checkpoint("in_progress")
                    progress.update(
                        len(new_episodes), episode_number, seed,
                        int(episode["frames"]), 1, 1,
                    )
                else:
                    archive_failure(episode, attempts)
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=config.workers, mp_context=context,
                initializer=_initialize_collection_worker,
            ) as pool:
                pending: dict[Any, tuple[int, int, int]] = {}

                def submit(episode_index: int) -> bool:
                    nonlocal attempts
                    if attempts >= maximum_attempts:
                        return False
                    seed = next_seed()
                    attempts += 1
                    future = pool.submit(_collect_episode, *episode_args(episode_index, seed))
                    pending[future] = (episode_index, seed, attempts)
                    progress.update(
                        len(new_episodes), episode_number_by_index[episode_index],
                        seed, 0, 0, 1,
                    )
                    return True

                remaining_indices = iter(episode_indices)
                for _ in range(min(config.workers, len(episode_indices))):
                    submit(next(remaining_indices))

                try:
                    while pending:
                        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        for future in sorted(completed, key=lambda f: pending[f][1]):
                            episode_index, seed, attempt_number = pending.pop(future)
                            try:
                                episode = future.result()
                            except Exception as error:
                                instruction_index = episode_index % len(instructions)
                                episode = _exception_episode(
                                    config, episode_index, seed, instruction_index,
                                    instructions[instruction_index], error,
                                )
                            episode_number = episode_number_by_index[episode_index]
                            if episode["success"]:
                                new_episodes.append(episode)
                                checkpoint("in_progress")
                                progress.update(
                                    len(new_episodes), episode_number, seed,
                                    int(episode["frames"]), 1, 1,
                                )
                                next_index = next(remaining_indices, None)
                                if next_index is not None:
                                    submit(next_index)
                            else:
                                archive_failure(episode, attempt_number)
                                submit(episode_index)
                except BaseException:
                    # Stop only workers owned by this batch. Partial recording
                    # directories are preserved for append-mode recovery.
                    for future in pending:
                        future.cancel()
                    for process in (getattr(pool, "_processes", None) or {}).values():
                        if process.is_alive():
                            process.terminate()
                    raise

        if len(new_episodes) < config.batch_size:
            checkpoint("exhausted")
            raise RuntimeError(
                f"Only collected {len(new_episodes)}/{config.batch_size} successful episodes "
                f"after {attempts} attempts (limit {maximum_attempts}); "
                f"partial progress is checkpointed in {manifest_path}; "
                f"see {failure_manifest_path}"
            )
    finally:
        progress.newline()
    checkpoint("complete")
    return manifest_path
