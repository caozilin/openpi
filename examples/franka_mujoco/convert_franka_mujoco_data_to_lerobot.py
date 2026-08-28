"""Convert TaskTol-VLA multi-robot MuJoCo schema 7.0 data to LeRobot format.

Example:
uv run python examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
    --raw-dir /path/to/franka_mujoco/datasets \
    --repo-id caozilin/franka_mujoco
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


SCHEMA_VERSION = "7.0"
DATASET_RELEASE_VERSION = "4.2"
FPS = 10
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
STATE_DIM = 7
ACTION_DIM = 7
JOINT_STATE_DIM = 7
ROBOT_ID_DIM = 1
PHASE_DIM = 1
ROTATION_6D_DIM = 6
STAGE_TARGET_POSE_DIM = ROTATION_6D_DIM
ROTATION_TOLERANCE_DIM = 6
PHASE_LABELS = {
    "pregrasp": "Pre-grasp",
    "grasp": "Grasp",
    "postgrasp": "Post-grasp",
    "release": "Release",
}
PHASE_TO_ID = {phase: index for index, phase in enumerate(PHASE_LABELS)}
ROBOT_SPECS = {
    "panda": {"id": 0, "arm_dof": 7},
    "xarm7": {"id": 1, "arm_dof": 7},
    "ur5e": {"id": 2, "arm_dof": 6},
}
ROTATION_TOLERANCE_PROFILE_ORDER = (
    "rx_negative",
    "rx_positive",
    "ry_negative",
    "ry_positive",
    "rz_negative",
    "rz_positive",
)
DEFAULT_ROTATION_TOLERANCE_PROFILES_RAD = {
    "pregrasp": [0.0, 0.0, np.pi / 6, np.pi / 6, 0.0, 0.0],
    "grasp": [0.0] * ROTATION_TOLERANCE_DIM,
    "postgrasp": [0.0, 0.0, 0.0, 0.0, np.pi / 4, np.pi / 4],
    "release": [0.0] * ROTATION_TOLERANCE_DIM,
}
SINGLE_AXIS_TO_INDICES = {
    "x": (0, 1),  # rx_negative, rx_positive
    "y": (2, 3),  # ry_negative, ry_positive
    "z": (4, 5),  # rz_negative, rz_positive
}
CONVERSION_STATE_PATH = Path("meta/conversion_state.jsonl")

EpisodeSpec = tuple[Path, dict[str, Any], dict[str, list[float]], str, int, int]

TOLERANCE_SUMMARY_COLUMNS = tuple(
    f"{stage}_r{axis}_{side}"
    for stage in ("pre", "post")
    for axis in "xyz"
    for side in ("neg", "pos")
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Failed to read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require_vector(value: Any, size: int, *, field: str, source: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{source}: {field} must be a finite vector with shape ({size},), got {array}")
    return array


def rotation_vector_to_6d(value: Any, *, field: str, source: Path) -> np.ndarray:
    """Encode an axis-angle rotation as the first two rotation-matrix columns."""
    rotation_vector = _require_vector(value, 3, field=field, source=source)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector.astype(np.float64))
    return np.concatenate((rotation_matrix[:, 0], rotation_matrix[:, 1])).astype(np.float32)


def _default_tolerance_profiles() -> dict[str, list[float]]:
    return {name: list(values) for name, values in DEFAULT_ROTATION_TOLERANCE_PROFILES_RAD.items()}


def _load_tolerance_summary(path: Path) -> dict[tuple[str, str, str], dict[str, list[float]]]:
    """Load authoritative task-level directional bounds from a summary CSV.

    CSV bounds are expressed in degrees and converted to non-negative radian
    magnitudes in the LeRobot feature order.
    """
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or ())
            required = {"robot_uid", "training_scene_id", "training_task_id", *TOLERANCE_SUMMARY_COLUMNS}
            missing = sorted(required - fieldnames)
            if missing:
                raise ValueError(f"{path}: missing required columns {missing}")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"Failed to read tolerance summary {path}: {error}") from error
    if not rows:
        raise ValueError(f"Tolerance summary is empty: {path}")

    result: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = (row["robot_uid"], row["training_scene_id"], row["training_task_id"])
        if not all(key):
            raise ValueError(f"{path}:{row_number}: empty robot or task identity")
        if key in result:
            raise ValueError(f"{path}:{row_number}: duplicate tolerance row {key}")

        profiles = _default_tolerance_profiles()
        for stage, phase in (("pre", "pregrasp"), ("post", "postgrasp")):
            values = []
            for axis in "xyz":
                for side in ("neg", "pos"):
                    column = f"{stage}_r{axis}_{side}"
                    try:
                        degrees = float(row[column])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{path}:{row_number}: invalid {column} value {row[column]!r}") from error
                    if not math.isfinite(degrees) or degrees < 0.0:
                        raise ValueError(f"{path}:{row_number}: {column} must be a finite non-negative value")
                    values.append(math.radians(degrees))
            profiles[phase] = values
        profiles["grasp"] = [0.0] * ROTATION_TOLERANCE_DIM
        profiles["release"] = [0.0] * ROTATION_TOLERANCE_DIM
        result[key] = profiles
    return result


def _report_unused_tolerance_summary_rows(
    summary_profiles: dict[tuple[str, str, str], dict[str, list[float]]],
    used_summary_keys: set[tuple[str, str, str]],
    *,
    raw_dir: Path,
) -> None:
    unused_summary_keys = set(summary_profiles) - used_summary_keys
    if unused_summary_keys:
        print(
            f"Ignoring {len(unused_summary_keys)} tolerance row(s) for tasks "
            f"not present under {raw_dir}."
        )


def _tolerance_profiles_from_metadata(metadata: dict[str, Any], *, source: Path) -> dict[str, list[float]]:
    annotation = metadata.get("annotation")
    profiles = annotation.get("rotation_tolerance_profiles_rad") if isinstance(annotation, dict) else None
    if not isinstance(profiles, dict):
        print(
            f"WARNING: {source} has no annotation.rotation_tolerance_profiles_rad; "
            "using the original default task tolerance profiles.",
        )
        return _default_tolerance_profiles()

    order = annotation.get("rotation_tolerance_profile_order")
    if order is not None and tuple(order) != ROTATION_TOLERANCE_PROFILE_ORDER:
        raise ValueError(
            f"{source}: rotation_tolerance_profile_order must be {ROTATION_TOLERANCE_PROFILE_ORDER}, got {order}"
        )
    validated = {}
    for phase in PHASE_LABELS:
        value = _require_vector(
            profiles.get(phase),
            ROTATION_TOLERANCE_DIM,
            field=f"annotation.rotation_tolerance_profiles_rad.{phase}",
            source=source,
        )
        if np.any(value < 0.0):
            raise ValueError(f"{source}: rotation tolerance profile {phase!r} must contain non-negative magnitudes")
        validated[phase] = value.tolist()
    return validated


def _load_tolerance_annotation(
    task_dir: Path,
    task_profiles: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Load one optimized task-level tolerance profile shared by all episodes."""
    annotation_path = task_dir / "tolerance_annotation.json"
    profiles = {name: list(values) for name, values in task_profiles.items()}
    if annotation_path.is_file():
        document = _load_json(annotation_path)
        order = document.get("rotation_tolerance_bounds_order")
        if order is not None and tuple(order) != ROTATION_TOLERANCE_PROFILE_ORDER:
            raise ValueError(
                f"{annotation_path}: rotation_tolerance_bounds_order must be "
                f"{ROTATION_TOLERANCE_PROFILE_ORDER}, got {order}"
            )

        task_bounds = document.get("rotation_tolerance_bounds_rad")
        if not isinstance(task_bounds, dict):
            raise ValueError(f"{annotation_path}: rotation_tolerance_bounds_rad must be an object")
        for phase, value in task_bounds.items():
            if phase not in PHASE_LABELS:
                raise ValueError(f"{annotation_path}: unknown task-level tolerance phase {phase!r}")
            bounds = _require_vector(
                value,
                ROTATION_TOLERANCE_DIM,
                field=f"rotation_tolerance_bounds_rad.{phase}",
                source=annotation_path,
            )
            if np.any(bounds < 0.0):
                raise ValueError(f"{annotation_path}: task-level tolerance phase {phase!r} must be non-negative")
            profiles[phase] = bounds.tolist()

        # Per-episode `episode_refinements` are deliberately not applied. The
        # converted dataset uses one counterfactual-search result for every episode
        # of the task so the labels do not depend on episode-specific optimization.
        refinements = document.get("episode_refinements")
        refinement_count = len(refinements) if isinstance(refinements, list) else 0
        print(
            f"Loaded {annotation_path}: using optimized task-level tolerances for all episodes; "
            f"ignored {refinement_count} per-episode refinements."
        )

    profiles = _load_single_axis_searches(task_dir, profiles)
    return profiles


def _load_single_axis_searches(
    task_dir: Path,
    profiles: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Merge single-axis search results into tolerance profiles.

    Scans ``task_dir`` for ``single_axis_*.json`` files.  For each file the
    ``single_axis_only`` field determines which rotation axes were searched and
    ``rotation_tolerance_bounds_rad`` supplies the optimized per-phase values.
    Only the searched axes override the existing profiles; unsearched axes are
    left unchanged.
    """
    merged = {name: list(values) for name, values in profiles.items()}
    for search_path in sorted(task_dir.glob("single_axis_*.json")):
        document = _load_json(search_path)
        if document.get("method") != "single_axis_only_search":
            continue

        search_config = document.get("search")
        if not isinstance(search_config, dict):
            raise ValueError(f"{search_path}: search must be an object")
        searched_axes = search_config.get("single_axis_only")
        if not isinstance(searched_axes, list):
            raise ValueError(f"{search_path}: search.single_axis_only must be a list")

        bounds_rad = document.get("rotation_tolerance_bounds_rad")
        if not isinstance(bounds_rad, dict):
            raise ValueError(f"{search_path}: rotation_tolerance_bounds_rad must be an object")

        axis_indices: list[int] = []
        for axis in searched_axes:
            try:
                indices = SINGLE_AXIS_TO_INDICES[axis]
            except KeyError as error:
                raise ValueError(
                    f"{search_path}: unknown single_axis_only value {axis!r}; "
                    f"expected one of {tuple(SINGLE_AXIS_TO_INDICES)}"
                ) from error
            axis_indices.extend(indices)

        for phase in PHASE_LABELS:
            phase_bounds = bounds_rad.get(phase)
            if phase_bounds is None:
                continue
            bounds = _require_vector(
                phase_bounds,
                ROTATION_TOLERANCE_DIM,
                field=f"rotation_tolerance_bounds_rad.{phase}",
                source=search_path,
            )
            for idx in axis_indices:
                merged[phase][idx] = float(bounds[idx])

        print(
            f"Loaded {search_path.name}: single-axis search for axes {searched_axes}; "
            f"merged into tolerance profiles."
        )
    return merged


def _episode_key(entry: dict[str, Any], *, source: Path) -> tuple[int, int]:
    episode_index = entry.get("episode_index")
    seed = entry.get("seed")
    if not isinstance(episode_index, int) or isinstance(episode_index, bool):
        raise ValueError(f"{source}: episode_index must be an integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{source}: seed must be an integer")
    return episode_index, seed


def state_from_frame(frame: dict[str, Any], *, source: Path) -> np.ndarray:
    """Return XYZ + base-frame rotation vector + one symmetric finger position."""
    position = _require_vector(frame["ee_state"]["position_m"], 3, field="ee_state.position_m", source=source)
    rotation = _require_vector(
        frame["ee_state"]["rotation_vector_base_rad"],
        3,
        field="ee_state.rotation_vector_base_rad",
        source=source,
    )
    fingers = _require_vector(
        frame["gripper_state"]["finger_positions_m"],
        2,
        field="gripper_state.finger_positions_m",
        source=source,
    )
    symmetric_finger_position = np.asarray([0.5 * float(np.sum(fingers))], dtype=np.float32)
    return np.concatenate((position, rotation, symmetric_finger_position), dtype=np.float32)


def joint_state_from_frame(
    frame: dict[str, Any],
    *,
    robot_uid: str,
    source: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the schema's fixed-width joint position and velocity vectors."""
    joint_state = frame.get("joint_state")
    if not isinstance(joint_state, dict):
        raise ValueError(f"{source}: joint_state must be an object")
    position = _require_vector(
        joint_state.get("position_rad"),
        JOINT_STATE_DIM,
        field="joint_state.position_rad",
        source=source,
    )
    velocity = _require_vector(
        joint_state.get("velocity_rad_s"),
        JOINT_STATE_DIM,
        field="joint_state.velocity_rad_s",
        source=source,
    )
    if robot_uid == "ur5e" and (position[-1] != 0.0 or velocity[-1] != 0.0):
        raise ValueError(f"{source}: UR5e joint state dimension 7 must be the zero padding value")
    return position, velocity


def phase_from_frame(
    frame: dict[str, Any],
    *,
    source: Path,
) -> np.ndarray:
    """Return the canonical four-stage phase ID."""
    phase = frame.get("phase")
    if phase not in PHASE_TO_ID:
        raise ValueError(
            f"{source}: phase must be one of {tuple(PHASE_TO_ID)}, got {phase!r}"
        )
    return np.asarray([PHASE_TO_ID[phase]], dtype=np.int64)


def _stage_annotation_lookup(document: dict[str, Any], *, source: Path) -> dict[int, dict[str, Any]]:
    values = document.get("stage_annotations")
    if not isinstance(values, list):
        raise ValueError(f"{source}: stage_annotations must be a list")
    annotations: dict[int, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{source}: every stage annotation must be an object")
        annotation_id = value.get("id")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool):
            raise ValueError(f"{source}: every stage annotation must have an integer id")
        annotations[annotation_id] = value
    return annotations


def tolerance_targets_from_frame(
    frame: dict[str, Any],
    *,
    annotations: dict[int, dict[str, Any]],
    tolerance_profiles: dict[str, Any],
    source: Path,
) -> dict[str, np.ndarray]:
    """Resolve the annotation referenced by a frame without inferring phase semantics."""
    annotation_id = frame.get("stage_annotation_id")
    try:
        annotation = annotations[annotation_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source}: unknown stage_annotation_id {annotation_id!r}") from error

    target_pose = annotation.get("stage_target_pose")
    if not isinstance(target_pose, dict):
        raise ValueError(f"{source}: annotation {annotation_id} is missing stage_target_pose")
    profile = annotation.get("rotation_tolerance_profile")
    try:
        tolerance = tolerance_profiles[profile]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source}: unknown rotation tolerance profile {profile!r}") from error

    return {
        "stage_target_pose": rotation_vector_to_6d(
            target_pose.get("rotation_vector_base_rad"),
            field="stage_target_pose.rotation_vector_base_rad",
            source=source,
        ),
        "rotation_tolerance": _require_vector(
            tolerance,
            ROTATION_TOLERANCE_DIM,
            field=f"rotation_tolerance_profiles_rad.{profile}",
            source=source,
        ),
    }


def _task_directories(raw_dir: Path) -> list[Path]:
    _EXCLUDED_PARENTS = {"failed_episodes", "single_axis_failure_videos"}

    task_dirs = sorted(
        path.parent
        for path in raw_dir.rglob("manifest.json")
        if path.parent.name not in _EXCLUDED_PARENTS
        and not any(part in _EXCLUDED_PARENTS for part in path.parent.parts)
    )
    if not task_dirs:
        raise ValueError(f"No task directories containing manifest.json found under {raw_dir}")
    return task_dirs


def _infer_robot_uid(task_dir: Path) -> str:
    for part in reversed(task_dir.parts):
        if part.lower() in ROBOT_SPECS:
            return part.lower()
    raise ValueError(f"{task_dir}: cannot infer robot UID from the directory hierarchy")


def _validate_robot(metadata: dict[str, Any], *, source: Path) -> str:
    robot = metadata.get("robot")
    if not isinstance(robot, dict):
        raise ValueError(f"{source}: robot must be an object")
    robot_uid = robot.get("uid")
    try:
        spec = ROBOT_SPECS[robot_uid]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source}: robot.uid must be one of {tuple(ROBOT_SPECS)}, got {robot_uid!r}") from error

    arm_dof = robot.get("arm_dof")
    joint_names = robot.get("joint_names")
    if arm_dof != spec["arm_dof"]:
        raise ValueError(f"{source}: robot.arm_dof must be {spec['arm_dof']} for {robot_uid}")
    if not isinstance(joint_names, list) or len(joint_names) != arm_dof:
        raise ValueError(f"{source}: robot.joint_names must contain {arm_dof} physical joints")
    if robot.get("joint_state_dimension") != JOINT_STATE_DIM:
        raise ValueError(f"{source}: robot.joint_state_dimension must be {JOINT_STATE_DIM}")
    return robot_uid


def _validate_task(task_dir: Path) -> tuple[dict[str, Any], dict[str, Any], str, int, int]:
    metadata_path = task_dir / "task_metadata.json"
    manifest_path = task_dir / "manifest.json"
    manifest = _load_json(manifest_path)

    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path}: expected schema_version {SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )
    if not metadata_path.is_file():
        robot_uid = _infer_robot_uid(task_dir)
        print(
            f"WARNING: {metadata_path} is missing; assuming {FPS} Hz, {IMAGE_WIDTH}x{IMAGE_HEIGHT} videos, "
            f"robot={robot_uid}, and the original default task tolerance profiles."
        )
        metadata = {"annotation": {"rotation_tolerance_profiles_rad": _default_tolerance_profiles()}}
        return metadata, manifest, robot_uid, IMAGE_WIDTH, IMAGE_HEIGHT

    metadata = _load_json(metadata_path)
    if str(metadata.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError(
            f"{metadata_path}: expected schema_version {SCHEMA_VERSION}, got {metadata.get('schema_version')}"
        )
    if str(metadata.get("dataset_release_version")) != DATASET_RELEASE_VERSION:
        raise ValueError(
            f"{metadata_path}: expected dataset_release_version {DATASET_RELEASE_VERSION}, "
            f"got {metadata.get('dataset_release_version')}"
        )
    if float(metadata.get("frequency_hz", -1)) != FPS:
        raise ValueError(f"{metadata_path}: expected frequency_hz {FPS}")
    robot_uid = _validate_robot(metadata, source=metadata_path)
    action_format = metadata.get("action_format")
    if not isinstance(action_format, dict) or int(action_format.get("dimension", -1)) != ACTION_DIM:
        raise ValueError(f"{metadata_path}: action_format.dimension must be {ACTION_DIM}")
    if metadata.get("phase_labels") != PHASE_LABELS:
        raise ValueError(f"{metadata_path}: phase_labels must define the canonical four schema 7.0 phases")

    tolerance_profiles = _tolerance_profiles_from_metadata(metadata, source=metadata_path)
    annotation = dict(metadata.get("annotation") or {})
    annotation["rotation_tolerance_profiles_rad"] = tolerance_profiles
    metadata = {**metadata, "annotation": annotation}

    video = metadata.get("video", {})
    resolution = video.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError(f"{metadata_path}: video.resolution must be [width, height]")
    width, height = (int(resolution[0]), int(resolution[1]))
    if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(
            f"{metadata_path}: videos must already be resize-with-pad processed to "
            f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}, got {width}x{height}"
        )
    if float(video.get("fps", -1)) != FPS:
        raise ValueError(f"{metadata_path}: invalid video resolution or fps")
    return metadata, manifest, robot_uid, width, height


def _episode_entries(task_dir: Path, manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{task_dir / 'manifest.json'}: episodes must be a list")
    for entry in episodes:
        if not isinstance(entry, dict) or entry.get("success") is not True:
            raise ValueError(f"{task_dir / 'manifest.json'}: manifest must contain only successful episodes")
        yield entry


def _open_video(path: Path, *, expected_width: int, expected_height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (expected_width, expected_height):
        capture.release()
        raise ValueError(f"{path}: expected video size {(expected_width, expected_height)}, got {(width, height)}")
    return capture


def _read_rgb(capture: cv2.VideoCapture, *, path: Path, index: int) -> np.ndarray:
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"{path}: failed to decode frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _prepare_episode(
    episode_dir: Path,
    manifest_entry: dict[str, Any],
    *,
    tolerance_profiles: dict[str, list[float]],
    robot_uid: str,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    trajectory_path = episode_dir / "trajectory.json"
    document = _load_json(trajectory_path)
    episode = document.get("episode")
    trajectory = document.get("trajectory")
    if not isinstance(episode, dict) or not isinstance(trajectory, list):
        raise ValueError(f"{trajectory_path}: expected episode object and trajectory list")
    if episode.get("result", {}).get("success") is not True:
        raise ValueError(f"{trajectory_path}: episode is not successful")
    if _episode_key(episode, source=trajectory_path) != _episode_key(
        manifest_entry, source=episode_dir.parent / "manifest.json"
    ):
        raise ValueError(f"{trajectory_path}: episode index or seed does not match the manifest")
    frame_count = int(episode.get("frame_count", -1))
    if frame_count != len(trajectory) or frame_count != int(manifest_entry.get("frames", -1)):
        raise ValueError(f"{trajectory_path}: inconsistent trajectory, episode, or manifest frame count")
    instruction = episode.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{trajectory_path}: episode instruction must be non-empty")
    if instruction != manifest_entry.get("instruction"):
        raise ValueError(f"{trajectory_path}: instruction does not match the manifest")
    if not all(isinstance(frame, dict) for frame in trajectory):
        raise ValueError(f"{trajectory_path}: every trajectory frame must be an object")
    annotations = _stage_annotation_lookup(document, source=trajectory_path)

    main_path = episode_dir / "main_rgb.mp4"
    wrist_path = episode_dir / "wrist_rgb.mp4"
    main_capture = _open_video(main_path, expected_width=width, expected_height=height)
    wrist_capture = _open_video(wrist_path, expected_width=width, expected_height=height)
    try:
        # 1. Read all frames first (parallel-safe, no lock needed)
        frame_data_list: list[dict[str, Any]] = []
        for index, frame in enumerate(trajectory):
            if frame.get("index") != index or frame.get("video_frame_index") != index:
                raise ValueError(f"{trajectory_path}: invalid index alignment at frame {index}")
            action = _require_vector(frame.get("action"), ACTION_DIM, field="action", source=trajectory_path)
            joint_state_from_frame(
                frame,
                robot_uid=robot_uid,
                source=trajectory_path,
            )
            tolerance_targets = tolerance_targets_from_frame(
                frame,
                annotations=annotations,
                tolerance_profiles=tolerance_profiles,
                source=trajectory_path,
            )
            frame_data_list.append({
                "image": _read_rgb(main_capture, path=main_path, index=index),
                "wrist_image": _read_rgb(wrist_capture, path=wrist_path, index=index),
                "state": state_from_frame(frame, source=trajectory_path),
                "robot_id": np.asarray([ROBOT_SPECS[robot_uid]["id"]], dtype=np.int64),
                "actions": action,
                "phase": phase_from_frame(frame, source=trajectory_path),
                **tolerance_targets,
                "task": instruction,
            })

        main_extra, _ = main_capture.read()
        wrist_extra, _ = wrist_capture.read()
        if main_extra or wrist_extra:
            raise ValueError(f"{episode_dir}: video contains more frames than trajectory.json")
    finally:
        main_capture.release()
        wrist_capture.release()

    if not frame_data_list:
        raise ValueError(f"{trajectory_path}: successful episode must contain at least one frame")
    return frame_data_list


def _write_episode(dataset: LeRobotDataset, frame_data_list: list[dict[str, Any]]) -> None:
    """Write one prepared episode from the main thread.

    LeRobotDataset owns a single mutable episode buffer, so add_frame and
    save_episode must never be called concurrently or interleaved across
    episodes.
    """
    # LeRobot normally concatenates every newly embedded episode onto
    # ``hf_dataset``.  The converter already persists one parquet per episode
    # and never reads the in-memory aggregate, so retaining the full history
    # only causes memory to grow with the dataset (including all image bytes).
    dataset.hf_dataset = dataset.create_hf_dataset()
    for frame_data in frame_data_list:
        dataset.add_frame(frame_data)
    dataset.save_episode()


def _dataset_features() -> dict[str, dict[str, Any]]:
    return {
        "image": {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
        "robot_id": {"dtype": "int64", "shape": (ROBOT_ID_DIM,), "names": ["robot_id"]},
        "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        "phase": {"dtype": "int64", "shape": (PHASE_DIM,), "names": ["phase"]},
        "stage_target_pose": {
            "dtype": "float32",
            "shape": (STAGE_TARGET_POSE_DIM,),
            "names": ["r1x", "r1y", "r1z", "r2x", "r2y", "r2z"],
        },
        "rotation_tolerance": {
            "dtype": "float32",
            "shape": (ROTATION_TOLERANCE_DIM,),
            "names": list(ROTATION_TOLERANCE_PROFILE_ORDER),
        },
    }


def _open_dataset_for_append(
    repo_id: str,
    output_path: Path,
    *,
    image_writer_threads: int,
    image_writer_processes: int,
) -> LeRobotDataset:
    """Open an existing LeRobot dataset without loading its parquet history."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=output_path)
    dataset.repo_id = dataset.meta.repo_id
    dataset.root = dataset.meta.root
    dataset.revision = None
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = None
    if image_writer_processes or image_writer_threads:
        dataset.start_image_writer(image_writer_processes, image_writer_threads)
    dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON line {line_number} in {path}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} in {path}")
            rows.append(row)
    return rows


def _write_jsonlines_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        file.flush()
    temporary.replace(path)


def _contiguous_rows(rows: list[dict[str, Any]], *, field: str) -> int:
    count = 0
    for row in rows:
        if row.get(field) != count:
            break
        count += 1
    return count


def _contiguous_parquet_count(output_path: Path, info: dict[str, Any]) -> int:
    pattern = info.get("data_path")
    chunk_size = int(info.get("chunks_size", 1000))
    if not isinstance(pattern, str) or chunk_size <= 0:
        raise ValueError(f"Invalid LeRobot data_path or chunks_size in {output_path / 'meta/info.json'}")
    count = 0
    while True:
        relative_path = pattern.format(episode_chunk=count // chunk_size, episode_index=count)
        if not (output_path / relative_path).is_file():
            return count
        count += 1


def _episode_index_from_name(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("episode_"):
        return None
    try:
        return int(stem.removeprefix("episode_"))
    except ValueError:
        return None


def _repair_partial_dataset(output_path: Path) -> int:
    """Roll back files written after the last fully committed LeRobot episode."""
    info_path = output_path / "meta/info.json"
    info = _load_json(info_path)
    episode_path = output_path / "meta/episodes.jsonl"
    stats_path = output_path / "meta/episodes_stats.jsonl"
    episode_rows = _read_jsonlines(episode_path)
    stats_rows = _read_jsonlines(stats_path)
    committed = min(
        _contiguous_rows(episode_rows, field="episode_index"),
        _contiguous_rows(stats_rows, field="episode_index"),
        _contiguous_parquet_count(output_path, info),
    )

    _write_jsonlines_atomic(episode_path, episode_rows[:committed])
    _write_jsonlines_atomic(stats_path, stats_rows[:committed])
    for parquet_path in output_path.glob("data/chunk-*/episode_*.parquet"):
        episode_index = _episode_index_from_name(parquet_path)
        if episode_index is not None and episode_index >= committed:
            parquet_path.unlink()
    images_path = output_path / "images"
    if images_path.exists():
        shutil.rmtree(images_path)

    task_rows = _read_jsonlines(output_path / "meta/tasks.jsonl")
    total_frames = sum(int(row["length"]) for row in episode_rows[:committed])
    chunk_size = int(info["chunks_size"])
    video_keys = [feature for feature in info["features"].values() if feature["dtype"] == "video"]
    info.update({
        "total_episodes": committed,
        "total_frames": total_frames,
        "total_tasks": len(task_rows),
        "total_videos": committed * len(video_keys),
        "total_chunks": math.ceil(committed / chunk_size),
        "splits": {"train": f"0:{committed}"},
    })
    temporary_info = info_path.with_name(".info.json.tmp")
    temporary_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    temporary_info.replace(info_path)
    return committed


def _source_id(raw_dir: Path, spec: EpisodeSpec) -> str:
    episode_dir, entry, _, robot_uid, _, _ = spec
    relative_path = episode_dir.resolve().relative_to(raw_dir)
    return f"{robot_uid}:{relative_path.as_posix()}:{entry['episode_index']}:{entry['seed']}"


def _fingerprint_arrays(states: np.ndarray, actions: np.ndarray, robot_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array, dtype in ((states, "<f4"), (actions, "<f4"), (robot_ids, "<i8")):
        normalized = np.ascontiguousarray(array, dtype=dtype).reshape(len(array), -1)
        digest.update(np.asarray(normalized.shape, dtype="<i8").tobytes())
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def _raw_episode_fingerprint(spec: EpisodeSpec) -> str:
    episode_dir, _, _, robot_uid, _, _ = spec
    trajectory_path = episode_dir / "trajectory.json"
    document = _load_json(trajectory_path)
    trajectory = document.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"{trajectory_path}: expected a non-empty trajectory list")
    states = np.stack([state_from_frame(frame, source=trajectory_path) for frame in trajectory])
    actions = np.stack([
        _require_vector(frame.get("action"), ACTION_DIM, field="action", source=trajectory_path)
        for frame in trajectory
    ])
    robot_ids = np.full((len(trajectory), ROBOT_ID_DIM), ROBOT_SPECS[robot_uid]["id"], dtype=np.int64)
    return _fingerprint_arrays(states, actions, robot_ids)


def _parquet_episode_fingerprint(path: Path) -> str:
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=["state", "actions", "robot_id"])
    return _fingerprint_arrays(
        np.asarray(table["state"].to_pylist(), dtype=np.float32),
        np.asarray(table["actions"].to_pylist(), dtype=np.float32),
        np.asarray(table["robot_id"].to_pylist(), dtype=np.int64),
    )


def _load_or_rebuild_conversion_state(
    output_path: Path,
    raw_dir: Path,
    episode_specs: list[EpisodeSpec],
    committed: int,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    state_path = output_path / CONVERSION_STATE_PATH
    rows = _read_jsonlines(state_path)
    if len(rows) > committed:
        rows = rows[:committed]
    planned_by_id = {_source_id(raw_dir, spec): spec for spec in episode_specs}
    if len(planned_by_id) != len(episode_specs):
        raise ValueError("Raw dataset contains duplicate episode identities")
    completed_ids = []
    for expected_index, row in enumerate(rows):
        source_id = row.get("source_id")
        if row.get("episode_index") != expected_index or source_id not in planned_by_id:
            raise ValueError(f"Invalid or stale resume state in {state_path} at episode {expected_index}")
        completed_ids.append(source_id)
    if len(set(completed_ids)) != len(completed_ids):
        raise ValueError(f"Duplicate source episode in {state_path}")

    if len(rows) < committed:
        print(
            f"Rebuilding resume index for {committed - len(rows)} existing episodes "
            "from trajectory fingerprints (one-time operation)..."
        )
        info = _load_json(output_path / "meta/info.json")
        chunk_size = int(info["chunks_size"])
        pattern = str(info["data_path"])
        missing_output_fingerprints: dict[str, list[int]] = {}
        for episode_index in range(len(rows), committed):
            parquet_path = output_path / pattern.format(
                episode_chunk=episode_index // chunk_size,
                episode_index=episode_index,
            )
            fingerprint = _parquet_episode_fingerprint(parquet_path)
            missing_output_fingerprints.setdefault(fingerprint, []).append(episode_index)

        unmatched_specs = [spec for spec in episode_specs if _source_id(raw_dir, spec) not in completed_ids]
        fingerprint_to_ids: dict[str, list[str]] = {}
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                fingerprints = executor.map(_raw_episode_fingerprint, unmatched_specs)
                for spec, fingerprint in zip(unmatched_specs, fingerprints, strict=True):
                    if fingerprint in missing_output_fingerprints:
                        fingerprint_to_ids.setdefault(fingerprint, []).append(_source_id(raw_dir, spec))
        else:
            for spec in unmatched_specs:
                fingerprint = _raw_episode_fingerprint(spec)
                if fingerprint in missing_output_fingerprints:
                    fingerprint_to_ids.setdefault(fingerprint, []).append(_source_id(raw_dir, spec))

        recovered: dict[int, str] = {}
        for fingerprint, output_indices in missing_output_fingerprints.items():
            source_ids = sorted(fingerprint_to_ids.get(fingerprint, []))
            if len(source_ids) < len(output_indices):
                raise ValueError(
                    "Could not match every existing LeRobot episode to a raw trajectory; "
                    "the raw dataset may have changed."
                )
            recovered.update(dict(zip(output_indices, source_ids, strict=False)))
        for episode_index in range(len(rows), committed):
            rows.append({"episode_index": episode_index, "source_id": recovered[episode_index]})
        _write_jsonlines_atomic(state_path, rows)
        print(f"Resume index rebuilt: {len(rows)} completed raw trajectories identified.")
    return rows


def _append_conversion_state(output_path: Path, episode_index: int, source_id: str) -> None:
    state_path = output_path / CONVERSION_STATE_PATH
    with state_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"episode_index": episode_index, "source_id": source_id}) + "\n")
        file.flush()


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    tolerance_summary: Path | None = None,
    push_to_hub: bool = False,
    private: bool = True,
    overwrite: bool = False,
    resume: bool = False,
    max_episodes: int | None = None,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
    workers: int = 1,
    progress_interval_seconds: float = 5.0,
) -> None:
    """Convert all successful episodes under raw_dir into one LeRobot dataset."""
    import datasets
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import tqdm

    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise ValueError(f"Raw dataset directory does not exist: {raw_dir}")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    # LeRobot invokes Hugging Face Dataset.map/to_parquet for every episode.
    # Suppress those nested bars so the terminal shows one stable global ETA.
    datasets.disable_progress_bars()

    if tolerance_summary is None:
        candidate = raw_dir / "tolerance_summary.csv"
        tolerance_summary = candidate if candidate.is_file() else None
    else:
        tolerance_summary = tolerance_summary.expanduser().resolve()
        if not tolerance_summary.is_file():
            raise ValueError(f"Tolerance summary does not exist: {tolerance_summary}")
    summary_profiles = _load_tolerance_summary(tolerance_summary) if tolerance_summary is not None else {}
    used_summary_keys: set[tuple[str, str, str]] = set()
    if tolerance_summary is not None:
        print(f"Loaded authoritative task tolerances from {tolerance_summary} ({len(summary_profiles)} rows).")

    task_specs = []
    for task_dir in _task_directories(raw_dir):
        metadata, manifest, robot_uid, width, height = _validate_task(task_dir)
        if summary_profiles:
            relative_parts = task_dir.relative_to(raw_dir).parts
            if len(relative_parts) != 3:
                raise ValueError(
                    f"{task_dir}: expected robot/scene/task hierarchy for tolerance-summary lookup"
                )
            summary_key = (robot_uid, relative_parts[1], relative_parts[2])
            try:
                task_profiles = summary_profiles[summary_key]
            except KeyError as error:
                raise ValueError(f"{tolerance_summary}: no tolerance row for task {summary_key}") from error
            used_summary_keys.add(summary_key)
        else:
            task_profiles = _load_tolerance_annotation(
                task_dir,
                metadata["annotation"]["rotation_tolerance_profiles_rad"],
            )
        task_specs.append((task_dir, manifest, robot_uid, width, height, task_profiles))
    _report_unused_tolerance_summary_rows(summary_profiles, used_summary_keys, raw_dir=raw_dir)

    # Collect all episode specs first
    episode_specs: list[EpisodeSpec] = []
    missing_trajectories: list[Path] = []
    for task_dir, manifest, robot_uid, width, height, task_profiles in task_specs:
        for entry in _episode_entries(task_dir, manifest):
            episode_dir = task_dir / str(entry["path"])
            if not episode_dir.is_dir():
                raise ValueError(f"Episode directory does not exist: {episode_dir}")
            if not (episode_dir / "trajectory.json").is_file():
                missing_trajectories.append(episode_dir)
                continue
            episode_specs.append((episode_dir, entry, task_profiles, robot_uid, width, height))

    if max_episodes is not None:
        episode_specs = episode_specs[:max_episodes]

    if missing_trajectories:
        examples = ", ".join(str(path) for path in missing_trajectories[:3])
        suffix = "" if len(missing_trajectories) <= 3 else ", ..."
        print(
            f"Warning: skipping {len(missing_trajectories)} manifest entries whose trajectory.json is missing. "
            f"Examples: {examples}{suffix}"
        )

    output_path = HF_LEROBOT_HOME / repo_id
    completed_rows: list[dict[str, Any]] = []
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        elif not resume:
            raise FileExistsError(
                f"Output dataset already exists: {output_path}. Pass --resume to continue or --overwrite to replace it."
            )

    if output_path.exists():
        committed = _repair_partial_dataset(output_path)
        print(f"Resume check: {committed} fully committed LeRobot episodes; rolled back any incomplete tail files.")
        completed_rows = _load_or_rebuild_conversion_state(
            output_path,
            raw_dir,
            episode_specs,
            committed,
            workers=workers,
        )
        dataset = _open_dataset_for_append(
            repo_id,
            output_path,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )
        expected_features = _dataset_features()
        for key, expected in expected_features.items():
            actual = dataset.meta.features.get(key)
            if actual != expected:
                raise ValueError(f"Existing dataset feature {key!r} is incompatible: {actual!r} != {expected!r}")
        if dataset.meta.robot_type != "tasktol_mujoco" or dataset.meta.fps != FPS:
            raise ValueError("Existing dataset robot_type or fps is incompatible with this converter")
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="tasktol_mujoco",
            fps=FPS,
            features=_dataset_features(),
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )
        _write_jsonlines_atomic(output_path / CONVERSION_STATE_PATH, [])

    completed_ids = {str(row["source_id"]) for row in completed_rows}
    remaining_specs = [spec for spec in episode_specs if _source_id(raw_dir, spec) not in completed_ids]
    if len(completed_ids) + len(remaining_specs) != len(episode_specs):
        raise ValueError("Resume state contains duplicate or unplanned raw episodes")

    total_episodes = len(episode_specs)
    total_frames = sum(int(entry.get("frames", 0)) for _, entry, _, _, _, _ in episode_specs)
    completed_frames = total_frames - sum(
        int(entry.get("frames", 0)) for _, entry, _, _, _, _ in remaining_specs
    )
    print(
        f"Planned conversion: {total_episodes} episodes, {total_frames} frames, workers={workers}; "
        f"completed={len(completed_ids)}, remaining={len(remaining_specs)}"
    )
    progress = tqdm.tqdm(
        total=total_frames,
        initial=completed_frames,
        desc="Converting",
        unit="frame",
        mininterval=progress_interval_seconds,
        maxinterval=max(10.0, progress_interval_seconds * 2),
        dynamic_ncols=True,
    )
    converted = len(completed_ids)
    progress.set_postfix_str(f"episodes={converted}/{total_episodes}", refresh=False)
    if workers > 1 and remaining_specs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            episode_iter = iter(remaining_specs)
            pending: list[tuple[EpisodeSpec, Any]] = []

            def submit_next() -> bool:
                try:
                    spec = next(episode_iter)
                except StopIteration:
                    return False
                episode_dir, entry, task_profiles, robot_uid, width, height = spec
                future = executor.submit(
                    _prepare_episode,
                    episode_dir,
                    entry,
                    tolerance_profiles=task_profiles,
                    robot_uid=robot_uid,
                    width=width,
                    height=height,
                )
                pending.append((spec, future))
                return True

            for _ in range(min(workers, len(remaining_specs))):
                submit_next()

            while pending:
                spec, future = pending.pop(0)
                episode_dir = spec[0]
                try:
                    frame_data_list = future.result()
                except Exception as error:
                    raise RuntimeError(f"Failed to prepare episode {episode_dir}") from error
                episode_index = dataset.meta.total_episodes
                _write_episode(dataset, frame_data_list)
                converted += 1
                _append_conversion_state(output_path, episode_index, _source_id(raw_dir, spec))
                progress.set_postfix_str(f"episodes={converted}/{total_episodes}", refresh=False)
                progress.update(len(frame_data_list))
                submit_next()
    else:
        for spec in remaining_specs:
            episode_dir, entry, task_profiles, robot_uid, width, height = spec
            frame_data_list = _prepare_episode(
                episode_dir,
                entry,
                tolerance_profiles=task_profiles,
                robot_uid=robot_uid,
                width=width,
                height=height,
            )
            episode_index = dataset.meta.total_episodes
            _write_episode(dataset, frame_data_list)
            converted += 1
            _append_conversion_state(output_path, episode_index, _source_id(raw_dir, spec))
            progress.set_postfix_str(f"episodes={converted}/{total_episodes}", refresh=False)
            progress.update(len(frame_data_list))

    progress.close()

    if total_episodes == 0:
        raise ValueError("No episodes were converted")
    if push_to_hub:
        dataset.push_to_hub(
            tags=["panda", "xarm7", "ur5e", "mujoco", "tasktol-vla"],
            private=private,
            push_videos=True,
            license="apache-2.0",
        )
    print(f"Dataset complete: {converted} episodes at {output_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
