"""Convert TaskTol-VLA multi-robot MuJoCo schema 8.0 data to LeRobot format.

Example:
uv run python examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
    --raw-dir /path/to/franka_mujoco/datasets \
    --repo-id caozilin/franka_mujoco
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from openpi.shared import franka_mujoco_dataset as dataset_format

if TYPE_CHECKING:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


SCHEMA_VERSION = dataset_format.SCHEMA_VERSION
DATASET_RELEASE_VERSION = dataset_format.DATASET_RELEASE_VERSION
FPS = 10
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
STATE_DIM = 7
ACTION_DIM = 7
JOINT_STATE_DIM = 7
ROBOT_ID_DIM = 1
PHASE_DIM = 1
ROTATION_TOLERANCE_DIM = 6
TRAJECTORY_TYPE_DIM = 1
NOMINAL_TRAJECTORY = "nominal"
TOLERANCE_TRAJECTORY = "tolerance"
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
ROTATION_TOLERANCE_PROFILE_ORDER = dataset_format.ROTATION_TOLERANCE_ORDER
CONVERSION_STATE_PATH = Path("meta/conversion_state.jsonl")

EpisodeSpec = tuple[Path, dict[str, Any], str, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Failed to read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def trajectory_is_tolerance(
    document: dict[str, Any],
    manifest_entry: dict[str, Any],
    *,
    source: Path,
) -> bool:
    """Resolve the schema 8.0 execution variant, independently of tolerance labels."""
    episode = document.get("episode")
    trajectory = document.get("trajectory")
    if not isinstance(episode, dict) or not isinstance(trajectory, list):
        raise ValueError(f"{source}: expected episode object and trajectory list")

    type_values = [
        record["execution_variant"]
        for record in (manifest_entry, episode, *trajectory)
        if isinstance(record, dict) and "execution_variant" in record
    ]
    invalid_types = sorted(
        {
            str(value)
            for value in type_values
            if value
            not in (
                NOMINAL_TRAJECTORY,
                TOLERANCE_TRAJECTORY,
            )
        }
    )
    if invalid_types:
        raise ValueError(f"{source}: invalid execution_variant value(s) {invalid_types}")
    if len(set(type_values)) > 1:
        raise ValueError(f"{source}: inconsistent execution_variant labels")
    # Base collection omits the variant; mixed datasets declare nominal/tolerance.
    return bool(type_values and type_values[0] == TOLERANCE_TRAJECTORY)


def _require_vector(value: Any, size: int, *, field: str, source: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{source}: {field} must be a finite vector with shape ({size},), got {array}")
    return array


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
        raise ValueError(f"{source}: phase must be one of {tuple(PHASE_TO_ID)}, got {phase!r}")
    return np.asarray([PHASE_TO_ID[phase]], dtype=np.int64)


def _task_directories(raw_dir: Path) -> list[Path]:
    excluded_parents = {"failed_episodes"}

    task_dirs = sorted(
        path.parent
        for path in raw_dir.rglob("manifest.json")
        if path.parent.name not in excluded_parents and not any(part in excluded_parents for part in path.parent.parts)
    )
    if not task_dirs:
        raise ValueError(f"No task directories containing manifest.json found under {raw_dir}")
    return task_dirs


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
    if action_format.get("rotation") != dataset_format.ACTION_ROTATION_CONVENTION:
        raise ValueError(f"{metadata_path}: actions must use base-frame left-multiplied rotation-vector increments")
    if metadata.get("phase_labels") != PHASE_LABELS:
        raise ValueError(f"{metadata_path}: phase_labels must define the canonical four schema 8.0 phases")

    if metadata.get("tolerance_labels") != dataset_format.tolerance_label_metadata():
        raise ValueError(f"{metadata_path}: invalid deferred tolerance label contract")
    if "annotation" in metadata:
        raise ValueError(f"{metadata_path}: task-level tolerance annotations are not part of schema 8.0")

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
    joint_task_tolerance: bool,
    robot_uid: str,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    trajectory_path = episode_dir / "trajectory.json"
    document, trajectory_sha256 = dataset_format.read_trajectory(episode_dir)
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
    tolerances = (
        dataset_format.load_rotation_tolerance(episode_dir, document, trajectory_sha256)
        if joint_task_tolerance
        else None
    )
    expected_video_counts = {"main_rgb.mp4": frame_count, "wrist_rgb.mp4": frame_count}
    if episode.get("video_frame_counts") != expected_video_counts:
        raise ValueError(f"{trajectory_path}: inconsistent video_frame_counts")
    is_tolerance = trajectory_is_tolerance(document, manifest_entry, source=trajectory_path)

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
            tolerance_targets = {} if tolerances is None else {"rotation_tolerance": tolerances[index]}
            frame_data_list.append(
                {
                    "image": _read_rgb(main_capture, path=main_path, index=index),
                    "wrist_image": _read_rgb(wrist_capture, path=wrist_path, index=index),
                    "state": state_from_frame(frame, source=trajectory_path),
                    "robot_id": np.asarray([ROBOT_SPECS[robot_uid]["id"]], dtype=np.int64),
                    "actions": action,
                    "phase": phase_from_frame(frame, source=trajectory_path),
                    "subtask_index": np.asarray([frame["subtask_index"]], dtype=np.int64),
                    "trajectory_is_tolerance": np.asarray([is_tolerance], dtype=np.int64),
                    **tolerance_targets,
                    "task": instruction,
                }
            )

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


def _dataset_features(*, joint_task_tolerance: bool) -> dict[str, dict[str, Any]]:
    features = {
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
        "subtask_index": {"dtype": "int64", "shape": (1,), "names": ["subtask_index"]},
        "trajectory_is_tolerance": {
            "dtype": "int64",
            "shape": (TRAJECTORY_TYPE_DIM,),
            "names": ["is_tolerance"],
        },
    }
    if joint_task_tolerance:
        features["rotation_tolerance"] = {
            "dtype": "float32",
            "shape": (ROTATION_TOLERANCE_DIM,),
            "names": list(ROTATION_TOLERANCE_PROFILE_ORDER),
        }
    return features


def _validate_dataset_features(info: dict[str, Any], *, joint_task_tolerance: bool) -> None:
    from lerobot.common.datasets.utils import DEFAULT_FEATURES

    dataset_format.validate_export_info(info, joint_task_tolerance=joint_task_tolerance)
    if info["joint_task_tolerance"] != joint_task_tolerance:
        raise ValueError("Cannot change conversion mode while resuming; use a new dataset or --overwrite")
    expected = {**_dataset_features(joint_task_tolerance=joint_task_tolerance), **DEFAULT_FEATURES}
    # Metadata JSON serializes tuple shapes as lists.
    expected = json.loads(json.dumps(expected))
    if info.get("features") != expected:
        raise ValueError("Existing dataset features are incompatible with the current conversion mode")
    if info.get("robot_type") != "tasktol_mujoco" or info.get("fps") != FPS:
        raise ValueError("Existing dataset robot_type or fps is incompatible")


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
    info.update(
        {
            "total_episodes": committed,
            "total_frames": total_frames,
            "total_tasks": len(task_rows),
            "total_videos": committed * len(video_keys),
            "total_chunks": math.ceil(committed / chunk_size),
            "splits": {"train": f"0:{committed}"},
        }
    )
    temporary_info = info_path.with_name(".info.json.tmp")
    temporary_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    temporary_info.replace(info_path)
    return committed


def _source_id(raw_dir: Path, spec: EpisodeSpec) -> str:
    episode_dir, entry, robot_uid, _, _ = spec
    relative_path = episode_dir.resolve().relative_to(raw_dir)
    return f"{robot_uid}:{relative_path.as_posix()}:{entry['episode_index']}:{entry['seed']}"


def _source_fingerprint(spec: EpisodeSpec, *, joint_task_tolerance: bool) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(spec[1], sort_keys=True, ensure_ascii=False).encode("utf-8"))
    names = ["trajectory.json"]
    if joint_task_tolerance:
        names.append(dataset_format.TOLERANCE_LABEL_FILE)
    for name in names:
        digest.update(name.encode())
        digest.update(hashlib.sha256((spec[0] / name).read_bytes()).digest())
    return digest.hexdigest()


def _load_conversion_state(
    output_path: Path,
    raw_dir: Path,
    episode_specs: list[EpisodeSpec],
    committed: int,
    *,
    joint_task_tolerance: bool,
) -> list[dict[str, Any]]:
    state_path = output_path / CONVERSION_STATE_PATH
    rows = _read_jsonlines(state_path)[:committed]
    planned = {_source_id(raw_dir, spec): spec for spec in episode_specs}
    if len(planned) != len(episode_specs) or len(rows) != committed:
        raise ValueError(f"{state_path}: incomplete conversion journal or duplicate source identities")
    completed_ids = set()
    for index, row in enumerate(rows):
        source_id = row.get("source_id")
        if row.get("episode_index") != index or source_id not in planned or source_id in completed_ids:
            raise ValueError(f"{state_path}: invalid resume state at episode {index}")
        if row.get("source_sha256") != _source_fingerprint(
            planned[source_id], joint_task_tolerance=joint_task_tolerance
        ):
            raise ValueError(f"{state_path}: manifest, trajectory or tolerance labels changed; rebuild the export")
        completed_ids.add(source_id)
    _write_jsonlines_atomic(state_path, rows)
    return rows


def _append_conversion_state(
    output_path: Path,
    episode_index: int,
    source_id: str,
    source_sha256: str,
) -> None:
    with (output_path / CONVERSION_STATE_PATH).open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "source_id": source_id,
                    "source_sha256": source_sha256,
                }
            )
            + "\n"
        )
        file.flush()


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    joint_task_tolerance: bool = True,
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

    episode_specs: list[EpisodeSpec] = []
    for task_dir in _task_directories(raw_dir):
        _, manifest, robot_uid, width, height = _validate_task(task_dir)
        for entry in _episode_entries(task_dir, manifest):
            episode_dir = task_dir / str(entry["path"])
            if not episode_dir.resolve().is_relative_to(task_dir.resolve()):
                raise ValueError(f"Episode path escapes task directory: {episode_dir}")
            episode_specs.append((episode_dir, entry, robot_uid, width, height))

    if max_episodes is not None:
        episode_specs = episode_specs[:max_episodes]
    if not episode_specs:
        raise ValueError("No successful episodes found")
    if len({_source_id(raw_dir, spec) for spec in episode_specs}) != len(episode_specs):
        raise ValueError("Raw dataset contains duplicate episode identities")
    for episode_dir, _, _, _, _ in episode_specs:
        for name in ("trajectory.json", "main_rgb.mp4", "wrist_rgb.mp4"):
            if not (episode_dir / name).is_file():
                raise FileNotFoundError(f"Required episode file is missing: {episode_dir / name}")
        document, digest = dataset_format.read_trajectory(episode_dir)
        if joint_task_tolerance:
            dataset_format.load_rotation_tolerance(episode_dir, document, digest)

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
        info = _load_json(output_path / "meta/info.json")
        _validate_dataset_features(info, joint_task_tolerance=joint_task_tolerance)
        committed = _repair_partial_dataset(output_path)
        print(f"Resume check: {committed} fully committed LeRobot episodes; rolled back any incomplete tail files.")
        completed_rows = _load_conversion_state(
            output_path,
            raw_dir,
            episode_specs,
            committed,
            joint_task_tolerance=joint_task_tolerance,
        )
        dataset = _open_dataset_for_append(
            repo_id,
            output_path,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="tasktol_mujoco",
            fps=FPS,
            features=_dataset_features(joint_task_tolerance=joint_task_tolerance),
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )
        from lerobot.common.datasets.utils import write_info

        dataset.meta.info.update(dataset_format.export_metadata(joint_task_tolerance=joint_task_tolerance))
        write_info(dataset.meta.info, output_path)
        _write_jsonlines_atomic(output_path / CONVERSION_STATE_PATH, [])

    completed_ids = {str(row["source_id"]) for row in completed_rows}
    remaining_specs = [spec for spec in episode_specs if _source_id(raw_dir, spec) not in completed_ids]
    if len(completed_ids) + len(remaining_specs) != len(episode_specs):
        raise ValueError("Resume state contains duplicate or unplanned raw episodes")

    total_episodes = len(episode_specs)
    total_frames = sum(int(entry.get("frames", 0)) for _, entry, _, _, _ in episode_specs)
    completed_frames = total_frames - sum(int(entry.get("frames", 0)) for _, entry, _, _, _ in remaining_specs)
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
                episode_dir, entry, robot_uid, width, height = spec
                future = executor.submit(
                    _prepare_episode,
                    episode_dir,
                    entry,
                    joint_task_tolerance=joint_task_tolerance,
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
                _append_conversion_state(
                    output_path,
                    episode_index,
                    _source_id(raw_dir, spec),
                    _source_fingerprint(spec, joint_task_tolerance=joint_task_tolerance),
                )
                _write_episode(dataset, frame_data_list)
                converted += 1
                progress.set_postfix_str(f"episodes={converted}/{total_episodes}", refresh=False)
                progress.update(len(frame_data_list))
                submit_next()
    else:
        for spec in remaining_specs:
            episode_dir, entry, robot_uid, width, height = spec
            frame_data_list = _prepare_episode(
                episode_dir,
                entry,
                joint_task_tolerance=joint_task_tolerance,
                robot_uid=robot_uid,
                width=width,
                height=height,
            )
            episode_index = dataset.meta.total_episodes
            _append_conversion_state(
                output_path,
                episode_index,
                _source_id(raw_dir, spec),
                _source_fingerprint(spec, joint_task_tolerance=joint_task_tolerance),
            )
            _write_episode(dataset, frame_data_list)
            converted += 1
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
