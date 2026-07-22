"""Convert TaskTol-VLA Franka MuJoCo schema 5.0 data to LeRobot format.

Example:
uv run examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
    --raw-dir /path/to/franka_mujoco/datasets \
    --repo-id caozilin/franka_mujoco
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


SCHEMA_VERSION = "5.0"
DATASET_RELEASE_VERSION = "2.0"
FPS = 10
STATE_DIM = 7
ACTION_DIM = 7
PHASE_DIM = 1
STAGE_TARGET_POSE_DIM = 3
TOLERANCE_FRAME_DIM = 3
ROTATION_TOLERANCE_DIM = 3
PHASE_LABELS = {
    "pregrasp": "Pre-grasp",
    "grasp": "Grasp",
    "postgrasp": "Post-grasp",
    "release": "Release",
}
PHASE_TO_ID = {phase: index for index, phase in enumerate(PHASE_LABELS)}


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
        "stage_target_pose": _require_vector(
            target_pose.get("rotation_vector_base_rad"),
            STAGE_TARGET_POSE_DIM,
            field="stage_target_pose.rotation_vector_base_rad",
            source=source,
        ),
        "tolerance_frame": _require_vector(
            annotation.get("tolerance_frame_rotation_vector_base_rad"),
            TOLERANCE_FRAME_DIM,
            field="tolerance_frame_rotation_vector_base_rad",
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
    task_dirs = sorted(path.parent for path in raw_dir.glob("*/manifest.json"))
    if not task_dirs:
        raise ValueError(f"No task directories containing manifest.json found under {raw_dir}")
    return task_dirs


def _validate_task(task_dir: Path) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    metadata_path = task_dir / "task_metadata.json"
    manifest_path = task_dir / "manifest.json"
    metadata = _load_json(metadata_path)
    manifest = _load_json(manifest_path)

    for source, document in ((metadata_path, metadata), (manifest_path, manifest)):
        if str(document.get("schema_version")) != SCHEMA_VERSION:
            actual_version = document.get("schema_version")
            raise ValueError(f"{source}: expected schema_version {SCHEMA_VERSION}, got {actual_version}")
    if str(metadata.get("dataset_release_version")) != DATASET_RELEASE_VERSION:
        raise ValueError(
            f"{metadata_path}: expected dataset_release_version {DATASET_RELEASE_VERSION}, "
            f"got {metadata.get('dataset_release_version')}"
        )
    if float(metadata.get("frequency_hz", -1)) != FPS:
        raise ValueError(f"{metadata_path}: expected frequency_hz {FPS}")
    action_format = metadata.get("action_format")
    if not isinstance(action_format, dict) or int(action_format.get("dimension", -1)) != ACTION_DIM:
        raise ValueError(f"{metadata_path}: action_format.dimension must be {ACTION_DIM}")
    if metadata.get("phase_labels") != PHASE_LABELS:
        raise ValueError(f"{metadata_path}: phase_labels must define the canonical four schema 5.0 phases")

    video = metadata.get("video", {})
    resolution = video.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError(f"{metadata_path}: video.resolution must be [width, height]")
    width, height = (int(resolution[0]), int(resolution[1]))
    if width <= 0 or height <= 0 or float(video.get("fps", -1)) != FPS:
        raise ValueError(f"{metadata_path}: invalid video resolution or fps")
    return metadata, manifest, width, height


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


def _convert_episode(
    dataset: LeRobotDataset,
    episode_dir: Path,
    manifest_entry: dict[str, Any],
    *,
    metadata: dict[str, Any],
    width: int,
    height: int,
) -> None:
    trajectory_path = episode_dir / "trajectory.json"
    document = _load_json(trajectory_path)
    episode = document.get("episode")
    trajectory = document.get("trajectory")
    if not isinstance(episode, dict) or not isinstance(trajectory, list):
        raise ValueError(f"{trajectory_path}: expected episode object and trajectory list")
    if episode.get("result", {}).get("success") is not True:
        raise ValueError(f"{trajectory_path}: episode is not successful")
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
    annotation_metadata = metadata.get("annotation")
    tolerance_profiles = (
        annotation_metadata.get("rotation_tolerance_profiles_rad")
        if isinstance(annotation_metadata, dict)
        else None
    )
    if not isinstance(tolerance_profiles, dict):
        raise ValueError(f"{episode_dir.parent / 'task_metadata.json'}: missing rotation tolerance profiles")

    main_path = episode_dir / "main_rgb.mp4"
    wrist_path = episode_dir / "wrist_rgb.mp4"
    main_capture = _open_video(main_path, expected_width=width, expected_height=height)
    wrist_capture = _open_video(wrist_path, expected_width=width, expected_height=height)
    try:
        for index, frame in enumerate(trajectory):
            if frame.get("index") != index or frame.get("video_frame_index") != index:
                raise ValueError(f"{trajectory_path}: invalid index alignment at frame {index}")
            action = _require_vector(frame.get("action"), ACTION_DIM, field="action", source=trajectory_path)
            tolerance_targets = tolerance_targets_from_frame(
                frame,
                annotations=annotations,
                tolerance_profiles=tolerance_profiles,
                source=trajectory_path,
            )
            dataset.add_frame(
                {
                    "image": _read_rgb(main_capture, path=main_path, index=index),
                    "wrist_image": _read_rgb(wrist_capture, path=wrist_path, index=index),
                    "state": state_from_frame(frame, source=trajectory_path),
                    "actions": action,
                    "phase": phase_from_frame(frame, source=trajectory_path),
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

    dataset.save_episode()


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    push_to_hub: bool = False,
    private: bool = True,
    overwrite: bool = False,
    max_episodes: int | None = None,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
) -> None:
    """Convert all successful episodes under raw_dir into one LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import tqdm

    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise ValueError(f"Raw dataset directory does not exist: {raw_dir}")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    task_specs = []
    common_size = None
    for task_dir in _task_directories(raw_dir):
        metadata, manifest, width, height = _validate_task(task_dir)
        if common_size is None:
            common_size = (width, height)
        elif common_size != (width, height):
            raise ValueError(f"All tasks must use one video resolution; got {common_size} and {(width, height)}")
        task_specs.append((task_dir, metadata, manifest))
    assert common_size is not None
    width, height = common_size

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="franka_panda_mujoco",
        fps=FPS,
        features={
            "image": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
            "phase": {
                "dtype": "int64",
                "shape": (PHASE_DIM,),
                "names": ["phase"],
            },
            "stage_target_pose": {
                "dtype": "float32",
                "shape": (STAGE_TARGET_POSE_DIM,),
                "names": ["rotation_vector_base_rad"],
            },
            "tolerance_frame": {
                "dtype": "float32",
                "shape": (TOLERANCE_FRAME_DIM,),
                "names": ["rotation_vector_base_rad"],
            },
            "rotation_tolerance": {
                "dtype": "float32",
                "shape": (ROTATION_TOLERANCE_DIM,),
                "names": ["x", "y", "z"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    converted = 0
    total = sum(len(list(_episode_entries(task_dir, manifest))) for task_dir, _, manifest in task_specs)
    if max_episodes is not None:
        total = min(total, max_episodes)
    progress = tqdm.tqdm(total=total, desc="Converting episodes")
    try:
        for task_dir, metadata, manifest in task_specs:
            for entry in _episode_entries(task_dir, manifest):
                if max_episodes is not None and converted >= max_episodes:
                    break
                episode_dir = task_dir / str(entry["path"])
                if not episode_dir.is_dir():
                    raise ValueError(f"Episode directory does not exist: {episode_dir}")
                _convert_episode(
                    dataset, episode_dir, entry, metadata=metadata, width=width, height=height
                )
                converted += 1
                progress.update()
            if max_episodes is not None and converted >= max_episodes:
                break
    finally:
        progress.close()

    if converted == 0:
        raise ValueError("No episodes were converted")
    if push_to_hub:
        dataset.push_to_hub(
            tags=["franka", "panda", "mujoco", "tasktol-vla"],
            private=private,
            push_videos=True,
            license="apache-2.0",
        )
    print(f"Converted {converted} episodes to {output_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
