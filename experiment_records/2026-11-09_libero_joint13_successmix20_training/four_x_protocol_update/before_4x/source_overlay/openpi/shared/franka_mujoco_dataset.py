"""TaskTol schema 8.0 source data and deferred six-dimensional supervision.

This reader has no simulation dependency. It implements the data contract in
TaskTol's tasktol/data/DATASET_FORMAT_ZH.md.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "8.0"
DATASET_RELEASE_VERSION = "5.0"
SOURCE_SCHEMA = "tasktol_mujoco/8.0"
ACTION_ROTATION_CONVENTION = "base-frame rotation-vector delta RxRyRz in radians; left-multiplied"
TOLERANCE_LABEL_FILE = "tolerance_labels.json"
ROTATION_TOLERANCE_ORDER = (
    "rx_negative",
    "rx_positive",
    "ry_negative",
    "ry_positive",
    "rz_negative",
    "rz_positive",
)
PHASES = ("pregrasp", "grasp", "postgrasp", "release")
LABELED_PHASES = frozenset(("pregrasp", "postgrasp"))


def tolerance_label_metadata() -> dict[str, Any]:
    return {
        "file": TOLERANCE_LABEL_FILE,
        "schema_version": "1.0",
        "dimension": 6,
        "units": "rad",
        "order": list(ROTATION_TOLERANCE_ORDER),
        "coordinate_convention": "fixed_axis_xyz_rpy",
        "rotation_composition": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "reference": "realtime_nominal",
        "assignment": "subtask_index_and_phase",
        "missing_labels": "unlabeled; never substitute zero for Pre/Post",
        "strict_phases": ["grasp", "release"],
    }


def export_metadata(*, joint_task_tolerance: bool) -> dict[str, Any]:
    return {
        "source_schema": SOURCE_SCHEMA,
        "source_dataset_release_version": DATASET_RELEASE_VERSION,
        "joint_task_tolerance": joint_task_tolerance,
        "tolerance_labels": tolerance_label_metadata() if joint_task_tolerance else None,
    }


def validate_export_info(info: dict[str, Any], *, joint_task_tolerance: bool | None = None) -> None:
    """Reject stale exports before loading their Parquet or Arrow samples."""
    mode = info.get("joint_task_tolerance")
    if type(mode) is not bool:
        raise ValueError("Missing current TaskTol export metadata; reconvert the schema 8.0 dataset")
    expected = export_metadata(joint_task_tolerance=mode)
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError("Incompatible TaskTol source schema or label contract; reconvert the schema 8.0 dataset")
    if joint_task_tolerance and not mode:
        raise ValueError("Joint training requires a labeled 7+6 export")


def read_trajectory(episode_dir: Path) -> tuple[dict[str, Any], str]:
    path = episode_dir / "trajectory.json"
    content = path.read_bytes()
    document = json.loads(content)
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: expected schema_version {SCHEMA_VERSION}")
    frames = document.get("trajectory")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path}: trajectory must be a non-empty list")
    if not isinstance(document.get("episode"), dict) or document["episode"].get("frame_count") != len(frames):
        raise ValueError(f"{path}: inconsistent episode frame_count")
    if "stage_annotations" in document:
        raise ValueError(f"{path}: stage annotations are not part of schema 8.0")
    forbidden = {
        "stage_annotation_id",
        "tolerance_frame_code",
        "tolerance_frame",
        "tolerance_frame_rotation_vector_base_rad",
        "rotation_tolerance_rad",
    }
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or forbidden.intersection(frame):
            raise ValueError(f"{path}: base frames must not contain stage annotations or tolerance labels")
        if any(type(frame.get(key)) is not int or frame[key] != index for key in ("index", "video_frame_index")):
            raise ValueError(f"{path}: invalid trajectory/video alignment at frame {index}")
        timestamp = frame.get("time_s")
        if type(timestamp) not in (int, float) or not math.isclose(timestamp, index / 10, abs_tol=1e-9):
            raise ValueError(f"{path}: invalid compact timestamp at frame {index}")
        subtask = frame.get("subtask_index")
        if type(subtask) is not int or subtask < 0 or frame.get("phase") not in PHASES:
            raise ValueError(f"{path}: invalid subtask_index or phase at frame {index}")
        action = np.asarray(frame.get("action"), dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError(f"{path}: action must contain seven finite values")
    return document, hashlib.sha256(content).hexdigest()


def load_rotation_tolerance(
    episode_dir: Path,
    document: dict[str, Any],
    trajectory_sha256: str,
) -> np.ndarray:
    """Expand completed subtask/phase labels to row-aligned [N,6] targets."""
    path = episode_dir / TOLERANCE_LABEL_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing tolerance labels; use action-only conversion or complete labeling")
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or any(
        labels.get(key) != value for key, value in tolerance_label_metadata().items()
    ):
        raise ValueError(f"{path}: invalid tolerance label metadata")
    frames = document["trajectory"]
    if labels.get("trajectory_sha256") != trajectory_sha256 or labels.get("frame_count") != len(frames):
        raise ValueError(f"{path}: labels do not match the base trajectory SHA-256 or frame_count")
    source = labels.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("method"), str) or not source["method"].strip():
        raise ValueError(f"{path}: label source must name its method")
    subtasks = labels.get("subtasks")
    if not isinstance(subtasks, list):
        raise ValueError(f"{path}: subtasks must be a list")
    observed = {frame["subtask_index"] for frame in frames}
    by_subtask = {}
    for item in subtasks:
        if not isinstance(item, dict) or set(item).difference({"subtask_index", *LABELED_PHASES}):
            raise ValueError(f"{path}: labels contain only subtask_index and Pre/Post bounds")
        index = item.get("subtask_index")
        if type(index) is not int or index not in observed or index in by_subtask:
            raise ValueError(f"{path}: duplicate or unknown labeled subtask")
        by_subtask[index] = {}
        for phase in LABELED_PHASES.intersection(item):
            values = np.asarray(item[phase], dtype=float)
            if values.shape != (6,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{path}: tolerance requires six finite nonnegative radian magnitudes")
            if np.any(values > np.finfo(np.float32).max):
                raise ValueError(f"{path}: tolerance is not representable as float32")
            by_subtask[index][phase] = values.astype(np.float32)
    result = np.zeros((len(frames), 6), dtype=np.float32)
    for index, frame in enumerate(frames):
        phase = frame["phase"]
        if phase in LABELED_PHASES:
            try:
                result[index] = by_subtask[frame["subtask_index"]][phase]
            except KeyError as error:
                raise ValueError(
                    f"{path}: missing tolerance labels for subtask {frame['subtask_index']} {phase}"
                ) from error
    return result
