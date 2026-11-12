"""Deferred six-dimensional labels, bound to an immutable base trajectory.

This module reads/writes data only. Search and simulation remain with their
existing backends; no tolerance frame is serialized or predicted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..tolerance.specification import (
    ROTATION_TOLERANCE_COMPOSITION,
    ROTATION_TOLERANCE_COORDINATE_CONVENTION,
    ROTATION_TOLERANCE_PROFILE_ORDER,
)


DATASET_SCHEMA_VERSION = "8.0"
TOLERANCE_LABEL_SCHEMA_VERSION = "1.0"
TOLERANCE_LABEL_FILE = "tolerance_labels.json"
_PHASES = {"pregrasp", "grasp", "postgrasp", "release"}
_LABELED_PHASES = {"pregrasp", "postgrasp"}


def tolerance_label_metadata() -> dict[str, Any]:
    """Describe the optional label file without fabricating supervision."""
    return {
        "file": TOLERANCE_LABEL_FILE,
        "schema_version": TOLERANCE_LABEL_SCHEMA_VERSION,
        "dimension": 6,
        "units": "rad",
        "order": list(ROTATION_TOLERANCE_PROFILE_ORDER),
        "coordinate_convention": ROTATION_TOLERANCE_COORDINATE_CONVENTION,
        "rotation_composition": ROTATION_TOLERANCE_COMPOSITION,
        "reference": "realtime_nominal",
        "assignment": "subtask_index_and_phase",
        "missing_labels": "unlabeled; never substitute zero for Pre/Post",
        "strict_phases": ["grasp", "release"],
    }


def _subtask_index(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("subtask_index must be a nonnegative integer")
    return value


def _base_trajectory(episode_dir: Path) -> tuple[dict[str, Any], str]:
    content = (Path(episode_dir) / "trajectory.json").read_bytes()
    document = json.loads(content)
    if document.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Expected base trajectory schema {DATASET_SCHEMA_VERSION}")
    frames = document.get("trajectory")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Base trajectory must contain frames")
    if document.get("episode", {}).get("frame_count") != len(frames):
        raise ValueError("Base trajectory frame_count is inconsistent")
    if "stage_annotations" in document:
        raise ValueError("Stage annotations are not part of the base dataset")
    forbidden = {
        "stage_annotation_id", "tolerance_frame_code", "tolerance_frame",
        "tolerance_frame_rotation_vector_base_rad", "rotation_tolerance_rad",
    }
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("Base trajectory frames must be objects")
        if forbidden.intersection(frame):
            raise ValueError("Base frames must not contain stage annotations or tolerance labels")
        if any(type(frame.get(key)) is not int or frame[key] != index for key in ("index", "video_frame_index")):
            raise ValueError(f"Invalid trajectory/video index at frame {index}")
        time = frame.get("time_s")
        if not isinstance(time, (float, int)) or not math.isclose(time, index / 10, abs_tol=1e-9):
            raise ValueError(f"Invalid compact timestamp at frame {index}")
        action = np.asarray(frame.get("action"), dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Base actions must contain seven finite values")
        _subtask_index(frame.get("subtask_index"))
        if frame.get("phase") not in _PHASES:
            raise ValueError("Unknown manipulation phase")
    return document, hashlib.sha256(content).hexdigest()


def trajectory_sha256(episode_dir: Path) -> str:
    """Capture this before submitting the base episode to a label/search job."""
    return _base_trajectory(episode_dir)[1]


def _validated_subtasks(
    subtasks: Any, frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(subtasks, list):
        raise ValueError("Tolerance subtasks must be a list")
    observed = {frame["subtask_index"] for frame in frames}
    required = {
        (frame["subtask_index"], frame["phase"])
        for frame in frames if frame["phase"] in _LABELED_PHASES
    }
    seen: set[int] = set()
    supplied: set[tuple[int, str]] = set()
    normalized = []
    for item in subtasks:
        if not isinstance(item, dict) or set(item) - {"subtask_index", *_LABELED_PHASES}:
            raise ValueError("Labels contain only subtask_index, pregrasp and postgrasp")
        index = _subtask_index(item.get("subtask_index"))
        if index in seen or index not in observed:
            raise ValueError("Duplicate or unknown labeled subtask")
        seen.add(index)
        result: dict[str, Any] = {"subtask_index": index}
        for phase in sorted(_LABELED_PHASES.intersection(item)):
            values = np.asarray(item[phase], dtype=float)
            if values.shape != (6,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Tolerance requires six finite nonnegative magnitudes in radians")
            result[phase] = values.tolist()
            supplied.add((index, phase))
        normalized.append(result)
    if not required.issubset(supplied):
        raise ValueError(f"Missing tolerance labels for {sorted(required - supplied)}")
    return sorted(normalized, key=lambda item: item["subtask_index"])


def write_tolerance_labels(
    episode_dir: Path,
    subtasks: list[dict[str, Any]],
    *,
    expected_trajectory_sha256: str,
    source: dict[str, Any],
    overwrite: bool = False,
) -> Path:
    """Attach completed labels without editing trajectory, manifest or videos.

    ``subtasks`` contains ``subtask_index`` and Pre/Post six-vectors in radians.
    The expected hash comes from the search job's original input, not a later
    copy of the episode. Missing stable phases are errors; strict phases are
    always resolved to zero and must not be supplied by the labeling script.
    """
    episode_dir = Path(episode_dir)
    base, digest = _base_trajectory(episode_dir)
    if expected_trajectory_sha256 != digest:
        raise ValueError("Base trajectory changed or labels target a different episode")
    if not isinstance(source, dict) or not isinstance(source.get("method"), str) or not source["method"].strip():
        raise ValueError("Label source must name its method")
    document = {
        **tolerance_label_metadata(),
        "trajectory_sha256": digest,
        "frame_count": len(base["trajectory"]),
        "source": source,
        "subtasks": _validated_subtasks(subtasks, base["trajectory"]),
    }
    content = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = episode_dir / TOLERANCE_LABEL_FILE
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Labels already exist: {destination}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=episode_dir, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_rotation_tolerance(
    episode_dir: Path, *, require_labels: bool = True,
) -> np.ndarray | None:
    """Return row-aligned [N,6] targets, or None for a wholly unlabeled episode."""
    base, digest = _base_trajectory(episode_dir)
    label_path = Path(episode_dir) / TOLERANCE_LABEL_FILE
    if not label_path.is_file():
        if require_labels:
            raise FileNotFoundError(f"Episode has no tolerance labels: {label_path}")
        return None
    document = json.loads(label_path.read_text(encoding="utf-8"))
    for key, expected in tolerance_label_metadata().items():
        if document.get(key) != expected:
            raise ValueError(f"Invalid tolerance label metadata: {key}")
    if not isinstance(document.get("source"), dict) or not document["source"].get("method"):
        raise ValueError("Tolerance labels must record their source method")
    if document.get("trajectory_sha256") != digest or document.get("frame_count") != len(base["trajectory"]):
        raise ValueError("Tolerance labels do not match the base trajectory")
    subtasks = _validated_subtasks(document.get("subtasks"), base["trajectory"])
    by_subtask = {item["subtask_index"]: item for item in subtasks}
    return np.asarray([
        by_subtask[frame["subtask_index"]][frame["phase"]]
        if frame["phase"] in _LABELED_PHASES else [0.0] * 6
        for frame in base["trajectory"]
    ], dtype=float)


def write_bayesian_tolerance_labels(
    episode_dir: Path,
    observation: dict[str, Any] | list[dict[str, Any]],
    *,
    expected_trajectory_sha256: str,
    overwrite: bool = False,
) -> Path:
    """Connect the existing selected-observation interface to dataset labels.

    Use the verified selected search caps, not executed release peaks. For long
    tasks supply one selected observation per subtask; never infer other
    subtasks' labels from their release diagnostics in a single trial.
    """
    from ..tolerance.labeling import label_from_observation

    observations = observation if isinstance(observation, list) else [observation]
    labels = [label_from_observation(item) for item in observations]
    subtasks = [{
        "subtask_index": item["subtask_index"],
        **{phase: np.radians(item[phase]).tolist() for phase in _LABELED_PHASES},
    } for label in labels for item in label["subtask_tolerance_labels_deg"]]
    sources = [{
        key: label[key] for key in (
            "label_policy", "raw_trial_path", "raw_trial_sha256", "velocity_rad_s",
            "semantic_subtask_index", "search_pre_caps_deg", "search_post_caps_deg",
            "actual_pre_release_peaks_deg", "actual_post_release_peaks_deg",
            "subtask_release_peaks_deg",
        ) if key in label
    } for label in labels]
    source = {"method": "bayesian_search"}
    if len(sources) == 1:
        source.update(sources[0])
    else:
        from ..tolerance.labeling import SEARCH_SOLUTION_LABEL_POLICY

        source.update(label_policy=SEARCH_SOLUTION_LABEL_POLICY, observations=sources)
    return write_tolerance_labels(
        episode_dir, subtasks,
        expected_trajectory_sha256=expected_trajectory_sha256,
        source=source,
        overwrite=overwrite,
    )
