"""Shared source-adjacent empty-frame filtering for dataset collection."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable


POSE_ACTION_ZERO_TOLERANCE = 0.0
JOINT_CHANGE_THRESHOLD_RAD = 0.002
GRIPPER_APERTURE_CHANGE_THRESHOLD_M = 0.0001
COMPARISON_ROUNDOFF_TOLERANCE = 1e-12


def _float_vector(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} contains a non-numeric value") from error
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} contains a non-finite value")
    return result


def is_empty_frame(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return whether ``current`` is empty relative to the prior source frame."""
    previous_action = _float_vector(previous["action"], field="previous.action")
    current_action = _float_vector(current["action"], field="current.action")
    if len(previous_action) != 7 or len(current_action) != 7:
        raise ValueError("actions must contain exactly seven values")
    if any(abs(value) > POSE_ACTION_ZERO_TOLERANCE for value in current_action[:6]):
        return False

    previous_joints = _float_vector(
        previous["joint_state"]["position_rad"],
        field="previous.joint_state.position_rad",
    )
    current_joints = _float_vector(
        current["joint_state"]["position_rad"],
        field="current.joint_state.position_rad",
    )
    if not previous_joints or len(previous_joints) != len(current_joints):
        raise ValueError("adjacent joint states must have the same non-zero length")
    if (
        max(abs(a - b) for a, b in zip(current_joints, previous_joints))
        > JOINT_CHANGE_THRESHOLD_RAD + COMPARISON_ROUNDOFF_TOLERANCE
    ):
        return False

    previous_aperture = float(previous["gripper_state"]["aperture_m"])
    current_aperture = float(current["gripper_state"]["aperture_m"])
    if not math.isfinite(previous_aperture) or not math.isfinite(current_aperture):
        raise ValueError("gripper aperture must be finite")
    if (
        abs(current_aperture - previous_aperture)
        > GRIPPER_APERTURE_CHANGE_THRESHOLD_M + COMPARISON_ROUNDOFF_TOLERANCE
    ):
        return False

    return current_action[6] == previous_action[6]


def retained_source_indices(trajectory: list[dict[str, Any]]) -> list[int]:
    """Select frames using source adjacency, never filtered adjacency."""
    if not trajectory:
        raise ValueError("trajectory must contain at least one frame")
    retained = [0]
    for index in range(1, len(trajectory)):
        if not is_empty_frame(trajectory[index - 1], trajectory[index]):
            retained.append(index)
    return retained


def retain_source_frame(
    previous_source: dict[str, Any] | None,
    current_source: dict[str, Any],
    *,
    output_index: int,
    frequency_hz: float,
) -> dict[str, Any] | None:
    """Filter one source frame and assign its compact output coordinates."""
    if previous_source is not None and is_empty_frame(previous_source, current_source):
        return None
    retained = current_source.copy()
    retained["index"] = output_index
    retained["video_frame_index"] = output_index
    retained["time_s"] = output_index / frequency_hz
    return retained


def frame_filter_metadata(
    source_frame_count: int,
    output_frame_count: int,
) -> dict[str, Any]:
    """Describe the common filter rule and its per-episode result."""
    if not 0 <= output_frame_count <= source_frame_count:
        raise ValueError("output frame count must be within the source frame count")
    return {
        "source_frame_count": source_frame_count,
        "removed_frame_count": source_frame_count - output_frame_count,
        "source_adjacency": True,
        "first_frame_retained": True,
        "extra_gripper_transition_frames_retained": 0,
        "pose_delta_action_zero_tolerance": POSE_ACTION_ZERO_TOLERANCE,
        "maximum_joint_change_rad": JOINT_CHANGE_THRESHOLD_RAD,
        "maximum_gripper_aperture_change_m": (
            GRIPPER_APERTURE_CHANGE_THRESHOLD_M
        ),
        "gripper_action_must_equal_previous": True,
    }


def filter_trajectory_payload(
    payload: dict[str, Any],
    retained: list[int],
    *,
    frequency_hz: float,
) -> dict[str, Any]:
    """Return a synchronized, compact copy of a trajectory payload."""
    source_frames = payload["trajectory"]
    filtered: list[dict[str, Any]] = []
    for new_index, source_index in enumerate(retained):
        frame = source_frames[source_index].copy()
        frame["index"] = new_index
        frame["video_frame_index"] = new_index
        frame["time_s"] = new_index / frequency_hz
        filtered.append(frame)

    result = payload.copy()
    episode = payload["episode"].copy()
    episode["frame_count"] = len(filtered)
    episode["frame_filter"] = frame_filter_metadata(
        len(source_frames), len(filtered),
    )
    result["episode"] = episode
    result["trajectory"] = filtered
    return result


def _compact_ranges(indices: Iterable[int]) -> list[tuple[int, int]]:
    values = sorted(set(indices))
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            ranges.append((start, end))
            start = end = value
    ranges.append((start, end))
    return ranges


def _video_select_filter(removed: list[int], frequency_hz: float) -> str:
    terms = []
    for start, end in _compact_ranges(removed):
        if start == end:
            terms.append(f"eq(n\\,{start})")
        else:
            terms.append(f"between(n\\,{start}\\,{end})")
    removal_expression = "+".join(terms)
    return f"select=not({removal_expression}),setpts=N/({frequency_hz:g}*TB)"


def filter_video(
    source: Path,
    destination: Path,
    removed: list[int],
    *,
    frequency_hz: float,
    ffmpeg: str,
) -> None:
    """Write a video containing exactly the retained source-frame indices."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not removed:
        shutil.copy2(source, destination)
        return

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(source),
        "-vf", _video_select_filter(removed, frequency_hz),
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", f"{frequency_hz:g}",
        "-fps_mode", "cfr",
        "-movflags", "+faststart",
        "-threads", "1",
        str(temporary),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"FFmpeg failed for {source}: {detail}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
