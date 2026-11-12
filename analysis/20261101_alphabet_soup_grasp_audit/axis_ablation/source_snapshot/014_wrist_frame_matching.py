"""Match wrist frames between tolerance and no-tolerance trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class WristFrameMatchingConfig:
    search_radius: int = 10
    comparison_width: int = 96
    comparison_height: int = 72
    query_condition: str = "with_tolerance"

    def __post_init__(self) -> None:
        if self.search_radius < 0:
            raise ValueError("search_radius must be non-negative")
        if self.comparison_width <= 0 or self.comparison_height <= 0:
            raise ValueError("comparison dimensions must be positive")
        if self.query_condition not in {"with_tolerance", "no_tolerance"}:
            raise ValueError(
                "query_condition must be with_tolerance or no_tolerance",
            )


def _video_frames(path: Path) -> Iterator[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        for encoded in reader:
            yield np.frombuffer(encoded, dtype=np.uint8).reshape(height, width, 3)
    finally:
        reader.close()


def _comparison_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    y_indices = np.linspace(0, source_height - 1, height).round().astype(int)
    x_indices = np.linspace(0, source_width - 1, width).round().astype(int)
    return frame[y_indices[:, None], x_indices[None, :], :].astype(np.float32) / 255.0


def _read_comparison_video(
    path: Path,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    frames = [
        _comparison_frame(frame, width=width, height=height)
        for frame in _video_frames(path)
    ]
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return np.stack(frames)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def match_wrist_videos(
    with_tolerance_video: Path,
    no_tolerance_video: Path,
    *,
    config: WristFrameMatchingConfig,
) -> dict[str, Any]:
    """Match every query-condition frame to a nearby frame in the other run."""
    tolerance_frames = _read_comparison_video(
        with_tolerance_video,
        width=config.comparison_width,
        height=config.comparison_height,
    )
    reference_frames = _read_comparison_video(
        no_tolerance_video,
        width=config.comparison_width,
        height=config.comparison_height,
    )
    frames_by_condition = {
        "with_tolerance": tolerance_frames,
        "no_tolerance": reference_frames,
    }
    query_condition = config.query_condition
    reference_condition = (
        "no_tolerance"
        if query_condition == "with_tolerance"
        else "with_tolerance"
    )
    query_frames = frames_by_condition[query_condition]
    candidate_frames = frames_by_condition[reference_condition]
    matches: list[dict[str, Any]] = []
    previous_reference_index: int | None = None
    for query_index, query_frame in enumerate(query_frames):
        start = max(0, query_index - config.search_radius)
        end = min(len(candidate_frames), query_index + config.search_radius + 1)
        candidates = candidate_frames[start:end]
        scores = np.mean(np.abs(candidates - query_frame), axis=(1, 2, 3))
        offsets = np.arange(start, end) - query_index
        tie_break = np.abs(offsets) * 1e-12 + np.arange(len(scores)) * 1e-15
        local_index = int(np.argmin(scores + tie_break))
        reference_index = start + local_index
        same_index = min(query_index, len(candidate_frames) - 1)
        same_index_mae = float(np.mean(np.abs(
            candidate_frames[same_index] - query_frame,
        )))
        sorted_scores = np.sort(scores)
        second_best_margin = (
            float(sorted_scores[1] - sorted_scores[0])
            if len(sorted_scores) > 1 else None
        )
        frame_indices = {
            query_condition: query_index,
            reference_condition: reference_index,
        }
        matches.append({
            "with_tolerance_frame_index": frame_indices["with_tolerance"],
            "no_tolerance_frame_index": frame_indices["no_tolerance"],
            "query_frame_index": query_index,
            "reference_frame_index": reference_index,
            "frame_offset": reference_index - query_index,
            "search_start_frame_index": start,
            "search_end_frame_index_exclusive": end,
            "match_mae": float(scores[local_index]),
            "same_index_mae": same_index_mae,
            "mae_improvement": same_index_mae - float(scores[local_index]),
            "second_best_mae_margin": second_best_margin,
            "radius_limit_hit": abs(reference_index - query_index) == config.search_radius,
            "search_edge_hit": reference_index in {start, end - 1},
            "reference_index_decreased": (
                previous_reference_index is not None
                and reference_index < previous_reference_index
            ),
        })
        previous_reference_index = reference_index

    offsets = np.asarray([item["frame_offset"] for item in matches], dtype=int)
    absolute_offsets = np.abs(offsets)
    match_mae = np.asarray([item["match_mae"] for item in matches], dtype=float)
    same_mae = np.asarray([item["same_index_mae"] for item in matches], dtype=float)
    return {
        "schema_version": "1.0",
        "matching": {
            "query": f"{query_condition}/wrist_rgb.mp4",
            "reference": f"{reference_condition}/wrist_rgb.mp4",
            "metric": "mean absolute RGB pixel error after deterministic nearest-neighbor resize",
            "search_radius_frames": config.search_radius,
            "comparison_size": [config.comparison_width, config.comparison_height],
            "tie_break": "smallest absolute frame offset, then earliest frame",
        },
        "frame_counts": {
            "with_tolerance": int(len(tolerance_frames)),
            "no_tolerance": int(len(reference_frames)),
        },
        "summary": {
            "matched_frames": len(matches),
            "mean_frame_offset": float(np.mean(offsets)),
            "median_frame_offset": float(np.median(offsets)),
            "mean_absolute_frame_offset": float(np.mean(absolute_offsets)),
            "p95_absolute_frame_offset": float(np.quantile(absolute_offsets, 0.95)),
            "maximum_absolute_frame_offset": int(np.max(absolute_offsets)),
            "radius_limit_hits": int(sum(item["radius_limit_hit"] for item in matches)),
            "radius_limit_hit_rate": float(np.mean([
                item["radius_limit_hit"] for item in matches
            ])),
            "search_edge_hits": int(sum(item["search_edge_hit"] for item in matches)),
            "reference_index_decreases": int(sum(
                item["reference_index_decreased"] for item in matches
            )),
            "mean_match_mae": float(np.mean(match_mae)),
            "mean_same_index_mae": float(np.mean(same_mae)),
            "mean_mae_improvement": float(np.mean(same_mae - match_mae)),
        },
        "matches": matches,
    }


def discover_seed_directories(root: Path) -> tuple[Path, ...]:
    if (root / "with_tolerance" / "wrist_rgb.mp4").is_file():
        return (root,)
    return tuple(sorted(
        path.parent.parent
        for path in root.glob("*/seed_*/with_tolerance/wrist_rgb.mp4")
        if (path.parent.parent / "no_tolerance" / "wrist_rgb.mp4").is_file()
    ))


def run_wrist_frame_matching(
    roots: tuple[Path, ...],
    *,
    config: WristFrameMatchingConfig,
    output_name: str = "wrist_frame_matches.json",
) -> tuple[Path, ...]:
    outputs: list[Path] = []
    for root in roots:
        seed_directories = discover_seed_directories(root)
        if not seed_directories:
            raise FileNotFoundError(f"no paired wrist videos found under {root}")
        run_records: list[dict[str, Any]] = []
        for seed_dir in seed_directories:
            result = match_wrist_videos(
                seed_dir / "with_tolerance" / "wrist_rgb.mp4",
                seed_dir / "no_tolerance" / "wrist_rgb.mp4",
                config=config,
            )
            result["task"] = seed_dir.parent.name
            result["seed"] = int(seed_dir.name.removeprefix("seed_"))
            output = seed_dir / output_name
            _write_json(output, result)
            outputs.append(output)
            run_records.append({
                "task": result["task"],
                "seed": result["seed"],
                "path": str(output),
                "summary": result["summary"],
            })
        summary_name = (
            "wrist_frame_matching_summary.json"
            if output_name == "wrist_frame_matches.json"
            else f"{Path(output_name).stem}_summary.json"
        )
        summary_path = root / summary_name
        _write_json(summary_path, {
            "schema_version": "1.0",
            "root": str(root),
            "matching": {
                "query_condition": config.query_condition,
                "search_radius_frames": config.search_radius,
                "comparison_size": [config.comparison_width, config.comparison_height],
            },
            "pairs": run_records,
        })
        outputs.append(summary_path)
    return tuple(outputs)
