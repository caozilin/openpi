"""Align wrist-frame matches with offline VLA actions and nominal actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACTION_LABELS = (
    "Δx [m]",
    "Δy [m]",
    "Δz [m]",
    "Δrx [rad]",
    "Δry [rad]",
    "Δrz [rad]",
    "gripper",
)
CONDITIONS = ("with_tolerance", "no_tolerance")
DEFAULT_MATCHING_NAME = "wrist_frame_matches_no_tolerance_query.json"
DEFAULT_ANALYSIS_NAME = "wrist_frame_action_analysis_no_tolerance_query.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _vector(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) < len(ACTION_LABELS):
        raise ValueError(f"{name} must contain at least 7 action values")
    return [float(item) for item in value[: len(ACTION_LABELS)]]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right, strict=True)]


def _inference_action(
    inferences: list[dict[str, Any]], frame_index: int,
) -> dict[str, Any]:
    for inference in reversed(inferences):
        chunk = inference.get("nominal_execution_chunk", {})
        start = int(chunk.get(
            "start_frame_index", inference.get("source_frame_index", 0),
        ))
        end = int(chunk.get("end_frame_index_exclusive", start))
        if start <= frame_index < end:
            row_index = frame_index - start
            actions = inference.get("response", {}).get("actions", [])
            if row_index >= len(actions):
                raise ValueError(
                    f"action block at frame {start} has no row {row_index}",
                )
            return {
                "inference_index": int(inference.get("inference_index", 0)),
                "source_frame_index": int(inference.get(
                    "source_frame_index", start,
                )),
                "action_row_index": row_index,
                "prediction": _vector(
                    actions[row_index], name=f"VLA action at frame {frame_index}",
                ),
            }
    raise ValueError(f"no inference action block covers frame {frame_index}")


def _frame_action(
    source: dict[str, Any], condition: str, frame_index: int,
) -> dict[str, Any]:
    inference = _inference_action(source["inference"]["inferences"], frame_index)
    rollout = source["rollout"]["rollout"]
    if frame_index >= len(rollout):
        raise ValueError(f"{condition} frame {frame_index} exceeds rollout length")
    truth = _vector(
        rollout[frame_index]["action"],
        name=f"nominal action at {condition} frame {frame_index}",
    )
    prediction = inference.pop("prediction")
    return {
        "frame_index": frame_index,
        **inference,
        "prediction": prediction,
        "nominal_action": truth,
        "prediction_error": _subtract(prediction, truth),
    }


def build_wrist_match_action_analysis(
    seed_directory: Path,
    matching_name: str = DEFAULT_MATCHING_NAME,
) -> dict[str, Any]:
    """Build per-paired-frame VLA/nominal action records for one task seed."""
    seed_directory = seed_directory.resolve()
    matching = _read_json(seed_directory / matching_name)
    query_condition = str(matching["matching"]["query"]).split("/", 1)[0]
    reference_condition = str(matching["matching"]["reference"]).split("/", 1)[0]
    condition_data: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        directory = seed_directory / condition
        condition_data[condition] = {
            "inference": _read_json(directory / "offline_inference.json"),
            "rollout": _read_json(directory / "rollout.json"),
        }

    records: list[dict[str, Any]] = []
    for match in matching["matches"]:
        frame_indices = {
            "with_tolerance": int(match["with_tolerance_frame_index"]),
            "no_tolerance": int(match["no_tolerance_frame_index"]),
        }
        aligned: dict[str, dict[str, Any]] = {}
        for condition, frame_index in frame_indices.items():
            source = condition_data[condition]
            aligned[condition] = _frame_action(source, condition, frame_index)

        with_action = aligned["with_tolerance"]
        without_action = aligned["no_tolerance"]
        records.append({
            "with_tolerance_frame_index": frame_indices["with_tolerance"],
            "no_tolerance_frame_index": frame_indices["no_tolerance"],
            "frame_offset": int(match["frame_offset"]),
            "match_mae": float(match["match_mae"]),
            "same_index_mae": float(match["same_index_mae"]),
            "mae_improvement": float(match["mae_improvement"]),
            "radius_limit_hit": bool(match["radius_limit_hit"]),
            "with_tolerance": with_action,
            "no_tolerance": without_action,
            "prediction_difference": _subtract(
                with_action["prediction"], without_action["prediction"],
            ),
            "nominal_action_difference": _subtract(
                with_action["nominal_action"], without_action["nominal_action"],
            ),
        })

    episode = condition_data["with_tolerance"]["rollout"].get("episode", {})
    inference_payload = condition_data["with_tolerance"]["inference"]
    condition_timelines = {
        condition: [
            _frame_action(source, condition, frame_index)
            for frame_index in range(len(source["rollout"]["rollout"]))
        ]
        for condition, source in condition_data.items()
    }
    return {
        "schema_version": "1.0",
        "task": matching.get("task", seed_directory.parent.name),
        "seed": matching.get("seed"),
        "frequency_hz": float(episode.get("frequency_hz", 10.0)),
        "replan_steps": int(inference_payload.get("replan_steps", 10)),
        "query_condition": query_condition,
        "reference_condition": reference_condition,
        "action_labels": list(ACTION_LABELS),
        "frame_counts": matching["frame_counts"],
        "matching": matching["matching"],
        "matching_summary": matching["summary"],
        "condition_timelines": condition_timelines,
        "records": records,
    }


def write_wrist_match_action_analysis(
    seed_directory: Path,
    output_name: str = DEFAULT_ANALYSIS_NAME,
    matching_name: str = DEFAULT_MATCHING_NAME,
) -> Path:
    output = seed_directory / output_name
    payload = build_wrist_match_action_analysis(seed_directory, matching_name)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def discover_seed_directories(
    root: Path,
    matching_name: str = DEFAULT_MATCHING_NAME,
) -> tuple[Path, ...]:
    root = root.resolve()
    if (root / matching_name).is_file():
        return (root,)
    return tuple(sorted(path.parent for path in root.rglob(matching_name)))


__all__ = [
    "ACTION_LABELS",
    "DEFAULT_ANALYSIS_NAME",
    "DEFAULT_MATCHING_NAME",
    "build_wrist_match_action_analysis",
    "discover_seed_directories",
    "write_wrist_match_action_analysis",
]
