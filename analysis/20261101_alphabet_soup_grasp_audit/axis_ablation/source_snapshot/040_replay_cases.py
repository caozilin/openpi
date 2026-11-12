"""Select representative successful and failed counterfactual replay cases."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np

from .intervention import COUNTERFACTUAL_STAGES, ROTATION_AXES
from .runner import CounterfactualTrialSpec


@dataclass(frozen=True)
class CounterfactualReplayCase:
    category: str
    label: str
    expected_success: bool
    spec: CounterfactualTrialSpec
    source: str


def _load_trials(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Counterfactual trial cache not found: {path}. "
            "Run annotate_tolerance.py first.",
        )
    trials = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not trials:
        raise ValueError(f"Counterfactual trial cache is empty: {path}")
    return trials


def _spec(payload: dict[str, Any]) -> CounterfactualTrialSpec:
    return CounterfactualTrialSpec(**payload["spec"])


def _vector(spec: CounterfactualTrialSpec) -> np.ndarray:
    return np.asarray((
        *spec.pregrasp_rotation_vector_deg,
        *spec.postgrasp_rotation_vector_deg,
    ), dtype=float)


def _is_joint(spec: CounterfactualTrialSpec) -> bool:
    return bool(np.any(np.abs(_vector(spec)) > 1e-12))


def _pick_payload(
    payloads: list[dict[str, Any]],
    preferred_seed: int | None,
) -> dict[str, Any] | None:
    if not payloads:
        return None
    return min(
        payloads,
        key=lambda payload: (
            preferred_seed is not None
            and int(payload["spec"]["seed"]) != preferred_seed,
            int(payload["spec"]["seed"]),
        ),
    )


def _signed_corners(
    stage_vectors: dict[str, list[float]],
) -> list[dict[str, list[float]]]:
    active = [
        (stage, axis)
        for stage in COUNTERFACTUAL_STAGES
        for axis, value in enumerate(stage_vectors[stage])
        if value > 0
    ]
    corners: list[dict[str, list[float]]] = []
    for signs in product((-1, 1), repeat=len(active)):
        corner = {stage: [0, 0, 0] for stage in COUNTERFACTUAL_STAGES}
        for (stage, axis), sign in zip(active, signs, strict=True):
            corner[stage][axis] = sign * stage_vectors[stage][axis]
        corners.append(corner)
    return corners


def _corner_vector(corner: dict[str, list[int]]) -> np.ndarray:
    return np.asarray((
        *corner["pregrasp"],
        *corner["postgrasp"],
    ), dtype=float)


def _corner_label(corner: dict[str, list[float]]) -> str:
    parts: list[str] = []
    for stage in COUNTERFACTUAL_STAGES:
        for axis, value in zip(ROTATION_AXES, corner[stage], strict=True):
            if value:
                parts.append(f"{stage}_{axis}_{value:+g}deg")
    return "__".join(parts) or "zero"


def _joint_pair(
    trials: list[dict[str, Any]],
    category: str,
    corner: dict[str, list[float]],
    preferred_seed: int | None,
) -> tuple[list[CounterfactualReplayCase], list[dict[str, Any]]]:
    target = _corner_vector(corner)
    support = np.abs(target) > 1e-12
    joint_trials = [
        payload
        for payload in trials
        if _is_joint(_spec(payload))
    ]
    exact_successes = [
        payload
        for payload in joint_trials
        if bool(payload["success"])
        and np.allclose(_vector(_spec(payload)), target, atol=1e-9)
    ]
    success_payload = _pick_payload(exact_successes, preferred_seed)
    cases: list[CounterfactualReplayCase] = []
    unavailable: list[dict[str, Any]] = []
    label = _corner_label(corner)
    if success_payload is None:
        unavailable.append({
            "category": category,
            "label": label,
            "outcome": "success",
            "reason": "no successful cached trial at the final joint corner",
        })
        success_seed = preferred_seed
    else:
        success_spec = _spec(success_payload)
        success_seed = success_spec.seed
        cases.append(CounterfactualReplayCase(
            category=category,
            label=label,
            expected_success=True,
            spec=success_spec,
            source="final_joint_corner",
        ))

    failure_candidates: list[tuple[float, bool, dict[str, Any]]] = []
    target_sign = np.sign(target[support])
    for payload in joint_trials:
        if bool(payload["success"]):
            continue
        candidate = _vector(_spec(payload))
        candidate_support = np.abs(candidate) > 1e-12
        if not np.array_equal(candidate_support, support):
            continue
        if not np.array_equal(np.sign(candidate[support]), target_sign):
            continue
        if np.any(np.abs(candidate[support]) + 1e-9 < np.abs(target[support])):
            continue
        distance = float(np.linalg.norm(candidate - target))
        same_seed = success_seed is not None and _spec(payload).seed == success_seed
        failure_candidates.append((distance, not same_seed, payload))
    failure_payload = (
        min(
            failure_candidates,
            key=lambda item: (
                item[0],
                item[1],
                int(item[2]["spec"]["seed"]),
            ),
        )[2]
        if failure_candidates
        else None
    )
    if failure_payload is None:
        unavailable.append({
            "category": category,
            "label": label,
            "outcome": "failure",
            "reason": "no cached failed trial at or outside this joint corner",
        })
    else:
        cases.append(CounterfactualReplayCase(
            category=category,
            label=label,
            expected_success=False,
            spec=_spec(failure_payload),
            source="nearest_cached_joint_failure",
        ))
    return cases, unavailable


def select_replay_cases(
    annotation: dict[str, Any],
    trials_path: Path,
    preferred_seed: int | None = None,
) -> tuple[list[CounterfactualReplayCase], list[dict[str, Any]]]:
    """Pair final successful boundaries with nearest observed failures."""
    trials = [
        payload
        for payload in _load_trials(trials_path)
        if _spec(payload).scene == annotation.get("scene")
        and _spec(payload).task == annotation.get("task")
        and _spec(payload).robot_uid == annotation.get("robot_uid")
    ]
    if not trials:
        raise ValueError(
            "Trial cache has no records matching the annotation scene/task/robot",
        )
    masks = annotation.get("stage_rotation_masks")
    cases: list[CounterfactualReplayCase] = []
    unavailable: list[dict[str, Any]] = []

    signed = annotation.get("signed_boundaries_deg")
    if signed is None:
        symmetric = annotation.get(
            "single_axis_symmetric_rotation_tolerance_deg", {},
        )
        for stage in COUNTERFACTUAL_STAGES:
            for axis_index, axis in enumerate(ROTATION_AXES):
                if masks is not None and not bool(masks[stage][axis_index]):
                    continue
                boundary = int(symmetric.get(stage, (0, 0, 0))[axis_index])
                matching = [
                    payload
                    for payload in trials
                    if not _is_joint(_spec(payload))
                    and _spec(payload).controller_mode == "sqp_specific"
                    and _spec(payload).stage == stage
                    and _spec(payload).axis == axis
                ]
                successes = [
                    payload
                    for payload in matching
                    if bool(payload["success"])
                    and _spec(payload).angle_deg == boundary
                ]
                success_payload = _pick_payload(successes, preferred_seed)
                selected_direction = (
                    _spec(success_payload).direction
                    if success_payload is not None else None
                )
                label = (
                    f"{stage}_{axis}_{selected_direction:+d}"
                    if selected_direction is not None
                    else f"{stage}_{axis}_sqp_directed"
                )
                if success_payload is None:
                    unavailable.append({
                        "category": "single_axis",
                        "label": label,
                        "outcome": "success",
                        "reason": (
                            "boundary is zero or has no successful cached trial"
                        ),
                    })
                    success_seed = preferred_seed
                else:
                    success_spec = _spec(success_payload)
                    success_seed = success_spec.seed
                    cases.append(CounterfactualReplayCase(
                        category="single_axis",
                        label=label,
                        expected_success=True,
                        spec=success_spec,
                        source="sqp_directed_boundary",
                    ))
                failures = [
                    payload
                    for payload in matching
                    if not bool(payload["success"])
                    and _spec(payload).angle_deg >= boundary
                    and (
                        selected_direction is None
                        or _spec(payload).direction == selected_direction
                    )
                ]
                failure_payload = min(
                    failures,
                    key=lambda payload: (
                        _spec(payload).angle_deg - boundary,
                        success_seed is not None
                        and _spec(payload).seed != success_seed,
                        _spec(payload).seed,
                    ),
                    default=None,
                )
                if failure_payload is None:
                    unavailable.append({
                        "category": "single_axis",
                        "label": label,
                        "outcome": "failure",
                        "reason": "no failed cached trial at or above the boundary",
                    })
                else:
                    cases.append(CounterfactualReplayCase(
                        category="single_axis",
                        label=label,
                        expected_success=False,
                        spec=_spec(failure_payload),
                        source="nearest_cached_failure",
                    ))

        logs = annotation.get("joint_validation_log", [])
        vectors_by_seed = logs[0].get("stage_vectors_deg_by_seed", {}) if logs else {}
        if vectors_by_seed:
            selected_seed = (
                str(preferred_seed)
                if preferred_seed is not None
                and str(preferred_seed) in vectors_by_seed
                else min(vectors_by_seed, key=int)
            )
            selected, missing = _joint_pair(
                trials,
                "cross_stage_joint",
                vectors_by_seed[selected_seed],
                int(selected_seed),
            )
            cases.extend(selected)
            unavailable.extend(missing)
        return cases, unavailable

    for stage in COUNTERFACTUAL_STAGES:
        for axis_index, axis in enumerate(ROTATION_AXES):
            if masks is not None and not bool(masks[stage][axis_index]):
                continue
            for direction, direction_name in ((-1, "negative"), (1, "positive")):
                boundary = int(signed[stage][axis][direction_name])
                label = f"{stage}_{axis}_{direction:+d}"
                matching = [
                    payload
                    for payload in trials
                    if not _is_joint(_spec(payload))
                    and _spec(payload).stage == stage
                    and _spec(payload).axis == axis
                    and _spec(payload).direction == direction
                ]
                successes = [
                    payload
                    for payload in matching
                    if bool(payload["success"])
                    and _spec(payload).angle_deg == boundary
                ]
                success_payload = _pick_payload(successes, preferred_seed)
                if success_payload is None:
                    unavailable.append({
                        "category": "single_axis",
                        "label": label,
                        "outcome": "success",
                        "reason": (
                            "boundary is zero or has no successful cached trial"
                        ),
                    })
                    success_seed = preferred_seed
                else:
                    success_spec = _spec(success_payload)
                    success_seed = success_spec.seed
                    cases.append(CounterfactualReplayCase(
                        category="single_axis",
                        label=label,
                        expected_success=True,
                        spec=success_spec,
                        source="signed_boundary",
                    ))

                failures = [
                    payload
                    for payload in matching
                    if not bool(payload["success"])
                    and _spec(payload).angle_deg >= boundary
                ]
                failure_payload = min(
                    failures,
                    key=lambda payload: (
                        _spec(payload).angle_deg - boundary,
                        success_seed is not None
                        and _spec(payload).seed != success_seed,
                        _spec(payload).seed,
                    ),
                    default=None,
                )
                if failure_payload is None:
                    unavailable.append({
                        "category": "single_axis",
                        "label": label,
                        "outcome": "failure",
                        "reason": "no failed cached trial at or above the boundary",
                    })
                else:
                    cases.append(CounterfactualReplayCase(
                        category="single_axis",
                        label=label,
                        expected_success=False,
                        spec=_spec(failure_payload),
                        source="nearest_cached_failure",
                    ))

    if str(annotation.get("schema_version")) in (
        "6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0",
    ):
        panels = [
            panel for panel in annotation.get("evaluation_log", [])
            if panel.get("label") in (
                "full-task bilateral SQP release",
                "held-out SQP bilateral release",
                "joint optimizer correction simulation",
            )
        ]
        if panels:
            details = [
                detail for detail in panels[-1].get("trials_detail", [])
                if "spec" in detail
            ]
            for expected_success in (True, False):
                matching = [
                    detail for detail in details
                    if bool(
                        detail.get("task_success")
                        and detail.get("target_reached")
                        and not detail.get("hard_failure_reasons")
                    ) == expected_success
                ]
                selected = min(
                    matching,
                    key=lambda detail: (
                        preferred_seed is not None
                        and int(detail["seed"]) != preferred_seed,
                        int(detail["seed"]),
                    ),
                    default=None,
                )
                if selected is None:
                    unavailable.append({
                        "category": "sqp_rollout",
                        "label": "full_task_bilateral_sqp",
                        "outcome": "success" if expected_success else "failure",
                        "reason": "no cached SQP rollout with this outcome",
                    })
                else:
                    cases.append(CounterfactualReplayCase(
                        category="sqp_rollout",
                        label="full_task_bilateral_sqp",
                        expected_success=expected_success,
                        spec=CounterfactualTrialSpec(**selected["spec"]),
                        source="sqp_rollout_panel",
                    ))
        return cases, unavailable

    if str(annotation.get("schema_version")) == "5.0":
        for panel in annotation.get("evaluation_log", []):
            label = str(panel.get("label", ""))
            if not label.startswith("hybrid corner="):
                continue
            details = [
                detail
                for detail in panel.get("trials_detail", [])
                if "spec" in detail
            ]
            for expected_success in (True, False):
                matching = [
                    detail
                    for detail in details
                    if bool(
                        detail.get("task_success")
                        and detail.get("target_reached")
                        and not detail.get("hard_failure_reasons")
                    ) == expected_success
                ]
                selected = min(
                    matching,
                    key=lambda detail: (
                        preferred_seed is not None
                        and int(detail["seed"]) != preferred_seed,
                        int(detail["seed"]),
                    ),
                    default=None,
                )
                if selected is None:
                    unavailable.append({
                        "category": "hybrid_joint",
                        "label": label,
                        "outcome": "success" if expected_success else "failure",
                        "reason": "no cached task-level replay with this outcome",
                    })
                else:
                    cases.append(CounterfactualReplayCase(
                        category="hybrid_joint",
                        label=label.replace(" ", "_"),
                        expected_success=expected_success,
                        spec=CounterfactualTrialSpec(**selected["spec"]),
                        source="hybrid_validation_panel",
                    ))
        return cases, unavailable

    same_stage = annotation.get("same_stage_joint_rotation_tolerance_deg", {})
    for stage in COUNTERFACTUAL_STAGES:
        values = list(same_stage.get(stage, (0, 0, 0)))
        if sum(value > 0 for value in values) <= 1:
            continue
        vectors = {
            other: (values if other == stage else [0, 0, 0])
            for other in COUNTERFACTUAL_STAGES
        }
        for corner in _signed_corners(vectors):
            selected, missing = _joint_pair(
                trials, "same_stage_joint", corner, preferred_seed,
            )
            cases.extend(selected)
            unavailable.extend(missing)

    cross_stage = annotation.get("symmetric_rotation_tolerance_deg", {})
    if sum(
        any(value > 0 for value in cross_stage.get(stage, (0, 0, 0)))
        for stage in COUNTERFACTUAL_STAGES
    ) > 1:
        vectors = {
            stage: list(cross_stage[stage])
            for stage in COUNTERFACTUAL_STAGES
        }
        for corner in _signed_corners(vectors):
            selected, missing = _joint_pair(
                trials, "cross_stage_joint", corner, preferred_seed,
            )
            cases.extend(selected)
            unavailable.extend(missing)

    return cases, unavailable
