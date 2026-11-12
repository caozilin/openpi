"""Per-episode rotational-tolerance annotation.

Every annotation starts from one nominal seed trajectory. Subtasks are
processed from last to first; within a subtask Post-grasp is finalized before
Pre-grasp. A stage starts with symmetric +/-30 degree release only on axes
enabled by its authored task mask; masked-out axes remain zero. Failed rollouts
are diagnosed by blocking each enabled axis in turn, then the unique successful
axis is tightened one signed side at a time from measured running release
extrema until the trajectory succeeds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..data.task_sets import resolve_training_task
from ..control.task_space.objective import DEFAULT_EMA_RELEASE_LOSS_WEIGHT
from ..simulation.randomization.routing import RandomizationSeedRouting
from ..simulation.robots import get_robot_spec
from ..tasks import registered_tasks
from ..tasks.planning.presets import PresetTaskExecutor
from ..tolerance.context import long_task_subtask_labels
from .intervention import COUNTERFACTUAL_STAGES, ROTATION_AXES
from .runner import (
    COUNTERFACTUAL_CACHE_VERSION,
    CounterfactualTrialResult,
    CounterfactualTrialSpec,
    run_counterfactual_trial,
)


ENVIRONMENT_ROUTING_ANNOTATION_VERSION = 2
ANNOTATION_SCHEMA_VERSION = "18.0"
DEFAULT_MAXIMUM_TOLERANCE_DEG = 30
DEFAULT_TOLERANCE_STEP_DEG = 5


class ToleranceAnnotationError(RuntimeError):
    """Raised when an episode cannot produce an unambiguous annotation."""


@dataclass(frozen=True)
class EpisodeAnnotationTarget:
    """One nominal seed trajectory and its annotation destination."""

    seed: int
    output_dir: Path
    episode_index: int | None = None
    trajectory_path: Path | None = None
    environment_seed: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.trajectory_path is not None:
            object.__setattr__(
                self, "trajectory_path", Path(self.trajectory_path),
            )


@dataclass(frozen=True)
class CounterfactualSearchConfig:
    """Configuration for seed-local annotation with no task-level voting."""

    scene: str = "geometry_plate_box_upright"
    task: str = "geometry_plate_box_upright"
    robot_uid: str = "panda"
    execution_mode: str = "kinematic"
    targets: tuple[EpisodeAnnotationTarget, ...] = ()
    seeds: tuple[int, ...] = (0,)
    output_dir: Path | None = None
    maximum_frames: int = 600
    maximum_tolerance_deg: int = DEFAULT_MAXIMUM_TOLERANCE_DEG
    tolerance_step_deg: int = DEFAULT_TOLERANCE_STEP_DEG
    environment_randomization_probability: float = 0.0
    scene_variant_index: int | None = None
    restart: bool = False
    annotation_filename: str = "tolerance_annotation.json"
    sqp_loss_profile: str = "delta_q_squared_conditioning_safeguard"
    sqp_loss_weights: tuple[float, float, float] = (1.0, 0.0, DEFAULT_EMA_RELEASE_LOSS_WEIGHT)
    ipopt_max_iterations: int = 20
    training_scene_id: str = field(init=False)
    training_task_id: str = field(init=False)
    subtask_labels: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        training_scene, training_task = resolve_training_task(
            self.scene, self.task,
        )
        object.__setattr__(self, "training_scene_id", training_scene.scene_id)
        object.__setattr__(self, "training_task_id", training_task.task_id)
        object.__setattr__(self, "scene", training_scene.runtime_scene)
        object.__setattr__(self, "task", training_task.runtime_task)
        get_robot_spec(self.robot_uid)
        if (
            self.task not in PresetTaskExecutor.task_names(self.scene)
            or (self.scene, self.task) not in registered_tasks()
        ):
            raise ValueError(
                "Tolerance annotation requires a registered preset task: "
                f"scene={self.scene!r}, task={self.task!r}",
            )
        if self.execution_mode not in ("simulation", "kinematic"):
            raise ValueError("execution_mode must be simulation or kinematic")
        if self.maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")
        if self.maximum_tolerance_deg <= 0:
            raise ValueError("maximum_tolerance_deg must be positive")
        if self.tolerance_step_deg <= 0:
            raise ValueError("tolerance_step_deg must be positive")
        if self.maximum_tolerance_deg % self.tolerance_step_deg:
            raise ValueError(
                "maximum_tolerance_deg must be divisible by tolerance_step_deg",
            )
        if not 0.0 <= self.environment_randomization_probability <= 1.0:
            raise ValueError(
                "environment_randomization_probability must be in [0, 1]",
            )
        if self.scene_variant_index is not None and self.scene_variant_index < 0:
            raise ValueError("scene_variant_index must be non-negative")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        labels = long_task_subtask_labels(self.task) or ("Subtask 1",)
        object.__setattr__(self, "subtask_labels", labels)
        if self.targets:
            object.__setattr__(self, "targets", tuple(self.targets))
        else:
            root = (
                Path(self.output_dir)
                if self.output_dir is not None
                else Path("analysis_results") / "tolerance_annotations"
                / self.robot_uid / self.training_scene_id
                / self.training_task_id
            )
            object.__setattr__(
                self,
                "targets",
                tuple(
                    EpisodeAnnotationTarget(
                        seed=int(seed),
                        output_dir=root / f"seed_{int(seed):02d}",
                        environment_seed=RandomizationSeedRouting(
                            int(seed),
                            self.environment_randomization_probability,
                        ).environment_seed,
                    )
                    for seed in self.seeds
                ),
            )
        if not self.targets:
            raise ValueError("At least one seed or episode target is required")
        keys = [(target.episode_index, target.seed) for target in self.targets]
        if len(keys) != len(set(keys)):
            raise ValueError("Episode annotation targets must be unique")


def _search_config_record(
    config: CounterfactualSearchConfig,
) -> dict[str, object]:
    """Return the JSON-safe production search configuration."""
    return {
        "scene": config.scene,
        "task": config.task,
        "training_scene_id": config.training_scene_id,
        "training_task_id": config.training_task_id,
        "robot_uid": config.robot_uid,
        "execution_mode": config.execution_mode,
        "maximum_frames": config.maximum_frames,
        "maximum_tolerance_deg": config.maximum_tolerance_deg,
        "tolerance_step_deg": config.tolerance_step_deg,
        "environment_randomization_probability": (
            config.environment_randomization_probability
        ),
        "scene_variant_index": config.scene_variant_index,
        "sqp_loss_profile": config.sqp_loss_profile,
        "sqp_loss_weights": list(config.sqp_loss_weights),
        "ipopt_max_iterations": config.ipopt_max_iterations,
    }


def _empty_bounds(subtask_count: int) -> list[dict[str, dict[str, list[int]]]]:
    return [
        {
            stage: {
                "negative": [0, 0, 0],
                "positive": [0, 0, 0],
            }
            for stage in COUNTERFACTUAL_STAGES
        }
        for _ in range(subtask_count)
    ]


def _copy_bounds(
    bounds: list[dict[str, dict[str, list[int]]]],
) -> list[dict[str, dict[str, list[int]]]]:
    return [
        {
            stage: {
                side: list(stage_bounds[stage][side])
                for side in ("negative", "positive")
            }
            for stage in COUNTERFACTUAL_STAGES
        }
        for stage_bounds in bounds
    ]


def _spec_for_bounds(
    config: CounterfactualSearchConfig,
    target: EpisodeAnnotationTarget,
    bounds: list[dict[str, dict[str, list[int]]]],
) -> CounterfactualTrialSpec:
    negative = tuple(
        tuple(
            tuple(float(value) for value in subtask[stage]["negative"])
            for stage in COUNTERFACTUAL_STAGES
        )
        for subtask in bounds
    )
    positive = tuple(
        tuple(
            tuple(float(value) for value in subtask[stage]["positive"])
            for stage in COUNTERFACTUAL_STAGES
        )
        for subtask in bounds
    )
    masks = tuple(
        tuple(
            tuple(int(negative[subtask_index][stage_index][axis] > 0.0
                      or positive[subtask_index][stage_index][axis] > 0.0)
                  for axis in range(3))
            for stage_index in range(2)
        )
        for subtask_index in range(len(bounds))
    )
    has_release = any(
        value
        for subtask in masks
        for stage in subtask
        for value in stage
    )
    return CounterfactualTrialSpec(
        scene=config.scene,
        task=config.task,
        robot_uid=config.robot_uid,
        seed=target.seed,
        environment_seed=target.environment_seed,
        environment_randomization_probability=(
            config.environment_randomization_probability
        ),
        scene_variant_index=config.scene_variant_index,
        stage="pregrasp",
        axis="x",
        direction=1,
        angle_deg=0,
        maximum_frames=config.maximum_frames,
        controller_mode="sqp_release" if has_release else "sqp_specific",
        execution_mode=config.execution_mode,
        sqp_subtask_stage_rotation_masks=masks,
        sqp_subtask_stage_negative_limits_deg=negative,
        sqp_subtask_stage_positive_limits_deg=positive,
        sqp_loss_profile=config.sqp_loss_profile,
        sqp_loss_weights=config.sqp_loss_weights,
        ipopt_max_iterations=config.ipopt_max_iterations,
    )


def _trial_key(spec: CounterfactualTrialSpec) -> str:
    payload = asdict(spec)
    payload["counterfactual_cache_version"] = COUNTERFACTUAL_CACHE_VERSION
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(record, dict)
            and record.get("counterfactual_cache_version")
            == COUNTERFACTUAL_CACHE_VERSION
            and isinstance(record.get("key"), str)
        ):
            cache[record["key"]] = record
    return cache


def _cache_record(
    key: str,
    result: CounterfactualTrialResult,
) -> dict[str, Any]:
    return {
        "counterfactual_cache_version": COUNTERFACTUAL_CACHE_VERSION,
        "key": key,
        "success": bool(result.success),
        "termination_reason": result.termination_reason,
        "hard_failure_reasons": list(result.hard_failure_reasons),
        "hard_failure_details": result.evaluation.get(
            "hard_failure_details", [],
        ),
        "runner_error": result.evaluation.get("error"),
        "release_extrema_deg": result.evaluation.get(
            "sqp_subtask_stage_release_extrema_deg", [],
        ),
        "declared_subtask_stage_rotation_masks": result.evaluation.get(
            "declared_subtask_stage_rotation_masks", [],
        ),
        "executed_frames": result.evaluation.get("executed_frames", 0),
        "expected_frames": result.evaluation.get("expected_frames"),
        "backend": result.evaluation.get("backend"),
        "trajectory_kinematics": getattr(result, "trajectory_kinematics", {}),
    }


def _dataset_frames(
    target: EpisodeAnnotationTarget,
) -> tuple[dict[str, Any], ...] | None:
    if target.trajectory_path is None:
        return None
    payload = json.loads(target.trajectory_path.read_text(encoding="utf-8"))
    frames = payload.get("trajectory")
    if not isinstance(frames, list) or not frames:
        raise ValueError(
            f"Dataset trajectory has no frames: {target.trajectory_path}",
        )
    return tuple(frame for frame in frames if isinstance(frame, dict))


def _release_peaks(
    record: dict[str, Any],
    subtask_index: int,
    stage: str,
    axis_index: int,
) -> tuple[float, float]:
    try:
        subtask = record["release_extrema_deg"][subtask_index]
        negative = float(subtask[stage]["negative"][axis_index])
        positive = float(subtask[stage]["positive"][axis_index])
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ToleranceAnnotationError(
            "Rollout did not report the required running release extrema",
        ) from error
    if not np.isfinite(negative) or not np.isfinite(positive):
        raise ToleranceAnnotationError("Release extrema must be finite")
    return max(0.0, negative), max(0.0, positive)


def _trial_summary(
    record: dict[str, Any],
    bounds: list[dict[str, dict[str, list[int]]]],
    *,
    purpose: str,
) -> dict[str, Any]:
    return {
        "purpose": purpose,
        "success": bool(record["success"]),
        "termination_reason": record.get("termination_reason"),
        "hard_failure_reasons": record.get("hard_failure_reasons", []),
        "hard_failure_details": record.get("hard_failure_details", []),
        "runner_error": record.get("runner_error"),
        "backend": record.get("backend"),
        "executed_frames": record.get("executed_frames"),
        "expected_frames": record.get("expected_frames"),
        "release_extrema_deg": record.get("release_extrema_deg", []),
        "declared_subtask_stage_rotation_masks": record.get(
            "declared_subtask_stage_rotation_masks", [],
        ),
        "bounds_deg": _copy_bounds(bounds),
        "trajectory_kinematics": record.get("trajectory_kinematics", {}),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _annotate_target(
    config: CounterfactualSearchConfig,
    target: EpisodeAnnotationTarget,
    trial_runner: Callable[..., CounterfactualTrialResult] | None = None,
    trial_context: dict[str, Any] | None = None,
) -> Path:
    target.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = target.output_dir / config.annotation_filename
    failure_path = target.output_dir / "tolerance_annotation_failure.json"
    cache_path = target.output_dir / "tolerance_trials.jsonl"
    progress_path = target.output_dir / "search_progress.jsonl"
    if annotation_path.is_file() and not config.restart:
        if trial_context is not None:
            existing = json.loads(annotation_path.read_text(encoding="utf-8"))
            if existing.get("trial_context") != trial_context:
                raise ValueError("Existing annotation has a different trial context")
        return annotation_path
    if config.restart:
        for path in (annotation_path, failure_path, cache_path, progress_path):
            if path.is_file():
                path.unlink()

    source_frames = _dataset_frames(target)
    subtask_count = (
        1 + max(
            int(frame.get("subtask_index", 0)) for frame in source_frames
        )
        if source_frames is not None
        else len(config.subtask_labels)
    )
    labels = tuple(
        config.subtask_labels[index]
        if index < len(config.subtask_labels)
        else f"Subtask {index + 1}"
        for index in range(subtask_count)
    )
    bounds = _empty_bounds(subtask_count)
    cache = _load_cache(cache_path)
    history: list[dict[str, Any]] = []

    def evaluate(
        candidate: list[dict[str, dict[str, list[int]]]],
        purpose: str,
    ) -> dict[str, Any]:
        spec = _spec_for_bounds(config, target, candidate)
        key = _trial_key(spec)
        if trial_context:
            key += json.dumps(trial_context, sort_keys=True)
        record = cache.get(key)
        cache_hit = record is not None
        if record is None:
            result = (trial_runner or run_counterfactual_trial)(
                spec,
                dataset_nominal_frames=source_frames,
            )
            record = _cache_record(key, result)
            cache[key] = record
            with cache_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = _trial_summary(
            record, candidate, purpose=purpose,
        )
        summary["cache_hit"] = cache_hit
        history.append(summary)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return record

    def fail(message: str, context: dict[str, Any]) -> None:
        payload = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "status": "failed",
            "message": message,
            "context": context,
            "seed": target.seed,
            "episode_index": target.episode_index,
            "trajectory_path": (
                None
                if target.trajectory_path is None
                else str(target.trajectory_path)
            ),
            "search_history": history,
            "trial_context": trial_context or {},
        }
        _write_json(failure_path, payload)
        raise ToleranceAnnotationError(message)

    nominal = evaluate(bounds, "nominal_zero_tolerance_preflight")
    if not nominal["success"]:
        fail(
            "Nominal zero-tolerance trajectory failed",
            {
                "termination_reason": nominal.get("termination_reason"),
                "hard_failure_reasons": nominal.get(
                    "hard_failure_reasons", [],
                ),
                "runner_error": nominal.get("runner_error"),
            },
        )

    declared_masks = np.asarray(
        nominal.get("declared_subtask_stage_rotation_masks", []),
        dtype=int,
    )
    if (
        declared_masks.shape != (subtask_count, 2, 3)
        or np.any((declared_masks != 0) & (declared_masks != 1))
    ):
        fail(
            "Nominal rollout did not report valid task masks",
            {
                "expected_shape": [subtask_count, 2, 3],
                "reported_masks": declared_masks.tolist(),
            },
        )

    stage_results: list[dict[str, Any]] = []
    for subtask_index in reversed(range(subtask_count)):
        for stage in ("postgrasp", "pregrasp"):
            stage_index = COUNTERFACTUAL_STAGES.index(stage)
            stage_mask = declared_masks[subtask_index, stage_index]
            for side in ("negative", "positive"):
                bounds[subtask_index][stage][side] = [
                    (
                        config.maximum_tolerance_deg
                        if stage_mask[axis_index] else 0
                    )
                    for axis_index in range(3)
                ]
            initial = evaluate(
                bounds,
                f"subtask_{subtask_index}_{stage}_masked_axes_30deg",
            )
            stage_history: list[dict[str, Any]] = [{
                "kind": "initial",
                "success": bool(initial["success"]),
                "bounds_deg": _copy_bounds(bounds),
                "release_extrema_deg": initial.get(
                    "release_extrema_deg", [],
                ),
            }]
            if initial["success"]:
                stage_results.append({
                    "subtask_index": subtask_index,
                    "stage": stage,
                    "status": "all_masked_axes_30deg_succeeded",
                    "history": stage_history,
                })
                continue

            successful_axes: list[int] = []
            isolation_records: dict[int, dict[str, Any]] = {}
            for axis_index, axis in enumerate(ROTATION_AXES):
                if not stage_mask[axis_index]:
                    continue
                isolated = _copy_bounds(bounds)
                isolated[subtask_index][stage]["negative"][axis_index] = 0
                isolated[subtask_index][stage]["positive"][axis_index] = 0
                record = evaluate(
                    isolated,
                    f"subtask_{subtask_index}_{stage}_{axis}_double_zero",
                )
                isolation_records[axis_index] = record
                stage_history.append({
                    "kind": "axis_isolation",
                    "axis": axis,
                    "success": bool(record["success"]),
                    "termination_reason": record.get("termination_reason"),
                })
                if record["success"]:
                    successful_axes.append(axis_index)
            if len(successful_axes) != 1:
                fail(
                    "Stage failure did not identify exactly one corrective axis",
                    {
                        "subtask_index": subtask_index,
                        "stage": stage,
                        "successful_zeroed_axes": [
                            ROTATION_AXES[index]
                            for index in successful_axes
                        ],
                        "axis_outcomes": {
                            ROTATION_AXES[index]: bool(record["success"])
                            for index, record in isolation_records.items()
                        },
                    },
                )

            axis_index = successful_axes[0]
            axis = ROTATION_AXES[axis_index]
            previous = initial
            iteration = 0
            while not previous["success"]:
                current_negative = bounds[subtask_index][stage][
                    "negative"
                ][axis_index]
                current_positive = bounds[subtask_index][stage][
                    "positive"
                ][axis_index]
                negative_peak, positive_peak = _release_peaks(
                    previous, subtask_index, stage, axis_index,
                )
                candidates = []
                if current_negative > 0:
                    candidates.append((
                        negative_peak,
                        current_negative,
                        "negative",
                    ))
                if current_positive > 0:
                    candidates.append((
                        positive_peak,
                        current_positive,
                        "positive",
                    ))
                if not candidates:
                    fail(
                        "Double-zero corrective axis unexpectedly still failed",
                        {
                            "subtask_index": subtask_index,
                            "stage": stage,
                            "axis": axis,
                        },
                    )
                _, current_value, side = max(
                    candidates,
                    key=lambda item: (
                        item[0], item[1], item[2] == "positive",
                    ),
                )
                selected_peak = (
                    negative_peak if side == "negative" else positive_peak
                )
                measured_floor = int(
                    math.floor(
                        (selected_peak + 1.0e-9)
                        / config.tolerance_step_deg
                    ) * config.tolerance_step_deg
                )
                new_value = max(
                    0,
                    min(
                        current_value - config.tolerance_step_deg,
                        measured_floor,
                    ),
                )
                bounds[subtask_index][stage][side][axis_index] = new_value
                iteration += 1
                previous = evaluate(
                    bounds,
                    f"subtask_{subtask_index}_{stage}_{axis}_{side}_"
                    f"{current_value}_to_{new_value}",
                )
                stage_history.append({
                    "kind": "signed_reduction",
                    "iteration": iteration,
                    "axis": axis,
                    "selected_side": side,
                    "negative_peak_deg": negative_peak,
                    "positive_peak_deg": positive_peak,
                    "previous_bound_deg": current_value,
                    "selected_bound_deg": new_value,
                    "success": bool(previous["success"]),
                    "termination_reason": previous.get("termination_reason"),
                    "bounds_deg": _copy_bounds(bounds),
                })
            stage_results.append({
                "subtask_index": subtask_index,
                "stage": stage,
                "status": "corrected",
                "corrective_axis": axis,
                "history": stage_history,
            })

    final = evaluate(bounds, "final_annotation_validation")
    if not final["success"]:
        fail(
            "Final annotation validation failed",
            {
                "termination_reason": final.get("termination_reason"),
                "hard_failure_reasons": final.get(
                    "hard_failure_reasons", [],
                ),
            },
        )

    subtasks = [
        {
            "index": index,
            "label": labels[index],
            "pregrasp": bounds[index]["pregrasp"],
            "postgrasp": bounds[index]["postgrasp"],
        }
        for index in range(subtask_count)
    ]
    payload = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "status": "complete",
        "method": f"per_episode_peak_guided_{config.tolerance_step_deg}deg_search",
        "definition": (
            "Process subtasks from last to first. For each subtask, search "
            "Post-grasp with Pre-grasp blocked, then search Pre-grasp while "
            "holding the finalized Post-grasp bounds. Each failed stage is "
            "diagnosed only on axes enabled by the task mask using bilateral "
            "zero trials, then corrected on the unique successful axis using "
            "running signed release extrema. Masked-out axes remain zero."
        ),
        "training_scene_id": config.training_scene_id,
        "training_task_id": config.training_task_id,
        "scene": config.scene,
        "task": config.task,
        "robot_uid": config.robot_uid,
        "execution_mode": config.execution_mode,
        "seed": target.seed,
        "episode_index": target.episode_index,
        "trajectory_path": (
            None
            if target.trajectory_path is None
            else str(target.trajectory_path)
        ),
        "search": _search_config_record(config),
        "trial_context": trial_context or {},
        "rotation_tolerance_bounds_order": [
            "rx_negative", "rx_positive",
            "ry_negative", "ry_positive",
            "rz_negative", "rz_positive",
        ],
        "subtasks": subtasks,
        "subtask_bounds_deg": _copy_bounds(bounds),
        "declared_subtask_stage_rotation_masks": declared_masks.tolist(),
        "search_order": [
            {
                "subtask_index": index,
                "stages": ["postgrasp", "pregrasp"],
            }
            for index in reversed(range(subtask_count))
        ],
        "stage_results": stage_results,
        "final_validation": _trial_summary(
            final, bounds, purpose="final_annotation_validation",
        ),
        "search_history": history,
        "failure_policy": {
            "nominal_failure": "exception",
            "zero_successful_axis_isolations": "exception",
            "multiple_successful_axis_isolations": "exception",
            "optimizer_infeasible": "trajectory_failure",
            "robot_self_collision": "trajectory_failure",
            "robot_table_collision": "trajectory_failure",
        },
        "release_measurement": (
            "per-stage running positive and negative maxima over every 10 Hz "
            "optimizer frame; endpoint-only release is not used"
        ),
    }
    _write_json(annotation_path, payload)
    if failure_path.is_file():
        failure_path.unlink()
    return annotation_path


def _index_root(targets: Iterable[EpisodeAnnotationTarget]) -> Path:
    parents = [target.output_dir.parent for target in targets]
    common = Path(parents[0])
    for parent in parents[1:]:
        while common != common.parent and not parent.is_relative_to(common):
            common = common.parent
    return common


def search_task_tolerance(
    config: CounterfactualSearchConfig,
    *,
    trial_runner: Callable[..., CounterfactualTrialResult] | None = None,
    trial_context: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Annotate every requested seed independently and return result paths."""
    results = tuple(
        _annotate_target(config, target, trial_runner, trial_context)
        for target in config.targets
    )
    root = _index_root(config.targets)
    index = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "training_scene_id": config.training_scene_id,
        "training_task_id": config.training_task_id,
        "scene": config.scene,
        "task": config.task,
        "robot_uid": config.robot_uid,
        "execution_mode": config.execution_mode,
        "annotation_count": len(results),
        "annotations": [
            {
                "seed": target.seed,
                "episode_index": target.episode_index,
                "path": str(path.relative_to(root)),
            }
            for target, path in zip(config.targets, results, strict=True)
        ],
    }
    _write_json(root / "tolerance_annotation_index.json", index)
    return results


__all__ = (
    "ANNOTATION_SCHEMA_VERSION",
    "CounterfactualSearchConfig",
    "ENVIRONMENT_ROUTING_ANNOTATION_VERSION",
    "EpisodeAnnotationTarget",
    "ToleranceAnnotationError",
    "search_task_tolerance",
)
