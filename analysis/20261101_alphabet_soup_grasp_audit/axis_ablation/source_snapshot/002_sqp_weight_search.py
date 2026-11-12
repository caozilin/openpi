"""Simulation-efficient search for SQP objective weights across tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from ..counterfactual.kinematic import (
    DEFAULT_KINEMATIC_WEIGHTS,
    KINEMATIC_METRIC_NAMES,
    TrajectoryKinematicMetrics,
    compare_trajectory_kinematics,
)
from ..counterfactual.runner import (
    COUNTERFACTUAL_CACHE_VERSION,
    CounterfactualTrialResult,
    CounterfactualTrialSpec,
    run_counterfactual_trial,
)
from ..data.task_sets import resolve_training_task


DEFAULT_OBJECTIVE_WEIGHTS = (
    50.0, 10.0, 0.7, 0.3, 1.0, 0.01, 1.0,
)
TUNED_WEIGHT_INDICES = (3, 4, 6)
TUNED_WEIGHT_NAMES = (
    "jerk_weight",
    "manipulability_weight",
    "release_loss_weight",
)
DEFAULT_MULTIPLIER_BOUNDS = (
    (0.25, 8.0),
    (0.25, 4.0),
    (0.25, 4.0),
)
_STAGES = ("pregrasp", "postgrasp")
_NUMERIC_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class TaskConfig:
    scene_id: str
    task_id: str
    runtime_scene: str
    runtime_task: str
    annotation_path: str
    masks: tuple[tuple[int, int, int], tuple[int, int, int]]
    negative_limits_deg: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    positive_limits_deg: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]

    @property
    def key(self) -> str:
        return f"{self.scene_id}/{self.task_id}"


@dataclass(frozen=True)
class WeightCandidate:
    candidate_id: str
    weights: tuple[float, ...]
    multipliers: tuple[float, float, float]


def _task_ref(value: str) -> tuple[str, str]:
    parts = value.split("/", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "task must use training_scene_id/training_task_id",
        )
    return parts[0].strip(), parts[1].strip()


def _tuple2x3(
    value: object,
    *,
    name: str,
    cast,
) -> tuple[tuple, tuple]:
    array = np.asarray(value)
    if array.shape != (2, 3):
        raise ValueError(f"{name} must have shape (2, 3)")
    return tuple(tuple(cast(item) for item in row) for row in array)  # type: ignore[return-value]


def load_task_config(
    scene_id: str,
    task_id: str,
    robot_uid: str,
    dataset_root: Path,
) -> TaskConfig:
    scene, task = resolve_training_task(scene_id, task_id)
    path = dataset_root / robot_uid / scene.scene_id / task.task_id / (
        "tolerance_annotation.json"
    )
    if not path.exists():
        legacy_matches: list[Path] = []
        robot_root = dataset_root / robot_uid
        for candidate_path in robot_root.rglob("tolerance_annotation.json"):
            try:
                candidate_payload = json.loads(
                    candidate_path.read_text(encoding="utf-8"),
                )
            except (OSError, ValueError):
                continue
            logical_match = (
                candidate_payload.get("training_scene_id") == scene.scene_id
                and candidate_payload.get("training_task_id") == task.task_id
            )
            runtime_match = (
                candidate_payload.get("scene") == scene.runtime_scene
                and candidate_payload.get("task") == task.runtime_task
            )
            if logical_match or runtime_match:
                legacy_matches.append(candidate_path)
        if len(legacy_matches) == 1:
            path = legacy_matches[0]
            print(f"[annotation] using legacy path for {scene.scene_id}/{task.task_id}: {path}")
        elif len(legacy_matches) > 1:
            raise RuntimeError(
                f"Multiple annotations match {scene.scene_id}/{task.task_id}: "
                + ", ".join(str(item) for item in legacy_matches),
            )
        else:
            raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    stage_masks = payload.get("stage_rotation_masks")
    if isinstance(stage_masks, Mapping):
        stage_masks = [stage_masks[stage] for stage in _STAGES]
    masks = _tuple2x3(stage_masks, name="stage_rotation_masks", cast=int)
    negative = _tuple2x3(
        payload.get("sqp_stage_negative_limits_deg"),
        name="sqp_stage_negative_limits_deg",
        cast=float,
    )
    positive = _tuple2x3(
        payload.get("sqp_stage_positive_limits_deg"),
        name="sqp_stage_positive_limits_deg",
        cast=float,
    )
    if not any(any(row) for row in masks):
        raise ValueError(f"{scene.scene_id}/{task.task_id} releases no SQP axis")
    return TaskConfig(
        scene.scene_id,
        task.task_id,
        scene.runtime_scene,
        task.runtime_task,
        str(path),
        masks,  # type: ignore[arg-type]
        negative,  # type: ignore[arg-type]
        positive,  # type: ignore[arg-type]
    )


def _candidate(
    candidate_id: str,
    multipliers: Sequence[float],
) -> WeightCandidate:
    multiplier_tuple = tuple(float(value) for value in multipliers)
    if len(multiplier_tuple) != len(TUNED_WEIGHT_INDICES):
        raise ValueError("Expected three SQP weight multipliers")
    weights = list(DEFAULT_OBJECTIVE_WEIGHTS)
    for index, multiplier in zip(
        TUNED_WEIGHT_INDICES, multiplier_tuple, strict=True,
    ):
        weights[index] *= multiplier
    return WeightCandidate(candidate_id, tuple(weights), multiplier_tuple)  # type: ignore[arg-type]


def latin_hypercube_candidates(
    count: int,
    seed: int,
) -> list[WeightCandidate]:
    """Generate deterministic log-uniform candidates, plus the incumbent."""
    if count < 0:
        raise ValueError("candidate count must be non-negative")
    result = [_candidate("default", (1.0, 1.0, 1.0))]
    if count == 0:
        return result
    rng = np.random.default_rng(seed)
    dimensions = len(TUNED_WEIGHT_INDICES)
    samples = np.empty((count, dimensions), dtype=float)
    for dimension in range(dimensions):
        strata = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(strata)
        lower, upper = DEFAULT_MULTIPLIER_BOUNDS[dimension]
        samples[:, dimension] = np.exp(
            math.log(lower)
            + strata * (math.log(upper) - math.log(lower))
        )
    result.extend(
        _candidate(f"lhs_{index:03d}", row)
        for index, row in enumerate(samples)
    )
    return result


def local_candidates(
    centre: WeightCandidate,
    count: int,
    seed: int,
    sigma: float = 0.35,
) -> list[WeightCandidate]:
    rng = np.random.default_rng(seed)
    centre_log = np.log(np.asarray(centre.multipliers))
    result: list[WeightCandidate] = []
    for index in range(count):
        proposal = np.exp(centre_log + rng.normal(0.0, sigma, 4))
        for dimension, (lower, upper) in enumerate(
            DEFAULT_MULTIPLIER_BOUNDS
        ):
            proposal[dimension] = np.clip(
                proposal[dimension], lower, upper,
            )
        result.append(_candidate(f"refine_{index:03d}", proposal))
    return result


def _case_seeds(seed: int, count: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    result: list[int] = []
    while len(result) < count:
        value = int(rng.integers(0, 2**32, dtype=np.uint32))
        if value not in result:
            result.append(value)
    return tuple(result)


def _trial_spec(
    task: TaskConfig,
    seed: int,
    candidate: WeightCandidate,
    maximum_frames: int,
    robot_uid: str,
) -> CounterfactualTrialSpec:
    return CounterfactualTrialSpec(
        scene=task.runtime_scene,
        task=task.runtime_task,
        robot_uid=robot_uid,
        seed=seed,
        stage="pregrasp",
        axis="x",
        direction=1,
        angle_deg=0,
        maximum_frames=maximum_frames,
        controller_mode="sqp_release",
        sqp_stage_rotation_masks=task.masks,
        sqp_stage_negative_limits_deg=task.negative_limits_deg,
        sqp_stage_positive_limits_deg=task.positive_limits_deg,
        sqp_objective_weights=candidate.weights,  # type: ignore[arg-type]
    )


def _record_key(
    task: TaskConfig,
    robot_uid: str,
    seed: int,
    candidate: WeightCandidate,
    maximum_frames: int,
) -> str:
    payload = {
        "cache_version": COUNTERFACTUAL_CACHE_VERSION,
        "task": task.key,
        "annotation": {
            "masks": task.masks,
            "negative": task.negative_limits_deg,
            "positive": task.positive_limits_deg,
        },
        "robot": robot_uid,
        "seed": seed,
        "weights": candidate.weights,
        "maximum_frames": maximum_frames,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _run_spec(spec: CounterfactualTrialSpec) -> CounterfactualTrialResult:
    return run_counterfactual_trial(spec)


@contextmanager
def _single_thread_workers() -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in _NUMERIC_THREAD_ENV}
    try:
        os.environ.update(_NUMERIC_THREAD_ENV)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TrialStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, object]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[str(record["key"])] = record

    def add(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records[str(record["key"])] = record


def _result_record(
    key: str,
    task: TaskConfig,
    candidate: WeightCandidate,
    result: CounterfactualTrialResult,
) -> dict[str, object]:
    metrics = result.evaluation.get(
        "optimized_10hz_trajectory_kinematics",
        result.trajectory_kinematics,
    )
    return {
        "key": key,
        "task": task.key,
        "candidate_id": candidate.candidate_id,
        "weights": list(candidate.weights),
        "multipliers": dict(zip(
            TUNED_WEIGHT_NAMES, candidate.multipliers, strict=True,
        )),
        "seed": result.spec.seed,
        "success": result.success,
        "task_success": result.task_success,
        "termination_reason": result.termination_reason,
        "hard_failure_reasons": list(result.hard_failure_reasons),
        "metrics": metrics,
        "stage_endpoint_release_deg": result.evaluation.get(
            "sqp_stage_endpoint_release_deg",
        ),
    }


def evaluate_missing(
    store: TrialStore,
    tasks: Sequence[TaskConfig],
    seeds: Sequence[int],
    candidates: Sequence[WeightCandidate],
    *,
    robot_uid: str,
    maximum_frames: int,
    workers: int,
) -> None:
    jobs: list[tuple[str, TaskConfig, WeightCandidate, CounterfactualTrialSpec]] = []
    for task in tasks:
        for seed in seeds:
            for candidate in candidates:
                key = _record_key(
                    task, robot_uid, seed, candidate, maximum_frames,
                )
                if key in store.records:
                    continue
                spec = _trial_spec(
                    task, seed, candidate, maximum_frames, robot_uid,
                )
                jobs.append((key, task, candidate, spec))
    if not jobs:
        return
    print(f"Running {len(jobs)} missing rollout(s) with {workers} workers")
    with _single_thread_workers(), ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(jobs), os.cpu_count() or 1)),
    ) as executor:
        futures = {
            executor.submit(_run_spec, spec): (key, task, candidate)
            for key, task, candidate, spec in jobs
        }
        completed = 0
        for future in as_completed(futures):
            key, task, candidate = futures[future]
            result = future.result()
            store.add(_result_record(key, task, candidate, result))
            completed += 1
            print(
                f"  {completed}/{len(jobs)} {task.key} "
                f"{candidate.candidate_id} seed={result.spec.seed} "
                f"success={result.success}",
                flush=True,
            )


def _release_vector(record: Mapping[str, object]) -> np.ndarray:
    payload = record.get("stage_endpoint_release_deg")
    if not isinstance(payload, Mapping):
        return np.zeros((2, 3))
    return np.asarray([payload.get(stage, (0.0, 0.0, 0.0)) for stage in _STAGES])


def release_retention_ratio(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    masks: Sequence[Sequence[int]],
) -> float:
    active = np.asarray(masks, dtype=bool)
    reference = _release_vector(baseline)[active]
    observed = _release_vector(candidate)[active]
    denominator = float(np.sum(np.abs(reference)))
    if denominator < 1.0:
        return 1.0
    aligned = np.maximum(observed * np.sign(reference), 0.0)
    return float(np.sum(aligned) / denominator)


def _score_pair(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> float:
    base_metrics = TrajectoryKinematicMetrics.from_mapping(
        baseline.get("metrics") if isinstance(baseline, Mapping) else None,
    )
    candidate_metrics = TrajectoryKinematicMetrics.from_mapping(
        candidate.get("metrics") if isinstance(candidate, Mapping) else None,
    )
    return compare_trajectory_kinematics(
        base_metrics,
        candidate_metrics,
        DEFAULT_KINEMATIC_WEIGHTS,
    ).weighted_loss_ratio


def rank_candidates(
    store: TrialStore,
    tasks: Sequence[TaskConfig],
    seeds: Sequence[int],
    candidates: Sequence[WeightCandidate],
    *,
    robot_uid: str,
    maximum_frames: int,
    minimum_release_retention: float,
) -> list[dict[str, object]]:
    default = _candidate("default", (1.0, 1.0, 1.0, 1.0))
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        scores: list[float] = []
        retentions: list[float] = []
        failures: list[dict[str, object]] = []
        for task in tasks:
            for seed in seeds:
                base_record = store.records[_record_key(
                    task, robot_uid, seed, default, maximum_frames,
                )]
                record = store.records[_record_key(
                    task, robot_uid, seed, candidate, maximum_frames,
                )]
                retention = release_retention_ratio(
                    base_record, record, task.masks,
                )
                success = bool(record.get("success"))
                if not success or retention < minimum_release_retention:
                    failures.append({
                        "task": task.key,
                        "seed": seed,
                        "success": success,
                        "retention": retention,
                        "reason": record.get("termination_reason"),
                    })
                scores.append(_score_pair(base_record, record))
                retentions.append(retention)
        feasible = not failures
        q80 = float(np.quantile(scores, 0.8, method="higher"))
        rows.append({
            "candidate_id": candidate.candidate_id,
            "weights": list(candidate.weights),
            "multipliers": dict(zip(
                TUNED_WEIGHT_NAMES, candidate.multipliers, strict=True,
            )),
            "feasible": feasible,
            "q80_score": q80,
            "mean_score": float(np.mean(scores)),
            "minimum_release_retention": float(min(retentions)),
            "failures": failures,
        })
    rows.sort(key=lambda row: (
        not bool(row["feasible"]),
        float(row["q80_score"]),
        float(row["mean_score"]),
    ))
    return rows


def _candidate_by_id(
    candidates: Sequence[WeightCandidate], candidate_id: str,
) -> WeightCandidate:
    return next(item for item in candidates if item.candidate_id == candidate_id)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", action="append", type=_task_ref, required=True,
        metavar="SCENE_ID/TASK_ID",
    )
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output", type=Path, default=Path("sqp_weight_optimization"),
    )
    parser.add_argument("--initial-candidates", type=int, default=8)
    parser.add_argument("--refine-candidates", type=int, default=4)
    parser.add_argument("--representative-tasks", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--validation-seeds", type=int, default=2)
    parser.add_argument("--release-retention", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-missing-annotations", action="store_true",
        help="warn and optimize only tasks whose tolerance annotation exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.release_retention <= 1.5:
        raise ValueError("--release-retention must be within [0, 1.5]")
    if min(
        args.initial_candidates,
        args.refine_candidates,
        args.validation_seeds,
    ) < 0:
        raise ValueError("candidate and validation counts must be non-negative")
    if args.validation_seeds == 0:
        raise ValueError("--validation-seeds must be at least 1")
    if min(args.representative_tasks, args.top_k, args.workers) <= 0:
        raise ValueError("representative tasks, top-k and workers must be positive")
    tasks: list[TaskConfig] = []
    missing: list[tuple[str, str, str]] = []
    for scene, task in args.task:
        try:
            tasks.append(load_task_config(
                scene, task, args.robot, args.dataset_root,
            ))
        except FileNotFoundError as error:
            missing.append((scene, task, str(error)))
    if missing:
        lines = ["Missing tolerance annotations:"]
        for scene, task, path in missing:
            lines.extend((
                f"  - {scene}/{task}: {path}",
                "    uv run python annotate_tolerance.py "
                f"--scene {scene} --task {task} --robot {args.robot}",
            ))
        message = "\n".join(lines)
        if not args.skip_missing_annotations:
            raise FileNotFoundError(
                message
                + "\nRun these commands first, or add "
                "--skip-missing-annotations to optimize the available tasks.",
            )
        print(f"[warning] {message}")
    if not tasks:
        raise RuntimeError("No task with a usable tolerance annotation remains")
    representative = tasks[:max(1, min(args.representative_tasks, len(tasks)))]
    seeds = _case_seeds(args.seed, 1 + args.validation_seeds)
    initial = latin_hypercube_candidates(args.initial_candidates, args.seed)
    estimated = (
        len(initial) * len(representative)
        + args.top_k * len(tasks)
        + args.refine_candidates * len(representative)
        + 2 * len(tasks) * args.validation_seeds
    )
    print(
        f"Tasks={len(tasks)} candidates={len(initial)} "
        f"estimated candidate rollouts<={estimated} "
        f"(+ cached/default baselines)",
    )
    if args.dry_run:
        print(json.dumps({
            "tasks": [task.key for task in tasks],
            "seeds": list(seeds),
            "initial_candidates": [asdict(item) for item in initial],
        }, ensure_ascii=False, indent=2))
        return

    store = TrialStore(args.output / "sqp_weight_trials.jsonl")
    default = initial[0]

    # Round 1: broad low-fidelity screen on representative tasks.
    evaluate_missing(
        store, representative, seeds[:1], initial,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        workers=args.workers,
    )
    round1 = rank_candidates(
        store, representative, seeds[:1], initial,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        minimum_release_retention=args.release_retention,
    )
    selected1 = [
        _candidate_by_id(initial, row["candidate_id"])
        for row in round1[:max(1, min(args.top_k, len(round1)))]
    ]
    if default not in selected1:
        selected1.append(default)

    # Round 2: task expansion for the survivors.
    evaluate_missing(
        store, tasks, seeds[:1], selected1,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        workers=args.workers,
    )
    round2 = rank_candidates(
        store, tasks, seeds[:1], selected1,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        minimum_release_retention=args.release_retention,
    )
    centre = _candidate_by_id(selected1, round2[0]["candidate_id"])

    # Round 3: local log-space refinement around the best survivor.
    refined = local_candidates(
        centre, args.refine_candidates, args.seed ^ 0x535150,
    )
    evaluate_missing(
        store, representative, seeds[:1], refined,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        workers=args.workers,
    )
    local_pool = list(dict.fromkeys((*selected1, *refined)))
    local_rank = rank_candidates(
        store, representative, seeds[:1], local_pool,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        minimum_release_retention=args.release_retention,
    )
    finalists = [
        _candidate_by_id(local_pool, row["candidate_id"])
        for row in local_rank[:2]
    ]
    evaluate_missing(
        store, tasks, seeds[:1], finalists,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        workers=args.workers,
    )
    training_rank = rank_candidates(
        store, tasks, seeds[:1], finalists,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        minimum_release_retention=args.release_retention,
    )

    # Final held-out validation. Default rollouts are automatically cached and
    # provide the paired denominator for every candidate.
    validation_candidates = list(dict.fromkeys((
        default,
        *[
            _candidate_by_id(finalists, row["candidate_id"])
            for row in training_rank[:2]
        ],
    )))
    validation_seed_panel = seeds[1:]
    evaluate_missing(
        store, tasks, validation_seed_panel, validation_candidates,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        workers=args.workers,
    )
    validation_rank = rank_candidates(
        store, tasks, validation_seed_panel, validation_candidates,
        robot_uid=args.robot, maximum_frames=args.max_frames,
        minimum_release_retention=args.release_retention,
    )
    best = validation_rank[0]
    report = {
        "schema_version": "1.0",
        "method": "paired_multifidelity_sqp_weight_search",
        "robot_uid": args.robot,
        "tasks": [asdict(task) for task in tasks],
        "training_seed": seeds[0],
        "validation_seeds": list(validation_seed_panel),
        "kinematic_weights": DEFAULT_KINEMATIC_WEIGHTS,
        "minimum_stage_endpoint_release_retention": args.release_retention,
        "tuned_weights": list(TUNED_WEIGHT_NAMES),
        "round1": round1,
        "round2": round2,
        "local_refinement": local_rank,
        "training_finalists": training_rank,
        "validation": validation_rank,
        "best": best,
    }
    _write_json(args.output / "sqp_weight_report.json", report)
    _write_json(args.output / "best_sqp_weights.json", {
        "weights_order": [
            "position", "rotation", "velocity", "jerk",
            "manipulability", "self_collision", "tolerance",
        ],
        **best,
    })
    print(
        f"Best={best['candidate_id']} q80={best['q80_score']:.4f} "
        f"mean={best['mean_score']:.4f} feasible={best['feasible']}",
    )
    print(f"Report: {args.output / 'sqp_weight_report.json'}")


if __name__ == "__main__":
    main()
