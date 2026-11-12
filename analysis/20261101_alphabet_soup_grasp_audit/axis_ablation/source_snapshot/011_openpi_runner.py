"""Batch orchestration for an OpenPI websocket policy and MuJoCo benchmark."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from ..data.task_sets import short_training_tasks
from ..policies.openpi.client import OpenPIClientConfig, OpenPIWebsocketPolicy
from .catalog import (
    benchmark_only_tasks_for_robot,
    benchmark_seed_plan,
    dataset_tasks_for_robot,
    resolve_dataset_task,
)
from .environment import (
    BenchmarkEnvironmentConfig,
    MuJoCoVLABenchmarkEnvironment,
)
from .metrics import BenchmarkEpisodeResult, aggregate_episode_results
from .runner import BenchmarkCase, BenchmarkRunner
from ..tolerance.oracle import (
    TOLERANCE_MODE_DATASET_ORACLE,
    TOLERANCE_MODE_OFF,
    TOLERANCE_MODE_POLICY_PREDICTED,
    TOLERANCE_MODES,
)
from ..tolerance.state import ORIENTATION_REFERENCE_STAGE_RELATIVE
from ..simulation.robots import get_robot_spec
from ..reachability.layout import (
    WORKSPACE_BOUNDARY_DISTANCE_MAX,
    WORKSPACE_BOUNDARY_DISTANCE_MIN,
    WORKSPACE_BOUNDARY_MODES,
)


WORKSPACE_BOUNDARY_GEOMETRY_TASKS = frozenset({
    "geometry_plate_ball",
    "geometry_plate_cylinder_upright",
    "geometry_plate_cylinder_upright_stable",
    "geometry_plate_cylinder_lying",
    "geometry_plate_cylinder_lying_stable",
    "cylinder_to_narrow_box",
    "geometry_plate_box_upright",
    "geometry_plate_box_upright_stable",
    "geometry_plate_box_lying",
    "geometry_plate_box_lying_stable",
    "geometry_region_box_upright",
    "geometry_region_box_lying",
    "geometry_region_cylinder_upright",
    "geometry_region_cylinder_lying",
    "box_to_narrow_box",
})
WORKSPACE_BOUNDARY_TASKS = frozenset(
    task.task_id for _, task in short_training_tasks()
)


def _workspace_boundary_task_family(task: Any) -> str:
    task_key = task.directory.name
    if task_key not in WORKSPACE_BOUNDARY_TASKS:
        raise ValueError(
            f"Task {task_key!r} is not one of the 22 short training "
            "workspace-boundary tasks",
        )
    if task_key not in WORKSPACE_BOUNDARY_GEOMETRY_TASKS:
        return "contextual"
    if task_key.endswith("_to_narrow_box"):
        return "box"
    return "region" if task_key.startswith("geometry_region_") else "tray"


def _workspace_boundary_mode(task: Any, episode_index: int) -> str:
    _workspace_boundary_task_family(task)
    return "pre_boundary"


@dataclass(frozen=True)
class OpenPIBenchmarkConfig:
    task: str
    count: int
    robot_uid: str = "panda"
    dataset_root: Path = Path("datasets")
    output_root: Path = Path("benchmark_results")
    seed: int = -1
    scene_variant_index: int | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    maximum_steps: int | None = None
    replan_steps: int = 10
    tolerance_mode: str = TOLERANCE_MODE_OFF
    environment_randomization_probability: float = 0.0
    # Compatibility for existing Python callers; new CLI code uses probability.
    environment_randomization: bool | None = None
    orientation_reference_mode: str = ORIENTATION_REFERENCE_STAGE_RELATIVE
    placement_difficulty: float = 0.0
    reachability_map_dir: Path | None = None
    workspace_boundary: bool = False
    workspace_boundary_distance_min: float = WORKSPACE_BOUNDARY_DISTANCE_MIN
    workspace_boundary_distance_max: float = WORKSPACE_BOUNDARY_DISTANCE_MAX
    run_name: str | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        get_robot_spec(self.robot_uid)
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.scene_variant_index is not None and self.scene_variant_index < 0:
            raise ValueError("scene_variant_index must be non-negative")
        if self.maximum_steps is not None and self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        if self.tolerance_mode not in TOLERANCE_MODES:
            raise ValueError(f"Unknown tolerance mode: {self.tolerance_mode}")
        if self.orientation_reference_mode != ORIENTATION_REFERENCE_STAGE_RELATIVE:
            raise ValueError(
                "Stage-relative release is the sole maintained reference mode",
            )
        probability = float(self.environment_randomization_probability)
        if self.environment_randomization is not None:
            legacy_probability = float(bool(self.environment_randomization))
            if probability != 0.0 and probability != legacy_probability:
                raise ValueError(
                    "environment_randomization and "
                    "environment_randomization_probability disagree",
                )
            probability = legacy_probability
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "environment_randomization_probability must be in [0, 1]",
            )
        object.__setattr__(
            self, "environment_randomization_probability", probability,
        )
        if not 0.0 <= self.placement_difficulty <= 1.0:
            raise ValueError("placement_difficulty must be in [0, 1]")
        if self.workspace_boundary and self.placement_difficulty:
            raise ValueError(
                "workspace-boundary suite uses its hard-coded difficulty and "
                "cannot be combined with --placement-difficulty",
            )
        if not (
            0.0 <= self.workspace_boundary_distance_min
            <= self.workspace_boundary_distance_max <= 2.0
        ):
            raise ValueError(
                "workspace-boundary distance factors must satisfy "
                "0 <= min <= max <= 2",
            )


def _safe_directory_label(value: object) -> str:
    """Return one compact cross-platform directory component."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return normalized.strip("._-")[:120]


def _checkpoint_version_label(checkpoint_value: object) -> str | None:
    """Extract a version token such as ``v2`` or ``v3`` from a checkpoint path."""
    version_pattern = re.compile(
        r"(?<![A-Za-z0-9])v\d+(?:[._-]\d+)*(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    for component in reversed(Path(str(checkpoint_value)).parts):
        match = version_pattern.search(component)
        if match is not None:
            return match.group(0).lower().replace("-", ".")
    return None


def _model_directory_label(
    config: OpenPIBenchmarkConfig,
    policy_metadata: dict[str, Any] | None = None,
) -> str:
    """Resolve model identity from CLI, server metadata, or the run label."""
    if config.model_name:
        return _safe_directory_label(config.model_name) or "openpi"
    metadata = policy_metadata or {}
    model_value = next((
        metadata.get(key)
        for key in ("model_name", "model", "policy_config", "config")
        if metadata.get(key)
    ), None)
    checkpoint_value = next((
        metadata.get(key)
        for key in ("checkpoint", "checkpoint_path", "policy_dir")
        if metadata.get(key)
    ), None)
    parts: list[str] = []
    if model_value is not None:
        parts.append(_safe_directory_label(model_value))
    if checkpoint_value is not None:
        version_label = _checkpoint_version_label(checkpoint_value)
        if version_label is not None:
            parts.append(version_label)
        checkpoint_name = Path(str(checkpoint_value)).name
        parts.append(_safe_directory_label(checkpoint_name))
    resolved = "_".join(part for part in parts if part)
    if resolved:
        return resolved
    if config.run_name:
        return _safe_directory_label(config.run_name) or "openpi"
    return "openpi"


def _timestamped_model_directory(
    config: OpenPIBenchmarkConfig,
    policy_metadata: dict[str, Any] | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tolerance_labels = {
        TOLERANCE_MODE_OFF: "off",
        TOLERANCE_MODE_DATASET_ORACLE: "dataset",
        TOLERANCE_MODE_POLICY_PREDICTED: "policy",
    }
    tolerance_label = tolerance_labels[config.tolerance_mode]
    return (
        f"{_model_directory_label(config, policy_metadata)}_"
        f"tolerance-{tolerance_label}_{timestamp}"
    )


def _read_policy_server_metadata(
    config: OpenPIBenchmarkConfig,
) -> dict[str, Any]:
    """Read handshake metadata without running a policy inference."""
    policy = OpenPIWebsocketPolicy(OpenPIClientConfig(
        host=config.host, port=config.port, api_key=config.api_key,
    ))
    try:
        return dict(policy.server_metadata)
    finally:
        policy.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _environment_randomization_plan(
    task_key: str,
    seeds: tuple[int, ...],
    probability: float,
) -> tuple[bool, ...]:
    """Select an exact, reproducible per-task fraction of rollout episodes."""
    enabled_count = int(np.floor(len(seeds) * float(probability) + 0.5))
    ranked = sorted(
        range(len(seeds)),
        key=lambda index: hashlib.blake2b(
            f"{task_key}\0{seeds[index]}\0{index}".encode("utf-8"),
            digest_size=16,
            person=b"tasktol-env-rand",
        ).digest(),
    )
    selected = set(ranked[:enabled_count])
    return tuple(index in selected for index in range(len(seeds)))


def _write_summary(
    path: Path,
    *,
    config: OpenPIBenchmarkConfig,
    task: Any,
    seeds: tuple[int, ...],
    policy_metadata: dict[str, Any],
    results: list[BenchmarkEpisodeResult],
    status: str,
    error: str | None = None,
) -> None:
    aggregate = aggregate_episode_results(results)
    environment_plan = _environment_randomization_plan(
        task.task_id, seeds, config.environment_randomization_probability,
    )
    workspace_plan = (
        tuple(_workspace_boundary_mode(task, index) for index in range(len(seeds)))
        if config.workspace_boundary else ()
    )
    payload = {
        "benchmark_version": "1.0",
        "status": status,
        "task": {
            "scene": task.scene,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "dataset_directory": str(task.directory),
            "metadata_robot": task.robot_uid,
            "benchmark_only": task.benchmark_only,
            "reference_task_directory": task.reference_task_directory,
        },
        "configuration": {
            "robot": {
                "uid": (spec := get_robot_spec(config.robot_uid)).uid,
                "name": spec.label,
                "arm_dof": spec.arm_dof,
            },
            "requested_episodes": config.count,
            "initial_seed": config.seed,
            "seeds": list(seeds),
            "scene_variant_index": config.scene_variant_index,
            "frequency_hz": 10.0,
            "maximum_steps": config.maximum_steps or task.maximum_steps,
            "maximum_duration_s": (
                (config.maximum_steps or task.maximum_steps) / 10.0
            ),
            "dataset_maximum_demonstration_duration_s": (
                task.maximum_demonstration_duration_s
            ),
            "dataset_duration_margin_s": 5.0,
            "maximum_duration_source": (
                "explicit_maximum_steps"
                if config.maximum_steps is not None
                else (
                    "reference_task_maximum_demonstration_duration_plus_5s"
                    if task.benchmark_only
                    else "dataset_maximum_demonstration_duration_plus_5s"
                )
            ),
            "replan_steps": config.replan_steps,
            "controller": "TaskTol stage-relative IPOPT",
            "optimizer_backend": "ipopt",
            "tolerance_mode": config.tolerance_mode,
            "tolerance_release": config.tolerance_mode != TOLERANCE_MODE_OFF,
            "orientation_reference_mode": config.orientation_reference_mode,
            "tolerance_parameter_source": (
                None if config.tolerance_mode == TOLERANCE_MODE_OFF
                else (
                    "seeded preset-demonstration target plus hard-coded "
                    "datasets/v1_tolerance_summary_v2.csv"
                    if config.tolerance_mode == TOLERANCE_MODE_DATASET_ORACLE
                    else "OpenPI 7D actions + 6D rotation_tolerance output"
                )
            ),
            "environment_randomization": (
                config.environment_randomization_probability > 0.0
            ),
            "environment_randomization_probability": (
                config.environment_randomization_probability
            ),
            "environment_randomization_episode_plan": list(environment_plan),
            "environment_randomized_episodes_requested": int(sum(
                environment_plan
            )),
            "placement_difficulty": config.placement_difficulty,
            "workspace_boundary": config.workspace_boundary,
            "workspace_boundary_distance_factor_min": (
                config.workspace_boundary_distance_min
                if config.workspace_boundary else None
            ),
            "workspace_boundary_distance_factor_max": (
                config.workspace_boundary_distance_max
                if config.workspace_boundary else None
            ),
            "workspace_boundary_factor_distance_scale_m": (
                0.50 if config.workspace_boundary else None
            ),
            "workspace_boundary_strategy_episode_plan": (
                list(workspace_plan) if config.workspace_boundary else None
            ),
            "workspace_boundary_strategy_counts": (
                {
                    mode: workspace_plan.count(mode)
                    for mode in WORKSPACE_BOUNDARY_MODES
                    if mode in workspace_plan
                }
                if config.workspace_boundary else None
            ),
            "reachability_map_dir": (
                None
                if config.reachability_map_dir is None
                else str(config.reachability_map_dir)
            ),
            "run_name": config.run_name,
            "model_name": _model_directory_label(config, policy_metadata),
            "seed_groups": {
                "environment": ["table", "front_camera", "lighting"],
                "task": ["object_layout"],
            },
        },
        "policy_server": {
            "host": config.host,
            "port": config.port,
            "metadata": _jsonable(policy_metadata),
        },
        "completed_episodes": len(results),
        "summary": aggregate.as_dict(),
        "episodes": [
            {
                **result.as_dict(),
                "path": f"episode_{index:04d}_seed_{result.seed}",
                "environment_randomization": environment_plan[index],
            }
            for index, result in enumerate(results)
        ],
    }
    if error is not None:
        payload["error"] = error
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    temporary.replace(path)


def run_openpi_benchmark(config: OpenPIBenchmarkConfig) -> Path:
    task = resolve_dataset_task(config.dataset_root, config.task, config.robot_uid)
    seeds = benchmark_seed_plan(config.count, config.seed)
    environment_plan = _environment_randomization_plan(
        task.task_id, seeds, config.environment_randomization_probability,
    )
    policy = OpenPIWebsocketPolicy(OpenPIClientConfig(
        host=config.host, port=config.port, api_key=config.api_key,
    ))
    policy_metadata = policy.server_metadata
    run_root = (
        config.output_root
        / task.directory.name
        / _timestamped_model_directory(config, policy_metadata)
    )
    run_root.mkdir(parents=True, exist_ok=False)
    summary_path = run_root / "benchmark.json"
    results: list[BenchmarkEpisodeResult] = []
    _write_summary(
        summary_path, config=config, task=task, seeds=seeds,
        policy_metadata=policy_metadata, results=results, status="in_progress",
    )

    maximum_steps = config.maximum_steps or task.maximum_steps
    try:
        for index, seed in enumerate(seeds):
            instruction = task.instructions[index % len(task.instructions)]
            episode_dir = run_root / f"episode_{index:04d}_seed_{seed}"
            environment = MuJoCoVLABenchmarkEnvironment(BenchmarkEnvironmentConfig(
                output_dir=episode_dir,
                robot_uid=config.robot_uid,
                width=task.width,
                height=task.height,
                tolerance_mode=config.tolerance_mode,
                dataset_task_id=task.directory.name,
                scene_variant_index=config.scene_variant_index,
                environment_randomization=environment_plan[index],
                orientation_reference_mode=config.orientation_reference_mode,
                placement_difficulty=config.placement_difficulty,
                reachability_map_dir=config.reachability_map_dir,
                workspace_boundary_mode=(
                    _workspace_boundary_mode(task, index)
                    if config.workspace_boundary else None
                ),
                workspace_boundary_distance_min=(
                    config.workspace_boundary_distance_min
                ),
                workspace_boundary_distance_max=(
                    config.workspace_boundary_distance_max
                ),
            ))
            try:
                result = BenchmarkRunner(environment).run_case(
                    policy,
                    BenchmarkCase(
                        scene=task.scene,
                        task=task.task_id,
                        seed=seed,
                        instruction=instruction,
                        maximum_steps=maximum_steps,
                        replan_steps=config.replan_steps,
                    ),
                )
            finally:
                environment.close()
            results.append(result)
            _write_summary(
                summary_path, config=config, task=task, seeds=seeds,
                policy_metadata=policy_metadata, results=results, status="in_progress",
            )
            print(
                f"[{index + 1}/{config.count}] seed={seed} "
                f"success={result.success} reason={result.termination_reason} "
                f"frames={result.frames}",
                flush=True,
            )
    except Exception as error:
        _write_summary(
            summary_path, config=config, task=task, seeds=seeds,
            policy_metadata=policy_metadata, results=results, status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        policy.close()

    _write_summary(
        summary_path, config=config, task=task, seeds=seeds,
        policy_metadata=policy_metadata, results=results, status="complete",
    )
    return summary_path


def run_openpi_training_suite(config: OpenPIBenchmarkConfig) -> Path:
    """Run every dataset-backed training task for the selected robot."""
    tasks = dataset_tasks_for_robot(config.dataset_root, config.robot_uid)
    if not tasks:
        raise ValueError(
            f"No dataset tasks found for robot {config.robot_uid!r} under "
            f"{config.dataset_root}"
        )
    if config.workspace_boundary:
        available = {task.directory.name: task for task in tasks}
        missing = sorted(WORKSPACE_BOUNDARY_TASKS - available.keys())
        if missing:
            raise ValueError(
                "Workspace-boundary suite is missing short training tasks: "
                + ", ".join(missing),
            )
        tasks = tuple(
            task for task in tasks
            if task.directory.name in WORKSPACE_BOUNDARY_TASKS
        )

    policy_metadata = _read_policy_server_metadata(config)
    label = _safe_directory_label(config.run_name or "openpi") or "openpi"
    suite_root = (
        config.output_root
        / (
            f"{label}_{config.robot_uid}_workspace_boundary"
            if config.workspace_boundary
            else f"{label}_{config.robot_uid}_training_tasks"
        )
        / _timestamped_model_directory(config, policy_metadata)
    )
    suite_root.mkdir(parents=True, exist_ok=False)
    suite_path = suite_root / "suite.json"
    entries: list[dict[str, Any]] = []

    def write_suite(status: str, error: str | None = None) -> None:
        completed_episodes = sum(int(entry["completed_episodes"]) for entry in entries)
        successful_episodes = sum(int(entry["successful_episodes"]) for entry in entries)
        payload = {
            "benchmark_version": "1.0",
            "series": (
                "workspace_boundary"
                if config.workspace_boundary else "training_tasks"
            ),
            "status": status,
            "run_name": config.run_name,
            "model_name": _model_directory_label(config, policy_metadata),
            "policy_server_metadata": _jsonable(policy_metadata),
            "robot": config.robot_uid,
            "tolerance_mode": config.tolerance_mode,
            "environment_randomization_probability": (
                config.environment_randomization_probability
            ),
            "workspace_boundary": config.workspace_boundary,
            "workspace_boundary_distance_factor_min": (
                config.workspace_boundary_distance_min
                if config.workspace_boundary else None
            ),
            "workspace_boundary_distance_factor_max": (
                config.workspace_boundary_distance_max
                if config.workspace_boundary else None
            ),
            "workspace_boundary_strategy_schedule": (
                "object_pre_boundary_only"
                if config.workspace_boundary else None
            ),
            "requested_tasks": len(tasks),
            "episodes_per_task": config.count,
            "requested_episodes": len(tasks) * config.count,
            "completed_tasks": len(entries),
            "completed_episodes": completed_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": (
                successful_episodes / completed_episodes
                if completed_episodes else 0.0
            ),
            "tasks": entries,
        }
        if error is not None:
            payload["error"] = error
        temporary = suite_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        temporary.replace(suite_path)

    write_suite("in_progress")
    try:
        for index, task in enumerate(tasks):
            print(
                f"[task {index + 1}/{len(tasks)}] {task.directory.name}: "
                f"{config.count} rollout(s), tolerance={config.tolerance_mode}",
                flush=True,
            )
            summary_path = run_openpi_benchmark(replace(
                config,
                task=task.directory.name,
                output_root=suite_root / "tasks",
            ))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            entries.append({
                "task_id": task.task_id,
                "task_name": task.task_name,
                "dataset_directory": str(task.directory),
                "summary_path": str(summary_path.relative_to(suite_root)),
                "completed_episodes": int(summary["completed_episodes"]),
                "successful_episodes": int(summary["summary"]["successes"]),
                "success_rate": float(summary["summary"]["success_rate"]),
            })
            write_suite("in_progress")
    except Exception as error:
        write_suite("failed", f"{type(error).__name__}: {error}")
        raise

    write_suite("complete")
    return suite_path


def run_openpi_workspace_boundary_suite(
    config: OpenPIBenchmarkConfig,
) -> Path:
    """Run all 15 Geometric Basis Tasks on the robot's IK boundary."""
    return run_openpi_training_suite(replace(
        config,
        workspace_boundary=True,
        placement_difficulty=0.0,
    ))


def run_openpi_unseen_object_suite(config: OpenPIBenchmarkConfig) -> Path:
    """Run the benchmark-only held-out-object series without demonstrations."""
    tasks = benchmark_only_tasks_for_robot(config.dataset_root, config.robot_uid)
    if not tasks:
        raise ValueError(
            f"No unseen-object tasks available for {config.robot_uid!r} under "
            f"{config.dataset_root}",
        )

    policy_metadata = _read_policy_server_metadata(config)
    label = _safe_directory_label(config.run_name or "openpi") or "openpi"
    suite_root = (
        config.output_root
        / f"{label}_{config.robot_uid}_unseen_objects"
        / _timestamped_model_directory(config, policy_metadata)
    )
    suite_root.mkdir(parents=True, exist_ok=False)
    suite_path = suite_root / "suite.json"
    entries: list[dict[str, Any]] = []

    def write_suite(status: str, error: str | None = None) -> None:
        completed = sum(int(item["completed_episodes"]) for item in entries)
        successes = sum(int(item["successful_episodes"]) for item in entries)
        payload = {
            "benchmark_version": "1.0",
            "series": "unseen_objects",
            "status": status,
            "run_name": config.run_name,
            "model_name": _model_directory_label(config, policy_metadata),
            "policy_server_metadata": _jsonable(policy_metadata),
            "robot": config.robot_uid,
            "tolerance_mode": config.tolerance_mode,
            "environment_randomization_probability": (
                config.environment_randomization_probability
            ),
            "requested_tasks": len(tasks),
            "episodes_per_task": config.count,
            "requested_episodes": len(tasks) * config.count,
            "completed_tasks": len(entries),
            "completed_episodes": completed,
            "successful_episodes": successes,
            "success_rate": successes / completed if completed else 0.0,
            "tasks": entries,
        }
        if error is not None:
            payload["error"] = error
        temporary = suite_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        temporary.replace(suite_path)

    write_suite("in_progress")
    try:
        for index, task in enumerate(tasks):
            print(
                f"[unseen task {index + 1}/{len(tasks)}] {task.directory.name}: "
                f"{config.count} rollout(s), tolerance={config.tolerance_mode}",
                flush=True,
            )
            summary_path = run_openpi_benchmark(replace(
                config,
                task=task.directory.name,
                output_root=suite_root / "tasks",
            ))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            entries.append({
                "task_id": task.task_id,
                "task_name": task.task_name,
                "benchmark_task": task.directory.name,
                "reference_task_directory": task.reference_task_directory,
                "summary_path": str(summary_path.relative_to(suite_root)),
                "completed_episodes": int(summary["completed_episodes"]),
                "successful_episodes": int(summary["summary"]["successes"]),
                "success_rate": float(summary["summary"]["success_rate"]),
            })
            write_suite("in_progress")
    except Exception as error:
        write_suite("failed", f"{type(error).__name__}: {error}")
        raise

    write_suite("complete")
    return suite_path
