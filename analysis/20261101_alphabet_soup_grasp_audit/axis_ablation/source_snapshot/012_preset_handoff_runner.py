"""Preset-trajectory warm start followed by closed-loop OpenPI inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from ..policies.base import PolicyStep
from ..policies.openpi.client import OpenPIClientConfig, OpenPIWebsocketPolicy
from ..simulation.robot import ee_pose
from ..simulation.robots import get_robot_spec
from ..tasks.planning.presets import PresetTaskExecutor
from ..tolerance.oracle import TOLERANCE_MODE_DATASET_ORACLE
from ..tolerance.state import ORIENTATION_REFERENCE_STAGE_RELATIVE
from .catalog import benchmark_seed_plan, resolve_dataset_task
from .environment import BenchmarkEnvironmentConfig, MuJoCoVLABenchmarkEnvironment
from .metrics import BenchmarkEpisodeResult, aggregate_episode_results
from .runner import BenchmarkCase, _policy_steps


@dataclass(frozen=True)
class PresetHandoffBenchmarkConfig:
    task: str
    count: int
    handoff_before_grasp: int = 10
    robot_uid: str = "panda"
    dataset_root: Path = Path("datasets/v1")
    output_root: Path = Path("benchmark_results")
    seed: int = -1
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    maximum_steps: int | None = None
    replan_steps: int = 10
    orientation_reference_mode: str = ORIENTATION_REFERENCE_STAGE_RELATIVE
    run_name: str = "preset_handoff"
    model_name: str | None = None

    def __post_init__(self) -> None:
        get_robot_spec(self.robot_uid)
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.handoff_before_grasp < 0:
            raise ValueError("handoff_before_grasp must be non-negative")
        if self.maximum_steps is not None and self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        if self.orientation_reference_mode != ORIENTATION_REFERENCE_STAGE_RELATIVE:
            raise ValueError(
                "Stage-relative release is the sole maintained reference mode",
            )


@dataclass(frozen=True)
class HandoffRecord:
    nominal_grasp_frame: int
    requested_frames_before_grasp: int
    handoff_frame: int
    preset_frames_executed: int
    vla_frames_executed: int
    vla_started: bool


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_label(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return result.strip("._-")[:120] or "openpi"


def _model_label(
    config: PresetHandoffBenchmarkConfig,
    policy_metadata: dict[str, Any],
) -> str:
    if config.model_name:
        return _safe_label(config.model_name)
    checkpoint = policy_metadata.get("checkpoint_path")
    if checkpoint:
        parent = Path(str(checkpoint)).parent.name
        step = Path(str(checkpoint)).name
        return _safe_label(f"{parent}_{step}")
    return _safe_label(policy_metadata.get("policy_config", "openpi"))


def _preset_actions(
    environment: MuJoCoVLABenchmarkEnvironment,
    task_id: str,
    scene: str,
) -> tuple[PolicyStep, ...]:
    assert environment.model is not None and environment.data is not None
    position, rotation = ee_pose(environment.data, environment.handles)
    executor = PresetTaskExecutor(environment.model, environment.handles)
    executor.start(task_id, scene, environment.data, position, rotation,
                   episode_seed=environment.case.seed)
    result: list[PolicyStep] = []
    for action in executor.actions:
        result.append(PolicyStep(np.asarray(action.value, dtype=float)))
    return tuple(result)


def _first_grasp_frame(actions: tuple[PolicyStep, ...]) -> int:
    for index, step in enumerate(actions):
        if float(step.action[6]) >= 0.0:
            return index
    raise ValueError("preset task contains no gripper-close transition")


def _annotate_rollout(
    episode_dir: Path,
    handoff: HandoffRecord,
) -> None:
    path = episode_dir / "rollout.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["episode"]["preset_handoff"] = asdict(handoff)
    for frame in payload.get("rollout", []):
        frame["action_source"] = (
            "preset_ipopt"
            if int(frame["index"]) < handoff.handoff_frame
            else "vla_closed_loop"
        )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_episode(
    policy: OpenPIWebsocketPolicy,
    environment: MuJoCoVLABenchmarkEnvironment,
    case: BenchmarkCase,
    *,
    handoff_before_grasp: int,
) -> tuple[BenchmarkEpisodeResult, HandoffRecord]:
    observation = environment.reset(case)
    preset_actions = _preset_actions(environment, case.task, case.scene)
    nominal_grasp_frame = _first_grasp_frame(preset_actions)
    handoff_frame = max(0, nominal_grasp_frame - handoff_before_grasp)
    handoff_frame = min(handoff_frame, case.maximum_steps)

    terminated = False
    for start in range(0, handoff_frame, case.replan_steps):
        block = preset_actions[start:min(
            start + case.replan_steps, handoff_frame,
        )]
        for policy_step in block:
            observation, terminated = environment.step(policy_step)
            if terminated:
                break
        if terminated:
            break

    vla_started = not terminated and handoff_frame < case.maximum_steps
    if vla_started:
        # Preserve the environment, tolerance, optimizer and evaluator state
        # across the boundary; reset only the VLA session.
        policy.reset()
        action_queue: list[PolicyStep] = []
        for _ in range(handoff_frame, case.maximum_steps):
            if not action_queue:
                started = time.perf_counter()
                inferred = _policy_steps(policy.infer(observation))
                environment.record_policy_latency(
                    (time.perf_counter() - started) * 1000.0,
                )
                action_queue.extend(inferred[:min(
                    case.replan_steps, len(inferred),
                )])
            observation, terminated = environment.step(action_queue.pop(0))
            if terminated:
                break
        if not terminated:
            environment.mark_timeout()
    elif not terminated:
        environment.mark_timeout()

    result = environment.result()
    preset_frames = min(handoff_frame, result.frames)
    handoff = HandoffRecord(
        nominal_grasp_frame=nominal_grasp_frame,
        requested_frames_before_grasp=handoff_before_grasp,
        handoff_frame=handoff_frame,
        preset_frames_executed=preset_frames,
        vla_frames_executed=max(0, result.frames - preset_frames),
        vla_started=vla_started,
    )
    _annotate_rollout(environment.config.output_dir, handoff)
    return result, handoff


def _write_summary(
    path: Path,
    *,
    config: PresetHandoffBenchmarkConfig,
    task: Any,
    seeds: tuple[int, ...],
    policy_metadata: dict[str, Any],
    results: list[BenchmarkEpisodeResult],
    handoffs: list[HandoffRecord],
    status: str,
    error: str | None = None,
) -> None:
    aggregate = aggregate_episode_results(results)
    spec = get_robot_spec(config.robot_uid)
    payload: dict[str, Any] = {
        "benchmark_version": "1.0",
        "status": status,
        "experiment": "preset stage-relative release warm start then closed-loop VLA",
        "task": {
            "scene": task.scene,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "dataset_directory": str(task.directory),
            "metadata_robot": task.robot_uid,
        },
        "configuration": {
            "robot": {
                "uid": spec.uid,
                "name": spec.label,
                "arm_dof": spec.arm_dof,
            },
            "requested_episodes": config.count,
            "initial_seed": config.seed,
            "seeds": list(seeds),
            "frequency_hz": 10.0,
            "maximum_steps": config.maximum_steps or task.maximum_steps,
            "replan_steps": config.replan_steps,
            "controller": "TaskTol stage-relative IPOPT",
            "optimizer_backend": "ipopt",
            "tolerance_mode": TOLERANCE_MODE_DATASET_ORACLE,
            "tolerance_release": True,
            "orientation_reference_mode": config.orientation_reference_mode,
            "environment_randomization": False,
            "preset_handoff_before_grasp": config.handoff_before_grasp,
            "preset_action_source": "seeded PresetTaskExecutor",
            "handoff_policy_reset": True,
            "run_name": config.run_name,
            "model_name": _model_label(config, policy_metadata),
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
                "environment_randomization": False,
                "preset_handoff": asdict(handoffs[index]),
            }
            for index, result in enumerate(results)
        ],
    }
    if error is not None:
        payload["error"] = error
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_preset_handoff_benchmark(
    config: PresetHandoffBenchmarkConfig,
) -> Path:
    task = resolve_dataset_task(config.dataset_root, config.task, config.robot_uid)
    seeds = benchmark_seed_plan(config.count, config.seed)
    policy = OpenPIWebsocketPolicy(OpenPIClientConfig(
        host=config.host, port=config.port, api_key=config.api_key,
    ))
    policy_metadata = policy.server_metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory_name = (
        f"{_safe_label(config.run_name)}_{_model_label(config, policy_metadata)}_"
        f"tolerance-dataset_handoff-pregrasp{config.handoff_before_grasp}_"
        f"{timestamp}"
    )
    run_root = config.output_root / task.directory.name / directory_name
    run_root.mkdir(parents=True, exist_ok=False)
    summary_path = run_root / "benchmark.json"
    results: list[BenchmarkEpisodeResult] = []
    handoffs: list[HandoffRecord] = []
    _write_summary(
        summary_path,
        config=config,
        task=task,
        seeds=seeds,
        policy_metadata=policy_metadata,
        results=results,
        handoffs=handoffs,
        status="in_progress",
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
                tolerance_mode=TOLERANCE_MODE_DATASET_ORACLE,
                dataset_task_id=task.directory.name,
                environment_randomization=False,
                orientation_reference_mode=config.orientation_reference_mode,
            ))
            case = BenchmarkCase(
                scene=task.scene,
                task=task.task_id,
                seed=seed,
                instruction=instruction,
                maximum_steps=maximum_steps,
                replan_steps=config.replan_steps,
            )
            try:
                result, handoff = _run_episode(
                    policy,
                    environment,
                    case,
                    handoff_before_grasp=config.handoff_before_grasp,
                )
            finally:
                environment.close()
            results.append(result)
            handoffs.append(handoff)
            _write_summary(
                summary_path,
                config=config,
                task=task,
                seeds=seeds,
                policy_metadata=policy_metadata,
                results=results,
                handoffs=handoffs,
                status="in_progress",
            )
            print(
                f"[{index + 1}/{config.count}] seed={seed} "
                f"handoff={handoff.handoff_frame} "
                f"grasp={handoff.nominal_grasp_frame} "
                f"success={result.success} reason={result.termination_reason} "
                f"frames={result.frames}",
                flush=True,
            )
    except Exception as error:
        _write_summary(
            summary_path,
            config=config,
            task=task,
            seeds=seeds,
            policy_metadata=policy_metadata,
            results=results,
            handoffs=handoffs,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        policy.close()

    _write_summary(
        summary_path,
        config=config,
        task=task,
        seeds=seeds,
        policy_metadata=policy_metadata,
        results=results,
        handoffs=handoffs,
        status="complete",
    )
    return summary_path


__all__ = [
    "PresetHandoffBenchmarkConfig",
    "run_preset_handoff_benchmark",
]
