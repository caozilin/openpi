"""Paired preset rollouts followed by read-only offline OpenPI inference.

The model never controls either rollout.  Both conditions execute the same
seeded preset action sequence; only the controller's tolerance release differs.
After both videos are closed, their decoded frames and aligned robot states are
sent to OpenPI at the configured action-chunk cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Iterable, Iterator

import imageio_ffmpeg
import numpy as np

from ..policies.openpi.client import (
    OpenPIClientConfig,
    OpenPIWebsocketPolicy,
    franka_mujoco_observation,
)
from ..policies.base import PolicyStep
from ..simulation.robot import ee_pose
from ..simulation.robots import get_robot_spec
from ..tasks.planning.presets import PresetTaskExecutor
from ..tolerance.oracle import (
    TOLERANCE_MODE_DATASET_ORACLE,
    TOLERANCE_MODE_OFF,
)
from .catalog import benchmark_seed_plan, resolve_dataset_task
from .environment import (
    BenchmarkEnvironmentConfig,
    MuJoCoVLABenchmarkEnvironment,
)
from .runner import BenchmarkCase


_CONDITIONS = (
    ("no_tolerance", TOLERANCE_MODE_OFF),
    ("with_tolerance", TOLERANCE_MODE_DATASET_ORACLE),
)


@dataclass(frozen=True)
class OfflineObservationBenchmarkConfig:
    tasks: tuple[str, ...]
    seeds_per_task: int
    robot_uid: str = "panda"
    dataset_root: Path = Path("datasets/v1")
    output_root: Path = Path("offline_observation_results")
    initial_seed: int = -1
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    replan_steps: int = 10
    maximum_steps: int | None = None
    run_name: str = "preset_observation_pair"

    def __post_init__(self) -> None:
        if not self.tasks or any(not task.strip() for task in self.tasks):
            raise ValueError("at least one non-empty task is required")
        if self.seeds_per_task <= 0:
            raise ValueError("seeds_per_task must be positive")
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        if self.maximum_steps is not None and self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        get_robot_spec(self.robot_uid)


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _video_frames(path: Path) -> Iterator[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        for encoded in reader:
            yield np.frombuffer(encoded, dtype=np.uint8).reshape(
                height, width, 3,
            )
    finally:
        reader.close()


def _episode_state(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ee_state = frame["ee_state"]
    gripper_state = frame["gripper_state"]
    fingers = gripper_state.get("finger_positions_m")
    if fingers is None:
        aperture = float(gripper_state["aperture_m"])
        fingers = (0.5 * aperture, 0.5 * aperture)
    return (
        np.asarray(ee_state["position_m"], dtype=np.float32),
        np.asarray(ee_state["rotation_vector_base_rad"], dtype=np.float32),
        np.asarray(fingers, dtype=np.float32),
    )


def _offline_infer_episode(
    policy: OpenPIWebsocketPolicy,
    episode_dir: Path,
    *,
    replan_steps: int,
) -> Path:
    rollout_path = episode_dir / "rollout.json"
    payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    frames = payload["rollout"]
    instruction = str(payload["episode"]["instruction"])
    policy.reset()
    records: list[dict[str, Any]] = []
    decoded_count = 0
    main_frames = _video_frames(episode_dir / "main_rgb.mp4")
    wrist_frames = _video_frames(episode_dir / "wrist_rgb.mp4")
    for index, (main_rgb, wrist_rgb) in enumerate(
        zip(main_frames, wrist_frames, strict=True),
    ):
        decoded_count += 1
        if index >= len(frames):
            raise ValueError(f"video contains more frames than {rollout_path}")
        if index % replan_steps:
            continue
        position, rotation, fingers = _episode_state(frames[index])
        observation = franka_mujoco_observation(
            position,
            rotation,
            fingers,
            main_rgb,
            wrist_rgb,
            instruction,
        )
        started = time.perf_counter()
        response = policy.infer(observation)
        latency_ms = (time.perf_counter() - started) * 1000.0
        records.append({
            "inference_index": len(records),
            "source_frame_index": index,
            "source_time_s": float(frames[index]["time_s"]),
            "nominal_execution_chunk": {
                "start_frame_index": index,
                "end_frame_index_exclusive": min(index + replan_steps, len(frames)),
            },
            "observation_state": _jsonable(observation["observation/state"]),
            "latency_ms": latency_ms,
            "response": _jsonable(response),
        })
    if decoded_count != len(frames):
        raise ValueError(
            f"video/rollout frame mismatch in {episode_dir}: "
            f"decoded {decoded_count}, JSON has {len(frames)}",
        )
    output = episode_dir / "offline_inference.json"
    _write_json(output, {
        "schema_version": "1.0",
        "observation_source": {
            "images": "decoded H.264 main_rgb.mp4 and wrist_rgb.mp4",
            "state": "aligned rollout.json frame",
            "model_outputs_do_not_control_rollout": True,
        },
        "replan_steps": replan_steps,
        "source_frame_count": len(frames),
        "inference_count": len(records),
        "inferences": records,
    })
    return output


def _preset_actions(
    environment: MuJoCoVLABenchmarkEnvironment,
    task_id: str,
    scene: str,
) -> tuple[PolicyStep, ...]:
    assert environment.model is not None and environment.data is not None
    position, rotation = ee_pose(environment.data, environment.handles)
    executor = PresetTaskExecutor(environment.model, environment.handles)
    executor.start(
        task_id, scene, environment.data, position, rotation,
        episode_seed=environment.case.seed,
    )
    result: list[PolicyStep] = []
    for action in executor.actions:
        result.append(PolicyStep(np.asarray(action.value, dtype=float)))
    return tuple(result)


def _generate_episode(
    *,
    episode_dir: Path,
    task: Any,
    seed: int,
    instruction: str,
    tolerance_mode: str,
    config: OfflineObservationBenchmarkConfig,
) -> tuple[Path, tuple[np.ndarray, ...]]:
    maximum_steps = config.maximum_steps or task.maximum_steps
    case = BenchmarkCase(
        scene=task.scene,
        task=task.task_id,
        seed=seed,
        instruction=instruction,
        maximum_steps=maximum_steps,
        replan_steps=config.replan_steps,
    )
    environment = MuJoCoVLABenchmarkEnvironment(BenchmarkEnvironmentConfig(
        output_dir=episode_dir,
        robot_uid=config.robot_uid,
        width=task.width,
        height=task.height,
        tolerance_mode=tolerance_mode,
        dataset_task_id=task.directory.name,
        environment_randomization=False,
        terminate_on_evaluation=False,
    ))
    try:
        environment.reset(case)
        actions = _preset_actions(environment, task.task_id, task.scene)
        actions = actions[:maximum_steps]
        for start in range(0, len(actions), config.replan_steps):
            block = actions[start:start + config.replan_steps]
            for policy_step in block:
                environment.step(policy_step)
        environment.result()
    finally:
        environment.close()
    return (
        episode_dir / "rollout.json",
        tuple(step.action.copy() for step in actions),
    )


def _action_response(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(record["response"]["actions"], dtype=float)


def _compare_inferences(
    first: Iterable[dict[str, Any]],
    second: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left, right in zip(first, second, strict=True):
        if left["source_frame_index"] != right["source_frame_index"]:
            raise ValueError("paired inference frame indices are not aligned")
        left_actions = _action_response(left)
        right_actions = _action_response(right)
        rows = min(len(left_actions), len(right_actions))
        columns = min(left_actions.shape[1], right_actions.shape[1], 7)
        delta = right_actions[:rows, :columns] - left_actions[:rows, :columns]
        result.append({
            "source_frame_index": left["source_frame_index"],
            "compared_action_rows": rows,
            "compared_action_columns": columns,
            "with_minus_without_tolerance": delta.tolist(),
            "per_row_l2": np.linalg.norm(delta, axis=1).tolist(),
            "mean_l2": float(np.mean(np.linalg.norm(delta, axis=1))),
            "maximum_l2": float(np.max(np.linalg.norm(delta, axis=1))),
        })
    return result


def run_offline_observation_benchmark(
    config: OfflineObservationBenchmarkConfig,
) -> Path:
    # Fail before expensive simulation if the requested server is unavailable.
    readiness = OpenPIWebsocketPolicy(OpenPIClientConfig(
        host=config.host, port=config.port, api_key=config.api_key,
    ))
    try:
        server_metadata = readiness.server_metadata
    finally:
        readiness.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = config.output_root / f"{config.run_name}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    summary_path = run_root / "paired_offline_benchmark.json"
    task_summaries: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "in_progress",
        "experiment": (
            "same seeded preset actions; compare observations and OpenPI outputs "
            "with versus without controller tolerance release"
        ),
        "configuration": {
            **_jsonable(config.__dict__),
            "environment_randomization": False,
            "conditions": [name for name, _ in _CONDITIONS],
        },
        "policy_server": {
            "host": config.host,
            "port": config.port,
            "metadata": _jsonable(server_metadata),
        },
        "tasks": task_summaries,
    }
    _write_json(summary_path, summary)
    try:
        for requested_task in config.tasks:
            task = resolve_dataset_task(
                config.dataset_root, requested_task, config.robot_uid,
            )
            seeds = benchmark_seed_plan(
                config.seeds_per_task, config.initial_seed,
            )
            task_dir = run_root / task.directory.name
            task_dir.mkdir()
            task_record: dict[str, Any] = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "dataset_directory": str(task.directory),
                "seeds": list(seeds),
                "pairs": [],
            }
            task_summaries.append(task_record)
            for seed_index, seed in enumerate(seeds):
                instruction = task.instructions[seed_index % len(task.instructions)]
                pair_dir = task_dir / f"seed_{seed}"
                pair_dir.mkdir()
                action_sequences: dict[str, tuple[np.ndarray, ...]] = {}
                condition_records: dict[str, dict[str, Any]] = {}
                for condition_name, tolerance_mode in _CONDITIONS:
                    episode_dir = pair_dir / condition_name
                    rollout_path, actions = _generate_episode(
                        episode_dir=episode_dir,
                        task=task,
                        seed=seed,
                        instruction=instruction,
                        tolerance_mode=tolerance_mode,
                        config=config,
                    )
                    action_sequences[condition_name] = actions
                    condition_records[condition_name] = {
                        "tolerance_mode": tolerance_mode,
                        "rollout": str(rollout_path.relative_to(run_root)),
                        "main_video": str(
                            (episode_dir / "main_rgb.mp4").relative_to(run_root)
                        ),
                        "wrist_video": str(
                            (episode_dir / "wrist_rgb.mp4").relative_to(run_root)
                        ),
                    }
                left = action_sequences["no_tolerance"]
                right = action_sequences["with_tolerance"]
                if len(left) != len(right) or any(
                    not np.array_equal(a, b) for a, b in zip(left, right)
                ):
                    raise RuntimeError(
                        f"preset action mismatch for {task.directory.name} seed {seed}",
                    )

                policy = OpenPIWebsocketPolicy(OpenPIClientConfig(
                    host=config.host, port=config.port, api_key=config.api_key,
                ))
                try:
                    for condition_name, _ in _CONDITIONS:
                        inference_path = _offline_infer_episode(
                            policy,
                            pair_dir / condition_name,
                            replan_steps=config.replan_steps,
                        )
                        condition_records[condition_name]["inference"] = str(
                            inference_path.relative_to(run_root)
                        )
                finally:
                    policy.close()
                inference_payloads = {
                    name: json.loads(
                        (pair_dir / name / "offline_inference.json").read_text(
                            encoding="utf-8",
                        )
                    )
                    for name, _ in _CONDITIONS
                }
                comparison = _compare_inferences(
                    inference_payloads["no_tolerance"]["inferences"],
                    inference_payloads["with_tolerance"]["inferences"],
                )
                comparison_path = pair_dir / "action_comparison.json"
                _write_json(comparison_path, {
                    "schema_version": "1.0",
                    "seed": seed,
                    "nominal_action_sequences_identical": True,
                    "replan_steps": config.replan_steps,
                    "comparisons": comparison,
                })
                task_record["pairs"].append({
                    "seed": seed,
                    "instruction": instruction,
                    "preset_action_count": len(left),
                    "nominal_action_sequences_identical": True,
                    "conditions": condition_records,
                    "comparison": str(comparison_path.relative_to(run_root)),
                })
                _write_json(summary_path, summary)
        summary["status"] = "complete"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        _write_json(summary_path, summary)
        raise
    _write_json(summary_path, summary)
    return summary_path
