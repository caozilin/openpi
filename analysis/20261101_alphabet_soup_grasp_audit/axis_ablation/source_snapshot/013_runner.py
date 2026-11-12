"""Policy-agnostic benchmark orchestration contracts."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

import numpy as np

from ..policies.base import ActionPolicy, PolicyStep
from .metrics import BenchmarkEpisodeResult


@dataclass(frozen=True)
class BenchmarkCase:
    scene: str
    task: str
    seed: int
    instruction: str
    maximum_steps: int = 600
    replan_steps: int = 10

    def __post_init__(self) -> None:
        if self.maximum_steps <= 0 or self.replan_steps <= 0:
            raise ValueError("maximum_steps and replan_steps must be positive")


class BenchmarkEnvironment(Protocol):
    """Simulation/real-robot boundary required by the generic runner."""

    def reset(self, case: BenchmarkCase) -> dict[str, Any]: ...

    def step(self, action: PolicyStep | np.ndarray) -> tuple[dict[str, Any], bool]: ...

    def result(self) -> BenchmarkEpisodeResult: ...

    def record_policy_latency(self, latency_ms: float) -> None: ...

    def mark_timeout(self) -> None: ...


class BenchmarkRunner:
    """Run one policy against cases without owning task success semantics."""

    def __init__(self, environment: BenchmarkEnvironment) -> None:
        self.environment = environment

    def run_case(
        self,
        policy: ActionPolicy,
        case: BenchmarkCase,
    ) -> BenchmarkEpisodeResult:
        observation = self.environment.reset(case)
        policy.reset()
        action_queue: list[PolicyStep] = []
        for _ in range(case.maximum_steps):
            if not action_queue:
                started = time.perf_counter()
                inferred = _policy_steps(policy.infer(observation))
                self.environment.record_policy_latency(
                    (time.perf_counter() - started) * 1000.0,
                )
                count = min(case.replan_steps, len(inferred))
                action_queue.extend(inferred[:count])
                set_action_block = getattr(
                    self.environment, "set_action_block", None,
                )
                if set_action_block is not None:
                    set_action_block(tuple(action_queue))
            observation, terminated = self.environment.step(action_queue.pop(0))
            if terminated:
                return self.environment.result()
        self.environment.mark_timeout()
        return self.environment.result()


def _matrix(value: Any, name: str, *, columns: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != columns:
        raise ValueError(f"Policy {name} must have non-empty shape [horizon, {columns}]")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Policy {name} must be finite")
    return result


def _policy_steps(output: np.ndarray | dict[str, Any]) -> list[PolicyStep]:
    """Keep every action row aligned with its optional TaskTol prediction."""
    if not isinstance(output, dict):
        actions = np.asarray(output, dtype=float)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7:
            raise ValueError("Policy actions must have non-empty shape [horizon, 7]")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Policy actions must be finite")
        return [PolicyStep(row.copy()) for row in actions]

    if "actions" not in output:
        raise KeyError("Policy output is missing 'actions'")
    actions = _matrix(output["actions"], "actions", columns=7)
    for key in ("stage_target_pose", "tolerance_frame_code", "tolerance_frame"):
        if key in output:
            raise KeyError(
                f"Policy output contains obsolete {key!r}; TaskTol requires only "
                "7D actions and 6D rotation_tolerance"
            )
    if "rotation_tolerance" not in output:
        return [PolicyStep(row.copy()) for row in actions]

    bounds = _matrix(output["rotation_tolerance"], "rotation_tolerance", columns=6)
    if bounds.shape[0] != actions.shape[0]:
        raise ValueError("TaskTol predictions must use the same horizon as actions")
    return [
        PolicyStep(
            actions[index].copy(),
            rotation_tolerance_rad=bounds[index].copy(),
        )
        for index in range(actions.shape[0])
    ]
