"""Static IK probing and zero-based, unbounded configuration badness."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from ..control.task_space.kinematics import RobotKinematics
from ..geometry import rotation_vector
from ..simulation.robot import RobotHandles


@dataclass(frozen=True)
class BadnessSettings:
    position_tolerance_m: float = 0.002
    rotation_tolerance_rad: float = math.radians(1.0)
    characteristic_length_m: float = 0.20
    safe_minimum_singular_value: float = 0.035
    safe_self_collision_clearance_m: float = 0.020
    self_collision_weight: float = 1.0
    maximum_iterations: int = 120
    restart_count: int = 9
    damping: float = 0.025
    maximum_joint_step_rad: float = 0.18

    def __post_init__(self) -> None:
        positive = (
            self.position_tolerance_m,
            self.rotation_tolerance_rad,
            self.characteristic_length_m,
            self.safe_minimum_singular_value,
            self.safe_self_collision_clearance_m,
            self.maximum_iterations,
            self.restart_count,
            self.damping,
            self.maximum_joint_step_rad,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Reachability badness settings must be positive")


@dataclass(frozen=True)
class ReachabilityProbeResult:
    target_position: np.ndarray
    q: np.ndarray
    reachable: bool
    badness: float
    position_error_m: float
    rotation_error_rad: float
    minimum_singular_value: float
    manipulability: float
    joint_margin_fraction: float
    self_collision_clearance_m: float
    iterations: int


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0
    while index:
        fraction /= base
        index, digit = divmod(index, base)
        result += digit * fraction
    return result


def _segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Shortest distance between two 3-D line segments."""
    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    aa = float(first @ first)
    bb = float(first @ second)
    cc = float(second @ second)
    dd = float(first @ offset)
    ee = float(second @ offset)
    denominator = aa * cc - bb * bb
    if aa <= 1e-16 and cc <= 1e-16:
        return float(np.linalg.norm(offset))
    if aa <= 1e-16:
        first_fraction = 0.0
        second_fraction = float(np.clip(ee / cc, 0.0, 1.0))
    elif cc <= 1e-16:
        second_fraction = 0.0
        first_fraction = float(np.clip(-dd / aa, 0.0, 1.0))
    else:
        first_fraction = float(np.clip((bb * ee - cc * dd) / max(denominator, 1e-16), 0.0, 1.0))
        second_fraction = float(np.clip((bb * first_fraction + ee) / cc, 0.0, 1.0))
        first_fraction = float(np.clip((bb * second_fraction - dd) / aa, 0.0, 1.0))
    delta = offset + first_fraction * first - second_fraction * second
    return float(np.linalg.norm(delta))


class ReachabilityProbe:
    """Find a feasible IK solution for a fixed-orientation Cartesian target."""

    _HALTON_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        target_rotation: np.ndarray,
        settings: BadnessSettings = BadnessSettings(),
    ) -> None:
        self.model = model
        self.handles = handles
        self.target_rotation = np.asarray(target_rotation, dtype=float).copy()
        if self.target_rotation.shape != (3, 3):
            raise ValueError("Reachability target rotation must be 3x3")
        self.settings = settings
        self.kinematics = RobotKinematics(model, handles)
        self.lower = handles.joint_lower.copy()
        self.upper = handles.joint_upper.copy()
        self.home = handles.spec.home_q.copy()
        self._span = self.upper - self.lower

    def _restart_seeds(
        self,
        warm_start: np.ndarray | None,
        use_restarts: bool,
    ) -> tuple[np.ndarray, ...]:
        seeds: list[np.ndarray] = []
        if warm_start is not None:
            warm = np.clip(np.asarray(warm_start, dtype=float), self.lower, self.upper)
            if warm.shape == self.home.shape:
                seeds.append(warm.copy())
        if not seeds or use_restarts:
            seeds.append(self.home.copy())
        if not use_restarts:
            return tuple(seeds)
        for restart in range(1, self.settings.restart_count + 1):
            unit = np.array([
                _halton(restart, self._HALTON_BASES[index])
                for index in range(self.home.size)
            ])
            seeds.append(self.lower + unit * self._span)
        return tuple(seeds)

    def _errors(self, q: np.ndarray, target_position: np.ndarray):
        state = self.kinematics.evaluate(
            q, include_manipulability=True, include_link_points=True,
        )
        position_error = target_position - state.position
        rotation_error = rotation_vector(state.rotation.T @ self.target_rotation)
        return state, position_error, rotation_error

    def _solve_once(
        self,
        target_position: np.ndarray,
        seed: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        q = np.clip(seed.copy(), self.lower, self.upper)
        length = self.settings.characteristic_length_m
        damping_sq = self.settings.damping**2
        best_q = q.copy()
        best_error = math.inf
        for iteration in range(1, self.settings.maximum_iterations + 1):
            state, position_error, rotation_error = self._errors(q, target_position)
            scaled_error = np.concatenate((position_error, length * rotation_error))
            error_norm = float(np.linalg.norm(scaled_error))
            if error_norm < best_error:
                best_error = error_norm
                best_q = q.copy()
            if (
                np.linalg.norm(position_error) <= self.settings.position_tolerance_m
                and np.linalg.norm(rotation_error) <= self.settings.rotation_tolerance_rad
            ):
                return q, iteration
            scaled_jacobian = state.jacobian.copy()
            scaled_jacobian[3:] *= length
            system = scaled_jacobian @ scaled_jacobian.T + damping_sq * np.eye(6)
            step = scaled_jacobian.T @ np.linalg.solve(system, scaled_error)
            # Resolve the seventh degree of freedom gently toward Home without
            # compromising the Cartesian correction.
            pseudo = scaled_jacobian.T @ np.linalg.solve(system, scaled_jacobian)
            step += 0.025 * (np.eye(q.size) - pseudo) @ (self.home - q)
            step_norm = float(np.linalg.norm(step))
            if step_norm > self.settings.maximum_joint_step_rad:
                step *= self.settings.maximum_joint_step_rad / step_norm
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                candidate = np.clip(q + scale * step, self.lower, self.upper)
                _, candidate_position_error, candidate_rotation_error = self._errors(
                    candidate, target_position,
                )
                candidate_error = float(np.linalg.norm(np.concatenate((
                    candidate_position_error, length * candidate_rotation_error,
                ))))
                if candidate_error < error_norm:
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                break
        return best_q, iteration

    def _self_collision_clearance(self, link_points: np.ndarray) -> float:
        segments = [
            (index, link_points[index], link_points[index + 1])
            for index in range(link_points.shape[0] - 1)
            if np.linalg.norm(link_points[index + 1] - link_points[index]) > 1e-9
        ]
        distances = []
        for first_order, (_, first_start, first_end) in enumerate(segments):
            for _, second_start, second_end in segments[first_order + 2:]:
                distances.append(_segment_distance(
                    first_start, first_end, second_start, second_end,
                ))
        # Match the controller's 5 cm centreline-capsule convention.
        return min(distances, default=math.inf) - 0.05

    @staticmethod
    def _log_barrier(safe: float, actual: float) -> float:
        if actual <= 0.0:
            return math.inf
        return max(math.log(safe / actual), 0.0)

    def _result(
        self,
        target_position: np.ndarray,
        q: np.ndarray,
        iterations: int,
    ) -> ReachabilityProbeResult:
        state, position_error, rotation_error = self._errors(q, target_position)
        position_norm = float(np.linalg.norm(position_error))
        rotation_norm = float(np.linalg.norm(rotation_error))
        scaled_jacobian = state.jacobian.copy()
        scaled_jacobian[3:] *= self.settings.characteristic_length_m
        minimum_singular = float(np.linalg.svd(scaled_jacobian, compute_uv=False)[-1])
        lower_margin = (q - self.lower) / self._span
        upper_margin = (self.upper - q) / self._span
        joint_margin = float(np.min(np.minimum(lower_margin, upper_margin)))
        assert state.link_points is not None
        collision_clearance = self._self_collision_clearance(state.link_points)
        position_violation = max(
            position_norm / self.settings.position_tolerance_m - 1.0, 0.0,
        )
        rotation_violation = max(
            rotation_norm / self.settings.rotation_tolerance_rad - 1.0, 0.0,
        )
        badness = 0.5 * (
            position_violation * position_violation
            + rotation_violation * rotation_violation
        )
        badness += self._log_barrier(
            self.settings.safe_minimum_singular_value, minimum_singular,
        )
        badness += self.settings.self_collision_weight * self._log_barrier(
            self.settings.safe_self_collision_clearance_m, collision_clearance,
        )
        reachable = (
            position_norm <= self.settings.position_tolerance_m
            and rotation_norm <= self.settings.rotation_tolerance_rad
        )
        return ReachabilityProbeResult(
            target_position.copy(), q.copy(), reachable, float(badness),
            position_norm, rotation_norm, minimum_singular,
            float(state.manipulability or 0.0), joint_margin,
            collision_clearance, iterations,
        )

    def probe(
        self,
        target_position: np.ndarray,
        warm_start: np.ndarray | None = None,
        *,
        use_restarts: bool = True,
    ) -> ReachabilityProbeResult:
        target = np.asarray(target_position, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("Reachability target position must contain finite XYZ")
        candidates = []
        for seed in self._restart_seeds(warm_start, use_restarts):
            q, iterations = self._solve_once(target, seed)
            candidates.append(self._result(target, q, iterations))
        reachable = [candidate for candidate in candidates if candidate.reachable]
        if reachable:
            return min(reachable, key=lambda item: (item.badness, item.position_error_m, item.rotation_error_rad))
        return min(candidates, key=lambda item: (
            item.position_error_m / self.settings.position_tolerance_m
            + item.rotation_error_rad / self.settings.rotation_tolerance_rad,
            item.badness,
        ))
