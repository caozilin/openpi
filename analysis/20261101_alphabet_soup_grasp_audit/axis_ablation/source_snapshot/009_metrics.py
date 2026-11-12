"""Controller-independent trajectory and task benchmark metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from ..config import TELEOP_DT
from ..control.task_space.conditioning import (
    SIGMA_MIN_NORMALIZATION, velocity_normalized_sigma_min,
)
from ..geometry import rotation_matrix, rotation_vector
from ..simulation.robot import RobotHandles, ee_pose


POSITION_TOLERANCE_M = 0.01
ROTATION_TOLERANCE_RAD = 0.10


def interpolate_pose_reference(
    start_position: np.ndarray,
    start_rotation: np.ndarray,
    goal_position: np.ndarray,
    goal_rotation: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a 10 Hz Cartesian reference at the physics sampling rate."""
    fraction = float(np.clip(alpha, 0.0, 1.0))
    position = (
        (1.0 - fraction) * start_position + fraction * goal_position
    )
    rotation_delta = rotation_vector(start_rotation.T @ goal_rotation)
    rotation = start_rotation @ rotation_matrix(fraction * rotation_delta)
    return position, rotation


@dataclass(frozen=True)
class EvaluationSummary:
    samples: int = 0
    mean_position_error_m: float = 0.0
    mean_rotation_error_rad: float = 0.0
    mean_joint_acceleration_rad_s2: float = 0.0
    mean_joint_jerk_rad_s3: float = 0.0
    mean_manipulability: float = 0.0
    exceed_tolerances: int = 0
    mean_joint_movement_rad: float = 0.0
    minimum_sigma_min_full: float | None = None
    joint_acceleration_samples: int = 0
    joint_jerk_samples: int = 0
    minimum_sigma_min_velocity_normalized: float | None = None
    sigma_min_normalization: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def standard_kinematic_metrics(summary: EvaluationSummary) -> dict[str, float | None]:
    """The standard four metrics, evaluated on optimized 10 Hz commands.

    Movement uses the common TELEOP_DT time scale, so dividing by it yields
    mean joint speed. sigma_min is the time minimum of per-state
    velocity-capability-normalized singular values. Historical raw-only
    results have no normalized metric; never substitute their raw value.
    """
    return {
        "velocity_rad_s": summary.mean_joint_movement_rad / TELEOP_DT,
        "acceleration_rad_s2": summary.mean_joint_acceleration_rad_s2,
        "mY": summary.mean_manipulability,
        "sigma_min": summary.minimum_sigma_min_velocity_normalized,
    }


@dataclass(frozen=True)
class BenchmarkEpisodeResult:
    """Policy-level outcome combined with controller trajectory quality."""

    scene: str
    task: str
    seed: int
    success: bool
    termination_reason: str
    collision_count: int
    elapsed_s: float
    frames: int
    trajectory: EvaluationSummary
    instruction: str = ""
    policy_calls: int = 0
    mean_policy_latency_ms: float = 0.0
    p95_policy_latency_ms: float = 0.0
    optimized_10hz_trajectory: EvaluationSummary = EvaluationSummary()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trajectory"] = self.trajectory.as_dict()
        return payload


@dataclass(frozen=True)
class BenchmarkSummary:
    episodes: int
    successes: int
    success_rate: float
    collision_episodes: int
    collision_rate: float
    mean_elapsed_s: float
    mean_joint_movement_rad: float
    mean_position_error_m: float
    mean_rotation_error_rad: float
    mean_joint_acceleration_rad_s2: float
    mean_joint_jerk_rad_s3: float
    mean_manipulability: float
    mean_policy_latency_ms: float
    mean_minimum_sigma_min_full: float = 0.0
    mean_minimum_sigma_min_velocity_normalized: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_episode_results(
    results: list[BenchmarkEpisodeResult],
) -> BenchmarkSummary:
    if not results:
        return BenchmarkSummary(
            episodes=0,
            successes=0,
            success_rate=0.0,
            collision_episodes=0,
            collision_rate=0.0,
            mean_elapsed_s=0.0,
            mean_joint_movement_rad=0.0,
            mean_position_error_m=0.0,
            mean_rotation_error_rad=0.0,
            mean_joint_acceleration_rad_s2=0.0,
            mean_joint_jerk_rad_s3=0.0,
            mean_manipulability=0.0,
            mean_policy_latency_ms=0.0,
            mean_minimum_sigma_min_full=0.0,
        )
    count = len(results)
    successes = sum(result.success for result in results)
    collisions = sum(result.collision_count > 0 for result in results)
    minimum_sigma_min_values = [
        result.optimized_10hz_trajectory.minimum_sigma_min_full
        for result in results
        if result.optimized_10hz_trajectory.minimum_sigma_min_full is not None
    ]
    normalized_sigma_values = [
        result.optimized_10hz_trajectory.minimum_sigma_min_velocity_normalized
        for result in results
        if result.optimized_10hz_trajectory.minimum_sigma_min_velocity_normalized is not None
    ]
    return BenchmarkSummary(
        episodes=count,
        successes=successes,
        success_rate=successes / count,
        collision_episodes=collisions,
        collision_rate=collisions / count,
        mean_elapsed_s=float(np.mean([result.elapsed_s for result in results])),
        mean_joint_movement_rad=float(np.mean([
            result.trajectory.mean_joint_movement_rad for result in results
        ])),
        mean_position_error_m=float(np.mean([
            result.trajectory.mean_position_error_m for result in results
        ])),
        mean_rotation_error_rad=float(np.mean([
            result.trajectory.mean_rotation_error_rad for result in results
        ])),
        mean_joint_acceleration_rad_s2=float(np.mean([
            result.trajectory.mean_joint_acceleration_rad_s2 for result in results
        ])),
        mean_joint_jerk_rad_s3=float(np.mean([
            result.trajectory.mean_joint_jerk_rad_s3 for result in results
        ])),
        mean_manipulability=float(np.mean([
            result.trajectory.mean_manipulability for result in results
        ])),
        mean_policy_latency_ms=float(np.mean([
            result.mean_policy_latency_ms for result in results
        ])),
        mean_minimum_sigma_min_full=(
            float(np.mean(minimum_sigma_min_values))
            if minimum_sigma_min_values else 0.0
        ),
        mean_minimum_sigma_min_velocity_normalized=(
            float(np.mean(normalized_sigma_values)) if normalized_sigma_values else None
        ),
    )


class EvaluationRecorder:
    """Accumulate physics-rate feedback metrics without sign cancellation."""

    def __init__(self, *, movement_time_scale_s: float | None = None) -> None:
        if movement_time_scale_s is not None and movement_time_scale_s <= 0.0:
            raise ValueError("movement_time_scale_s must be positive")
        self.recording = False
        self.summary = EvaluationSummary()
        self._movement_time_scale_s = movement_time_scale_s
        self._previous_q: np.ndarray | None = None
        self._previous_velocity: np.ndarray | None = None
        self._previous_acceleration: np.ndarray | None = None
        self._sums = np.zeros(6)
        self._acceleration_samples = 0
        self._jerk_samples = 0
        self._minimum_sigma_min_full: float | None = None
        self._minimum_sigma_min_velocity_normalized: float | None = None
        self._exceeded = 0
        self.last_acceleration_norm = float("nan")
        self.last_jerk_norm = float("nan")

    def toggle(self) -> bool:
        if self.recording:
            self.stop()
        else:
            self.start()
        return self.recording

    def start(self) -> None:
        """Start a fresh measurement, discarding the previous live summary."""
        self.recording = True
        self.summary = EvaluationSummary()
        self._previous_q = self._previous_velocity = self._previous_acceleration = None
        self._sums.fill(0.0)
        self._acceleration_samples = 0
        self._jerk_samples = 0
        self._minimum_sigma_min_full = None
        self._minimum_sigma_min_velocity_normalized = None
        self._exceeded = 0
        self.last_acceleration_norm = float("nan")
        self.last_jerk_norm = float("nan")

    def stop(self) -> None:
        """Freeze the current summary without clearing it."""
        self.recording = False

    def sample(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        actual_position: np.ndarray,
        actual_rotation: np.ndarray,
        q: np.ndarray,
        manipulability: float | None,
        dt: float,
        *,
        commanded: bool,
        sigma_min_full: float | None = None,
        sigma_min_velocity_normalized: float | None = None,
    ) -> EvaluationSummary:
        if not self.recording:
            return self.summary
        if sigma_min_full is not None:
            sigma_min_full = float(sigma_min_full)
            if not np.isfinite(sigma_min_full) or sigma_min_full < 0.0:
                raise ValueError("sigma_min_full must be finite and non-negative")
            self._minimum_sigma_min_full = (
                sigma_min_full
                if self._minimum_sigma_min_full is None
                else min(self._minimum_sigma_min_full, sigma_min_full)
            )
        if sigma_min_velocity_normalized is not None:
            value = float(sigma_min_velocity_normalized)
            if not np.isfinite(value) or value < 0:
                raise ValueError("sigma_min_velocity_normalized must be finite and non-negative")
            self._minimum_sigma_min_velocity_normalized = (
                value if self._minimum_sigma_min_velocity_normalized is None
                else min(self._minimum_sigma_min_velocity_normalized, value)
            )
        # ``commanded`` is retained for call-site compatibility and metadata;
        # measurement is deliberately controller-independent, so every frame
        # inside the recording window contributes to the statistics.
        position_error = float(np.linalg.norm(target_position - actual_position))
        rotation_error = float(np.linalg.norm(rotation_vector(target_rotation.T @ actual_rotation)))
        # Do not manufacture zero-valued derivative samples at startup.  A
        # backward second difference is valid only after three q samples, and
        # a backward third difference only after four.  Each metric therefore
        # owns its denominator instead of dividing by the state-sample count.
        velocity = (
            None if self._previous_q is None
            else (q - self._previous_q) / dt
        )
        acceleration = (
            None
            if velocity is None or self._previous_velocity is None
            else (velocity - self._previous_velocity) / dt
        )
        jerk = (
            None
            if acceleration is None or self._previous_acceleration is None
            else (acceleration - self._previous_acceleration) / dt
        )
        self.last_acceleration_norm = (
            float("nan")
            if acceleration is None else float(np.linalg.norm(acceleration))
        )
        self.last_jerk_norm = (
            float("nan") if jerk is None else float(np.linalg.norm(jerk))
        )
        movement = 0.0 if self._previous_q is None else float(np.linalg.norm(q - self._previous_q))
        normalized_movement = (
            movement
            if self._movement_time_scale_s is None
            else movement * self._movement_time_scale_s / dt
        )
        self._previous_q = q.copy()
        if velocity is not None:
            self._previous_velocity = velocity
        if acceleration is not None:
            self._previous_acceleration = acceleration
        values = np.array((
            position_error,
            rotation_error,
            0.0 if acceleration is None else np.linalg.norm(acceleration),
            0.0 if jerk is None else np.linalg.norm(jerk),
            manipulability or 0.0,
            normalized_movement,
        ))
        self._sums += np.abs(values)
        self._acceleration_samples += int(acceleration is not None)
        self._jerk_samples += int(jerk is not None)
        self._exceeded += int(
            position_error > POSITION_TOLERANCE_M
            or rotation_error > ROTATION_TOLERANCE_RAD
        )
        count = self.summary.samples + 1
        means = self._sums / count
        means[2] = (
            self._sums[2] / self._acceleration_samples
            if self._acceleration_samples else 0.0
        )
        means[3] = (
            self._sums[3] / self._jerk_samples
            if self._jerk_samples else 0.0
        )
        # Joint movement is defined over intervals, not state samples.  Divide
        # by N-1 so its common 0.1 s time scale is invariant to 10/100/1000 Hz
        # sampling instead of being biased by the initial zero-delta sample.
        means[5] = (
            self._sums[5] / (count - 1)
            if count > 1 else 0.0
        )
        self.summary = EvaluationSummary(
            count,
            *means[:5],
            self._exceeded,
            means[5],
            minimum_sigma_min_full=self._minimum_sigma_min_full,
            minimum_sigma_min_velocity_normalized=self._minimum_sigma_min_velocity_normalized,
            sigma_min_normalization=SIGMA_MIN_NORMALIZATION if self._minimum_sigma_min_velocity_normalized is not None else None,
            joint_acceleration_samples=self._acceleration_samples,
            joint_jerk_samples=self._jerk_samples,
        )
        return self.summary


@dataclass(frozen=True)
class MultiRateEvaluationSummary:
    """Controller-independent views of one physical execution."""

    optimized_10hz: EvaluationSummary = EvaluationSummary()
    actual_10hz: EvaluationSummary = EvaluationSummary()
    actual_100hz: EvaluationSummary = EvaluationSummary()
    filtered_100hz: EvaluationSummary = EvaluationSummary()
    actual_1khz: EvaluationSummary = EvaluationSummary()

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimized_10hz": self.optimized_10hz.as_dict(),
            "actual_10hz": self.actual_10hz.as_dict(),
            "actual_100hz": self.actual_100hz.as_dict(),
            "filtered_100hz": self.filtered_100hz.as_dict(),
            "actual_1khz": self.actual_1khz.as_dict(),
        }


class _ButterworthLowPass4:
    """Fourth-order Butterworth low-pass as two stable biquad sections."""

    _SECTION_Q = (0.541196100146197, 1.306562964876377)

    def __init__(self, cutoff_hz: float, sample_hz: float) -> None:
        if not 0.0 < cutoff_hz < 0.5 * sample_hz:
            raise ValueError("cutoff_hz must be between zero and Nyquist")
        omega = 2.0 * np.pi * cutoff_hz / sample_hz
        cosine = float(np.cos(omega))
        sine = float(np.sin(omega))
        self._coefficients: list[tuple[float, float, float, float, float]] = []
        for quality in self._SECTION_Q:
            alpha = sine / (2.0 * quality)
            a0 = 1.0 + alpha
            self._coefficients.append((
                (1.0 - cosine) / (2.0 * a0),
                (1.0 - cosine) / a0,
                (1.0 - cosine) / (2.0 * a0),
                -2.0 * cosine / a0,
                (1.0 - alpha) / a0,
            ))
        self._history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None

    def reset(self) -> None:
        self._history = None

    def sample(self, value: np.ndarray) -> np.ndarray:
        current = np.asarray(value, dtype=float).copy()
        if self._history is None:
            # Constant-state initialization removes the artificial start-up
            # transient while preserving the Butterworth DC gain of one.
            self._history = [
                (current.copy(), current.copy(), current.copy(), current.copy())
                for _ in self._coefficients
            ]
            return current
        next_history = []
        for coefficients, history in zip(
            self._coefficients, self._history, strict=True,
        ):
            b0, b1, b2, a1, a2 = coefficients
            x1, x2, y1, y2 = history
            output = b0 * current + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            next_history.append((current.copy(), x1, output.copy(), y1))
            current = output
        self._history = next_history
        return current


class MultiRateEvaluationRecorder:
    """Measure optimizer waypoints and physical feedback on aligned clocks.

    Controllers provide only their next optimized joint waypoint.  This class
    owns FK, measured-state sampling, reference interpolation, derivative
    histories, and aggregation, so every controller is evaluated identically.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        control_dt: float,
    ) -> None:
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        self.model = model
        self.handles = handles
        self.control_dt = float(control_dt)
        self.physics_dt = float(model.opt.timestep)
        recorder_options = {"movement_time_scale_s": self.control_dt}
        self._optimized = EvaluationRecorder(**recorder_options)
        self._actual_10hz = EvaluationRecorder(**recorder_options)
        self._actual_100hz = EvaluationRecorder(**recorder_options)
        self._filtered_100hz = EvaluationRecorder(**recorder_options)
        self._actual_1khz = EvaluationRecorder(**recorder_options)
        self._downsample_factor = 10
        self._downsample_dt = self.physics_dt * self._downsample_factor
        self._downsample_count = 0
        self._filter = _ButterworthLowPass4(20.0, 1.0 / self.physics_dt)
        self._scratch = mujoco.MjData(model)
        self._jac_pos = np.zeros((3, model.nv))
        self._jac_rot = np.zeros((3, model.nv))
        self._reference_start_position = np.zeros(3)
        self._reference_goal_position = np.zeros(3)
        self._reference_start_rotation = np.eye(3)
        self._reference_goal_rotation = np.eye(3)
        self._optimized_q: np.ndarray | None = None
        self._pending_endpoint = False
        self._control_interval_index = -1
        self._derivative_traces: dict[str, list[tuple[int, float, float]]] = {
            "optimized_10hz": [],
            "filtered_100hz": [],
            "actual_1khz": [],
        }

    @property
    def recording(self) -> bool:
        return self._actual_1khz.recording

    @property
    def summary(self) -> MultiRateEvaluationSummary:
        return MultiRateEvaluationSummary(
            self._optimized.summary,
            self._actual_10hz.summary,
            self._actual_100hz.summary,
            self._filtered_100hz.summary,
            self._actual_1khz.summary,
        )

    def start(self) -> None:
        self._optimized.start()
        self._actual_10hz.start()
        self._actual_100hz.start()
        self._filtered_100hz.start()
        self._actual_1khz.start()
        self._pending_endpoint = False
        self._downsample_count = 0
        self._filter.reset()
        self._control_interval_index = -1
        for trace in self._derivative_traces.values():
            trace.clear()

    def stop(self) -> None:
        self._optimized.stop()
        self._actual_10hz.stop()
        self._actual_100hz.stop()
        self._filtered_100hz.stop()
        self._actual_1khz.stop()
        self._pending_endpoint = False

    def toggle(self) -> bool:
        if self.recording:
            self.stop()
        else:
            self.start()
        return self.recording

    def begin_control_interval(
        self,
        reference_start_position: np.ndarray,
        reference_start_rotation: np.ndarray,
        reference_goal_position: np.ndarray,
        reference_goal_rotation: np.ndarray,
        optimized_q: np.ndarray | None,
        *,
        interval_index: int | None = None,
    ) -> None:
        """Register one controller interval without depending on its solver."""
        self._reference_start_position = reference_start_position.copy()
        self._reference_start_rotation = reference_start_rotation.copy()
        self._reference_goal_position = reference_goal_position.copy()
        self._reference_goal_rotation = reference_goal_rotation.copy()
        self._optimized_q = (
            None if optimized_q is None else np.asarray(optimized_q, dtype=float).copy()
        )
        self._control_interval_index = (
            self._control_interval_index + 1
            if interval_index is None else int(interval_index)
        )
        self._pending_endpoint = self.recording

    def derivative_trace(self, domain: str) -> np.ndarray:
        """Return ``[control_interval, ||a||, ||j||]`` diagnostic samples."""
        if domain not in self._derivative_traces:
            raise ValueError(f"Unsupported derivative trace domain: {domain!r}")
        return np.asarray(self._derivative_traces[domain], dtype=float).reshape(-1, 3)

    def boundary_extrema(
        self,
        interval_index: int,
        *,
        radius_control_intervals: int = 3,
    ) -> dict[str, dict[str, float | int]]:
        """Return local derivative peaks around one semantic phase boundary."""
        if radius_control_intervals < 0:
            raise ValueError("Boundary radius must be non-negative")
        lower = int(interval_index) - radius_control_intervals
        upper = int(interval_index) + radius_control_intervals
        result: dict[str, dict[str, float | int]] = {}
        for domain in self._derivative_traces:
            values = self.derivative_trace(domain)
            selected = values[
                (values[:, 0] >= lower) & (values[:, 0] <= upper)
            ] if values.size else values
            acceleration = (
                selected[np.isfinite(selected[:, 1]), 1]
                if selected.size else np.empty(0)
            )
            jerk = (
                selected[np.isfinite(selected[:, 2]), 2]
                if selected.size else np.empty(0)
            )
            result[domain] = {
                "samples": int(len(selected)),
                "acceleration_samples": int(len(acceleration)),
                "jerk_samples": int(len(jerk)),
                "max_joint_acceleration_norm_rad_s2": (
                    float(np.max(acceleration)) if len(acceleration) else 0.0
                ),
                "max_joint_jerk_norm_rad_s3": (
                    float(np.max(jerk)) if len(jerk) else 0.0
                ),
            }
        return result

    def _state_metrics(
        self,
        data: mujoco.MjData,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        position, rotation = ee_pose(data, self.handles)
        mujoco.mj_jacSite(
            self.model, data, self._jac_pos, self._jac_rot,
            self.handles.ee_site_id,
        )
        jacobian = np.vstack((
            self._jac_pos[:, self.handles.dof_ids],
            self._jac_rot[:, self.handles.dof_ids],
        ))
        determinant = float(np.linalg.det(jacobian @ jacobian.T))
        manipulability = float(np.sqrt(np.clip(determinant, 0.0, 1.0)))
        q = data.qpos[self.handles.qpos_ids].copy()
        return position, rotation, q, manipulability

    def _joint_state_metrics(
        self,
        q: np.ndarray,
        *,
        include_sigma_min: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float | None, float | None]:
        """Evaluate one joint waypoint, optionally including full-Jacobian σmin."""
        self._scratch.qpos[self.handles.qpos_ids] = q
        mujoco.mj_forward(self.model, self._scratch)
        position, rotation = ee_pose(self._scratch, self.handles)
        mujoco.mj_jacSite(
            self.model, self._scratch, self._jac_pos, self._jac_rot,
            self.handles.ee_site_id,
        )
        jacobian = np.vstack((
            self._jac_pos[:, self.handles.dof_ids],
            self._jac_rot[:, self.handles.dof_ids],
        ))
        determinant = float(np.linalg.det(jacobian @ jacobian.T))
        manipulability = float(np.sqrt(np.clip(determinant, 0.0, 1.0)))
        sigma_min_full = (
            float(np.linalg.svd(jacobian, compute_uv=False)[-1])
            if include_sigma_min else None
        )
        normalized = (
            velocity_normalized_sigma_min(jacobian, self.handles.spec.joint_velocity_limits)
            if include_sigma_min else None
        )
        return position, rotation, manipulability, sigma_min_full, normalized

    def sample_physics(
        self,
        data: mujoco.MjData,
        alpha: float,
        *,
        commanded: bool,
    ) -> MultiRateEvaluationSummary:
        """Sample immediately after one physics step."""
        if not self.recording:
            return self.summary
        actual_position, actual_rotation, actual_q, actual_manipulability = (
            self._state_metrics(data)
        )
        reference_position, reference_rotation = interpolate_pose_reference(
            self._reference_start_position,
            self._reference_start_rotation,
            self._reference_goal_position,
            self._reference_goal_rotation,
            alpha,
        )
        self._actual_1khz.sample(
            reference_position,
            reference_rotation,
            actual_position,
            actual_rotation,
            actual_q,
            actual_manipulability,
            self.physics_dt,
            commanded=commanded,
        )
        self._derivative_traces["actual_1khz"].append((
            self._control_interval_index,
            self._actual_1khz.last_acceleration_norm,
            self._actual_1khz.last_jerk_norm,
        ))
        filtered_q = self._filter.sample(actual_q)
        self._downsample_count += 1
        if self._downsample_count % self._downsample_factor == 0:
            self._actual_100hz.sample(
                reference_position,
                reference_rotation,
                actual_position,
                actual_rotation,
                actual_q,
                actual_manipulability,
                self._downsample_dt,
                commanded=commanded,
            )
            (
                filtered_position,
                filtered_rotation,
                filtered_manipulability,
                _,
                _,
            ) = self._joint_state_metrics(filtered_q)
            self._filtered_100hz.sample(
                reference_position,
                reference_rotation,
                filtered_position,
                filtered_rotation,
                filtered_q,
                filtered_manipulability,
                self._downsample_dt,
                commanded=commanded,
            )
            self._derivative_traces["filtered_100hz"].append((
                self._control_interval_index,
                self._filtered_100hz.last_acceleration_norm,
                self._filtered_100hz.last_jerk_norm,
            ))
        if self._pending_endpoint and alpha >= 1.0 - 1e-12:
            self._actual_10hz.sample(
                self._reference_goal_position,
                self._reference_goal_rotation,
                actual_position,
                actual_rotation,
                actual_q,
                actual_manipulability,
                self.control_dt,
                commanded=True,
            )
            if self._optimized_q is not None:
                (
                    position,
                    rotation,
                    manipulability,
                    sigma_min_full,
                    sigma_min_velocity_normalized,
                ) = self._joint_state_metrics(
                    self._optimized_q,
                    include_sigma_min=True,
                )
                self._optimized.sample(
                    self._reference_goal_position,
                    self._reference_goal_rotation,
                    position,
                    rotation,
                    self._optimized_q,
                    manipulability,
                    self.control_dt,
                    commanded=True,
                    sigma_min_full=sigma_min_full,
                    sigma_min_velocity_normalized=sigma_min_velocity_normalized,
                )
                self._derivative_traces["optimized_10hz"].append((
                    self._control_interval_index,
                    self._optimized.last_acceleration_norm,
                    self._optimized.last_jerk_norm,
                ))
            self._pending_endpoint = False
        return self.summary
