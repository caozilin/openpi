"""Transactional reference updates for the local LIBERO benchmark client.

The frozen motion generator is left untouched. Failed candidate references are
rolled back before advancing physics with the last accepted actuator reference.
"""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MAX_CONSECUTIVE_SOLVER_FAILURES = 20
_REFERENCE_FIELDS = ("target", "tolerance", "release", "stage_key", "subtask_index")


def validate_failure_policy_request(request, limit):
    """A changed failure policy must be bound into the next run's manifest."""
    path = Path(__file__).resolve()
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    if request.get("source_sha256", {}).get(str(path)) != expected:
        raise ValueError(
            "A new run request must freeze solver_failure_hold.py; historical runs cannot be resumed with this policy"
        )
    if (
        request.get("solver_failure_policy") != "hold_last_accepted_reference"
        or request.get("max_consecutive_solver_failures") != limit
    ):
        raise ValueError("The requested solver failure policy/limit does not match the client configuration")


class SolverFailureHoldSession:
    """Wrap a native session; solver failure consumes time, not a pose increment."""

    def __init__(self, session, *, max_consecutive_solver_failures=DEFAULT_MAX_CONSECUTIVE_SOLVER_FAILURES):
        if type(max_consecutive_solver_failures) is not int or max_consecutive_solver_failures <= 0:
            raise ValueError("The consecutive solver failure limit must be a positive integer")
        self.session = session
        self.max_consecutive_solver_failures = max_consecutive_solver_failures
        self.consecutive_solver_failures = 0
        self.total_solver_failures = 0
        self.maximum_solver_failure_streak = 0
        # Before the first accepted solve, hold the initial measured joint pose.
        self._joint_reference = session.data.qpos[session.handles.qpos_ids].copy()
        self._gripper_reference = -1.0
        self._accepted_row = None
        self._terminated = False

    def __getattr__(self, name):
        return getattr(self.session, name)

    def policy_metadata(self):
        return {
            "solver_failure_policy": "hold_last_accepted_reference",
            "max_consecutive_solver_failures": self.max_consecutive_solver_failures,
            "solver_failure_timeout_seconds": self.max_consecutive_solver_failures * self.dt,
            "total_solver_failures": self.total_solver_failures,
            "maximum_solver_failure_streak": self.maximum_solver_failure_streak,
        }

    def step(self, action, **kwargs):
        if self._terminated:
            raise RuntimeError("The consecutive solver failure threshold already terminated this episode")
        s = self.session
        snapshot = deepcopy({name: getattr(s, name) for name in _REFERENCE_FIELDS})
        observation, row = s.step(action, **kwargs)
        failed = not bool(row["optimizer"]["feasible"])
        if not failed:
            self.consecutive_solver_failures = 0
            self._joint_reference = np.asarray(row["joint_target_rad"], dtype=float).copy()
            self._gripper_reference = float(action[6])
        else:
            if row["physics_steps"] != 0:
                raise RuntimeError("A rejected solve unexpectedly advanced physics")
            row["rejected_nominal_target"] = deepcopy(row["nominal_target"])
            row["rejected_joint_target_rad"] = row["joint_target_rad"]
            for name, value in snapshot.items():
                setattr(s, name, value)
            self.consecutive_solver_failures += 1
            self.total_solver_failures += 1
            self.maximum_solver_failure_streak = max(
                self.maximum_solver_failure_streak, self.consecutive_solver_failures
            )
            observation = self._advance_held_reference(row)
        row.update(
            solver_failed=failed,
            reference_held=failed,
            nominal_reference_updated=not failed,
            applied_gripper_command=self._gripper_reference,
            consecutive_solver_failures=self.consecutive_solver_failures,
            max_consecutive_solver_failures=self.max_consecutive_solver_failures,
            solver_failure_limit_reached=self.consecutive_solver_failures >= self.max_consecutive_solver_failures,
        )
        # A native task success observed during a hold still counts as success.
        row["aborted"] = bool(row["solver_failure_limit_reached"] and not row["success"])
        self._terminated = row["aborted"]
        if not failed:
            self._accepted_row = deepcopy(row)
        return observation, row

    def _advance_held_reference(self, row):
        from tasktol.geometry import rotation_vector

        s = self.session
        q = self._joint_reference
        command = s.native_joint_action(q, self._gripper_reference)
        before_time = float(s.data.time)
        observation, _, _, _ = s.env.step(command.tolist())
        physics_steps = round((float(s.data.time) - before_time) / s.model.opt.timestep)
        if physics_steps != s.physics_steps:
            raise RuntimeError("A held reference did not advance one complete control interval")
        held = s.controller.joint_target if s.joint_control_mode == "osc_task_joint_null" else s.controller.goal_qpos
        if not np.allclose(held, q, atol=1e-10, rtol=0):
            raise RuntimeError("The native controller did not retain the accepted joint reference")
        actual = s.ik.kinematics.evaluate(
            s.data.qpos[s.handles.qpos_ids], include_manipulability=False, include_link_points=False
        )
        reference = s.ik.kinematics.evaluate(q, include_manipulability=False, include_link_points=False)
        row.update(
            nominal_target={
                "position_m": s.target.position.tolist(),
                "rotation_vector_rad": rotation_vector(s.target.rotation).tolist(),
            },
            joint_target_rad=q.tolist(),
            post_action={
                "position_m": actual.position.tolist(),
                "rotation_vector_rad": rotation_vector(actual.rotation).tolist(),
                "joint_position_rad": s.data.qpos[s.handles.qpos_ids].tolist(),
                "finger_qpos_m": s.data.qpos[s.handles.finger_qpos_ids].tolist(),
            },
            simulation_time_s=float(s.data.time),
            physics_steps=physics_steps,
            native_joint_action=command.tolist(),
            native_joint_goal_error_rad=float(np.max(np.abs(held - q))),
            applied_gripper_command=self._gripper_reference,
            held_reference_phase=None if self._accepted_row is None else self._accepted_row["phase"],
            held_reference_subtask_index=None if self._accepted_row is None else self._accepted_row["subtask_index"],
            held_reference_bounds_rad=None if self._accepted_row is None else deepcopy(self._accepted_row.get("bounds_rad")),
            tracking_position_error_m=float(np.linalg.norm(actual.position - s.target.position)),
            tracking_rotation_error_rad=float(np.linalg.norm(rotation_vector(reference.rotation.T @ actual.rotation))),
            success=bool(s.env.check_success()),
        )
        # Failed candidates must not appear as accepted tolerance realizations.
        for field in ("effective_mask", "rotation_release_rad", "predicted_tolerance_violation_rad"):
            if field in row:
                row[f"rejected_{field}"] = row.pop(field)
            if self._accepted_row is not None and field in self._accepted_row:
                row[field] = deepcopy(self._accepted_row[field])
        row.setdefault("effective_mask", "000")
        return observation
