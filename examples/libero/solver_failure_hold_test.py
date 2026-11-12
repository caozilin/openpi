from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.libero.solver_failure_hold import SolverFailureHoldSession
from examples.libero.solver_failure_hold import validate_failure_policy_request


class FakeSession:
    """Rejected solver candidates mutate reference state, but never physics."""

    dt = 0.05
    physics_steps = 25
    joint_control_mode = "native"

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.target = SimpleNamespace(position=np.zeros(3), rotation=np.eye(3))
        self.tolerance = {"intent_ema": 0}
        self.release = {"accepted_rotation": 0}
        self.stage_key = None
        self.subtask_index = None
        self.frame_index = 0
        self.handles = SimpleNamespace(qpos_ids=np.arange(7), finger_qpos_ids=np.array([7, 8]))
        self.data = SimpleNamespace(qpos=np.zeros(9), time=0.0)
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.controller = SimpleNamespace(goal_qpos=np.zeros(7))
        self.ik = SimpleNamespace(
            kinematics=SimpleNamespace(
                evaluate=lambda q, **kw: SimpleNamespace(position=np.asarray(q[:3]), rotation=np.eye(3))
            )
        )
        self.env = SimpleNamespace(step=self.physics, check_success=lambda: self.task_success)
        self.commands = []
        self.task_success = False

    def native_joint_action(self, q, gripper):
        return np.r_[q, gripper]

    def physics(self, command):
        q = np.asarray(command[:7])
        self.controller.goal_qpos = q.copy()
        # A holding servo still settles toward the retained target.
        self.data.qpos[:7] += 0.5 * (q - self.data.qpos[:7])
        self.data.time += self.dt
        self.commands.append(np.asarray(command).copy())
        return {"time": self.data.time}, 0, False, {}

    def step(self, action, phase="APP", subtask_index=0, **kwargs):
        feasible = next(self.outcomes)
        self.target.position += action[:3]
        self.target.rotation = np.diag([1.0, -1.0, -1.0])
        self.tolerance["intent_ema"] += 1
        self.release["accepted_rotation"] += 1
        self.stage_key = (subtask_index, phase)
        self.subtask_index = subtask_index
        q = np.full(7, self.target.position[0])
        if feasible:
            self.physics(self.native_joint_action(q, action[6]))
        row = {
            "index": self.frame_index,
            "action": list(action),
            "phase": phase,
            "bounds_rad": kwargs.get('bounds_rad'),
            "subtask_index": subtask_index,
            "optimizer": {"feasible": feasible, "status": "success" if feasible else "infeasible"},
            "nominal_target": {"position_m": self.target.position.tolist(), "rotation_vector_rad": [np.pi, 0.0, 0.0]},
            "joint_target_rad": q.tolist(),
            "physics_steps": 25 if feasible else 0,
            "effective_mask": "111",
            "rotation_release_rad": [0.1] * 3,
            "predicted_tolerance_violation_rad": [0.0] * 3,
            "aborted": not feasible,
            "success": self.task_success,
        }
        self.frame_index += 1
        return {"time": self.data.time}, row


@pytest.fixture(autouse=True)
def geometry(monkeypatch):
    # Geometry is supplied by the frozen runtime in the real client.
    def rotvec(rotation):
        return np.zeros(3) if np.allclose(rotation, np.eye(3)) else np.array([np.pi, 0.0, 0.0])

    monkeypatch.setitem(sys.modules, "tasktol.geometry", SimpleNamespace(rotation_vector=rotvec))


def test_rejected_pose_and_tolerance_handoff_are_rolled_back_while_physics_runs():
    session = FakeSession([True, False, True])
    client = SolverFailureHoldSession(session)
    _, accepted = client.step(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
    snapshot = deepcopy((session.target, session.tolerance, session.release, session.stage_key, session.subtask_index))
    _, failed = client.step(np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), phase="EXEC", subtask_index=1)
    np.testing.assert_array_equal(session.target.position, snapshot[0].position)
    np.testing.assert_array_equal(session.target.rotation, snapshot[0].rotation)
    assert (session.tolerance, session.release, session.stage_key, session.subtask_index) == snapshot[1:]
    assert failed["nominal_target"] == accepted["nominal_target"]
    assert failed["rejected_nominal_target"]["position_m"][0] == pytest.approx(0.9)
    np.testing.assert_array_equal(session.commands[-1], session.commands[0])
    assert failed["physics_steps"] == 25
    assert session.data.time == pytest.approx(0.1)
    assert failed["reference_held"]
    assert not failed["nominal_reference_updated"]
    assert not failed["aborted"]
    assert failed["held_reference_phase"] == "APP"
    assert failed["held_reference_subtask_index"] == 0
    assert failed["post_action"]["joint_position_rad"][0] == pytest.approx(0.075)
    _, recovered = client.step(np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert recovered["nominal_target"]["position_m"][0] == pytest.approx(0.12)
    assert recovered["consecutive_solver_failures"] == 0
    assert recovered["nominal_reference_updated"]
    assert not recovered["reference_held"]


def test_failure_20_is_exactly_one_second_and_success_resets_streak():
    client = SolverFailureHoldSession(FakeSession([False] * 19 + [True] + [False] * 20))
    zero = np.zeros(7)
    for i in range(19):
        _, row = client.step(zero)
        assert row["consecutive_solver_failures"] == i + 1
        assert not row["aborted"]
    _, row = client.step(zero)
    assert row["consecutive_solver_failures"] == 0
    start = client.data.time
    for i in range(20):
        _, row = client.step(zero)
        assert row["aborted"] == (i == 19)
        assert row["physics_steps"] == 25
    assert client.data.time - start == pytest.approx(1.0)
    assert client.policy_metadata()["total_solver_failures"] == 39
    assert client.policy_metadata()["maximum_solver_failure_streak"] == 20
    with pytest.raises(RuntimeError, match="terminated"):
        client.step(zero)


def test_initial_rejection_holds_measured_pose_and_native_success_is_preserved():
    session = FakeSession([False, False])
    session.data.qpos[:7] = 0.3
    client = SolverFailureHoldSession(session, max_consecutive_solver_failures=2)
    for i in range(2):
        session.task_success = i == 1
        _, row = client.step(np.array([1.0, 0, 0, 0, 0, 0, 1.0]))
    assert row["success"]
    assert row["solver_failure_limit_reached"]
    assert not row["aborted"]
    assert row["nominal_target"]["position_m"] == [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(session.commands[0], np.r_[np.full(7, 0.3), -1.0])


def test_hold_keeps_accepted_angle_bounds_separate_from_rejected_phase_settings():
    client = SolverFailureHoldSession(FakeSession([True, False]))
    accepted = np.deg2rad([5,5,5,5,15,15]).tolist()
    attempted = [0.]*6
    client.step(np.zeros(7),phase='EXEC',bounds_rad=accepted)
    _, row = client.step(np.zeros(7),phase='ENG',bounds_rad=attempted)
    assert row['bounds_rad'] == attempted
    assert row['held_reference_bounds_rad'] == accepted
    assert row['held_reference_phase'] == 'EXEC'


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_bad_failure_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="positive integer"):
        SolverFailureHoldSession(FakeSession([]), max_consecutive_solver_failures=limit)


def test_changed_policy_cannot_silently_resume_historical_requests():
    from examples.libero import solver_failure_hold

    with pytest.raises(ValueError, match="new run request"):
        validate_failure_policy_request({}, 20)
    p = Path(solver_failure_hold.__file__).resolve()
    request = {
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()},
        "solver_failure_policy": "hold_last_accepted_reference",
        "max_consecutive_solver_failures": 20,
    }
    validate_failure_policy_request(request, 20)
    with pytest.raises(ValueError, match="does not match"):
        validate_failure_policy_request(request, 19)
