from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from examples.libero import success_continuation as policy


def execute(success_frames, *, limit=100, abort_at=None, seconds=1.0):
    continuation = policy.SuccessContinuation(seconds)
    rows = []
    frame = 0
    while continuation.should_run(frame, limit):
        row = dict(success=frame in success_frames, aborted=frame == abort_at)
        stopped = continuation.observe(frame, row)
        rows.append(row)
        frame += 1
        if stopped:
            break
    result = dict(success=continuation.success, termination=continuation.termination,
                  max_steps=limit, steps=len(rows), **continuation.metadata())
    return result, rows


def test_exactly_twenty_more_actions_even_when_native_success_disappears():
    result, rows = execute({3})
    assert len(rows) == 24  # first successful action plus twenty more
    assert rows[3]['post_success_step'] == 0
    assert rows[-1]['post_success_step'] == 20
    assert result['success'] and not result['final_native_success']
    assert result['post_success_complete']
    assert result['post_success_elapsed_seconds'] == 1.0
    policy.audit_record(result, rows, False)


def test_tail_can_extend_past_original_limit_but_no_success_cannot():
    result, rows = execute({4, 24}, limit=5)
    assert len(rows) == 25
    assert result['final_native_success']
    policy.audit_record(result, rows, True)
    result, rows = execute(set(), limit=5)
    assert len(rows) == 5 and not result['success']
    assert result['termination'] == 'step_limit'
    policy.audit_record(result, rows, False)


def test_successive_native_hits_do_not_restart_the_tail():
    result, rows = execute(set(range(1, 100)))
    assert len(rows) == 22
    assert result['first_success_frame'] == 1
    assert result['post_success_steps_completed'] == 20
    policy.audit_record(result, rows, True)


def test_continuation_and_real_hold_wrapper_advance_a_full_simulated_second(monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    from examples.libero.solver_failure_hold import SolverFailureHoldSession
    from examples.libero.solver_failure_hold_test import FakeSession
    monkeypatch.setitem(sys.modules, 'tasktol.geometry', SimpleNamespace(rotation_vector=lambda r: np.zeros(3)))
    session = FakeSession([True] * 21)
    client = SolverFailureHoldSession(session)
    continuation = policy.SuccessContinuation()
    first_time = None
    index = 0
    while continuation.should_run(index, 5):
        session.task_success = index == 0
        _, row = client.step(np.array([.01, 0, 0, 0, 0, 0, -1.0]))
        if index == 0:
            first_time = session.data.time
        stop = continuation.observe(index, row)
        index += 1
        if stop:
            break
    assert session.data.time - first_time == pytest.approx(1.0)
    assert len(session.commands) == 21
    assert not np.array_equal(session.commands[0], session.commands[-1])
    assert continuation.success and not continuation.final_native_success


def test_solver_failure_stop_is_retained_and_tail_interruption_is_explicit():
    result, rows = execute({1}, abort_at=5)
    assert result['success']
    assert result['post_success_steps_completed'] == 4
    assert not result['post_success_complete']
    assert result['post_success_interrupted_reason'] == 'ik_failure_streak'
    policy.audit_record(result, rows, False)
    result, rows = execute(set(), abort_at=5)
    assert not result['success'] and result['termination'] == 'ik_failure_streak'


def test_zero_tail_is_explicit_and_legacy_records_still_audit():
    result, rows = execute({1}, seconds=0)
    assert len(rows) == 2 and result['post_success_steps_completed'] == 0
    policy.audit_record(result, rows, True)
    policy.audit_record(dict(success=True, max_steps=100), [dict(success=False),dict(success=True)], True)


def test_audit_detects_truncation_false_final_success_and_reset_countdown():
    result, rows = execute({1, 21})
    for key, value in [('post_success_steps_completed', 19), ('first_success_frame', 3), ('final_native_success', False)]:
        broken = dict(result, **{key: value})
        with pytest.raises(AssertionError):
            policy.audit_record(broken, rows, True)
    broken_rows = deepcopy(rows)
    broken_rows[3]['post_success_step'] = 0
    with pytest.raises(AssertionError):
        policy.audit_record(result, broken_rows, True)


def test_new_request_must_freeze_helper_and_explicit_policy():
    with pytest.raises(ValueError, match='historical'):
        policy.validate_request({})
    path = Path(policy.__file__).resolve()
    request = dict(source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()}, **policy.request_fields())
    assert policy.validate_request(request)['post_success_steps'] == 20
    with pytest.raises(ValueError, match='explicitly'):
        policy.validate_request(dict(request, post_success_steps=19))


@pytest.mark.parametrize('seconds', [-1, True, float('nan'), float('inf'), 0.03])
def test_invalid_durations_cannot_silently_round_or_become_unbounded(seconds):
    with pytest.raises(ValueError):
        policy.SuccessContinuation(seconds)
