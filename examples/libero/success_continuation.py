"""Continue the same closed-loop policy for one simulated second after success."""
import hashlib
import math
from pathlib import Path

DEFAULT_POST_SUCCESS_SECONDS = 1.0
CONTROL_HZ = 20
SUCCESS_CRITERION = 'first_native_success'


def request_fields(seconds=DEFAULT_POST_SUCCESS_SECONDS):
    steps = seconds_to_steps(seconds)
    return dict(post_success_seconds=float(seconds), post_success_steps=steps,
                success_criterion=SUCCESS_CRITERION)


def seconds_to_steps(seconds):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('post_success_seconds must be finite and nonnegative')
    steps = round(seconds * CONTROL_HZ)
    if not math.isclose(steps / CONTROL_HZ, seconds, abs_tol=1e-10):
        raise ValueError('post_success_seconds must contain complete 20Hz control intervals')
    return steps


def validate_request(request):
    path = Path(__file__).resolve()
    if request.get('source_sha256', {}).get(str(path)) != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Create a new evaluation request freezing success_continuation.py; do not resume historical runs')
    expected = request_fields(request.get('post_success_seconds', DEFAULT_POST_SUCCESS_SECONDS))
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError(f'The new request must explicitly include the success continuation policy: {expected}')
    return expected


class SuccessContinuation:
    """Latch task success while recording the final native predicate separately.

    The first successful action is step zero of the tail, followed by exactly
    twenty additional actions. A pre-success step limit is allowed to extend
    only for that tail. Existing solver-failure termination remains active.
    """
    def __init__(self, seconds=DEFAULT_POST_SUCCESS_SECONDS):
        self.requested_steps = seconds_to_steps(seconds)
        self.first_success_frame = None
        self.completed_steps = 0
        self.steps_seen = 0
        self.final_native_success = False
        self.stopped = False
        self.termination = 'step_limit'
        self.interrupted_reason = None

    @property
    def success(self):
        return self.first_success_frame is not None

    def should_run(self, frame, nominal_limit):
        return not self.stopped and (frame < nominal_limit or self.success)

    def observe(self, frame, row):
        if self.stopped or frame != self.steps_seen:
            raise ValueError('Success continuation requires sequential, unconsumed action frames')
        self.steps_seen += 1
        self.final_native_success = bool(row['success'])
        if not self.success and self.final_native_success:
            self.first_success_frame = frame
        elif self.success:
            self.completed_steps += 1
        row.update(first_native_success_frame=self.first_success_frame,
                   post_success_step=self.completed_steps if self.success else None)
        if row['aborted']:
            self.stopped = True
            self.termination = 'task_success' if self.success else 'ik_failure_streak'
            if self.success and self.completed_steps < self.requested_steps:
                self.interrupted_reason = 'ik_failure_streak'
        elif self.success and self.completed_steps >= self.requested_steps:
            self.stopped = True
            self.termination = 'task_success'
        return self.stopped

    def metadata(self):
        return dict(**request_fields(self.requested_steps / CONTROL_HZ),
                    first_success_frame=self.first_success_frame,
                    first_success_time_s=None if not self.success else (self.first_success_frame + 1) / CONTROL_HZ,
                    post_success_steps_completed=self.completed_steps,
                    post_success_elapsed_seconds=self.completed_steps / CONTROL_HZ,
                    post_success_complete=bool(self.success and self.completed_steps == self.requested_steps),
                    post_success_interrupted_reason=self.interrupted_reason,
                    final_native_success=self.final_native_success)


def audit_record(result, rows, native_final_success, *, request=None):
    """Check recorded termination independently, including legacy first-hit runs."""
    first = next((i for i, row in enumerate(rows) if row['success']), None)
    assert bool(rows[-1]['success']) == bool(native_final_success)
    if 'post_success_seconds' not in result:
        assert len(rows) <= result['max_steps']
        assert first is None or first == len(rows) - 1
        assert bool(result['success']) == bool(native_final_success)
        return
    requested = seconds_to_steps(result['post_success_seconds'])
    if request is not None:
        settings = validate_request(request)
        assert all(result[key] == value for key, value in settings.items())
    assert result['post_success_steps'] == requested
    assert result['success_criterion'] == SUCCESS_CRITERION
    assert result['first_success_frame'] == first
    assert result['success'] == (first is not None)
    assert result['final_native_success'] == bool(native_final_success)
    completed = 0 if first is None else len(rows) - first - 1
    assert result['post_success_steps_completed'] == completed <= requested
    assert result['post_success_elapsed_seconds'] == completed / CONTROL_HZ
    assert result['post_success_complete'] == (first is not None and completed == requested)
    assert result['first_success_time_s'] == (None if first is None else (first + 1) / CONTROL_HZ)
    assert len(rows) <= result['max_steps'] + requested
    assert first is None or first < result['max_steps']
    for i, row in enumerate(rows):
        assert row['first_native_success_frame'] == (first if first is not None and i >= first else None)
        assert row['post_success_step'] == (i - first if first is not None and i >= first else None)
    if first is not None:
        assert result['termination'] == 'task_success'
        if completed < requested:
            assert rows[-1]['aborted']
            assert result['post_success_interrupted_reason'] == 'ik_failure_streak'
        else:
            assert result['post_success_interrupted_reason'] is None


def audit_first_success_state(env, pre_action_states, final_state, result):
    """Recompute first-hit native success at LIBERO's cached Euler sample time."""
    if not result.get('success') or 'post_success_seconds' not in result:
        return
    import mujoco
    import numpy as np
    first = result['first_success_frame']
    after = pre_action_states[first + 1] if first + 1 < len(pre_action_states) else final_state
    env.set_state(after)
    model, data = env.sim.model._model, env.sim.data._data
    assert int(model.opt.integrator) == 0
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    mujoco.mj_integratePos(model, data.qpos, qvel, -model.opt.timestep)
    roundtrip = data.qpos.copy()
    mujoco.mj_integratePos(model, roundtrip, qvel, model.opt.timestep)
    assert np.allclose(roundtrip, qpos, atol=1e-12, rtol=0)
    env.sim.forward()
    data.qpos[:] = qpos
    assert bool(env.check_success()), 'Recorded first-success state fails the native predicate'
