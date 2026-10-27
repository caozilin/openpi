from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from rerecord_task_pose_batch import accepts
from rerecord_task_pose_reference import corrected_goals
import probe_original_speed_reference as projection


def test_acceptance_retains_success_and_rejects_infeasible_prefix():
    old = dict(success=True, objective=2.)
    trial = dict(success=True, objective=1., full_length=True, rejected=False)
    assert accepts(old, trial)
    for update in (dict(success=False), dict(rejected=True), dict(full_length=False), dict(objective=3.)):
        assert not accepts(old, {**trial, **update})


def test_projection_has_no_source_joint_penalty(monkeypatch):
    class Kinematics:
        def evaluate(self, q, **kwargs):
            return SimpleNamespace(position=np.asarray(q[:3]), rotation=np.eye(3))

    calls = []
    def solve(residual, seed, **kwargs):
        changed = seed.copy(); changed[6] += .2
        np.testing.assert_array_equal(residual(seed), residual(changed))
        calls.append(True)
        return SimpleNamespace(x=seed)
    monkeypatch.setattr(projection, 'least_squares', solve)
    base = dict(positions=np.zeros((6, 3)), rotations=np.tile(np.eye(3), (6, 1, 1)), q=np.zeros((6, 7)))
    handles = SimpleNamespace(joint_lower=np.full(7, -1.), joint_upper=np.full(7, 1.))
    actions = np.zeros((6, 7)); actions[:, 6] = [-1, -1, 1, 1, -1, 1]
    result = projection.make_reference(base, actions, handles, Kinematics(), 'preview', 1, 0, posture_weight=0.)
    assert len(calls) == 5
    np.testing.assert_array_equal(result['actions'][:, 6], actions[:5, 6])


def test_reference_update_is_bounded_and_keeps_timeline():
    count = 7
    ref = dict(actions=np.zeros((count, 7)), source_positions_m=np.zeros((count+1, 3)),
               source_rotations=np.tile(np.eye(3), (count+1, 1, 1)),
               positions=np.zeros((count+1, 3)), rotations=np.tile(np.eye(3), (count+1, 1, 1)))
    rows = [dict(actual_position_m=[.1, 0, 0], actual_rotation=np.eye(3), aborted=False) for _ in range(count)]
    p, r, correction_p, correction_r = corrected_goals(ref, rows, .5)
    assert p.shape == (count, 3) and r.shape == (count, 3, 3)
    np.testing.assert_allclose(np.linalg.norm(correction_p, axis=1), .005)
    np.testing.assert_array_equal(correction_r, np.zeros((count, 3)))
