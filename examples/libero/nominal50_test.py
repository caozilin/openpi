import pytest

from examples.libero.eval_nominal50 import SUITES, assignments
from examples.libero.nominal50_protocol import merge


def rows(seeds):
    return [dict(suite=s, task_id=t, seed=z, solver_failure_policy='hold_last_accepted_reference', success=False)
        for s in SUITES for t in range(10) for z in seeds]


def test_partition_covers_exactly1600_new_episodes_and_balances_six_replicas():
    groups = assignments()
    jobs = [j for group in groups.values() for j in group]
    keys = [(j['suite'], j['task_id'], j['seed']) for j in jobs]
    assert len(keys) == len(set(keys)) == 1600
    assert set(keys) == {(s, t, z) for s in SUITES for t in range(10) for z in range(10, 50)}
    assert sorted(map(len, groups.values())) == [266, 266, 267, 267, 267, 267]


def test_merge_keeps_failed_baseline_without_outcome_selection():
    old, new = rows(range(10)), rows(range(10, 50))
    for r in new:
        r['success'] = True
    merged = merge(old, new)
    assert len(merged) == 2000 and sum(r['success'] for r in merged) == 1600
    assert all(r is old[old.index(r)] for r in merged if r['seed'] < 10)
    for s in SUITES:
        for t in range(10):
            assert {r['seed'] for r in merged if r['suite'] == s and r['task_id'] == t} == set(range(50))


def test_merge_rejects_duplicate_missing_and_overlapping_seed():
    old, new = rows(range(10)), rows(range(10, 50))
    with pytest.raises(AssertionError):
        merge(old, new[:-1] + [new[0]])
    with pytest.raises(AssertionError):
        merge(old, new[:-1])
    with pytest.raises(AssertionError):
        merge(old, rows(range(9, 49)))
