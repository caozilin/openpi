import pytest

from examples.libero.eval_formal_nominal import assignments, SUITES
from examples.libero.summarize_formal_nominal import merge_formal


def test_all_first_six_seeds_rerun_regardless_of_previous_outcome():
    groups=assignments()
    assert len(groups)==6 and all(len(g)==40 for g in groups.values())
    actual={(r['suite'],r['task_id'],r['seed']) for g in groups.values() for r in g}
    assert len(actual)==240
    assert actual=={(s,t,z) for s in SUITES for t in range(10) for z in range(6)}
    assert all({r['seed'] for r in group}=={int(i)} for i,group in groups.items())


def test_formal_merge_retains_failures_and_rejects_selective_replacements_or_duplicate_trials():
    a=[dict(suite=s,task_id=t,seed=z,success=False,solver_failure_policy='hold_last_accepted_reference') for s in SUITES for t in range(10) for z in range(6)]
    b=[dict(suite=s,task_id=t,seed=z,success=True,solver_failure_policy='hold_last_accepted_reference') for s in SUITES for t in range(10) for z in range(6,10)]
    rows=merge_formal(a,b)
    assert len(rows)==400 and sum(r['success'] for r in rows)==160
    with pytest.raises(AssertionError):merge_formal(a[1:],b)
    with pytest.raises(AssertionError):merge_formal(a+[a[0]],b)
    with pytest.raises(AssertionError):merge_formal(a,b[:-1]+[a[0]])
    a[0]['solver_failure_policy']='stop_on_first_rejected_solve'
    with pytest.raises(AssertionError):merge_formal(a,b)
