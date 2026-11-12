import pytest
from examples.libero.summarize_nominal10 import SUITES,merge_views


def test_retries_replace_original_keys_and_never_become_417_trials():
    original=[dict(suite=s,task_id=t,seed=z,success=(s,t,z)!=('libero_spatial',4,2)) for s in SUITES for t in range(10) for z in range(6)]
    extension=[dict(suite=s,task_id=t,seed=z,success=True) for s in SUITES for t in range(10) for z in range(6,10)]
    retry=[dict(suite='libero_spatial',task_id=4,seed=2,success=True)]
    historical,latest=merge_views(original,retry,extension)
    assert len(historical)==len(latest)==400
    assert sum(r['success'] for r in historical)==399
    assert sum(r['success'] for r in latest)==400
    with pytest.raises(AssertionError):merge_views(original,retry,extension+extension[:1])
    with pytest.raises(AssertionError):merge_views(original,[],extension)
