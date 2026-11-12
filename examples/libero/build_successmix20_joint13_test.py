import copy,random
import pytest
from examples.libero.build_successmix20_joint13 import select_per_task

def pool():
    return [dict(suite=s,task_id=t,demo=f'demo_{i}',success=True,execution_rejected=False) for s in ['libero_spatial','libero_object','libero_goal','libero_10'] for t in range(10) for i in range(12+t)]

def test_per_task_rounding_denominator_and_sampling_without_replacement():
    chosen,tasks=select_per_task(pool(),42)
    for task in tasks:
        assert task['selected']==round(task['candidates']*.2)
        assert len(set(task['selected_demos']))==task['selected']
        assert set(task['selected_demos'])<=set(task['candidate_demos'])
    assert len(chosen)==sum(t['selected'] for t in tasks)

def test_selection_is_reproducible_independent_of_source_completion_order():
    p=pool();first=select_per_task(p,42);random.Random(123).shuffle(p)
    assert select_per_task(p,42)==first
    assert select_per_task(p,43)!=first

def test_failed_or_duplicate_pool_entries_cannot_be_selected():
    p=pool();p[0]['success']=False
    with pytest.raises(AssertionError):select_per_task(p,42)
    p=pool();p.append(copy.deepcopy(p[0]))
    with pytest.raises(ValueError,match='Duplicate'):select_per_task(p,42)
