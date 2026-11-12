"""Semantic assignment regressions without physics or IPOPT solves."""
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.libero_tasktol_compat.rollout_confirmed_tolerance400 import confirmed_masks,make_phase_policy

ROOT=Path(__file__).resolve().parents[2]

def test_user_confirmed_changes_and_merged_goal03():
    source=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/mask_annotations.json').read_text())
    masks=confirmed_masks(source)['tasks']
    def values(i):return [(s['APP'],s['EXEC'])for s in masks[i-1]['subtasks']]
    assert values(15)==[('011','111')]
    assert values(24)==[('010','000')]
    assert values(26)==[('000','000')]
    assert values(30)==[('011','000')]
    assert values(33)==[('010','000'),('010','000')]
    assert values(34)==[('010','000'),('000','000')]
    assert values(39)==[('010','000'),('010','000')]
    assert values(40)==[('010','000'),('000','000')]

def test_regrasp_identity_and_live_transition_gating():
    subtasks=[dict(APP='011',ENG='000',EXEC='111',DIS='000'),
              dict(APP='010',ENG='000',EXEC='001',DIS='000')]
    policy=make_phase_policy(subtasks,[0,1,2,3],[0,1,1])
    def obs(key):return SimpleNamespace(phase=SimpleNamespace(key=key))
    assert policy(2,obs('postgrasp'),None)['semantic_mask']=='001'
    assert policy(2,obs('grasp'),None)['semantic_mask']=='000'
    assert policy(2,obs('release'),None)['phase']=='DIS'
    assert policy(3,obs('pregrasp'),None)['phase']=='outside_subtask'

def test_goal03_entire_open_approach_remains_pre_until_live_bowl_grasp():
    policy=make_phase_policy([dict(APP='010',ENG='000',EXEC='000',DIS='000')],[0]*4,[0])
    for index,key in enumerate(['pregrasp','grasp','postgrasp','release']):
        result=policy(index,SimpleNamespace(phase=SimpleNamespace(key=key)),None)
        assert result['semantic_mask']==('010' if index==0 else '000')
