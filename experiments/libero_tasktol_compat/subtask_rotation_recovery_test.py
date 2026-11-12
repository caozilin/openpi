from types import SimpleNamespace
import numpy as np
from experiments.libero_tasktol_compat.subtask_rotation_recovery import RotationRecovery,RecoverySettings,exp,log

def test_thirty_degrees_recovers_in_twenty_frames_without_translation_or_gripper_changes():
    r=RotationRecovery(np.eye(3),[1]);current=exp([0,np.deg2rad(30),0]);r.start(0,1,current)
    assert r.events[0]['planned_frames']==20
    for _ in range(20):
        source=np.array([.001,-.002,.003,0,0,0,-1.])
        action,detail=r.advance(source,current);current=exp(action[3:6])@current
        np.testing.assert_array_equal(action[:3],source[:3]);assert action[6]==-1
        assert np.linalg.norm(action[3:6])<=np.deg2rad(45)*.05+1e-10
    np.testing.assert_allclose(current,np.eye(3),atol=1e-12)
    assert r.events[0]['completed']

def test_noncommuting_nominal_motion_is_composed_not_added_as_euler_angles():
    r=RotationRecovery(exp([.2,-.1,.3]),[1]);current=r.original_rotation@exp([.25,.15,-.2]);r.start(0,1,current)
    for _ in range(200):
        source=np.r_[np.zeros(3),[.006,-.004,.005],-1.]
        action,detail=r.advance(source,current);current=exp(action[3:6])@current
        if r.active is None:break
        assert np.linalg.norm(action[3:6])<=np.deg2rad(45)*.05+1e-9
    assert r.active is None
    np.testing.assert_allclose(current,r.original_rotation,atol=2e-12)

def test_overspeed_source_rotation_is_preserved_and_recovery_paused():
    r=RotationRecovery(np.eye(3),[1]);current=exp([0,.5,0]);r.start(0,1,current)
    source=np.array([0,0,0,.1,0,0,-1.]);action,detail=r.advance(source,current)
    np.testing.assert_allclose(action,source,atol=1e-15)
    assert detail['progress']==0 and r.events[0]['source_over_speed_frames']==1

def test_next_grasp_interrupts_recovery_without_retiming():
    r=RotationRecovery(np.eye(3),[1]);current=exp([0,.5,0]);r.start(0,1,current)
    session=SimpleNamespace(target=SimpleNamespace(rotation=current),release=SimpleNamespace(accepted_rotation=current))
    phase=SimpleNamespace(subtask_index=1,phase=SimpleNamespace(key='grasp'),aperture_m=.07)
    source=np.array([0,0,0,0,0,0,1.]);action,info=r(0,source,phase,dict(phase='ENG',semantic_mask='000'),session)
    np.testing.assert_array_equal(action,source)
    assert r.active is None and not r.events[0]['completed']
    assert r.events[0]['interrupted_by']=='grasp'

def test_single_subtask_does_not_trigger_recovery():
    r=RotationRecovery(np.eye(3),[]);r.had_execution=True
    current=exp([0,.5,0]);session=SimpleNamespace(target=SimpleNamespace(rotation=current),release=SimpleNamespace(accepted_rotation=current))
    phase=SimpleNamespace(subtask_index=1,phase=SimpleNamespace(key='pregrasp'),aperture_m=.08)
    source=np.array([0,0,0,.001,.002,0,-1.]);action,info=r(0,source,phase,dict(phase='outside_subtask',semantic_mask='000'),session)
    np.testing.assert_array_equal(action,source);assert not r.events
