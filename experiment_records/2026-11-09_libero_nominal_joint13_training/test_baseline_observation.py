"""Baseline observations cannot make rejected candidates admissible."""
import copy
import pytest
from tasktol.tolerance.clearance_policy import make_baseline, BaselineUnavailable, relative_feedback
from benchmark_joint_pre_post_t1_60 import result_outcome

def observation(complete=True):
 return {'termination_reason':'optimizer_infeasible','trajectory_kinematics':{'mean_joint_velocity_rad_s':1.},'evaluation':{'physics_steps':0,'trajectory_complete':False,'kinematic_feasible_without_table':False,'baseline_observation_only':True,'baseline_observation_complete':complete,'baseline_rejection_policy':'hold_last_accepted_joints','robot_table_clearance':{'distribution':{'minimum_m':-.002},'samples':[{'frame_index':0,'distance_m':.01},{'frame_index':1,'distance_m':-.002}]}}}

def test_complete_failed_baseline_allowed_only_explicitly():
 r=observation()
 with pytest.raises(BaselineUnavailable):make_baseline(r,[8,9])
 p=make_baseline(r,[8,9],require_feasible=False,epsilon_m=.0001)
 assert p['baseline_clearance_m']==[.01,-.002]
 assert p['required_clearance_m']==[.005,-.002]

@pytest.mark.parametrize('corruption',['truncated','nonfinite','unmarked','notcomplete','physics'])
def test_incomplete_or_unmarked_observation_cannot_fabricate_reference(corruption):
 r=observation();e=r['evaluation']
 if corruption=='truncated':e['robot_table_clearance']['samples'].pop()
 if corruption=='nonfinite':e['robot_table_clearance']['samples'][1]['distance_m']=float('nan')
 if corruption=='unmarked':e['baseline_observation_only']=False
 if corruption=='notcomplete':e['baseline_observation_complete']=False
 if corruption=='physics':e['physics_steps']=1
 with pytest.raises(ValueError):make_baseline(r,[8,9],require_feasible=False)

def test_failed_zero_candidate_stays_failed_even_with_complete_clearance():
 r=observation();p=make_baseline(r,[8,9],require_feasible=False)
 row=relative_feedback(result_outcome(r),r,p)
 assert not row['success'] and row['velocity_rad_s'] is None
 assert row['clearance_margin_m'] is None

def test_candidate_still_must_satisfy_relative_clearance():
 b=observation();p=make_baseline(b,[8,9],require_feasible=False,epsilon_m=.0001)
 r=copy.deepcopy(b);e=r['evaluation'];e.update(trajectory_complete=True,kinematic_feasible_without_table=True,baseline_observation_only=False)
 e['robot_table_clearance']['samples'][1]['distance_m']=-.0022
 e['robot_table_clearance']['distribution']['minimum_m']=-.0022
 row=relative_feedback(result_outcome(r),r,p)
 assert not row['success'] and row['clearance_margin_m']<0
