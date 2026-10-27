from copy import deepcopy

import numpy as np
import pytest

from experiments.libero_tasktol_compat.paired_eef_metrics import aligned_eef_metrics, aligned_tracking_metrics


def trajectories():
    identity=np.eye(3).tolist()
    left=[dict(pre_q=[i],post_q=[i+1],actual_position_m=[i+1,0,0],
               actual_tool_rotation=identity,physics_steps=25) for i in range(2)]
    positions=[0,0.999,1.997]
    rotations=[[0,0,0],[0,0,np.pi/2],[0,0,np.pi/2]]
    right=[dict(
        pre_action=dict(position_m=[positions[i],0,0],rotation_vector_rad=rotations[i]),
        post_action=dict(position_m=[positions[i+1],0,0],rotation_vector_rad=rotations[i+1]),
        aborted=False,physics_steps=25,
    ) for i in range(2)]
    return left,right


def test_metrics_match_pre_action_video_and_actual_terminal_picture():
    left,right=trajectories()
    rows=aligned_eef_metrics(left,right,np.zeros(3),np.eye(3),tail=2)
    assert rows[0]['actual_position_difference_mm']==0
    assert rows[1]['actual_position_difference_mm']==pytest.approx(1)
    assert rows[1]['actual_rotation_difference_deg']==pytest.approx(90)
    assert rows[2]['actual_position_difference_mm']==pytest.approx(3)
    assert rows[2]['comparison_kind']=='synchronized_final'
    assert rows[2]['left_pose_time_s']==rows[2]['right_pose_time_s']==0.1
    assert rows[3]['actual_position_difference_mm']==rows[2]['actual_position_difference_mm']


def test_aborted_tail_is_a_held_picture_not_a_synchronous_state():
    left,right=trajectories()
    right[-1]['aborted']=True
    right[-1]['physics_steps']=0
    right[-1]['post_action']=deepcopy(right[-1]['pre_action'])
    rows=aligned_eef_metrics(left,right,np.zeros(3),np.eye(3),tail=2)
    assert rows[1]['synchronized']
    assert rows[1]['actual_position_difference_mm']==pytest.approx(1)
    assert not rows[2]['synchronized']
    assert rows[2]['comparison_kind']=='held_after_ipopt_abort'
    assert rows[2]['left_pose_time_s']==0.1
    assert rows[2]['right_pose_time_s']==0.05
    assert rows[2]['actual_position_difference_mm']==pytest.approx(1001)


def test_unexplained_short_trajectory_is_rejected():
    left,right=trajectories()
    with pytest.raises(ValueError,match='without recorded abort'):
        aligned_eef_metrics(left,right[:1],np.zeros(3),np.eye(3))


def reference_trajectories():
    left,right=trajectories()
    # Distinct current/upcoming goals expose accidental use of the upcoming
    # command with an old video picture. Right and left also have OWN goals.
    for i,row in enumerate(left):
        row.update(native_goal_position_m=[10*(i+1),0,0],native_goal_rotation=np.eye(3).tolist())
    for i,row in enumerate(right):
        row['nominal_target']=dict(position_m=[20*(i+1),0,0],rotation_vector_rad=[0,0,np.pi/4])
    return left,right


def test_six_dimensional_errors_use_own_already_issued_reference():
    left,right=reference_trajectories()
    rows=aligned_tracking_metrics(left,right,np.zeros(3),np.eye(3),tail=2)
    np.testing.assert_allclose(rows[0]['left_tracking_error_6d'],np.zeros(6),atol=1e-12)
    np.testing.assert_allclose(rows[0]['right_tracking_error_6d'],np.zeros(6),atol=1e-12)
    np.testing.assert_allclose(rows[1]['left_tracking_error_6d'],[-9000,0,0,0,0,0],atol=1e-10)
    np.testing.assert_allclose(rows[1]['right_tracking_error_6d'],[-19001,0,0,0,0,45],atol=1e-10)
    assert rows[1]['left_reference_index']==rows[1]['right_reference_index']==0
    np.testing.assert_allclose(rows[2]['left_tracking_error_6d'],[-18000,0,0,0,0,0],atol=1e-10)


def test_rejected_ipopt_target_never_replaces_last_issued_reference():
    left,right=reference_trajectories()
    right[-1].update(aborted=True,physics_steps=0)
    right[-1]['post_action']=deepcopy(right[-1]['pre_action'])
    rows=aligned_tracking_metrics(left,right,np.zeros(3),np.eye(3),tail=2)
    assert rows[2]['right_reference_index']==0
    assert rows[2]['left_reference_index']==1
    np.testing.assert_allclose(rows[2]['right_tracking_error_6d'],rows[1]['right_tracking_error_6d'],atol=1e-12)
