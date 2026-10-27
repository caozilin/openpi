import numpy as np

from experiments.libero_tasktol_compat.reconstruct_joint_servo import accepts, corrected


def test_partial_rejection_uses_only_executed_post_states():
    count=8
    ref=dict(actions=np.zeros((count,7)),source_positions_m=np.zeros((count+1,3)),
             source_rotations=np.tile(np.eye(3),(count+1,1,1)))
    rows=[dict(aborted=False,post_action=dict(position_m=[.1,0,0],rotation_vector_rad=[0,0,0])) for _ in range(4)]
    # A failed solve has no executed post-state and must not enter the error.
    rows.append(dict(aborted=True,post_action=None))
    p,r,measured=corrected(ref,rows,1.)
    assert measured==4 and p.shape==r.shape==(count,3)
    np.testing.assert_allclose(p,np.tile([-.02,0,0],(count,1)))
    np.testing.assert_array_equal(r,np.zeros((count,3)))


def test_selection_preserves_success_and_demands_full_execution():
    old=dict(success=True,objective=10.)
    candidate=dict(success=True,objective=1.,full_length=True,rejected=False)
    assert accepts(old,candidate)
    for update in [dict(success=False),dict(full_length=False),dict(rejected=True),dict(objective=11.)]:
        assert not accepts(old,{**candidate,**update})
