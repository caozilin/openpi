import json
from types import SimpleNamespace

import h5py
import numpy as np

from experiments.libero_tasktol_compat.reconstruct_joint_servo_dataset import ObservationWriter,select_candidate


def test_success_recovery_precedes_score_and_never_accepts_partial():
    previous=dict(success=False,full_length=True,rejected=False,objective=.1)
    recovered=dict(success=True,full_length=True,rejected=False,objective=10.)
    assert select_candidate(previous,recovered)
    assert not select_candidate(recovered,previous)
    assert not select_candidate(None,{**recovered,'full_length':False})
    assert not select_candidate(None,{**recovered,'rejected':True})
    assert select_candidate(recovered,{**recovered,'objective':1.})


def test_observer_keeps_native_images_and_pre_action_alignment(tmp_path):
    image=np.arange(4*4*3,dtype=np.uint8).reshape(4,4,3)
    obs=dict(agentview_image=image,robot0_eye_in_hand_image=image+1,
        robot0_gripper_qpos=np.array([.04,-.04]),robot0_joint_pos=np.zeros(7),
        robot0_eef_pos=np.array([1.,2.,3.]),robot0_eef_quat=np.array([0.,0.,0.,1.]))
    state=np.arange(20,dtype=float)
    sim=SimpleNamespace(get_state=lambda:SimpleNamespace(flatten=lambda:state),
        model=SimpleNamespace(get_xml=lambda:'<mujoco/>'))
    writer=ObservationWriter(tmp_path/'recording.hdf5','demo_0')
    writer(0,np.zeros(7),obs,SimpleNamespace(sim=sim));writer.close()
    with h5py.File(tmp_path/'recording.hdf5')as f:
        g=f['data/demo_0']
        np.testing.assert_array_equal(g['obs/agentview_rgb'][0],image)
        np.testing.assert_array_equal(g['obs/eye_in_hand_rgb'][0],image+1)
        np.testing.assert_array_equal(g['states'][0],state)
        np.testing.assert_array_equal(g['obs/ee_states'][0],[1,2,3,0,0,0])
