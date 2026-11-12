import math
import pytest
from experiments.libero_tasktol_compat.annotated_tolerance_policy import label_policy

def labels(bounds=None):
 return dict(control_hz=20,frames=[dict(source_frame_index=0,phase='pregrasp',subtask_index=2,
 rotation_tolerance_rad=bounds if bounds is not None else [0,0,.1,.2,0,0])])

def test_uses_signed_values_and_original_subtask():
 p=label_policy(labels())(0,None,None)
 assert p['semantic_mask']=='010' and p['bounds_rad']==[0,0,.1,.2,0,0]
 assert p['phase']=='APP' and p['label_subtask_index']==2
 assert p['label_source_frame_index']==0

def test_strict_zero_preserved():
 assert label_policy(labels([0]*6))(0,None,None)['semantic_mask']=='000'

@pytest.mark.parametrize('bounds',[[0]*5,[-1]*6,[math.nan]*6,[1]*6])
def test_invalid_directional_values_rejected(bounds):
 with pytest.raises(ValueError):label_policy(labels(bounds))

def test_gap_rejected():
 l=labels();l['frames'][0]['source_frame_index']=1
 with pytest.raises(ValueError):label_policy(l)

def test_strict_phase_cannot_release():
 l=labels();l['frames'][0]['phase']='grasp'
 with pytest.raises(ValueError):label_policy(l)
