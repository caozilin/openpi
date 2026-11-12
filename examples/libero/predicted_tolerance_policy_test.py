from types import SimpleNamespace
import numpy as np
import pytest
from examples.libero.predicted_tolerance_policy import decode_prediction, PredictedTolerancePolicy, MAX_BOUND_RAD

class Classifier:
    def __init__(self, phase): self.phase = phase
    def update(self, fingers, closed):
        return SimpleNamespace(phase=SimpleNamespace(key=self.phase),subtask_index=2,aperture_m=.03,motion_m=0.)


def test_pairing_and_units():
    a=np.arange(70,dtype=float).reshape(10,7)/1000;a[:,6]=np.linspace(-1,1,10)
    b=np.arange(60,dtype=float).reshape(10,6)/100
    pairs=decode_prediction({'actions':a,'rotation_tolerance':b})
    assert len(pairs)==5
    for i,(decoded,bounds,raw) in enumerate(pairs):
        np.testing.assert_array_equal(decoded[:6],a[i,:6]);np.testing.assert_array_equal(raw,a[i])
        np.testing.assert_array_equal(bounds,b[i]);assert decoded[6] in [-1.,0.,1.]

@pytest.mark.parametrize('shape',[(10,5),(5,6)])
def test_reject_bad_shape(shape):
    with pytest.raises(ValueError):decode_prediction({'actions':np.zeros((10,7)),'rotation_tolerance':np.zeros(shape)})


def test_nonfinite_is_infrastructure_error():
    b=np.zeros((10,6));b[3,2]=np.nan
    with pytest.raises(ValueError):decode_prediction({'actions':np.zeros((10,7)),'rotation_tolerance':b})


def test_signed_order_clipping_and_no_task_table():
    p=PredictedTolerancePolicy(Classifier('pregrasp'))
    out=p.resolve([.02,-.02],[0,0,0,0,0,0,-1],[-.1,.1,0,0,.2,2])
    np.testing.assert_allclose(out['bounds_rad'],[0,.1,0,0,.2,MAX_BOUND_RAD])
    assert out['semantic_mask']=='101' and out['subtask_index']==2
    assert out['predicted_bounds_raw_rad']==[-.1,.1,0,0,.2,2] and out['predicted_bounds_clipped']

@pytest.mark.parametrize('phase',['grasp','release'])
def test_existing_phase_guard(phase):
    out=PredictedTolerancePolicy(Classifier(phase)).resolve([.02,-.02],[0]*7,[.1]*6)
    assert out['semantic_mask']=='000' and out['bounds_rad']==[.1]*6
