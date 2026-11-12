import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
SPEC=importlib.util.spec_from_file_location('joint13_receipt_test',ROOT/'examples/libero/train_successmix20_joint13_pi05_server.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

@pytest.fixture
def receipt_case(tmp_path,monkeypatch):
    monkeypatch.setattr(MODULE,'DATA_HOME',tmp_path)
    request={'config':{'name':'test_joint13','model':{'action_dim':32,'action_horizon':10},'data':{'repo_id':'test/dataset'},'batch_size':72,'num_workers':8,'seed':42}}
    p=tmp_path/'assets/test_joint13/test/dataset/norm_stats.json';p.parent.mkdir(parents=True)
    stats={'norm_stats':{key:{'mean':[0]*width,'std':[1]*width,'q01':[0]*width,'q99':[1]*width} for key,width in [('state',8),('actions',13)]}}
    p.write_text(json.dumps(stats))
    return request,{'transfer_manifest_sha256':'frozen-dataset'},p

def test_only_loss_weight_is_excluded_from_norm_fingerprint(receipt_case):
    request,verify,p=receipt_case
    original=MODULE.normalization_receipt(p.parent,request,verify)
    changed=copy.deepcopy(request);changed['config']['data']['tolerance_trajectory_action_loss_weight']=4.0
    assert MODULE.normalization_receipt(p.parent,changed,verify)==original
    assert 'tolerance_trajectory_action_loss_weight' in changed['config']['data']

@pytest.mark.parametrize('field,value',[('seed',43),('batch_size',144),('num_workers',4)])
def test_other_norm_inputs_cannot_be_silently_reused(receipt_case,field,value):
    request,verify,p=receipt_case
    original=MODULE.normalization_receipt(p.parent,request,verify)
    changed=copy.deepcopy(request);changed['config'][field]=value
    assert MODULE.normalization_receipt(p.parent,changed,verify)['inputs_sha256']!=original['inputs_sha256']

def test_horizon_and_dataset_manifest_remain_bound(receipt_case):
    request,verify,p=receipt_case
    original=MODULE.normalization_receipt(p.parent,request,verify)
    changed=copy.deepcopy(request);changed['config']['model']['action_horizon']=20
    assert MODULE.normalization_receipt(p.parent,changed,verify)['inputs_sha256']!=original['inputs_sha256']
    assert MODULE.normalization_receipt(p.parent,request,{'transfer_manifest_sha256':'changed'})['inputs_sha256']!=original['inputs_sha256']

def test_wrong_dimensions_or_nonfinite_stats_cannot_be_reused(receipt_case):
    request,verify,p=receipt_case
    stats=json.loads(p.read_text());stats['norm_stats']['actions']['mean']=[0]*7;p.write_text(json.dumps(stats))
    with pytest.raises(AssertionError):MODULE.normalization_receipt(p.parent,request,verify)
    stats['norm_stats']['actions']['mean']=[float('nan')]*13;p.write_text(json.dumps(stats))
    with pytest.raises(AssertionError):MODULE.normalization_receipt(p.parent,request,verify)
