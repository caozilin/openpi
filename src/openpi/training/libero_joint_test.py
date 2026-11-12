import dataclasses
import jax
import numpy as np
import pytest
from openpi import transforms
from openpi.models import model, pi0_config
from openpi.policies.libero_joint_policy import LiberoJointInputs, LiberoJointOutputs
from openpi.shared import normalize
from openpi.training import config
from openpi.training.libero_joint import LeRobotLiberoJointDataConfig

def sample():
    return {'observation/image':np.zeros((3,16,16),np.float32),'observation/wrist_image':np.zeros((3,16,16),np.float32),
            'observation/state':np.arange(8,dtype=np.float32),'actions':np.arange(70,dtype=np.float32).reshape(10,7)/100,
            'rotation_tolerance':np.full((10,6),.2,np.float32),'trajectory_is_tolerance':np.array([0]),'prompt':'pick up bowl'}

def test_joint_targets_and_normalization_preserve_action7_state8_and_rad6():
    d=sample();x=LiberoJointInputs(model.ModelType.PI05)(d)
    np.testing.assert_array_equal(x['actions'][:,:7],d['actions']);np.testing.assert_array_equal(x['actions'][:,7:],d['rotation_tolerance'])
    stats={'actions':normalize.NormStats(mean=np.zeros(13),std=np.ones(13),q01=-np.ones(13),q99=np.ones(13))}
    packed=transforms.PadStatesAndActions(32)(transforms.Normalize(stats,use_quantiles=True)(x))
    assert packed['actions'].shape==(10,32);assert packed['state'].shape==(32,)
    np.testing.assert_array_equal(packed['actions'][:,13:],0)
    restored=LiberoJointOutputs()(transforms.Unnormalize(stats,use_quantiles=True)({'actions':packed['actions']}))
    np.testing.assert_allclose(restored['actions'],d['actions'],atol=1e-6);np.testing.assert_allclose(restored['rotation_tolerance'],d['rotation_tolerance'],atol=1e-6)

@pytest.mark.parametrize('bad',[np.full((10,6),np.nan),np.full((10,6),30.),np.full((10,6),-.01),np.zeros((9,6))])
def test_missing_wrong_units_and_misaligned_labels_rejected(bad):
    d=sample();d['rotation_tolerance']=bad
    with pytest.raises(ValueError):LiberoJointInputs(model.ModelType.PI05)(d)

@pytest.mark.parametrize('bad',[np.zeros(7),np.full(8,np.nan)])
def test_state8_contract_does_not_reuse_franka7state(bad):
    d=sample();d['observation/state']=bad
    with pytest.raises(ValueError):LiberoJointInputs(model.ModelType.PI05)(d)

def test_strict_phase_valid_zeros_are_preserved():
    d=sample();d['rotation_tolerance'][:]=0
    np.testing.assert_array_equal(LiberoJointInputs(model.ModelType.PI05)(d)['actions'][:,7:],0)

def test_config_sequences_both_physical_actions_and_tolerance(monkeypatch,tmp_path):
    monkeypatch.setattr(config.DataConfigFactory,'create_base_config',lambda *_:config.DataConfig())
    monkeypatch.setattr(config,'ModelTransformFactory',lambda:lambda _:transforms.Group())
    cfg=LeRobotLiberoJointDataConfig();m=pi0_config.Pi0Config(pi05=True,action_dim=32,action_horizon=10)
    created=cfg.create(tmp_path,m)
    assert created.action_sequence_keys==('actions','rotation_tolerance') and created.source_schema=='libero_joint13/1.0'
    with pytest.raises(ValueError,match='32-D'):cfg.create(tmp_path,dataclasses.replace(m,action_dim=13))


def test_default_tolerance_trajectory_keeps_nominal_targets_and_unit_weight():
    d=sample();d['trajectory_is_tolerance']=np.array([1])
    physical=d['actions'].copy();out=LiberoJointInputs(model.ModelType.PI05)(d)
    np.testing.assert_array_equal(out['actions'][:,:7],physical)
    np.testing.assert_array_equal(out['actions'][:,7:],d['rotation_tolerance'])
    assert out['action_loss_weight'] == 1.0


@pytest.mark.parametrize('flag,weight', [(0, 1.0), (1, 4.0)])
def test_four_x_uses_trajectory_flag_only_and_survives_normalize_pad_observation(flag, weight):
    d = sample(); d['trajectory_is_tolerance'] = np.array([flag])
    # Valid strict zero tolerance still gets4x when the observation came from a tolerance rollout.
    d['rotation_tolerance'][:] = 0
    output = LiberoJointInputs(model.ModelType.PI05, 4.0)(d)
    reference = LiberoJointInputs(model.ModelType.PI05, 1.0)(d)
    for key in ('state', 'actions'):
        np.testing.assert_array_equal(output[key], reference[key])
    stats = {'actions': normalize.NormStats(mean=np.zeros(13), std=np.ones(13), q01=-np.ones(13), q99=np.ones(13))}
    output = transforms.PadStatesAndActions(32)(transforms.Normalize(stats, use_quantiles=True)(output))
    assert output['actions'].shape == (10, 32)
    batched = jax.tree.map(lambda x: np.asarray(x)[None], {k: v for k, v in output.items() if k != 'prompt'})
    observation = model.Observation.from_dict(batched)
    assert float(observation.action_loss_weight[0]) == weight


@pytest.mark.parametrize('bad', [-1.0, np.nan, np.inf])
def test_nonfinite_or_negative_sample_weight_rejected(bad):
    with pytest.raises(ValueError, match='finite and non-negative'):
        LiberoJointInputs(model.ModelType.PI05, bad)(sample())
