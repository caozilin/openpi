import dataclasses

import jax
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.policies import franka_mujoco_policy
from openpi.shared import normalize
from openpi.training import config
from openpi.training import weight_loaders
from scripts import train


def test_all_franka_configs_keep_pretrained_32_dimensional_interface():
    for joint in config._CONFIGS:
        if not isinstance(joint.data, config.LeRobotFrankaMujocoDataConfig):
            continue
        assert joint.model.action_dim == 32
        assert isinstance(joint.weight_loader, weight_loaders.CheckpointWeightLoader)
        _, action_spec = joint.model.inputs_spec(batch_size=2)
        assert action_spec.shape == (2, joint.model.action_horizon, 32)
    for name in ('pi05_franka_mujoco_joint13', 'pi05_franka_mujoco_joint13_state', 'pi05_tasktol_mix20_4xloss_7gpu_9k'):
        with pytest.raises(ValueError, match='not found'):
            config.get_config(name)


@pytest.mark.parametrize('dimension', [7, 13, 15, 19])
def test_franka_cli_model_dimension_override_cannot_reinitialize_heads(tmp_path, dimension):
    joint = config.get_config('pi05_franka_mujoco_joint32')
    with pytest.raises(ValueError, match='pretrained 32-D'):
        joint.data.create(tmp_path, dataclasses.replace(joint.model, action_dim=dimension, action_loss_groups=None))


def test_pretrained_action_projection_values_are_loaded(monkeypatch):
    loaded = {
        'action_in_proj': {'kernel': np.full((32, 1024), 3., np.float32)},
        'action_out_proj': {'kernel': np.full((1024, 32), 4., np.float32), 'bias': np.full(32, 5., np.float32)},
    }
    reference = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), loaded)
    monkeypatch.setattr(model_lib, 'restore_params', lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(weight_loaders.download, 'maybe_download', lambda path: path)
    result = train._load_weights_and_validate(weight_loaders.CheckpointWeightLoader('fixture'), reference)
    for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(loaded), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_native13_checkpoint_is_rejected_instead_of_silently_reinitialized(monkeypatch):
    loaded = {'action_out_proj': {'kernel': np.ones((1024, 13), np.float32)}}
    reference = {'action_out_proj': {'kernel': jax.ShapeDtypeStruct((1024, 32), np.float32)}}
    monkeypatch.setattr(model_lib, 'restore_params', lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(weight_loaders.download, 'maybe_download', lambda path: path)
    with pytest.raises(ValueError, match='Shape mismatch'):
        train._load_weights_and_validate(weight_loaders.CheckpointWeightLoader('fixture'), reference)


def test_effective_targets_normalize_then_pad_and_outputs_ignore_padding():
    physical = np.arange(14, dtype=np.float32).reshape(2, 7) / 20
    tolerance = np.full((2, 6), .2, np.float32)
    packed = franka_mujoco_policy.FrankaMujocoInputs(model_type=model_lib.ModelType.PI05, joint_task_tolerance=True)({
        'observation/image': np.zeros((20, 30, 3), np.uint8),
        'observation/wrist_image': np.zeros((20, 30, 3), np.uint8),
        'observation/state': np.zeros(7, np.float32),
        'actions': physical, 'rotation_tolerance': tolerance,
    })
    stats = {'actions': normalize.NormStats(mean=np.zeros(13), std=np.ones(13), q01=-np.ones(13), q99=np.ones(13))}
    packed = transforms.Normalize(stats, use_quantiles=True)(packed)
    packed = transforms.PadStatesAndActions(32)(packed)
    assert packed['state'].shape == (32,)
    assert packed['actions'].shape == (2, 32)
    np.testing.assert_array_equal(packed['actions'][:, 13:], 0)
    result = transforms.Unnormalize(stats, use_quantiles=True)({'actions': packed['actions']})
    result = franka_mujoco_policy.FrankaMujocoOutputs(joint_task_tolerance=True)(result)
    np.testing.assert_allclose(result['actions'], physical, atol=1e-6)
    np.testing.assert_allclose(result['rotation_tolerance'], tolerance, atol=1e-6)


def test_official_style_mean_counts_padding_but_existing_groups_ignore_it():
    squared_error = jax.numpy.concatenate([jax.numpy.zeros((1, 1, 13)), jax.numpy.ones((1, 1, 19))], axis=-1)
    full, _ = pi0._grouped_action_loss_with_breakdown(squared_error, None)
    grouped, _ = pi0._grouped_action_loss_with_breakdown(squared_error, (('action', 0, 7, 1.), ('rotation_tolerance', 7, 13, .5)))
    assert float(full[0, 0]) == pytest.approx(19 / 32)
    assert float(grouped[0, 0]) == 0


@pytest.mark.parametrize('field', ['mean', 'std', 'q01', 'q99'])
@pytest.mark.parametrize('dimension', [7, 15, 32])
def test_joint32_still_requires_13_dimensional_effective_statistics(monkeypatch, tmp_path, field, dimension):
    fields = {name: np.ones(13) for name in ('mean', 'std', 'q01', 'q99')}
    fields[field] = np.ones(dimension)
    monkeypatch.setattr(config.LeRobotFrankaMujocoDataConfig, 'create_base_config',
                        lambda *_: config.DataConfig(norm_stats={'actions': normalize.NormStats(**fields)}))
    joint = config.get_config('pi05_franka_mujoco_joint32')
    with pytest.raises(ValueError, match='Recompute normalization statistics'):
        joint.data.create(tmp_path, joint.model)


def test_action_only_config_sequences_no_unlabeled_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(config.LeRobotFrankaMujocoDataConfig, 'create_base_config', lambda *_: config.DataConfig())
    monkeypatch.setattr(config, 'ModelTransformFactory', lambda: lambda _: transforms.Group())
    baseline = config.get_config('pi05_franka_mujoco')
    data = baseline.data.create(tmp_path, baseline.model)
    assert data.action_sequence_keys == ('actions',)
    assert data.source_schema == 'tasktol_mujoco/8.0'
