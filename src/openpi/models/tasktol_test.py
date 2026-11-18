import dataclasses

import flax.nnx as nnx
import flax.traverse_util as traverse
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client.ttc_action_chunk_broker import TTCActionChunkBroker
import pytest
import tyro

from openpi.models import model as model_api
from openpi.models import rtc
from openpi.models.rtc_test import small_pi05_config  # noqa: F401
from openpi.policies import policy_config
from openpi.policies.franka_mujoco_policy import FrankaMujocoOutputs
from openpi.policies.policy import Policy
from openpi.shared import nnx_utils
from openpi.training import config as training_config
from openpi.training import weight_loaders


@pytest.fixture
def dual(small_pi05_config):  # noqa: F811
    config = dataclasses.replace(
        small_pi05_config,
        action_dim=32,
        tolerance_tokens=True,
        action_loss_groups=(("action", 0, 7, 1.0), ("rotation_tolerance", 7, 13, 0.5)),
    )
    return config, config.create(jax.random.key(7)), config.fake_obs(2)


def test_noise_and_directional_attention_with_two_prefixes(dual):
    _, model, obs = dual
    # New adaRMS gates start at zero. Exercise actual cross-token attention.
    state = nnx.state(model)
    params = traverse.flatten_dict(state.to_pure_dict(), sep="/")
    (gate,) = [k for k in params if k.endswith("pre_attention_norm_1/Dense_0/bias")]
    params[gate] = jnp.ones_like(params[gate])
    state.replace_by_pure_dict(traverse.unflatten_dict(params, sep="/"))
    nnx.update(model, state)
    model.ttc = rtc.TTCConfig(enabled=True)
    key = jax.random.key(55)
    noise = model.sample_noise(key, (2, 5, 32))
    a_key, r_key = jax.random.split(key)
    np.testing.assert_array_equal(noise[..., :7], jax.random.normal(a_key, (2, 5, 7)))
    np.testing.assert_array_equal(noise[..., 7:13], jax.random.normal(r_key, (2, 5, 6)))
    lengths = jnp.array([0, 2])
    prior = model.sample_noise(jax.random.key(12), noise.shape)
    sample = nnx_utils.module_jit(model.sample_actions)

    def run(z, prefix):
        return sample(key, obs, num_steps=3, noise=z, action_prefix=prefix, prefix_lengths=lengths)

    baseline = run(noise, prior)
    changed_r = run(noise.at[..., 7:13].add(0.5), prior.at[..., 7:13].add(2.0))
    np.testing.assert_array_equal(changed_r[..., :7], baseline[..., :7])
    assert not np.allclose(changed_r[:, 2:, 7:13], baseline[:, 2:, 7:13])
    changed_a = run(noise.at[..., :7].add(1.0), prior)
    assert not np.allclose(changed_a[:, 2:, 7:13], baseline[:, 2:, 7:13])
    np.testing.assert_array_equal(baseline[1, :2], prior[1, :2])


@pytest.mark.parametrize("ttc", [False, True])
def test_dual_loss_gradients_weights_and_prefix_mask(dual, ttc):
    config, model, obs = dual
    obs = dataclasses.replace(obs, action_loss_weight=jnp.array([1.0, 4.0]))
    model.ttc = rtc.TTCConfig(enabled=ttc, target_delay_probabilities=(0.0, 0.0, 1.0))
    targets = config.fake_act(2)

    def objective(net):
        loss, metrics = net.compute_loss_with_breakdown(
            jax.random.key(5), obs, targets, train=True, training_step=80, total_training_steps=100
        )
        return loss.mean(), metrics

    (loss, metrics), grads = nnx.jit(nnx.value_and_grad(objective, has_aux=True))(model)
    assert np.isfinite(loss)
    for proj in (grads.action_in_proj, grads.action_out_proj, grads.tolerance_in_proj, grads.tolerance_out_proj):
        assert np.any(proj.kernel.value != 0)
    expected = metrics["loss/action"] * jnp.array([[1.0], [4.0]]) + 0.5 * metrics["loss/rotation_tolerance"]
    np.testing.assert_allclose(metrics["loss/total"], expected, rtol=1e-6)
    if ttc:
        np.testing.assert_array_equal(metrics["loss/total"][:, :2], 0)
    assert np.all(metrics["loss/total"][:, 2:] > 0)


def test_two_named_priors_and_robot_outputs(dual):
    _, model, obs = dual
    model.ttc = rtc.TTCConfig(enabled=True)
    policy = Policy(model, output_transforms=[FrankaMujocoOutputs(joint_task_tolerance=True)])
    observation = jax.tree.map(lambda x: np.asarray(x[0]), obs.to_dict())
    envelope = {"version": 1, "space": "model_normalized"}
    first = policy.infer({**observation, "_ttc": {**envelope, "delay": 0, "prefixes": {}}})
    assert policy.metadata["ttc"]["blocks"] == {"actions": 7, "rotation_tolerance": 6}
    prefixes = {name: np.roll(block, -2, axis=0) for name, block in first["ttc"]["blocks"].items()}
    second = policy.infer({**observation, "_ttc": {**envelope, "delay": 2, "prefixes": prefixes}})
    for name, prefix in prefixes.items():
        np.testing.assert_array_equal(second["ttc"]["blocks"][name][:2], prefix[:2])
        np.testing.assert_array_equal(second[name][:2], first[name][2:4])
    with pytest.raises(ValueError, match="prior blocks"):
        policy.infer({"_ttc": {**envelope, "delay": 2, "prefixes": {"actions": prefixes["actions"]}}})

    class RecordingClient:
        def __init__(self):
            self.calls = []

        def infer(self, obs):
            result = policy.infer(obs)
            self.calls.append((obs["_ttc"], result))
            return result

        def get_server_metadata(self):
            return policy.metadata

    client = RecordingClient()
    broker = TTCActionChunkBroker(client, replan_steps=2, delay_steps=2)
    try:
        steps = [broker.infer(observation) for _ in range(7)]
    finally:
        broker.close()
    assert len(client.calls) >= 3
    for i, (request, response) in enumerate(client.calls[1:], start=1):
        for name, prior in request["prefixes"].items():
            np.testing.assert_array_equal(prior[:2], client.calls[i - 1][1]["ttc"]["blocks"][name][2:4])
            if 2 * i + 2 < len(steps):
                np.testing.assert_array_equal(steps[2 * i + 2][name], response[name][2])


def test_checkpoint_init_and_serving_use_correct_projections(dual, tmp_path, monkeypatch):
    config, model, _ = dual
    weights = nnx.state(model).to_pure_dict()
    base = {k: v for k, v in weights.items() if not k.startswith("tolerance_")}
    monkeypatch.setattr(model_api, "restore_params", lambda *args, **kwargs: base)
    initialized = weight_loaders.CheckpointWeightLoader(str(tmp_path)).load(weights)
    jax.tree.map(np.testing.assert_array_equal, initialized, weights)

    preset = training_config.get_config("pi05_franka_mujoco_joint32")
    preset = dataclasses.replace(
        preset, model=dataclasses.replace(config, tolerance_tokens=False), data=training_config.FakeDataConfig()
    )
    monkeypatch.setattr(model_api, "restore_params", lambda *args, **kwargs: weights)
    policy = policy_config.create_trained_policy(preset, tmp_path, norm_stats={})
    restored = policy._model  # noqa: SLF001
    assert restored.block_dims == {"actions": 7, "rotation_tolerance": 6}
    jax.tree.map(np.testing.assert_array_equal, nnx.state(restored).to_pure_dict(), weights)


def test_training_cli_selects_architecture_independently_of_ttc():
    preset = training_config.get_config("pi05_franka_mujoco_joint32")
    options = {preset.name: ("TaskTol", preset)}
    for ttc in (False, True):
        args = [preset.name, "--exp-name", "smoke", "--model.tolerance-tokens"] + (
            ["--model.ttc.enabled"] if ttc else []
        )
        config = tyro.extras.overridable_config_cli(options, args=args)
        assert config.model.tolerance_tokens
        assert config.model.ttc.enabled == ttc
    assert not preset.model.tolerance_tokens
