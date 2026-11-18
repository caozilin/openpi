import dataclasses

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import rtc
from openpi.shared import nnx_utils


@pytest.mark.parametrize(
    ("step", "alpha", "expected"),
    [
        (0, 0, [1, 0, 0, 0]),
        (19, 0, [1, 0, 0, 0]),
        (20, 0, [1, 0, 0, 0]),
        (40, 0.5, [0.55, 0.025, 0.4, 0.025]),
        (60, 1, [0.1, 0.05, 0.8, 0.05]),
        (99, 1, [0.1, 0.05, 0.8, 0.05]),
    ],
)
def test_curriculum(step, alpha, expected):
    config = rtc.TTCConfig()
    actual = jax.jit(lambda t: rtc.curriculum_alpha(t, 100, config))(step)
    assert float(actual) == pytest.approx(alpha)
    np.testing.assert_allclose(rtc.delay_probabilities(actual, config), expected, atol=1e-7)


def test_delay_sampling_matches_target_and_warmup():
    config = rtc.TTCConfig()
    sample = jax.jit(lambda key, alpha: rtc.sample_delays(key, (100_000,), rtc.delay_probabilities(alpha, config)))
    assert np.all(sample(jax.random.key(42), 0.0) == 0)
    delays = np.asarray(sample(jax.random.key(42), 1.0))
    np.testing.assert_allclose(np.bincount(delays, minlength=4) / len(delays), [0.1, 0.05, 0.8, 0.05], atol=0.005)


@pytest.mark.parametrize("probabilities", [(), (0.2, 0.2), (-0.1, 1.1), (float("nan"), 1)])
def test_invalid_distribution(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        rtc.TTCConfig(target_delay_probabilities=probabilities)


def test_config_guards():
    with pytest.raises(ValueError, match="TTC requires"):
        pi0_config.Pi0Config(ttc=rtc.TTCConfig(enabled=True))
    with pytest.raises(ValueError, match="TTC requires"):
        pi0_config.Pi0Config(pi05=True, action_horizon=3, ttc=rtc.TTCConfig(enabled=True))
    with pytest.raises(ValueError, match="fraction"):
        rtc.TTCConfig(warmup_fraction=0.6, ramp_end_fraction=0.2)


def test_independent_blocks_share_prefix_lengths_without_sharing_noise_or_time():
    lengths = jnp.asarray([0, 2, 3])
    for dim, time in [(7, 0.25), (6, 0.75)]:
        actions = jnp.arange(3 * 5 * dim, dtype=jnp.float32).reshape(3, 5, dim)
        noise = jnp.full_like(actions, -1)
        x, velocity, token_time, valid = jax.jit(rtc.flow_inputs)(actions, noise, jnp.full((3,), time), lengths)
        mask = np.arange(5)[None] < np.asarray(lengths)[:, None]
        np.testing.assert_array_equal(x[mask], actions[mask])
        np.testing.assert_allclose(x[~mask], (time * noise + (1 - time) * actions)[~mask])
        np.testing.assert_array_equal(token_time[mask], 0)
        np.testing.assert_array_equal(valid, ~mask)
        np.testing.assert_array_equal(velocity, noise - actions)


def test_valid_token_loss_and_gradients_keep_existing_group_weights():
    errors = jnp.ones((2, 4, 13))
    valid = jnp.arange(4)[None] >= jnp.asarray([0, 2])[:, None]
    groups = (("action", 0, 7, 1.0), ("rotation_tolerance", 7, 13, 0.5))

    def objective(values):
        loss, _ = pi0._grouped_action_loss_with_breakdown(values**2, groups, jnp.asarray([1.0, 4.0]))  # noqa: SLF001
        return rtc.mask_loss(loss, valid).mean()

    assert float(jax.jit(objective)(errors)) == pytest.approx((4 * 1.5 + 2 * 4.5) / 6)
    gradients = jax.jit(jax.grad(objective))(errors)
    np.testing.assert_array_equal(gradients[1, :2], 0)
    assert float(gradients[1, 2, 0] / gradients[0, 0, 0]) == pytest.approx(4)
    assert float(gradients[1, 2, 7] / gradients[0, 0, 7]) == pytest.approx(1)


class TinyImage(nn.Module):
    width: int

    @nn.compact
    def __call__(self, image, *, train=False):
        del train
        return nn.Dense(self.width)(jnp.mean(image, axis=(1, 2))[:, None]), None


@pytest.fixture
def small_pi05_config(monkeypatch):
    # Keep actual pi05, Gemma attention, time MLP, adaRMS, and cache code.
    # Only reduce dimensions and replace the image tower; no external weights.
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    monkeypatch.setattr(gemma, "get_config", lambda variant: gemma.Config(32, 1, 64, 2, 1, 16))
    monkeypatch.setattr(pi0._siglip, "Module", lambda **kwargs: TinyImage(kwargs["num_classes"]))  # noqa: SLF001
    return pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=4,
        action_horizon=5,
        max_token_len=4,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )


@pytest.fixture
def small_pi05(small_pi05_config):
    config = small_pi05_config
    model = config.create(jax.random.key(7))
    obs = config.fake_obs(2)
    obs = dataclasses.replace(obs, tokenized_prompt=jnp.ones((2, 4), jnp.int32))
    actions = jax.random.normal(jax.random.key(8), (2, 5, 4))
    return model, obs, actions


def test_pi05_warmup_is_ordinary_training_with_same_rng_and_parameters(small_pi05):
    model, obs, actions = small_pi05
    key = jax.random.key(42)

    def run(act):
        return model.compute_loss(key, obs, act, train=True, training_step=10, total_training_steps=100).mean()

    baseline, baseline_grad = jax.jit(jax.value_and_grad(run))(actions)
    params_before = nnx.state(model).to_pure_dict()
    model.ttc = rtc.TTCConfig(enabled=True)
    warmup, warmup_grad = jax.jit(jax.value_and_grad(run))(actions)
    np.testing.assert_array_equal(warmup, baseline)
    # The conditional changes XLA fusion; FP32 backward may round differently.
    np.testing.assert_allclose(warmup_grad, baseline_grad, rtol=1e-6, atol=1e-8)
    jax.tree.map(np.testing.assert_array_equal, params_before, nnx.state(model).to_pure_dict())


def test_pi05_ttc_jit_loss_parameter_gradient_and_sampling(small_pi05):
    model, obs, actions = small_pi05
    model.ttc = rtc.TTCConfig(enabled=True)

    def objective(net):
        loss, info = net.compute_loss_with_breakdown(
            jax.random.key(42), obs, actions, train=True, training_step=80, total_training_steps=100
        )
        return loss.mean(), info

    (value, info), grads = nnx.jit(nnx.value_and_grad(objective, has_aux=True))(model)
    assert np.isfinite(value)
    assert float(info["ttc/alpha"]) == 1
    assert np.any(grads.action_out_proj.kernel.value != 0)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    prefix = jnp.full_like(actions, 0.25)
    lengths = jnp.asarray([0, 3], jnp.int32)
    sample = nnx_utils.module_jit(model.sample_actions)
    generated = sample(jax.random.key(5), obs, action_prefix=prefix, prefix_lengths=lengths, num_steps=3)
    np.testing.assert_array_equal(generated[1, :3], prefix[1, :3])
    assert np.isfinite(generated).all()
    assert not np.all(generated[1, 3:] == 0.25)
    with pytest.raises(ValueError, match="requires training_step"):
        model.compute_loss(jax.random.key(42), obs, actions, train=True)


def test_sample_without_prefix_unchanged_when_ttc_enabled(small_pi05):
    model, obs, _ = small_pi05
    key = jax.random.key(42)
    old = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=3)
    model.ttc = rtc.TTCConfig(enabled=True)
    new = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=3)
    np.testing.assert_array_equal(old, new)


def test_adarms_per_token_matches_broadcast_condition_and_preserves_parameter_shapes():
    norm = gemma.RMSNorm()
    x = jax.random.normal(jax.random.key(0), (2, 5, 4))
    cond = jax.random.normal(jax.random.key(1), (2, 6))
    variables = norm.init(jax.random.key(2), x, cond)
    # Nonzero modulation catches incorrect scale/shift/gate broadcasting.
    variables["params"]["Dense_0"]["kernel"] = jnp.full((6, 12), 0.1)
    scalar = norm.apply(variables, x, cond)
    per_token = norm.apply(variables, x, jnp.broadcast_to(cond[:, None], (2, 5, 6)))
    np.testing.assert_array_equal(scalar[0], per_token[0])
    np.testing.assert_array_equal(jnp.broadcast_to(scalar[1], (2, 5, 4)), per_token[1])


def test_sharded_loss_uses_global_valid_token_denominator():
    if jax.device_count() < 2:
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=2")
    mesh = jax.make_mesh((2,), ("data",))
    shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    loss = jnp.asarray([[1.0] * 4, [10.0] * 4])
    valid = jnp.asarray([[True] * 4, [False, False, False, True]])
    compute = jax.jit(lambda x, mask: rtc.mask_loss(x, mask).mean(), in_shardings=(shard, shard))
    assert float(compute(jax.device_put(loss, shard), jax.device_put(valid, shard))) == pytest.approx(14 / 5)
    grad = jax.jit(jax.grad(lambda x, mask: rtc.mask_loss(x, mask).mean()), in_shardings=(shard, shard))(loss, valid)
    np.testing.assert_allclose(grad, np.asarray(valid) / 5)


def test_policy_transports_model_space_priors_before_robot_output_transforms(small_pi05):
    from openpi.policies.policy import Policy

    model, obs, _ = small_pi05
    model.ttc = rtc.TTCConfig(enabled=True)
    # Slice and scale controls to prove wire priors retain all model dimensions.
    policy = Policy(model, output_transforms=[lambda data: {"actions": data["actions"][:, :2] * 10}])
    observation = jax.tree.map(lambda x: np.asarray(x[0]), obs.to_dict())
    first = policy.infer(
        {
            **observation,
            "_ttc": {
                "version": 1,
                "space": "model_normalized",
                "delay": 0,
                "prefixes": {},
            },
        }
    )
    block = first["ttc"]["blocks"]["actions"]
    assert block.shape == (5, 4)
    np.testing.assert_allclose(first["actions"], block[:, :2] * 10)
    next_result = policy.infer(
        {
            **observation,
            "_ttc": {
                "version": 1,
                "space": "model_normalized",
                "delay": 2,
                "prefixes": {"actions": block},
            },
        }
    )
    np.testing.assert_array_equal(next_result["ttc"]["blocks"]["actions"][:2], block[:2])
    assert "ttc" not in policy.infer(observation)
    assert policy.metadata["ttc"]["max_delay"] == 3


@pytest.mark.parametrize(
    "patch",
    [
        {"delay": True},
        {"delay": 4},
        {"delay": -1},
        {"delay": 1},
        {"space": "physical"},
        {"prefixes": {"tolerance": np.zeros((5, 6))}},
        {"delay": 1, "prefixes": {"actions": np.zeros((5, 2))}},
        {"delay": 1, "prefixes": {"actions": np.full((5, 4), np.nan)}},
    ],
)
def test_policy_rejects_bad_ttc_requests_before_sampling(small_pi05, patch):
    from openpi.policies.policy import Policy

    model, _, _ = small_pi05
    model.ttc = rtc.TTCConfig(enabled=True)
    policy = Policy(model)
    request = {"version": 1, "space": "model_normalized", "delay": 0, "prefixes": {}, **patch}
    with pytest.raises(ValueError, match="TTC|pi05"):
        policy.infer({"_ttc": request})
