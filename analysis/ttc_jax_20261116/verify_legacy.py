"""Compare disabled TTC to the exact pre-change Git source, using a small real action expert."""
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import gemma, pi0, pi0_config, rtc
from openpi.models.rtc_test import TinyImage
from openpi.shared import nnx_utils

root = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location("baseline_" + name, root / ("baseline_" + name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

old_gemma = load("gemma")
old_pi0 = load("pi0")
old_pi0._gemma = old_gemma
for module in (old_gemma, gemma):
    module.PALIGEMMA_VOCAB_SIZE = 32
    module.get_config = lambda variant, module=module: module.Config(32, 1, 64, 2, 1, 16)
pi0._siglip.Module = lambda **kwargs: TinyImage(kwargs["num_classes"])
results = []
for dtype in ("float32", "bfloat16"):
    cfg = pi0_config.Pi0Config(pi05=True, dtype=dtype, action_dim=4, action_horizon=5,
                              max_token_len=4, paligemma_variant="dummy", action_expert_variant="dummy")
    old = old_pi0.Pi0(cfg, rngs=nnx.Rngs(jax.random.key(7)))
    new = cfg.create(jax.random.key(7))
    old_params = nnx.state(old).to_pure_dict()
    jax.tree.map(np.testing.assert_array_equal, old_params, nnx.state(new).to_pure_dict())
    # Exercise actual BaseModelConfig.load parameter-tree validation too.
    loaded = cfg.load(old_params)
    obs = dataclasses.replace(cfg.fake_obs(2), tokenized_prompt=jnp.ones((2, 4), jnp.int32))
    actions = jax.random.normal(jax.random.key(8), (2, 5, 4))
    key = jax.random.key(42)
    old_loss = nnx_utils.module_jit(old.compute_loss)
    new_loss = nnx_utils.module_jit(loaded.compute_loss)
    left, left_grad = jax.value_and_grad(lambda a: old_loss(key, obs, a).mean())(actions)
    right, right_grad = jax.value_and_grad(lambda a: new_loss(key, obs, a).mean())(actions)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left_grad, right_grad)
    old_sample = nnx_utils.module_jit(old.sample_actions)(key, obs, num_steps=3)
    new_sample = nnx_utils.module_jit(loaded.sample_actions)(key, obs, num_steps=3)
    np.testing.assert_array_equal(old_sample, new_sample)
    loaded.ttc = rtc.TTCConfig(enabled=True)
    prior = jnp.full((2,5,4), .25)
    conditioned = nnx_utils.module_jit(loaded.sample_actions)(
        key, obs, num_steps=3, action_prefix=prior, prefix_lengths=jnp.asarray([0,3]))
    np.testing.assert_array_equal(conditioned[1,:3], prior[1,:3])
    assert np.isfinite(conditioned).all()
    results.append({"dtype": dtype, "parameter_tree_equal": True, "legacy_load_passed": True,
                    "disabled_loss_bitwise_equal": True, "disabled_input_grad_bitwise_equal": True,
                    "disabled_sample_bitwise_equal": True, "ttc_sample_prefix_exact": True})
(root / "legacy_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
