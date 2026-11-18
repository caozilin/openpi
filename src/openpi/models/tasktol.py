"""π0.5 action and rotation-tolerance tokens with one-way block attention."""

from typing import ClassVar

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.pi0 import Pi0


class Pi05ToleranceTokens(Pi0):
    # The existing joint13 data, loss, normalization and Euler solver all use
    # feature-packed arrays. Only the expert operates on two token sequences.
    block_dims: ClassVar[dict[str, int]] = {"actions": 7, "rotation_tolerance": 6}

    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        width = self.action_in_proj.out_features
        self.tolerance_in_proj = nnx.Linear(6, width, rngs=rngs)
        self.tolerance_out_proj = nnx.Linear(width, 6, rngs=rngs)

    def sample_noise(self, key, shape):
        action_key, tolerance_key = jax.random.split(key)
        actions = jax.random.normal(action_key, (*shape[:-1], 7))
        tolerance = jax.random.normal(tolerance_key, (*shape[:-1], 6))
        return self.pack_blocks(actions, tolerance)

    def pack_blocks(self, actions, tolerance):
        padding = jnp.zeros((*actions.shape[:-1], self.action_dim - 13), dtype=actions.dtype)
        return jnp.concatenate((actions, tolerance, padding), axis=-1)

    def embed_suffix(self, obs, noisy_actions, timestep):
        # Only seven physical components may influence the pretrained action projection.
        action_input = jnp.zeros_like(noisy_actions).at[..., :7].set(noisy_actions[..., :7])
        actions, mask, block_start, cond = super().embed_suffix(obs, action_input, timestep)
        tolerance = self.tolerance_in_proj(noisy_actions[..., 7:13])
        # A second block start gives R access to A, while hiding R from A in every layer.
        if cond.ndim == 3:
            cond = jnp.concatenate((cond, cond), axis=1)
        return (
            jnp.concatenate((actions, tolerance), axis=1),
            jnp.concatenate((mask, mask), axis=1),
            jnp.concatenate((block_start, block_start)),
            cond,
        )

    def project_suffix(self, suffix_out):
        actions, tolerance = jnp.split(suffix_out, 2, axis=1)
        return self.pack_blocks(self.action_out_proj(actions)[..., :7], self.tolerance_out_proj(tolerance))
