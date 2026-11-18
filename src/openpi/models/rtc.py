"""Training-time RTC primitives, shared by independently denoised future blocks."""

import dataclasses
import math

from flax import struct
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class TTCConfig:
    enabled: bool = False
    # Index is delay in control steps. Curriculum mixes this with delta_0.
    target_delay_probabilities: tuple[float, ...] = (0.10, 0.05, 0.80, 0.05)
    warmup_fraction: float = 0.2
    ramp_end_fraction: float = 0.6

    def __post_init__(self):
        probabilities = self.target_delay_probabilities
        if (
            not probabilities
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("TTC delay probabilities must be finite, nonnegative, and sum to one")
        if not 0 <= self.warmup_fraction < self.ramp_end_fraction <= 1:
            raise ValueError("TTC requires 0 <= warmup_fraction < ramp_end_fraction <= 1")

    @property
    def max_delay(self) -> int:
        return max(index for index, probability in enumerate(self.target_delay_probabilities) if probability > 0)


@struct.dataclass
class PrefixCondition:
    """One block's model-space prior. Different blocks may use distinct D and time."""

    values: jax.Array  # [B,H,D], fixed shape for JIT
    lengths: jax.Array  # [B], shared across blocks by the current protocol

    def mask(self) -> jax.Array:
        return jnp.arange(self.values.shape[-2]) < self.lengths[..., None]

    def clamp(self, values: jax.Array) -> jax.Array:
        return jnp.where(self.mask()[..., None], self.values, values)


def curriculum_alpha(step, total_steps, config: TTCConfig):
    progress = jnp.asarray(step, jnp.float32) / total_steps
    return jnp.clip((progress - config.warmup_fraction) / (config.ramp_end_fraction - config.warmup_fraction), 0.0, 1.0)


def delay_probabilities(alpha, config: TTCConfig):
    probabilities = alpha * jnp.asarray(config.target_delay_probabilities, jnp.float32)
    return probabilities.at[0].add(1.0 - alpha)


def sample_delays(key, batch_shape, probabilities):
    return jax.random.categorical(key, jnp.log(probabilities), shape=batch_shape).astype(jnp.int32)


def flow_inputs(actions, noise, time, lengths):
    """Clean prefix at t=0, original noise-at-one flow on the postfix."""
    condition = PrefixCondition(actions, lengths)
    token_time = jnp.where(condition.mask(), 0.0, time[..., None])
    x_t = token_time[..., None] * noise + (1.0 - token_time[..., None]) * actions
    return x_t, noise - actions, token_time, ~condition.mask()


def mask_loss(loss, valid):
    """Keep [B,H] API; mean(result) equals the global valid-token mean."""
    scale = valid.size / jnp.maximum(jnp.sum(valid), 1)
    return jnp.where(valid, loss, 0.0) * scale
