import enum

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class RTCAttentionSchedule(enum.IntEnum):
    ZEROS = 0
    ONES = 1
    LINEAR = 2
    EXP = 3


def schedule_from_string(value: str | RTCAttentionSchedule) -> RTCAttentionSchedule:
    if isinstance(value, RTCAttentionSchedule):
        return value
    normalized = str(value).strip().upper()
    try:
        return RTCAttentionSchedule[normalized]
    except KeyError as exc:
        valid = ", ".join(schedule.name.lower() for schedule in RTCAttentionSchedule)
        raise ValueError(f"Invalid RTC prefix attention schedule '{value}'. Valid values: {valid}") from exc


def get_prefix_weights(
    start: at.Int[at.Array, ""],
    end: at.Int[at.Array, ""],
    total: int,
    schedule: at.Int[at.Array, ""],
) -> at.Float[at.Array, " total"]:
    """Build RTC prefix weights with the same semantics as LeRobot's RTC implementation."""
    start = jnp.minimum(start, end)
    idx = jnp.arange(total)
    zeros = jnp.where(idx < start, 1.0, 0.0)
    ones = jnp.where(idx < end, 1.0, 0.0)

    denom = jnp.maximum(end - start + 1, 1)
    linear = 1.0 - (idx - start + 1) / denom
    linear = jnp.where(idx < start, 1.0, jnp.where(idx >= end, 0.0, linear))
    exp = linear * jnp.expm1(linear) / (jnp.e - 1.0)
    exp = jnp.where(idx < start, 1.0, jnp.where(idx >= end, 0.0, exp))

    choices = jnp.stack([zeros, ones, linear, exp])
    schedule = jnp.clip(schedule, 0, len(RTCAttentionSchedule) - 1)
    return choices[schedule]


def guidance_weight(time: at.Float[at.Array, ""], max_guidance_weight: at.Float[at.Array, ""]) -> at.Float[at.Array, ""]:
    """RTC guidance schedule for OpenPI's t=1 noise -> t=0 action convention."""
    tau = 1.0 - time
    squared_one_minus_tau = (1.0 - tau) ** 2
    inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
    c = jnp.nan_to_num((1.0 - tau) / tau, posinf=max_guidance_weight)
    weight = jnp.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
    return jnp.minimum(weight, max_guidance_weight)


def guided_denoise_step(
    x_t: at.Float[at.Array, "b ah ad"],
    *,
    time: at.Float[at.Array, ""],
    prev_chunk_left_over: at.Float[at.Array, "b ah ad"],
    inference_delay: at.Int[at.Array, ""],
    execution_horizon: at.Int[at.Array, ""],
    max_guidance_weight: at.Float[at.Array, ""],
    prefix_attention_schedule: at.Int[at.Array, ""],
    guidance_mask: at.Float[at.Array, " action_dim"],
    denoise_fn,
) -> at.Float[at.Array, "b ah ad"]:
    """Apply Real-Time Chunking guidance around a denoiser."""
    action_horizon = x_t.shape[1]
    execution_horizon = jnp.minimum(execution_horizon, action_horizon)
    weights = get_prefix_weights(
        inference_delay,
        execution_horizon,
        action_horizon,
        prefix_attention_schedule,
    )[None, :, None]
    weights = weights * guidance_mask[None, None, :]

    def x1_fn(input_x_t):
        v_t = denoise_fn(input_x_t)
        return input_x_t - time * v_t

    x1_t, pullback = jax.vjp(x1_fn, x_t)
    err = (prev_chunk_left_over - x1_t) * weights
    correction = pullback(err)[0]
    v_t = denoise_fn(x_t)
    guided_v_t = v_t - guidance_weight(time, max_guidance_weight) * correction
    return guided_v_t.astype(v_t.dtype)
