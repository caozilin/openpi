import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import rtc


def test_schedule_from_string():
    assert rtc.schedule_from_string("linear") == rtc.RTCAttentionSchedule.LINEAR
    assert rtc.schedule_from_string(" EXP ") == rtc.RTCAttentionSchedule.EXP

    with pytest.raises(ValueError, match="Invalid RTC prefix attention schedule"):
        rtc.schedule_from_string("unknown")


def test_get_prefix_weights():
    np.testing.assert_allclose(
        rtc.get_prefix_weights(1, 4, 6, rtc.RTCAttentionSchedule.ZEROS),
        np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        rtc.get_prefix_weights(1, 4, 6, rtc.RTCAttentionSchedule.ONES),
        np.array([1, 1, 1, 1, 0, 0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        rtc.get_prefix_weights(1, 4, 6, rtc.RTCAttentionSchedule.LINEAR),
        np.array([1, 0.75, 0.5, 0.25, 0, 0], dtype=np.float32),
    )


def test_guided_denoise_step_uses_masked_prefix():
    x_t = jnp.zeros((1, 4, 3), dtype=jnp.float32)
    prev_chunk = jnp.ones((1, 4, 3), dtype=jnp.float32)

    guided = rtc.guided_denoise_step(
        x_t,
        time=jnp.asarray(0.5, dtype=jnp.float32),
        prev_chunk_left_over=prev_chunk,
        inference_delay=jnp.asarray(0, dtype=jnp.int32),
        execution_horizon=jnp.asarray(2, dtype=jnp.int32),
        max_guidance_weight=jnp.asarray(10.0, dtype=jnp.float32),
        prefix_attention_schedule=jnp.asarray(rtc.RTCAttentionSchedule.ONES, dtype=jnp.int32),
        guidance_mask=jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32),
        denoise_fn=lambda x: jnp.zeros_like(x),
    )

    np.testing.assert_allclose(
        guided,
        np.array([[[-2, 0, -2], [-2, 0, -2], [0, 0, 0], [0, 0, 0]]], dtype=np.float32),
    )


def test_guided_denoise_step_is_jittable():
    x_t = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    prev_chunk = jnp.ones((1, 2, 2), dtype=jnp.float32)

    guided = jax.jit(rtc.guided_denoise_step, static_argnames=("denoise_fn",))(
        x_t,
        time=jnp.asarray(0.5, dtype=jnp.float32),
        prev_chunk_left_over=prev_chunk,
        inference_delay=jnp.asarray(0, dtype=jnp.int32),
        execution_horizon=jnp.asarray(1, dtype=jnp.int32),
        max_guidance_weight=jnp.asarray(10.0, dtype=jnp.float32),
        prefix_attention_schedule=jnp.asarray(rtc.RTCAttentionSchedule.LINEAR, dtype=jnp.int32),
        guidance_mask=jnp.ones((2,), dtype=jnp.float32),
        denoise_fn=lambda x: jnp.zeros_like(x),
    )

    assert guided.shape == x_t.shape
