from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_config: dict[str, Any] | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            rtc_config: Optional server-side Real-Time Chunking configuration for JAX models.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._rtc_config = rtc_config or {}

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        rtc_request = obs.get("rtc") if isinstance(obs, dict) else None
        inputs = {key: value for key, value in obs.items() if key != "rtc"}
        inputs = jax.tree.map(lambda x: x, inputs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if not self._is_pytorch_model:
            sample_kwargs.update(self._make_rtc_sample_kwargs(rtc_request, inputs["state"].shape[0]))

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        model_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": model_actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            model_actions_np = outputs["actions"].copy()
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            model_actions_np = outputs["actions"].copy()

        outputs = self._output_transform(outputs)
        if self._rtc_config.get("enabled", False):
            outputs["rtc"] = {"model_actions": model_actions_np}
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _make_rtc_sample_kwargs(self, rtc_request: dict | None, batch_size: int) -> dict[str, Any]:
        if not self._rtc_config.get("enabled", False) or not rtc_request:
            return {}
        if not rtc_request.get("enabled", True):
            return {}
        prev_chunk = rtc_request.get("prev_chunk_left_over")
        if prev_chunk is None:
            return {}

        prev_chunk = np.asarray(prev_chunk, dtype=np.float32)
        if prev_chunk.ndim == 2:
            prev_chunk = prev_chunk[None, ...]
        if prev_chunk.ndim != 3:
            raise ValueError(f"RTC prev_chunk_left_over must have shape [T,A] or [B,T,A], got {prev_chunk.shape}")

        action_horizon = int(self._model.action_horizon)
        action_dim = int(self._model.action_dim)
        normalized = np.zeros((batch_size, action_horizon, action_dim), dtype=np.float32)
        steps = min(prev_chunk.shape[1], action_horizon)
        dims = min(prev_chunk.shape[2], action_dim)
        if prev_chunk.shape[0] == 1 and batch_size != 1:
            prev_chunk = np.repeat(prev_chunk, batch_size, axis=0)
        if prev_chunk.shape[0] != batch_size:
            raise ValueError(f"RTC prev_chunk batch size {prev_chunk.shape[0]} does not match input batch {batch_size}")
        normalized[:, :steps, :dims] = prev_chunk[:, :steps, :dims]

        guidance_dims = int(self._rtc_config.get("guidance_dims", action_dim))
        guidance_mask = np.zeros((action_dim,), dtype=np.float32)
        guidance_mask[: min(max(guidance_dims, 0), action_dim)] = 1.0

        prefix_attention_schedule = rtc_request.get(
            "prefix_attention_schedule",
            self._rtc_config.get("prefix_attention_schedule", int(_rtc.RTCAttentionSchedule.LINEAR)),
        )
        if isinstance(prefix_attention_schedule, str):
            prefix_attention_schedule = int(_rtc.schedule_from_string(prefix_attention_schedule))

        return {
            "rtc_active": jnp.ones((), dtype=jnp.bool_),
            "rtc_prev_chunk_left_over": jnp.asarray(normalized),
            "rtc_inference_delay": jnp.asarray(
                int(rtc_request.get("inference_delay", self._rtc_config.get("inference_delay", 0))),
                dtype=jnp.int32,
            ),
            "rtc_execution_horizon": jnp.asarray(
                int(rtc_request.get("execution_horizon", self._rtc_config.get("execution_horizon", action_horizon))),
                dtype=jnp.int32,
            ),
            "rtc_max_guidance_weight": jnp.asarray(
                float(rtc_request.get("max_guidance_weight", self._rtc_config.get("max_guidance_weight", 10.0))),
                dtype=jnp.float32,
            ),
            "rtc_prefix_attention_schedule": jnp.asarray(int(prefix_attention_schedule), dtype=jnp.int32),
            "rtc_guidance_mask": jnp.asarray(guidance_mask),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
