import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import pi0
from openpi.models import tokenizer
from openpi.training import config

CONFIG_NAME = "pi05_franka_mujoco_state_action7_lora"


def test_action7_state_pipeline_uses_only_physical_actions(monkeypatch, tmp_path):
    baseline = config.get_config(CONFIG_NAME)
    captured_states = []

    class CaptureTokenizer:
        def __init__(self, max_len):
            assert max_len == 200

        def tokenize(self, prompt, state):
            assert prompt == "pick up the object"
            captured_states.append(state.copy())
            return np.zeros(200, np.int32), np.ones(200, bool)

    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", CaptureTokenizer)
    monkeypatch.setattr(config.LeRobotFrankaMujocoDataConfig, "_load_norm_stats", lambda *_: None)
    data = baseline.data.create(tmp_path, baseline.model)
    state = np.arange(7, dtype=np.float32) / 10
    physical = np.ones((16, 7), np.float32)
    sample = {
        "image": np.zeros((224, 224, 3), np.uint8),
        "wrist_image": np.zeros((224, 224, 3), np.uint8),
        "state": state,
        "actions": physical,
        "prompt": "pick up the object",
        # These labels must have no effect on the action-only pipeline.
        "rotation_tolerance": np.full((16, 6), 99, np.float32),
        "trajectory_is_tolerance": np.ones(1, np.int64),
    }
    sample = transforms.compose(data.repack_transforms.inputs)(sample)
    sample = transforms.compose(data.data_transforms.inputs)(sample)
    sample = transforms.compose(data.model_transforms.inputs)(sample)
    assert data.action_sequence_keys == ("actions",)
    assert not data.data_transforms.outputs[0].joint_task_tolerance
    assert float(sample["action_loss_weight"]) == 1.0
    assert sample["actions"].shape == (16, 32)
    np.testing.assert_array_equal(sample["actions"][:, :7], physical)
    np.testing.assert_array_equal(sample["actions"][:, 7:], 0)
    assert len(captured_states) == 1
    np.testing.assert_array_equal(captured_states[0], state)
    assert sample["state"].shape == (32,)


def test_action7_loss_has_zero_gradient_for_all_padding_channels():
    baseline = config.get_config(CONFIG_NAME)
    prediction = jnp.arange(64, dtype=jnp.float32).reshape(1, 2, 32) / 10

    def loss(values):
        return pi0._grouped_action_loss(values**2, baseline.model.action_loss_groups).mean()  # noqa: SLF001

    expected = np.asarray(prediction)[..., :7]
    np.testing.assert_allclose(loss(prediction), np.mean(expected**2), rtol=1e-6)
    gradient = np.asarray(jax.grad(loss)(prediction))
    np.testing.assert_allclose(gradient[..., :7], 2 * expected / expected.size, rtol=1e-6)
    np.testing.assert_array_equal(gradient[..., 7:], 0)


def test_action7_model_keeps_lora_and_pretrained_interface():
    baseline = config.get_config(CONFIG_NAME)
    previous = config.get_config("pi05_franka_mujoco")
    assert baseline.model.action_dim == 32
    assert baseline.model.pi05
    assert baseline.model.discrete_state_input
    assert baseline.model.paligemma_variant == "gemma_2b_lora"
    assert baseline.model.action_expert_variant == "gemma_300m_lora"
    assert baseline.model.action_horizon == 16
    assert baseline.weight_loader == previous.weight_loader
    assert baseline.data.assets.assets_dir != previous.data.assets.assets_dir
    assert dataclasses.replace(baseline, batch_size=28, num_train_steps=30_000).batch_size == 28
