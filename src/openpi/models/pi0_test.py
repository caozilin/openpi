import flax.nnx as nnx
import jax
import pytest

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_action_loss_groups_validate_ranges_and_overlap():
    config = _pi0_config.Pi0Config(action_loss_groups=(("action", 0, 7, 1.0), ("tolerance", 7, 10, 0.25)))
    assert config.action_loss_groups == (("action", 0, 7, 1.0), ("tolerance", 7, 10, 0.25))

    with pytest.raises(ValueError, match="must not overlap"):
        _pi0_config.Pi0Config(action_loss_groups=(("action", 0, 7, 1.0), ("tolerance", 6, 10, 0.25)))
    with pytest.raises(ValueError, match="non-empty and unique"):
        _pi0_config.Pi0Config(action_loss_groups=(("action", 0, 7, 1.0), ("action", 7, 10, 0.25)))
    with pytest.raises(ValueError, match="non-empty"):
        _pi0_config.Pi0Config(action_loss_groups=())


def test_grouped_action_loss_applies_each_weight_and_ignores_other_dimensions():
    squared_error = jax.numpy.asarray([[[1.0, 4.0, 9.0, 16.0, 1000.0]]])
    loss, breakdown = _pi0._grouped_action_loss_with_breakdown(
        squared_error, (("action", 0, 2, 1.0), ("tolerance", 2, 4, 0.5))
    )

    assert loss.shape == (1, 1)
    assert float(loss[0, 0]) == pytest.approx(8.75)
    assert float(breakdown["loss/action"][0, 0]) == pytest.approx(2.5)
    assert float(breakdown["loss/tolerance"][0, 0]) == pytest.approx(12.5)
    assert float(breakdown["loss_weighted/action"][0, 0]) == pytest.approx(2.5)
    assert float(breakdown["loss_weighted/tolerance"][0, 0]) == pytest.approx(6.25)
    assert float(breakdown["loss/total"][0, 0]) == pytest.approx(8.75)
