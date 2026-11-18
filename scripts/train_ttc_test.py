import dataclasses
import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
import tyro

from openpi.models import pi0_config
from openpi.models import rtc
from openpi.training import checkpoints
from openpi.training import config as training_config
from openpi.training import sharding
from scripts import train
from scripts.train_test import MiniModel


class MiniTTCModel(MiniModel):
    def __init__(self, config):
        super().__init__(config)
        self.ttc = config.ttc

    def compute_loss_with_breakdown(
        self, rng, observation, actions, *, train=False, training_step=None, total_training_steps=None
    ):
        loss = self.compute_loss(rng, observation, actions, train=train)
        alpha = rtc.curriculum_alpha(training_step, total_training_steps, self.ttc) if train else jnp.asarray(1.0)
        return loss, {"ttc/alpha": alpha}


@dataclasses.dataclass(frozen=True)
class MiniTTCConfig(pi0_config.Pi0Config):
    def create(self, rng):
        del rng
        return MiniTTCModel(self)


def test_real_train_step_restored_step_drives_curriculum_and_checkpoint_metadata(tmp_path):
    cfg = dataclasses.replace(
        training_config.get_config("debug_pi05"),
        model=MiniTTCConfig(
            pi05=True, action_dim=4, action_horizon=5, max_token_len=4, ttc=rtc.TTCConfig(enabled=True)
        ),
        num_train_steps=100,
        ema_decay=None,
    )
    mesh = sharding.make_mesh(1)
    state, _ = train.init_train_state(cfg, jax.random.key(0), mesh, resume=False)
    update = jax.jit(
        lambda current: train.train_step(
            cfg, jax.random.key(42), current, (cfg.model.fake_obs(2), cfg.model.fake_act(2))
        )
    )
    for step, alpha in [(0, 0), (19, 0), (20, 0), (40, 0.5), (60, 1), (99, 1)]:
        _, info = update(dataclasses.replace(state, step=jnp.asarray(step)))
        assert float(info["ttc/alpha"]) == pytest.approx(alpha)

    class Loader:
        def data_config(self):
            return training_config.DataConfig()

    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoint", keep_period=1, overwrite=False, resume=False
    )
    metadata = {
        "version": 1,
        "config": dataclasses.asdict(cfg.model.ttc),
        "total_training_steps": 100,
        "completed_step": 40,
    }
    checkpoints.save_state(
        manager, dataclasses.replace(state, step=jnp.asarray(40)), Loader(), 40, ttc_metadata=metadata
    )
    manager.wait_until_finished()
    saved = json.loads((tmp_path / "checkpoint/40/assets/ttc.json").read_text())
    assert saved["config"]["target_delay_probabilities"] == [0.1, 0.05, 0.8, 0.05]
    restored = checkpoints.restore_state(manager, state, Loader(), 40)
    assert int(restored.step) == 40
    _, info = update(restored)
    assert float(info["ttc/alpha"]) == pytest.approx(0.5)
    assert nnx.state(nnx.merge(restored.model_def, restored.params))
    manager.close()


def test_ttc_cli_opt_in_and_custom_target_distribution():
    configs = {"debug_pi05": ("test", training_config.get_config("debug_pi05"))}
    default = tyro.extras.overridable_config_cli(configs, args=["debug_pi05"])
    assert not default.model.ttc.enabled
    enabled = tyro.extras.overridable_config_cli(
        configs,
        args=[
            "debug_pi05",
            "--model.ttc.enabled",
            "--model.ttc.target-delay-probabilities",
            "0.1",
            "0.05",
            "0.8",
            "0.05",
        ],
    )
    assert enabled.model.ttc.enabled
    assert enabled.model.ttc.target_delay_probabilities == (0.1, 0.05, 0.8, 0.05)
