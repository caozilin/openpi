import dataclasses
import json

import pytest

from openpi.models import pi0_config
from openpi.models import rtc
from openpi.policies import policy_config
from openpi.training import config


@pytest.mark.parametrize(
    ("enabled", "completed", "expected"),
    [
        (False, 100, False),
        (True, 20, False),
        (True, 21, False),
        (True, 22, True),
        (True, 100, True),
    ],
)
def test_checkpoint_ttc_metadata_restores_config_without_changing_model_shapes(
    tmp_path, monkeypatch, enabled, completed, expected
):
    checkpoint = tmp_path
    if enabled:
        (checkpoint / "assets").mkdir()
        (checkpoint / "assets/ttc.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "config": dataclasses.asdict(rtc.TTCConfig(enabled=True)),
                    "total_training_steps": 100,
                    "completed_step": completed,
                }
            )
        )
    loaded = []

    def load(model_config, params):
        loaded.append(model_config)
        return object()

    monkeypatch.setattr(pi0_config.Pi0Config, "load", load)
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda value: checkpoint)
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})  # noqa: SLF001
    monkeypatch.setattr(policy_config._policy, "Policy", lambda model, **kwargs: kwargs)  # noqa: SLF001
    cfg = config.get_config("debug_pi05")
    policy_config.create_trained_policy(cfg, checkpoint, norm_stats={})
    assert loaded[0].ttc.enabled == expected
    assert loaded[0].action_dim == cfg.model.action_dim
    assert loaded[0].action_horizon == cfg.model.action_horizon
    assert not cfg.model.ttc.enabled


def test_cannot_advertise_untrained_checkpoint_as_ttc(tmp_path, monkeypatch):
    cfg = config.get_config("debug_pi05")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, ttc=rtc.TTCConfig(enabled=True)))
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda value: tmp_path)
    with pytest.raises(ValueError, match="ttc.json"):
        policy_config.create_trained_policy(cfg, tmp_path, norm_stats={})
