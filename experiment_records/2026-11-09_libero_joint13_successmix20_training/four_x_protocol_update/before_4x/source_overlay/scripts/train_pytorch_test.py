import dataclasses

import torch

from openpi.models import model as _model
from openpi.training import config as _config

from . import train_pytorch as train


class MiniModel(torch.nn.Module):
    def __init__(self, _config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, _observation, actions):
        return (actions - self.weight).square()


def test_pytorch_swanlab_training_and_resume(tmp_path, monkeypatch):
    """Exercise image logging, real CPU updates, and checkpoint round trips."""
    logged_metrics = []
    monkeypatch.setattr(train, "setup_ddp", lambda: (False, 0, torch.device("cpu")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", MiniModel)
    monkeypatch.setattr(train.swanlab, "log", lambda metrics, *, step: logged_metrics.append((step, metrics)))
    observation = _model.Observation(
        images={"camera": torch.zeros((2, 3, 4, 4))},
        image_masks={"camera": torch.ones(2, dtype=torch.bool)},
        state=torch.zeros((2, 1)),
    )
    loader = [(observation, torch.ones((2, 1, 1)))]
    monkeypatch.setattr(train, "build_datasets", lambda _train_config: (loader, _config.DataConfig()))
    monkeypatch.setattr(train._data, "create_data_loader", lambda *_args, **_kwargs: loader)  # noqa: SLF001

    default = _config.get_config("debug")
    config = dataclasses.replace(
        default,
        model=dataclasses.replace(default.model),
        exp_name="swanlab-pytorch-test",
        checkpoint_base_dir=str(tmp_path),
        batch_size=2,
        num_train_steps=2,
        log_interval=1,
        save_interval=1,
        overwrite=False,
        swanlab_enabled=True,
        swanlab_mode="disabled",
    )
    train.train_loop(config)
    train.train_loop(dataclasses.replace(config, resume=True, num_train_steps=3))
    assert sum("camera_views" in metrics for _, metrics in logged_metrics) == 1
    assert [step for step, metrics in logged_metrics if "loss" in metrics] == [0, 1, 2]
    assert [metrics["checkpoint_step"] for _, metrics in logged_metrics if "checkpoint_step" in metrics] == [1, 2, 3]
    assert all("grad_norm" in metrics for _, metrics in logged_metrics if "loss" in metrics)
    assert (config.checkpoint_dir / "3" / "model.safetensors").is_file()
    assert not (config.checkpoint_dir / "swanlab_id.txt").exists()


def test_pytorch_worker_does_not_save_or_log(tmp_path, monkeypatch):
    def unexpected_log(*_args, **_kwargs):
        raise AssertionError("A secondary worker must not log a checkpoint")

    monkeypatch.setattr(train.swanlab, "log", unexpected_log)
    config = dataclasses.replace(_config.get_config("debug"), exp_name="worker", checkpoint_base_dir=str(tmp_path))
    train.save_checkpoint(None, None, 1, config, is_main=False, data_config=None)
    assert not config.checkpoint_dir.exists()
