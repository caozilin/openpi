import dataclasses
import os
import pathlib

import pytest
import tyro

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _config

from . import train


class MiniModel(_model.BaseModel):
    def __init__(self, config):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.kernel = nnx.Param(jnp.zeros(config.action_dim))

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, observation, train
        return jnp.mean(jnp.square(actions - self.kernel.value), axis=-1)

    def sample_actions(self, rng, observation, **kwargs):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class MiniModelConfig(pi0_config.Pi0Config):
    def create(self, rng):
        return MiniModel(self)


@pytest.mark.parametrize("validation_count", [0, 1, 2])
def test_train(tmp_path: pathlib.Path, monkeypatch, validation_count: int):
    # Exercise the real optimizer, JIT, and checkpoint serialization with a small NNX model.
    logged_metrics = []
    terminal_messages = []
    monkeypatch.setattr(train.swanlab, "log", lambda metrics, *, step: logged_metrics.append((step, metrics)))

    class Progress:
        def __init__(self, iterable, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def write(self, message):
            terminal_messages.append(message)

    monkeypatch.setattr(train.tqdm, "tqdm", Progress)
    create_loader = train._data_loader.create_torch_data_loader  # noqa: SLF001

    def create_fake_loader(data_config, **kwargs):
        # Exercise real batching and JAX validation while replacing external LeRobot data.
        return create_loader(dataclasses.replace(data_config, repo_id="fake"), **kwargs)

    monkeypatch.setattr(train._data_loader, "create_torch_data_loader", create_fake_loader)  # noqa: SLF001
    config = dataclasses.replace(
        _config.get_config("debug"),
        model=MiniModelConfig(action_dim=8, action_horizon=4, max_token_len=8),
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=3,
        log_interval=1,
        save_interval=2,
        num_workers=0,
        validation_repo_id="owner/heldout" if validation_count >= 1 else None,
        validation2_repo_id="owner/shifted" if validation_count == 2 else None,
        validation_num_batches=1,
        swanlab_enabled=True,
        swanlab_mode="disabled",
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=5)
    train.main(config)

    validation_names = ("val1", "val2")[:validation_count]
    # Training and all validation metrics must be committed once per step in SwanLab.
    step_logs = [(step, metrics) for step, metrics in logged_metrics if step > 0]
    assert [step for step, _ in step_logs] == [1, 2, 3, 4, 5]
    for step, metrics in step_logs:
        assert "loss" in metrics
        names = {key.split("/")[1] for key in metrics if key.startswith("validation/")}
        assert names == (set(validation_names) if step >= 2 else set())
        for name in names:
            assert f"validation/{name}/loss" in metrics
            assert metrics[f"validation/{name}/batch_size"] == 5
            assert metrics[f"validation/{name}/num_samples"] == 5
    validation_messages = [message for message in terminal_messages if "validation [" in message]
    assert len(validation_messages) == 4 * validation_count
    for name in validation_names:
        assert all(
            any(f"Step {step} validation [{name}]" in message for message in validation_messages)
            for step in (2, 3, 4, 5)
        )


def test_run_validation_averages_fixed_batches():
    batches = [1, 3]

    def validation_step(_state, batch, _rng):
        return {"loss": jnp.asarray(batch, dtype=jnp.float32)}

    metrics = train.run_validation(batches, validation_step, None, jax.random.key(0), name="heldout")
    assert metrics["validation/heldout/loss"] == 2.0
    assert metrics["validation/heldout/num_batches"] == 2
    assert metrics["validation/heldout/duration_seconds"] >= 0


@pytest.mark.parametrize("second_only", [False, True])
def test_validation_cli_accepts_two_named_datasets(second_only):
    args = ["debug", "--validation2-repo-id=owner/shifted", "--validation2-name=shifted"]
    if not second_only:
        args += ["--validation-repo-id=owner/heldout", "--validation-name=heldout"]
    cfg = tyro.extras.overridable_config_cli({"debug": ("debug", _config.get_config("debug"))}, args=args)
    assert [dataset.name for dataset in cfg.validation_datasets] == (
        ["shifted"] if second_only else ["heldout", "shifted"]
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"validation2_lerobot_dataset_root": "/validation"}, "validation2_repo_id"),
        ({"validation_repo_id": "owner/val", "validation_name": "has/slash"}, "names must contain"),
        ({"validation_repo_id": ""}, "repo_id must not be empty"),
        (
            {"validation_repo_id": "owner/val1", "validation2_repo_id": "owner/val2", "validation2_name": "val1"},
            "names must be unique",
        ),
    ],
)
def test_validation_configuration_rejects_ambiguous_logging(overrides, error):
    with pytest.raises(ValueError, match=error):
        dataclasses.replace(_config.get_config("debug"), **overrides)


def test_validation_loaders_keep_training_transforms_and_stats(monkeypatch):
    cfg = dataclasses.replace(
        _config.get_config("debug"),
        validation_repo_id="owner/val1",
        validation_lerobot_dataset_root="/val1",
        validation_lerobot_arrow_cache_dir="/arrow1",
        validation2_repo_id="owner/val2",
        validation2_lerobot_dataset_root="/val2",
        validation2_lerobot_arrow_cache_dir="/arrow2",
    )
    train_data_config = _config.DataConfig(repo_id="owner/train", norm_stats={})
    calls = []

    def create_loader(data_config, **kwargs):
        calls.append((data_config, kwargs))
        return object()

    monkeypatch.setattr(train._data_loader, "create_torch_data_loader", create_loader)  # noqa: SLF001
    loaders, batch_size = train.create_validation_loaders(cfg, train_data_config, None)
    assert list(loaders) == ["val1", "val2"]
    assert batch_size == 5
    for index, (data_config, kwargs) in enumerate(calls, start=1):
        assert data_config.repo_id == f"owner/val{index}"
        assert data_config.lerobot_dataset_root == f"/val{index}"
        assert data_config.lerobot_arrow_cache_dir == f"/arrow{index}"
        assert data_config.norm_stats is train_data_config.norm_stats
        assert data_config.data_transforms is train_data_config.data_transforms
        assert kwargs["repeat"] is False
        assert kwargs["shuffle"] is False


@pytest.mark.parametrize("same_training_source", [False, True])
def test_validation_loaders_reject_reused_sources_before_loading(monkeypatch, tmp_path, same_training_source):
    cfg = dataclasses.replace(
        _config.get_config("debug"),
        validation_repo_id="owner/val1",
        validation_lerobot_dataset_root=str(tmp_path / "val"),
        validation2_repo_id="owner/val2",
        validation2_lerobot_dataset_root=str(tmp_path / "val" / ".." / "val"),
    )
    train_data_config = _config.DataConfig(
        repo_id="owner/train",
        lerobot_dataset_root=str(tmp_path / "val" if same_training_source else tmp_path / "train"),
    )

    def unexpected_loader(*_args, **_kwargs):
        pytest.fail("Duplicate datasets must be rejected before loading data.")

    monkeypatch.setattr(train._data_loader, "create_torch_data_loader", unexpected_loader)  # noqa: SLF001
    with pytest.raises(ValueError, match="must be distinct"):
        train.create_validation_loaders(cfg, train_data_config, None)


@pytest.mark.parametrize(
    ("train_batch_size", "factor", "num_devices", "expected"),
    [(24, 1.0, 1, 24), (24, 0.5, 1, 12), (24, 2.0, 1, 48), (24, 1.5, 4, 36)],
)
def test_validation_batch_size(train_batch_size, factor, num_devices, expected):
    assert train.validation_batch_size(train_batch_size, factor, num_devices) == expected


def test_default_validation_batch_size_factor():
    cfg = _config.get_config("pi05_franka_mujoco_joint13_state")
    assert cfg.validation_batch_size_factor == 2.5
    val_batch_size = train.validation_batch_size(cfg.batch_size, cfg.validation_batch_size_factor, 1)
    assert val_batch_size == 60
    assert train.validation_batch_count(cfg.batch_size, cfg.validation_num_batches, val_batch_size) == 7


@pytest.mark.parametrize("factor", [0, -0.5, float("nan"), float("inf")])
def test_validation_batch_size_factor_rejects_invalid_values(factor):
    with pytest.raises(ValueError, match="validation_batch_size_factor"):
        dataclasses.replace(_config.get_config("debug"), validation_batch_size_factor=factor)


@pytest.mark.parametrize(
    ("val_batch_size", "expected_batches"),
    [(12, 32), (24, 16), (36, 11), (48, 8)],
)
def test_validation_batch_count_preserves_sample_budget(val_batch_size, expected_batches):
    assert train.validation_batch_count(24, 16, val_batch_size) == expected_batches
