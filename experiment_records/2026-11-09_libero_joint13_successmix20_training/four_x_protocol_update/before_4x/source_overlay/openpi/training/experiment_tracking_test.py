import dataclasses
import json
from pathlib import Path
import types

import numpy as np
import pytest
import swanlab
from swanlab.data.porter.datastore import DataStore
import tyro

from openpi.training import config as _config
from openpi.training import experiment_tracking as tracking


@pytest.fixture
def config(tmp_path, monkeypatch):
    for key in ("SWANLAB_MODE", "SWANLAB_RUN_ID", "SWANLAB_RESUME"):
        monkeypatch.delenv(key, raising=False)
    config = dataclasses.replace(
        _config.get_config("debug"),
        exp_name="swanlab-test",
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        swanlab_enabled=True,
        swanlab_log_dir=str(tmp_path / "swanlog"),
    )
    config.checkpoint_dir.mkdir(parents=True)
    return config


@pytest.fixture
def init_calls(monkeypatch):
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(id="new-swanlab-id")

    monkeypatch.setattr(swanlab, "init", init)
    return calls


def test_new_experiment_records_config_and_cloud_id(config, init_calls):
    config = dataclasses.replace(config, swanlab_workspace="robotics")
    tracking.init_swanlab(config, resuming=False)
    kwargs = init_calls[0]
    assert kwargs["project"] == config.project_name
    assert kwargs["experiment_name"] == config.exp_name
    assert kwargs["workspace"] == "robotics"
    assert kwargs["logdir"] == config.swanlab_log_dir
    assert kwargs["config"]["num_train_steps"] == config.num_train_steps
    assert "resume" not in kwargs
    assert (config.checkpoint_dir / "swanlab_id.txt").read_text() == "new-swanlab-id"


def test_resume_uses_swanlab_id(config, init_calls):
    (config.checkpoint_dir / "swanlab_id.txt").write_text("existing-id\n")
    tracking.init_swanlab(config, resuming=True)
    assert init_calls[0]["id"] == "existing-id"
    assert init_calls[0]["resume"] == "must"


@pytest.mark.parametrize("legacy_id", [None, "old-wandb-id"])
def test_checkpoint_without_swanlab_id_starts_new_experiment(config, init_calls, legacy_id):
    path = config.checkpoint_dir / "wandb_id.txt"
    if legacy_id is not None:
        path.write_text(legacy_id)
    tracking.init_swanlab(config, resuming=True)
    assert "id" not in init_calls[0]
    assert "resume" not in init_calls[0]
    assert (config.checkpoint_dir / "swanlab_id.txt").is_file()
    if legacy_id is not None:
        assert path.read_text() == legacy_id


@pytest.mark.parametrize("mode", ["offline", "local"])
def test_offline_resume_preserves_cloud_id(config, init_calls, mode):
    path = config.checkpoint_dir / "swanlab_id.txt"
    path.write_text("cloud-id")
    tracking.init_swanlab(dataclasses.replace(config, swanlab_mode=mode), resuming=True)
    assert init_calls[0]["mode"] == mode
    assert "id" not in init_calls[0]
    assert "resume" not in init_calls[0]
    assert path.read_text() == "cloud-id"


@pytest.mark.parametrize("overrides", [{"swanlab_enabled": False}, {"swanlab_mode": "disabled"}])
def test_disabled_tracking_does_not_require_checkpoint(config, init_calls, overrides):
    config.checkpoint_dir.rmdir()
    tracking.init_swanlab(dataclasses.replace(config, **overrides), resuming=True)
    assert init_calls == [{"mode": "disabled"}]
    assert not config.checkpoint_dir.exists()


def test_environment_mode_and_explicit_override(config, init_calls, monkeypatch):
    monkeypatch.setenv("SWANLAB_MODE", "disabled")
    tracking.init_swanlab(config, resuming=False)
    assert init_calls == [{"mode": "disabled"}]
    tracking.init_swanlab(dataclasses.replace(config, swanlab_mode="offline"), resuming=False)
    assert init_calls[1]["mode"] == "offline"
    assert not (config.checkpoint_dir / "swanlab_id.txt").exists()


def test_cloud_environment_alias_persists_id(config, init_calls, monkeypatch):
    monkeypatch.setenv("SWANLAB_MODE", "cloud")
    tracking.init_swanlab(config, resuming=False)
    assert init_calls[0]["mode"] == "online"
    assert (config.checkpoint_dir / "swanlab_id.txt").is_file()


def test_empty_cloud_id_fails_before_initialization(config, init_calls):
    (config.checkpoint_dir / "swanlab_id.txt").write_text("\n")
    with pytest.raises(ValueError, match="experiment ID is empty"):
        tracking.init_swanlab(config, resuming=True)
    assert not init_calls


def test_camera_image_preserves_normalized_colors():
    pixels = np.asarray([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
    image = tracking.camera_image(pixels)
    np.testing.assert_array_equal(np.asarray(image.image_data), [[[0, 127, 255]]])
    uint8_pixels = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    np.testing.assert_array_equal(np.asarray(tracking.camera_image(uint8_pixels).image_data), uint8_pixels)


def test_swanlab_cli_flags():
    cfg = tyro.extras.overridable_config_cli(
        {"debug": ("debug", _config.get_config("debug"))},
        args=[
            "debug",
            "--swanlab-enabled",
            "--swanlab-mode=offline",
            "--swanlab-workspace=robotics",
            "--swanlab-log-dir=/tmp/robotics-swanlog",
        ],
    )
    assert cfg.swanlab_enabled
    assert cfg.swanlab_mode == "offline"
    assert cfg.swanlab_workspace == "robotics"
    assert cfg.swanlab_log_dir == "/tmp/robotics-swanlog"


def test_real_offline_sdk_writes_metrics_images_and_config(config):
    config = dataclasses.replace(config, swanlab_mode="offline")
    run = tracking.init_swanlab(config, resuming=False)
    try:
        assert run.config["num_train_steps"] == config.num_train_steps
        swanlab.log({"camera_views": [tracking.camera_image(np.zeros((4, 4, 3), dtype=np.float32))]}, step=0)
        swanlab.log({"loss": 0.5, "validation/val1/loss": 0.25}, step=7)
    finally:
        swanlab.finish()
    log_dir = Path(config.swanlab_log_dir)
    assert list(log_dir.rglob("*.png"))
    store = DataStore()
    store.open_for_scan(str(next(log_dir.rglob("backup.swanlab"))))
    try:
        records = [json.loads(record) for record in store]
    finally:
        store.close()
    metrics = {
        (record["data"]["key"], record["data"]["step"]): record["data"]["metric"]["data"]
        for record in records
        if record["model_type"] == "Scalar"
    }
    assert metrics["loss", 7] == 0.5
    assert metrics["validation/val1/loss", 7] == 0.25
    assert not (config.checkpoint_dir / "swanlab_id.txt").exists()
