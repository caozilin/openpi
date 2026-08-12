import dataclasses
import json

import datasets
import jax
import pyarrow as pa
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_find_arrow_cache_entry_requires_complete_train_shards(tmp_path):
    cache_entry = tmp_path / "parquet" / "default-fingerprint" / "0.0.0" / "builder-fingerprint"
    cache_entry.mkdir(parents=True)
    (cache_entry / "dataset_info.json").write_text(
        json.dumps({"splits": {"train": {"num_examples": 3, "num_shards": 2}}}), encoding="utf-8"
    )
    (cache_entry / "parquet-train-00000-of-00002.arrow").touch()
    (cache_entry / "cache-transform.arrow").touch()

    with pytest.raises(FileNotFoundError):
        _data_loader._find_arrow_cache_entry(tmp_path, expected_num_rows=3)

    (cache_entry / "parquet-train-00001-of-00002.arrow").touch()
    found_entry, shards = _data_loader._find_arrow_cache_entry(tmp_path, expected_num_rows=3)

    assert found_entry == cache_entry
    assert [path.name for path in shards] == [
        "parquet-train-00000-of-00002.arrow",
        "parquet-train-00001-of-00002.arrow",
    ]


def test_arrow_shards_are_loaded_in_dataset_order(tmp_path):
    cache_entry = tmp_path / "cache_entry"
    cache_entry.mkdir()
    for shard_index, value in [(1, 20), (0, 10)]:
        path = cache_entry / f"parquet-train-{shard_index:05d}-of-00002.arrow"
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_stream(sink, pa.table({"value": [value]}).schema) as writer:
                writer.write_table(pa.table({"value": [value]}))
    (cache_entry / "dataset_info.json").write_text(
        json.dumps({"splits": {"train": {"num_examples": 2, "num_shards": 2}}}), encoding="utf-8"
    )

    _, paths = _data_loader._find_arrow_cache_entry(cache_entry, expected_num_rows=2)
    dataset = datasets.concatenate_datasets([datasets.Dataset.from_file(str(path)) for path in paths])

    assert dataset["value"] == [10, 20]


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_create_torch_dataset_uses_explicit_dataset_and_cache_paths(monkeypatch, tmp_path):
    dataset_root = tmp_path / "lerobot" / "owner" / "dataset"
    dataset_root.mkdir(parents=True)
    cache_dir = tmp_path / "datasets_cache"
    cache_dir.mkdir()
    calls = {}

    class FakeMetadata:
        fps = 20

        def __init__(self, repo_id, *, root=None):
            calls["metadata"] = (repo_id, root)

    class FakeLeRobotDataset:
        def __init__(self, repo_id, *, root=None, delta_timestamps=None):
            calls["dataset"] = (repo_id, root, delta_timestamps)

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", FakeLeRobotDataset)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", datasets.config.HF_DATASETS_CACHE)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets.config.HF_DATASETS_CACHE))

    data_config = _config.DataConfig(
        repo_id="owner/dataset",
        lerobot_dataset_root=str(dataset_root),
        hf_datasets_cache_dir=str(cache_dir),
        action_sequence_keys=("actions",),
    )
    model_config = pi0_config.Pi0Config(action_dim=7, action_horizon=3, max_token_len=48)
    _data_loader.create_torch_dataset(data_config, action_horizon=3, model_config=model_config)

    resolved_root = dataset_root.resolve()
    assert calls["metadata"] == ("owner/dataset", resolved_root)
    assert calls["dataset"] == (
        "owner/dataset",
        resolved_root,
        {"actions": [0.0, 0.05, 0.1]},
    )
    assert str(cache_dir.resolve()) == datasets.config.HF_DATASETS_CACHE
    assert cache_dir.is_dir()


def test_create_torch_dataset_can_use_arrow_cache_without_source_parquet(monkeypatch, tmp_path):
    dataset_root = tmp_path / "lerobot" / "owner" / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    arrow_cache_dir = tmp_path / "datasets_cache"
    arrow_cache_dir.mkdir()
    calls = {}

    class FakeMetadata:
        fps = 20

        def __init__(self, repo_id, *, root=None):
            calls["metadata"] = (repo_id, root)

    def fake_create(repo_id, root, cache_root, delta_timestamps, dataset_meta):
        calls["arrow"] = (repo_id, root, cache_root, delta_timestamps, dataset_meta)
        return object()

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader, "_create_arrow_cached_lerobot_dataset", fake_create)

    data_config = _config.DataConfig(
        repo_id="owner/dataset",
        lerobot_dataset_root=str(dataset_root),
        lerobot_arrow_cache_dir=str(arrow_cache_dir),
        action_sequence_keys=("actions",),
    )
    model_config = pi0_config.Pi0Config(action_dim=7, action_horizon=3, max_token_len=48)
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon=3, model_config=model_config)

    assert dataset is not None
    assert calls["metadata"] == ("owner/dataset", dataset_root.resolve())
    assert calls["arrow"][:4] == (
        "owner/dataset",
        dataset_root.resolve(),
        arrow_cache_dir.resolve(),
        {"actions": [0.0, 0.05, 0.1]},
    )


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
