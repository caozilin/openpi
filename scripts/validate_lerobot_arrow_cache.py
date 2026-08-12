"""Lightweight validation of a completed Hugging Face Arrow cache for a LeRobot dataset."""

import argparse
import json
import pathlib
import re

import datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import hf_transform_to_torch


def main(
    dataset_root: str,
    arrow_cache_dir: str,
    repo_id: str = "caozilin/franka_mujoco",
    expected_frames: int | None = None,
    sample_indices: tuple[int, ...] = (0, -1),
) -> None:
    """Validate metadata, Arrow completeness, schema, and a few decoded samples without loading a model."""
    root = pathlib.Path(dataset_root).expanduser().resolve()
    cache_root = pathlib.Path(arrow_cache_dir).expanduser().resolve()
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    expected_frames = expected_frames or metadata.total_frames

    candidates = []
    info_paths = [cache_root / "dataset_info.json"] if (cache_root / "dataset_info.json").is_file() else []
    if not info_paths:
        info_paths = sorted(cache_root.rglob("dataset_info.json"))
    for info_path in info_paths:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        split = info.get("splits", {}).get("train", {})
        shard_lengths = split.get("shard_lengths")
        num_rows = split.get("num_examples", sum(shard_lengths) if shard_lengths is not None else None)
        shards = sorted(
            (
                path
                for path in info_path.parent.glob("*.arrow")
                if re.fullmatch(r"parquet-train-\d+-of-\d+\.arrow", path.name)
            ),
            key=lambda path: int(path.name.split("-")[2]),
        )
        expected_shards = split.get("num_shards", len(shard_lengths) if shard_lengths is not None else None)
        if num_rows == expected_frames and shards and (expected_shards is None or len(shards) == expected_shards):
            candidates.append((info_path.parent, shards))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one complete {expected_frames}-frame cache, found {len(candidates)}")

    cache_entry, shard_paths = candidates[0]
    shards = [datasets.Dataset.from_file(str(path)) for path in shard_paths]
    dataset = datasets.concatenate_datasets(shards) if len(shards) > 1 else shards[0]
    if len(dataset) != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} frames, found {len(dataset)}")
    dataset.set_transform(hf_transform_to_torch)

    required_columns = {"timestamp", "episode_index", "frame_index", "task_index", "actions"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    print(f"cache_entry={cache_entry}")
    print(f"arrow_shards={len(shard_paths)}")
    print(f"frames={len(dataset)}")
    print(f"episodes={metadata.total_episodes}")
    print(f"columns={dataset.column_names}")
    for requested_index in sample_indices:
        index = requested_index if requested_index >= 0 else len(dataset) + requested_index
        sample = dataset[index]
        summary = {
            "index": index,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "task_index": int(sample["task_index"]),
            "action_shape": tuple(sample["actions"].shape),
        }
        for image_key in metadata.image_keys:
            summary[f"{image_key}_shape"] = tuple(sample[image_key].shape)
        print(f"sample={summary}")
    print("ARROW_CACHE_VALIDATION_OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--arrow-cache-dir", required=True)
    parser.add_argument("--repo-id", default="caozilin/franka_mujoco")
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--sample-indices", type=int, nargs="*", default=(0, -1))
    main(**vars(parser.parse_args()))
