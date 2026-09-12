"""Compute Franka MuJoCo normalization files from LeRobot Parquet or Arrow.

Both backends read only the numeric columns needed for normalization and never
decode images. The 15-D action target is computed once, then its first seven
dimensions are reused for the original pi0.5 normalization file.
"""

from collections.abc import Iterator
import json
import os
import pathlib
import re
from typing import Literal

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize

STATE_DIM = 7
ACTION_DIM = 7
TOLERANCE_FRAME_CODE_DIM = 2
ROTATION_TOLERANCE_DIM = 6
JOINT_ACTION_DIM = ACTION_DIM + TOLERANCE_FRAME_CODE_DIM + ROTATION_TOLERANCE_DIM
PARQUET_COLUMNS = ("state", "actions", "tolerance_frame_code", "rotation_tolerance")
ARROW_COLUMNS = (*PARQUET_COLUMNS, "episode_index")


def _vector_column_to_numpy(column: pa.ChunkedArray, *, name: str, dimension: int) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        if array.type.list_size != dimension:
            raise ValueError(f"Expected {name} vectors of length {dimension}, got {array.type}")
        flat_values = array.values
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
        if not np.all(lengths == dimension):
            raise ValueError(f"Expected every {name} vector to have length {dimension}")
        flat_values = pc.list_flatten(array)
    else:
        raise ValueError(f"Expected {name} to be a list-valued column, got {array.type}")
    values = flat_values.to_numpy(zero_copy_only=False).reshape(len(array), dimension)
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Column {name} contains non-finite values")
    return values


def _arrays_from_table(table: pa.Table, *, source: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    missing = set(PARQUET_COLUMNS).difference(table.column_names)
    if missing:
        raise ValueError(f"{source}: missing numeric columns {sorted(missing)}")
    state = _vector_column_to_numpy(table["state"], name="state", dimension=STATE_DIM)
    actions = _vector_column_to_numpy(table["actions"], name="actions", dimension=ACTION_DIM)
    frame_code = _vector_column_to_numpy(
        table["tolerance_frame_code"], name="tolerance_frame_code", dimension=TOLERANCE_FRAME_CODE_DIM
    )
    tolerance = _vector_column_to_numpy(
        table["rotation_tolerance"], name="rotation_tolerance", dimension=ROTATION_TOLERANCE_DIM
    )
    row_count = len(state)
    if not (row_count == len(actions) == len(frame_code) == len(tolerance)):
        raise ValueError(f"Numeric column lengths do not match in {source}")
    return state, np.concatenate((actions, frame_code, tolerance), axis=-1)


def _iter_lerobot_episodes(
    dataset_root: pathlib.Path, expected_episodes: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquet_paths) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episode Parquets, found {len(parquet_paths)}")
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=list(PARQUET_COLUMNS))
        yield _arrays_from_table(table, source=parquet_path)


def _find_arrow_shards(arrow_cache_dir: pathlib.Path, expected_frames: int) -> tuple[pathlib.Path, list[pathlib.Path]]:
    info_paths = (
        [arrow_cache_dir / "dataset_info.json"]
        if (arrow_cache_dir / "dataset_info.json").is_file()
        else sorted(arrow_cache_dir.rglob("dataset_info.json"))
    )
    matches: list[tuple[pathlib.Path, list[pathlib.Path]]] = []
    for info_path in info_paths:
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            split = info.get("splits", {}).get("train", {})
            shard_lengths = split.get("shard_lengths")
            num_rows = split.get("num_examples")
            if num_rows is None and shard_lengths is not None:
                num_rows = sum(shard_lengths)
        except (OSError, TypeError, ValueError):
            continue
        shards = sorted(
            (
                path
                for path in info_path.parent.glob("*.arrow")
                if re.fullmatch(r"parquet-train-\d+-of-\d+\.arrow", path.name)
            ),
            key=lambda path: int(path.name.split("-")[2]),
        )
        expected_shards = split.get("num_shards")
        if expected_shards is None and shard_lengths is not None:
            expected_shards = len(shard_lengths)
        if num_rows == expected_frames and shards and (expected_shards is None or len(shards) == expected_shards):
            matches.append((info_path.parent, shards))
    if not matches:
        raise FileNotFoundError(
            f"No complete Arrow cache with {expected_frames} rows was found under {arrow_cache_dir}"
        )
    if len(matches) > 1:
        locations = "\n".join(f"  - {entry}" for entry, _ in matches)
        raise RuntimeError(f"Multiple matching Arrow caches found; pass one exact cache directory:\n{locations}")
    return matches[0]


def _iter_arrow_episodes(
    arrow_cache_dir: pathlib.Path, expected_frames: int, expected_episodes: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    cache_entry, shard_paths = _find_arrow_shards(arrow_cache_dir, expected_frames)
    tables = []
    for shard_path in shard_paths:
        shard = datasets.Dataset.from_file(str(shard_path))
        missing = set(ARROW_COLUMNS).difference(shard.column_names)
        if missing:
            raise ValueError(f"{shard_path}: missing Arrow columns {sorted(missing)}")
        tables.append(shard.select_columns(list(ARROW_COLUMNS)).data.table)
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if len(table) != expected_frames:
        raise ValueError(f"Expected {expected_frames} Arrow rows, found {len(table)} in {cache_entry}")
    episode_indices = np.asarray(table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64)
    if len(episode_indices) == 0 or np.any(np.diff(episode_indices) < 0):
        raise ValueError("Arrow episode_index must be non-empty and monotonically non-decreasing")
    boundaries = np.flatnonzero(np.diff(episode_indices)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(episode_indices)]))
    unique_episode_indices = episode_indices[starts]
    if len(starts) != expected_episodes or not np.array_equal(
        unique_episode_indices, np.arange(expected_episodes, dtype=np.int64)
    ):
        raise ValueError(
            f"Expected contiguous episode indices 0..{expected_episodes - 1}, got {unique_episode_indices.tolist()}"
        )
    numeric_table = table.select(list(PARQUET_COLUMNS))
    for start, stop in zip(starts, stops, strict=True):
        yield _arrays_from_table(numeric_table.slice(int(start), int(stop - start)), source=cache_entry)


def _action_chunks(values: np.ndarray, action_horizon: int) -> np.ndarray:
    """Create LeRobot-style future chunks, repeating the final episode frame."""
    if len(values) == 0:
        raise ValueError("An episode cannot be empty")
    indices = np.minimum(
        np.arange(len(values), dtype=np.int64)[:, None] + np.arange(action_horizon, dtype=np.int64)[None, :],
        len(values) - 1,
    )
    return values[indices]


def _slice_stats(stats: normalize.NormStats, stop: int) -> normalize.NormStats:
    return normalize.NormStats(
        mean=np.asarray(stats.mean)[..., :stop],
        std=np.asarray(stats.std)[..., :stop],
        q01=None if stats.q01 is None else np.asarray(stats.q01)[..., :stop],
        q99=None if stats.q99 is None else np.asarray(stats.q99)[..., :stop],
    )


def _stage_json(path: pathlib.Path, stats: dict[str, normalize.NormStats]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(normalize.serialize_json(stats), encoding="utf-8")
    return temporary_path


def main(
    dataset_root: pathlib.Path,
    source: Literal["auto", "lerobot", "arrow"] = "auto",
    arrow_cache_dir: pathlib.Path | None = None,
    action_horizon: int = 16,
    batch_size: int = 24,
    joint_output_dir: pathlib.Path = pathlib.Path("assets/pi05_franka_mujoco_joint15/caozilin/franka_mujoco"),
    pi05_output_dir: pathlib.Path = pathlib.Path("assets/pi05_franka_mujoco/caozilin/franka_mujoco"),
) -> None:
    """Scan one LeRobot dataset or completed Arrow cache and write both stats files."""
    dataset_root = dataset_root.expanduser().resolve()
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata does not exist: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected_episodes = int(info["total_episodes"])
    expected_frames = int(info["total_frames"])
    if source == "auto":
        source = "arrow" if arrow_cache_dir is not None else "lerobot"
    if source == "arrow":
        if arrow_cache_dir is None:
            raise ValueError("--arrow-cache-dir is required when --source arrow")
        arrow_cache_dir = arrow_cache_dir.expanduser().resolve()
        if not arrow_cache_dir.is_dir():
            raise FileNotFoundError(f"Arrow cache directory does not exist: {arrow_cache_dir}")
        episodes = _iter_arrow_episodes(arrow_cache_dir, expected_frames, expected_episodes)
    else:
        episodes = _iter_lerobot_episodes(dataset_root, expected_episodes)

    state_stats = normalize.RunningStats()
    joint_action_stats = normalize.RunningStats()
    processed_frames = 0
    pending_states = np.empty((0, STATE_DIM), dtype=np.float32)
    pending_joint_actions = np.empty((0, action_horizon, JOINT_ACTION_DIM), dtype=np.float32)
    progress = tqdm.tqdm(total=expected_frames, unit="frames", desc="Computing Joint15 stats")
    try:
        for state, joint_actions in episodes:
            row_count = len(state)
            pending_states = np.concatenate((pending_states, state), axis=0)
            pending_joint_actions = np.concatenate(
                (pending_joint_actions, _action_chunks(joint_actions, action_horizon)), axis=0
            )
            complete_count = len(pending_states) // batch_size * batch_size
            for offset in range(0, complete_count, batch_size):
                state_stats.update(pending_states[offset : offset + batch_size])
                joint_action_stats.update(pending_joint_actions[offset : offset + batch_size])
            pending_states = pending_states[complete_count:]
            pending_joint_actions = pending_joint_actions[complete_count:]
            processed_frames += row_count
            progress.update(row_count)
    finally:
        progress.close()

    if processed_frames != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, processed {processed_frames}")
    if len(pending_states):
        # The standard OpenPI script uses floor(len(dataset) / batch_size), so it
        # also omits an incomplete final batch. The production dataset has no remainder.
        print(f"Omitting {len(pending_states)} trailing frames to match the standard OpenPI script")

    joint_stats = {
        "state": state_stats.get_statistics(),
        "actions": joint_action_stats.get_statistics(),
    }
    pi05_stats = {
        "state": joint_stats["state"],
        "actions": _slice_stats(joint_stats["actions"], ACTION_DIM),
    }
    joint_path = joint_output_dir.expanduser().resolve() / "norm_stats.json"
    pi05_path = pi05_output_dir.expanduser().resolve() / "norm_stats.json"
    staged_joint_path = _stage_json(joint_path, joint_stats)
    staged_pi05_path = _stage_json(pi05_path, pi05_stats)
    os.replace(staged_joint_path, joint_path)
    os.replace(staged_pi05_path, pi05_path)

    print(f"Processed {processed_frames} frames from {expected_episodes} episodes using {source}")
    print(f"Joint15 stats (state=7, actions=15): {joint_path}")
    print(f"pi0.5 stats (state=7, actions=7): {pi05_path}")


if __name__ == "__main__":
    tyro.cli(main)
