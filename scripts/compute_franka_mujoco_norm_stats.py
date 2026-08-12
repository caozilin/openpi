"""Compute both Franka MuJoCo normalization files directly from LeRobot Parquet.

This intentionally bypasses ``datasets.load_dataset``: it reads only the four
numeric columns needed for normalization and never decodes images or creates an
Arrow cache. The 19-D action target is computed once, then its first seven
dimensions are reused for the original pi0.5 normalization file.
"""

import json
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize

STATE_DIM = 7
ACTION_DIM = 7
STAGE_TARGET_POSE_DIM = 6
ROTATION_TOLERANCE_DIM = 6
JOINT_ACTION_DIM = ACTION_DIM + STAGE_TARGET_POSE_DIM + ROTATION_TOLERANCE_DIM
PARQUET_COLUMNS = ("state", "actions", "stage_target_pose", "rotation_tolerance")


def _fixed_size_list_to_numpy(column: pa.ChunkedArray, *, name: str, dimension: int) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_fixed_size_list(array.type) or array.type.list_size != dimension:
        raise ValueError(f"Expected {name} to be fixed_size_list[{dimension}], got {array.type}")
    values = array.values.to_numpy(zero_copy_only=False).reshape(len(array), dimension)
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Column {name} contains non-finite values")
    return values


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
    action_horizon: int = 16,
    batch_size: int = 24,
    joint_output_dir: pathlib.Path = pathlib.Path(
        "assets/pi05_franka_mujoco_joint19/caozilin/franka_mujoco"
    ),
    pi05_output_dir: pathlib.Path = pathlib.Path("assets/pi05_franka_mujoco/caozilin/franka_mujoco"),
) -> None:
    """Scan one local LeRobot dataset and atomically write Joint19 and pi0.5 stats."""
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

    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquet_paths) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episode Parquets, found {len(parquet_paths)}")

    state_stats = normalize.RunningStats()
    joint_action_stats = normalize.RunningStats()
    processed_frames = 0
    pending_states = np.empty((0, STATE_DIM), dtype=np.float32)
    pending_joint_actions = np.empty((0, action_horizon, JOINT_ACTION_DIM), dtype=np.float32)
    progress = tqdm.tqdm(total=expected_frames, unit="frames", desc="Computing Joint19 stats")
    try:
        for parquet_path in parquet_paths:
            table = pq.read_table(parquet_path, columns=list(PARQUET_COLUMNS))
            state = _fixed_size_list_to_numpy(table["state"], name="state", dimension=STATE_DIM)
            actions = _fixed_size_list_to_numpy(table["actions"], name="actions", dimension=ACTION_DIM)
            target_pose = _fixed_size_list_to_numpy(
                table["stage_target_pose"], name="stage_target_pose", dimension=STAGE_TARGET_POSE_DIM
            )
            tolerance = _fixed_size_list_to_numpy(
                table["rotation_tolerance"], name="rotation_tolerance", dimension=ROTATION_TOLERANCE_DIM
            )
            row_count = len(state)
            if not (row_count == len(actions) == len(target_pose) == len(tolerance)):
                raise ValueError(f"Numeric column lengths do not match in {parquet_path}")

            joint_actions = np.concatenate((actions, target_pose, tolerance), axis=-1)
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

    print(f"Processed {processed_frames} frames from {expected_episodes} episodes")
    print(f"Joint19 stats (state=7, actions=19): {joint_path}")
    print(f"pi0.5 stats (state=7, actions=7): {pi05_path}")


if __name__ == "__main__":
    tyro.cli(main)
