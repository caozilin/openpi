import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.shared import franka_mujoco_dataset as dataset_format
from openpi.shared import normalize
from scripts import compute_franka_mujoco_norm_stats as compute_stats


def test_action_chunks_repeat_episode_end():
    values = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)

    chunks = compute_stats._action_chunks(values, action_horizon=3)  # noqa: SLF001

    np.testing.assert_array_equal(
        chunks,
        np.asarray([[[1.0], [2.0], [3.0]], [[2.0], [3.0], [3.0]], [[3.0], [3.0], [3.0]]]),
    )


def test_slice_stats_keeps_first_dimensions():
    stats = normalize.NormStats(
        mean=np.arange(13),
        std=np.arange(13) + 1,
        q01=np.arange(13) + 2,
        q99=np.arange(13) + 3,
    )

    sliced = compute_stats._slice_stats(stats, 7)  # noqa: SLF001

    np.testing.assert_array_equal(sliced.mean, np.arange(7))
    np.testing.assert_array_equal(sliced.std, np.arange(7) + 1)
    np.testing.assert_array_equal(sliced.q01, np.arange(7) + 2)
    np.testing.assert_array_equal(sliced.q99, np.arange(7) + 3)


def _fixed_size_list(values: np.ndarray) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[-1])


def test_main_writes_joint13_and_derived_pi05_stats(tmp_path):
    dataset_root = tmp_path / "dataset"
    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()
    frame_count = 24
    state = np.arange(frame_count * 7, dtype=np.float32).reshape(frame_count, 7)
    actions = state + 1_000
    tolerance = np.arange(frame_count * 6, dtype=np.float32).reshape(frame_count, 6) + 3_000
    table = pa.table(
        {
            "state": _fixed_size_list(state),
            "actions": _fixed_size_list(actions),
            "rotation_tolerance": _fixed_size_list(tolerance),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": frame_count,
                **dataset_format.export_metadata(joint_task_tolerance=True),
            }
        )
    )
    joint_output = tmp_path / "joint13"
    pi05_output = tmp_path / "pi05"

    compute_stats.main(
        dataset_root,
        action_horizon=3,
        batch_size=24,
        joint_output_dir=joint_output,
        pi05_output_dir=pi05_output,
    )

    joint = normalize.load(joint_output)
    pi05 = normalize.load(pi05_output)
    assert joint["state"].mean.shape == (7,)
    assert joint["actions"].mean.shape == (13,)
    assert pi05["state"].mean.shape == (7,)
    assert pi05["actions"].mean.shape == (7,)
    expected_actions = compute_stats._action_chunks(  # noqa: SLF001
        np.concatenate((actions, tolerance), axis=-1), action_horizon=3
    )
    np.testing.assert_allclose(joint["actions"].mean, expected_actions.mean(axis=(0, 1)))
    for field in ("mean", "std", "q01", "q99"):
        np.testing.assert_array_equal(getattr(pi05["state"], field), getattr(joint["state"], field))
        np.testing.assert_array_equal(getattr(pi05["actions"], field), getattr(joint["actions"], field)[:7])


@pytest.mark.parametrize("mode", [False, True])
def test_arrow_and_lerobot_backends_produce_identical_stats(tmp_path, mode):
    dataset_root = tmp_path / "dataset"
    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    arrow_dir = tmp_path / "arrow"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()
    arrow_dir.mkdir()

    episode_lengths = (5, 7)
    tables = []
    for episode_index, frame_count in enumerate(episode_lengths):
        offset = episode_index * 1_000
        state = np.arange(frame_count * 7, dtype=np.float32).reshape(frame_count, 7) + offset
        actions = state + 100
        tolerance = np.arange(frame_count * 6, dtype=np.float32).reshape(frame_count, 6) + offset + 300
        table = pa.table(
            {
                "state": _fixed_size_list(state),
                "actions": _fixed_size_list(actions),
                "rotation_tolerance": _fixed_size_list(tolerance),
                "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
            }
        )
        if not mode:
            table = table.drop_columns(["rotation_tolerance"])
        tables.append(table)
        pq.write_table(
            table.drop_columns(["episode_index"]),
            data_dir / f"episode_{episode_index:06d}.parquet",
        )

    total_frames = sum(episode_lengths)
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": len(episode_lengths),
                "total_frames": total_frames,
                **dataset_format.export_metadata(joint_task_tolerance=mode),
            }
        )
    )
    arrow_path = arrow_dir / "parquet-train-00000-of-00001.arrow"
    with pa.OSFile(str(arrow_path), "wb") as sink, pa.ipc.new_stream(sink, tables[0].schema) as writer:
        for table in tables:
            writer.write_table(table)
    (arrow_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "num_examples": total_frames,
                        "num_shards": 1,
                        "shard_lengths": [total_frames],
                    }
                }
            }
        )
    )

    lerobot_joint = tmp_path / "lerobot_joint"
    lerobot_pi05 = tmp_path / "lerobot_pi05"
    arrow_joint = tmp_path / "arrow_joint"
    arrow_pi05 = tmp_path / "arrow_pi05"
    kwargs = {"action_horizon": 3, "batch_size": 4, "joint_task_tolerance": mode}
    compute_stats.main(
        dataset_root,
        source="lerobot",
        joint_output_dir=lerobot_joint,
        pi05_output_dir=lerobot_pi05,
        **kwargs,
    )
    compute_stats.main(
        dataset_root,
        source="arrow",
        arrow_cache_dir=arrow_dir,
        joint_output_dir=arrow_joint,
        pi05_output_dir=arrow_pi05,
        **kwargs,
    )

    pairs = [(normalize.load(lerobot_pi05), normalize.load(arrow_pi05))]
    if mode:
        pairs.append((normalize.load(lerobot_joint), normalize.load(arrow_joint)))
    for lerobot_stats, arrow_stats in pairs:
        for key in ("state", "actions"):
            for field in ("mean", "std", "q01", "q99"):
                np.testing.assert_allclose(
                    getattr(lerobot_stats[key], field), getattr(arrow_stats[key], field), rtol=0, atol=0
                )


def test_action_only_statistics_need_no_tolerance_columns(tmp_path):
    dataset_root = tmp_path / "dataset"
    data = dataset_root / "data/chunk-000"
    meta = dataset_root / "meta"
    data.mkdir(parents=True)
    meta.mkdir()
    values = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)
    pq.write_table(
        pa.table({"state": _fixed_size_list(values), "actions": _fixed_size_list(values)}),
        data / "episode_000000.parquet",
    )
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 8,
                **dataset_format.export_metadata(joint_task_tolerance=False),
            }
        )
    )
    joint = tmp_path / "joint"
    baseline = tmp_path / "baseline"
    compute_stats.main(
        dataset_root, joint_task_tolerance=False, batch_size=24, joint_output_dir=joint, pi05_output_dir=baseline
    )
    assert normalize.load(baseline)["actions"].mean.shape == (7,)
    np.testing.assert_allclose(normalize.load(baseline)["state"].mean, values.mean(axis=0))
    assert not joint.exists()
