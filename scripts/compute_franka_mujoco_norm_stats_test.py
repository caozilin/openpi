import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
        mean=np.arange(15),
        std=np.arange(15) + 1,
        q01=np.arange(15) + 2,
        q99=np.arange(15) + 3,
    )

    sliced = compute_stats._slice_stats(stats, 7)  # noqa: SLF001

    np.testing.assert_array_equal(sliced.mean, np.arange(7))
    np.testing.assert_array_equal(sliced.std, np.arange(7) + 1)
    np.testing.assert_array_equal(sliced.q01, np.arange(7) + 2)
    np.testing.assert_array_equal(sliced.q99, np.arange(7) + 3)


def _fixed_size_list(values: np.ndarray) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[-1])


def test_main_writes_joint15_and_derived_pi05_stats(tmp_path):
    dataset_root = tmp_path / "dataset"
    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()
    frame_count = 24
    state = np.arange(frame_count * 7, dtype=np.float32).reshape(frame_count, 7)
    actions = state + 1_000
    frame_code = np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2) + 2_000
    tolerance = np.arange(frame_count * 6, dtype=np.float32).reshape(frame_count, 6) + 3_000
    table = pa.table(
        {
            "state": _fixed_size_list(state),
            "actions": _fixed_size_list(actions),
            "tolerance_frame_code": _fixed_size_list(frame_code),
            "rotation_tolerance": _fixed_size_list(tolerance),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    (meta_dir / "info.json").write_text(json.dumps({"total_episodes": 1, "total_frames": frame_count}))
    joint_output = tmp_path / "joint15"
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
    assert joint["actions"].mean.shape == (15,)
    assert pi05["state"].mean.shape == (7,)
    assert pi05["actions"].mean.shape == (7,)
    for field in ("mean", "std", "q01", "q99"):
        np.testing.assert_array_equal(getattr(pi05["state"], field), getattr(joint["state"], field))
        np.testing.assert_array_equal(getattr(pi05["actions"], field), getattr(joint["actions"], field)[:7])


def test_arrow_and_lerobot_backends_produce_identical_stats(tmp_path):
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
        frame_code = np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2) + offset + 200
        tolerance = np.arange(frame_count * 6, dtype=np.float32).reshape(frame_count, 6) + offset + 300
        table = pa.table(
            {
                "state": _fixed_size_list(state),
                "actions": _fixed_size_list(actions),
                "tolerance_frame_code": _fixed_size_list(frame_code),
                "rotation_tolerance": _fixed_size_list(tolerance),
                "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
            }
        )
        tables.append(table)
        pq.write_table(table.select(compute_stats.PARQUET_COLUMNS), data_dir / f"episode_{episode_index:06d}.parquet")

    total_frames = sum(episode_lengths)
    (meta_dir / "info.json").write_text(
        json.dumps({"total_episodes": len(episode_lengths), "total_frames": total_frames})
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
    kwargs = {"action_horizon": 3, "batch_size": 4}
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

    for lerobot_stats, arrow_stats in (
        (normalize.load(lerobot_joint), normalize.load(arrow_joint)),
        (normalize.load(lerobot_pi05), normalize.load(arrow_pi05)),
    ):
        for key in ("state", "actions"):
            for field in ("mean", "std", "q01", "q99"):
                np.testing.assert_allclose(
                    getattr(lerobot_stats[key], field), getattr(arrow_stats[key], field), rtol=0, atol=0
                )
