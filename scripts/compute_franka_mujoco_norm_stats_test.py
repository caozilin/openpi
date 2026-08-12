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
        mean=np.arange(19),
        std=np.arange(19) + 1,
        q01=np.arange(19) + 2,
        q99=np.arange(19) + 3,
    )

    sliced = compute_stats._slice_stats(stats, 7)  # noqa: SLF001

    np.testing.assert_array_equal(sliced.mean, np.arange(7))
    np.testing.assert_array_equal(sliced.std, np.arange(7) + 1)
    np.testing.assert_array_equal(sliced.q01, np.arange(7) + 2)
    np.testing.assert_array_equal(sliced.q99, np.arange(7) + 3)


def _fixed_size_list(values: np.ndarray) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[-1])


def test_main_writes_joint19_and_derived_pi05_stats(tmp_path):
    dataset_root = tmp_path / "dataset"
    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()
    frame_count = 24
    state = np.arange(frame_count * 7, dtype=np.float32).reshape(frame_count, 7)
    actions = state + 1_000
    target_pose = np.arange(frame_count * 6, dtype=np.float32).reshape(frame_count, 6) + 2_000
    tolerance = target_pose + 1_000
    table = pa.table(
        {
            "state": _fixed_size_list(state),
            "actions": _fixed_size_list(actions),
            "stage_target_pose": _fixed_size_list(target_pose),
            "rotation_tolerance": _fixed_size_list(tolerance),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    (meta_dir / "info.json").write_text(json.dumps({"total_episodes": 1, "total_frames": frame_count}))
    joint_output = tmp_path / "joint19"
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
    assert joint["actions"].mean.shape == (19,)
    assert pi05["state"].mean.shape == (7,)
    assert pi05["actions"].mean.shape == (7,)
    for field in ("mean", "std", "q01", "q99"):
        np.testing.assert_array_equal(getattr(pi05["state"], field), getattr(joint["state"], field))
        np.testing.assert_array_equal(getattr(pi05["actions"], field), getattr(joint["actions"], field)[:7])
