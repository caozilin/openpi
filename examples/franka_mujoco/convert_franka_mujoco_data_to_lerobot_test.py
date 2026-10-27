# ruff: noqa: SLF001
import dataclasses
import hashlib
import json
from pathlib import Path

import cv2
import datasets
import numpy as np
import pytest

from examples.franka_mujoco import convert_franka_mujoco_data_to_lerobot as converter
from openpi import transforms
from openpi.shared import franka_mujoco_dataset as dataset_format
from openpi.training import config
from openpi.training import data_loader


def make_task(root: Path) -> Path:
    task = root / "panda" / "scene" / "task"
    task.mkdir(parents=True)
    metadata = {
        "schema_version": "8.0",
        "dataset_release_version": "5.0",
        "frequency_hz": 10,
        "robot": {
            "uid": "panda",
            "arm_dof": 7,
            "joint_names": [f"joint{i}" for i in range(7)],
            "joint_state_dimension": 7,
        },
        "action_format": {
            "dimension": 7,
            "rotation": "base-frame rotation-vector delta RxRyRz in radians; left-multiplied",
        },
        "phase_labels": converter.PHASE_LABELS,
        "video": {"resolution": [224, 224], "fps": 10},
        "tolerance_labels": dataset_format.tolerance_label_metadata(),
    }
    (task / "task_metadata.json").write_text(json.dumps(metadata))
    (task / "manifest.json").write_text(json.dumps({"schema_version": "8.0", "episodes": []}))
    return task


def make_episode(task: Path, index: int = 0, *, labeled: bool = True) -> tuple[Path, dict]:
    episode = task / f"episode_{index:04d}_seed_{index + 7}"
    episode.mkdir()
    phases = ["pregrasp", "grasp", "postgrasp", "release"] * 2
    frames = [
        {
            "index": i,
            "video_frame_index": i,
            "time_s": i / 10,
            "phase": phase,
            "subtask_index": i // 4,
            "action": [i * 0.001, 0.0, 0.0, 0.01, 0.02, 0.03, -1.0],
            "ee_state": {"position_m": [0.4, 0.0, 0.5], "rotation_vector_base_rad": [3.1, 0.0, 0.0]},
            "gripper_state": {"finger_positions_m": [0.03, 0.04]},
            "joint_state": {"position_rad": [0.0] * 7, "velocity_rad_s": [0.0] * 7},
            "nominal_target_pose": {"position_m": [0.4, 0.0, 0.5], "rotation_vector_base_rad": [3.11, 0.02, 0.03]},
        }
        for i, phase in enumerate(phases)
    ]
    entry = {
        "episode_index": index,
        "seed": index + 7,
        "frames": 8,
        "success": True,
        "path": episode.name,
        "instruction": "pick up the block",
    }
    document = {
        "schema_version": "8.0",
        "episode": {
            "episode_index": index,
            "seed": index + 7,
            "frame_count": 8,
            "video_frame_counts": {"main_rgb.mp4": 8, "wrist_rgb.mp4": 8},
            "instruction": entry["instruction"],
            "result": {"success": True},
        },
        "trajectory": frames,
    }
    path = episode / "trajectory.json"
    path.write_text(json.dumps(document))
    if labeled:
        labels = {
            **dataset_format.tolerance_label_metadata(),
            "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frame_count": 8,
            "source": {"method": "external_label_script"},
            "subtasks": [
                {"subtask_index": i, "pregrasp": [0.1 + i / 10] * 6, "postgrasp": [0.3 + i / 10] * 6} for i in range(2)
            ],
        }
        (episode / "tolerance_labels.json").write_text(json.dumps(labels))
    for name in ("main_rgb.mp4", "wrist_rgb.mp4"):
        writer = cv2.VideoWriter(str(episode / name), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224))
        assert writer.isOpened()
        for i in range(8):
            writer.write(np.full((224, 224, 3), 20 * i, dtype=np.uint8))
        writer.release()
    manifest_path = task / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"].append(entry)
    manifest_path.write_text(json.dumps(manifest))
    return episode, entry


@pytest.fixture
def source(tmp_path):
    task = make_task(tmp_path / "raw")
    episode, entry = make_episode(task)
    return task, episode, entry


def test_prepare_episode_uses_deferred_subtask_labels_and_aligned_video(source):
    _, episode, entry = source
    original = {p.name: p.read_bytes() for p in episode.iterdir()}
    frames = converter._prepare_episode(
        episode,
        entry,
        joint_task_tolerance=True,
        robot_uid="panda",
        width=224,
        height=224,
    )
    assert len(frames) == 8
    np.testing.assert_allclose([f["rotation_tolerance"][0] for f in frames], [0.1, 0.0, 0.3, 0.0, 0.2, 0.0, 0.4, 0.0])
    np.testing.assert_array_equal([f["subtask_index"][0] for f in frames], [0] * 4 + [1] * 4)
    np.testing.assert_allclose(frames[0]["state"], [0.4, 0.0, 0.5, 3.1, 0.0, 0.0, 0.035])
    np.testing.assert_allclose(frames[0]["actions"][3:6], [0.01, 0.02, 0.03])
    for i, frame in enumerate(frames):
        assert abs(float(frame["image"].mean()) - i * 20) < 5
        np.testing.assert_array_equal(frame["image"], frame["wrist_image"])
    assert {p.name: p.read_bytes() for p in episode.iterdir()} == original


def test_action_only_preparation_never_fabricates_tolerance(source):
    _, episode, entry = source
    (episode / "tolerance_labels.json").unlink()
    frames = converter._prepare_episode(
        episode,
        entry,
        joint_task_tolerance=False,
        robot_uid="panda",
        width=224,
        height=224,
    )
    assert all("rotation_tolerance" not in frame for frame in frames)
    with pytest.raises(FileNotFoundError, match="missing tolerance labels"):
        converter._prepare_episode(
            episode,
            entry,
            joint_task_tolerance=True,
            robot_uid="panda",
            width=224,
            height=224,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "7.0"),
        ("dataset_release_version", "4.2"),
        ("video", {"resolution": [640, 480], "fps": 10}),
        ("tolerance_labels", {}),
    ],
)
def test_task_metadata_rejects_incompatible_format(source, field, value):
    task, _, _ = source
    path = task / "task_metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=".+"):
        converter._validate_task(task)


def test_task_metadata_is_required(source):
    task, _, _ = source
    (task / "task_metadata.json").unlink()
    with pytest.raises(ValueError, match="Failed to read JSON"):
        converter._validate_task(task)


@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("variant", ["nominal", "tolerance"])
def test_conversion_export_and_resume(tmp_path, monkeypatch, mode, variant):
    import lerobot.common.datasets.lerobot_dataset as lerobot
    import pyarrow.parquet as pq

    raw = tmp_path / "raw"
    task = make_task(raw)
    episode, _ = make_episode(task, labeled=mode)
    manifest_path = task / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][0]["execution_variant"] = variant
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(lerobot, "HF_LEROBOT_HOME", tmp_path / "exports")
    kwargs = {
        "raw_dir": raw,
        "repo_id": "owner/test",
        "joint_task_tolerance": mode,
        "workers": 2,
        "image_writer_threads": 0,
        "image_writer_processes": 0,
    }
    converter.main(**kwargs)
    root = tmp_path / "exports/owner/test"
    info = json.loads((root / "meta/info.json").read_text())
    dataset_format.validate_export_info(info, joint_task_tolerance=mode)
    converter._validate_dataset_features(info, joint_task_tolerance=mode)
    assert info["total_frames"] == 8
    table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
    assert ("rotation_tolerance" in table.column_names) == mode
    np.testing.assert_array_equal(table["trajectory_is_tolerance"].to_numpy(), int(variant == "tolerance"))
    assert table["actions"].to_pylist()[0][3:6] == pytest.approx([0.01, 0.02, 0.03])
    cache = tmp_path / "hf_cache"
    cache.mkdir()
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(cache))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache))
    cfg = config.get_config("pi05_franka_mujoco_joint13" if mode else "pi05_franka_mujoco")
    factory = dataclasses.replace(
        cfg.data,
        repo_id="owner/test",
        lerobot_dataset_root=str(root),
        tolerance_trajectory_action_loss_weight=4.0,
    )
    data_config = factory.create(tmp_path / "assets", cfg.model)
    loaded = data_loader.create_torch_dataset(data_config, action_horizon=3, model_config=cfg.model)
    transformed = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ]
    )(loaded[0])
    assert transformed["actions"].shape == (3, 13 if mode else 7)
    assert transformed["action_loss_weight"] == (4.0 if variant == "tolerance" else 1.0)
    if mode:
        np.testing.assert_allclose(transformed["actions"][:, 7:], [[0.1] * 6, [0.0] * 6, [0.3] * 6])
    before = (root / "data/chunk-000/episode_000000.parquet").read_bytes()
    converter.main(**kwargs, resume=True)
    make_episode(task, 1, labeled=mode)
    converter.main(**kwargs, resume=True)
    assert (root / "data/chunk-000/episode_000000.parquet").read_bytes() == before
    assert json.loads((root / "meta/info.json").read_text())["total_episodes"] == 2
    if mode:
        with pytest.raises(ValueError, match="mode"):
            converter.main(**{**kwargs, "joint_task_tolerance": False}, resume=True)
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][0]["execution_variant"] = "tolerance" if variant == "nominal" else "nominal"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest.*changed"):
        converter.main(**kwargs, resume=True)
    manifest["episodes"][0]["execution_variant"] = variant
    manifest_path.write_text(json.dumps(manifest))
    changed = episode / ("tolerance_labels.json" if mode else "trajectory.json")
    document = json.loads(changed.read_text())
    document["source" if mode else "note"] = {"method": "changed"}
    changed.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="changed"):
        converter.main(**kwargs, resume=True)


@pytest.mark.parametrize("location", ["manifest", "episode", "frame"])
@pytest.mark.parametrize("variant", ["nominal", "tolerance"])
def test_execution_variant_maps_to_binary_label(location, variant):
    document = {"episode": {}, "trajectory": [{}]}
    manifest = {}
    record = {"manifest": manifest, "episode": document["episode"], "frame": document["trajectory"][0]}[location]
    record["execution_variant"] = variant
    assert converter.trajectory_is_tolerance(
        document,
        manifest,
        source=Path("trajectory.json"),
    ) == (variant == "tolerance")


def test_base_collection_without_variant_is_nominal():
    assert not converter.trajectory_is_tolerance(
        {"episode": {}, "trajectory": [{}]},
        {},
        source=Path("trajectory.json"),
    )


@pytest.mark.parametrize("value", ["boundary_tolerance", "unknown", "", None, True, 1, [], {}])
def test_invalid_execution_variant_is_rejected(value):
    with pytest.raises(ValueError, match="invalid execution_variant"):
        converter.trajectory_is_tolerance(
            {"episode": {}, "trajectory": [{}]},
            {"execution_variant": value},
            source=Path("trajectory.json"),
        )


def test_conflicting_execution_variants_are_rejected():
    with pytest.raises(ValueError, match="inconsistent execution_variant"):
        converter.trajectory_is_tolerance(
            {"episode": {"execution_variant": "nominal"}, "trajectory": [{}]},
            {"execution_variant": "tolerance"},
            source=Path("trajectory.json"),
        )


def test_robot_joint_padding_is_validated():
    frame = {"joint_state": {"position_rad": [0.0] * 7, "velocity_rad_s": [0.0] * 7}}
    converter.joint_state_from_frame(frame, robot_uid="ur5e", source=Path("trajectory.json"))
    frame["joint_state"]["position_rad"][-1] = 0.1
    with pytest.raises(ValueError, match="zero padding"):
        converter.joint_state_from_frame(frame, robot_uid="ur5e", source=Path("trajectory.json"))


def test_repair_partial_dataset_rolls_back_uncommitted_tail(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset"
    meta_path = output_path / "meta"
    data_path = output_path / "data/chunk-000"
    image_path = output_path / "images/image/episode_000002"
    meta_path.mkdir(parents=True)
    data_path.mkdir(parents=True)
    image_path.mkdir(parents=True)
    (meta_path / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 3,
                "total_frames": 999,
                "total_tasks": 1,
                "total_videos": 0,
                "total_chunks": 1,
                "chunks_size": 1000,
                "splits": {"train": "0:3"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {"state": {"dtype": "float32"}},
            }
        ),
        encoding="utf-8",
    )
    (meta_path / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, "tasks": ["a"], "length": 5}),
                json.dumps({"episode_index": 1, "tasks": ["a"], "length": 6}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (meta_path / "episodes_stats.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, "stats": {}}),
                json.dumps({"episode_index": 1, "stats": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (meta_path / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "a"}) + "\n", encoding="utf-8")
    for episode_index in range(3):
        (data_path / f"episode_{episode_index:06d}.parquet").write_bytes(b"parquet")
    (image_path / "frame_000000.png").write_bytes(b"png")

    committed = converter._repair_partial_dataset(output_path)

    assert committed == 2
    assert (data_path / "episode_000001.parquet").is_file()
    assert not (data_path / "episode_000002.parquet").exists()
    assert not (output_path / "images").exists()
    repaired_info = json.loads((meta_path / "info.json").read_text(encoding="utf-8"))
    assert repaired_info["total_episodes"] == 2
    assert repaired_info["total_frames"] == 11
    assert repaired_info["splits"] == {"train": "0:2"}


def test_write_episode_discards_accumulated_hf_dataset_before_saving() -> None:
    empty_dataset = object()

    class FakeDataset:
        hf_dataset = object()

        def create_hf_dataset(self) -> object:
            return empty_dataset

        def add_frame(self, frame: dict) -> None:
            assert self.hf_dataset is empty_dataset
            assert frame == {"frame": 1}

        def save_episode(self) -> None:
            assert self.hf_dataset is empty_dataset

    dataset = FakeDataset()
    converter._write_episode(dataset, [{"frame": 1}])

    assert dataset.hf_dataset is empty_dataset


@pytest.mark.parametrize("mode", [False, True])
def test_protected_append_uses_current_schema_and_journal(tmp_path, monkeypatch, mode):
    import importlib
    import sys

    import lerobot.common.datasets.lerobot_dataset as lerobot

    monkeypatch.setitem(sys.modules, "convert_franka_mujoco_data_to_lerobot", converter)
    append = importlib.import_module("examples.franka_mujoco.append_missing_franka_mujoco_episodes")
    raw = tmp_path / "raw"
    task = make_task(raw)
    make_episode(task, labeled=mode)
    make_episode(task, 1, labeled=mode)
    monkeypatch.setattr(lerobot, "HF_LEROBOT_HOME", tmp_path / "exports")
    kwargs = {
        "raw_dir": raw,
        "repo_id": "owner/append",
        "joint_task_tolerance": mode,
        "image_writer_threads": 0,
        "image_writer_processes": 0,
    }
    converter.main(**kwargs, max_episodes=1)
    root = tmp_path / "exports/owner/append"
    parquet = root / "data/chunk-000/episode_000000.parquet"
    before = parquet.read_bytes()
    baseline = tmp_path / "baseline.sha256"
    baseline.write_text(f"{hashlib.sha256(before).hexdigest()}  data/chunk-000/episode_000000.parquet\n")
    append.main(
        raw_task_dir=task,
        repo_id="owner/append",
        baseline_path=baseline,
        raw_dir=raw,
        expected_existing_episodes=1,
        expected_existing_frames=8,
        append_start=1,
        append_count=1,
        expected_append_frames=8,
        joint_task_tolerance=mode,
        image_writer_threads=0,
        workers=2,
        execute=True,
    )
    assert parquet.read_bytes() == before
    assert json.loads((root / "meta/info.json").read_text())["total_episodes"] == 2
    converter.main(**kwargs, resume=True)
    baseline.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
            for path in sorted((root / "data").rglob("episode_*.parquet"))
        )
    )
    with pytest.raises(ValueError, match="already converted"):
        append.main(
            raw_task_dir=task,
            repo_id="owner/append",
            baseline_path=baseline,
            raw_dir=raw,
            expected_existing_episodes=2,
            expected_existing_frames=16,
            append_start=1,
            append_count=1,
            expected_append_frames=8,
            joint_task_tolerance=mode,
            execute=False,
        )
