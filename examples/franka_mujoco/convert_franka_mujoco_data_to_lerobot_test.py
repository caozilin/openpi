from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import cv2
import numpy as np

from examples.franka_mujoco import convert_franka_mujoco_data_to_lerobot as converter


def test_state_from_frame_uses_ee_pose_and_symmetric_finger_position() -> None:
    state = converter.state_from_frame(
        {
            "ee_state": {
                "position_m": [0.1, 0.2, 0.3],
                "rotation_vector_base_rad": [3.1, 0.2, -0.1],
            },
            "gripper_state": {"finger_positions_m": [0.03, 0.04]},
        },
        source=Path("trajectory.json"),
    )

    np.testing.assert_allclose(state, [0.1, 0.2, 0.3, 3.1, 0.2, -0.1, 0.035])
    assert state.dtype == np.float32


def test_rotation_vector_to_6d_round_trips_through_gram_schmidt() -> None:
    rotation_vector = np.asarray([0.4, -0.3, 1.2], dtype=np.float32)
    encoded = converter.rotation_vector_to_6d(
        rotation_vector,
        field="rotation",
        source=Path("trajectory.json"),
    )

    first = encoded[:3] / np.linalg.norm(encoded[:3])
    second_raw = encoded[3:] - np.dot(first, encoded[3:]) * first
    second = second_raw / np.linalg.norm(second_raw)
    decoded = np.column_stack((first, second, np.cross(first, second)))
    expected, _ = cv2.Rodrigues(rotation_vector.astype(np.float64))

    np.testing.assert_allclose(decoded, expected, atol=1e-6)
    np.testing.assert_allclose(decoded.T @ decoded, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(decoded), 1.0, atol=1e-6)
    assert encoded.shape == (6,)
    assert encoded.dtype == np.float32


def test_joint_state_from_frame_preserves_fixed_width_and_ur5e_padding() -> None:
    position, velocity = converter.joint_state_from_frame(
        {
            "joint_state": {
                "position_rad": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
                "velocity_rad_s": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
            }
        },
        robot_uid="ur5e",
        source=Path("trajectory.json"),
    )

    np.testing.assert_allclose(position, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])
    np.testing.assert_allclose(velocity, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
    assert position.dtype == velocity.dtype == np.float32


def test_joint_state_from_frame_rejects_nonzero_ur5e_padding() -> None:
    try:
        converter.joint_state_from_frame(
            {
                "joint_state": {
                    "position_rad": [0.0] * 6 + [0.1],
                    "velocity_rad_s": [0.0] * 7,
                }
            },
            robot_uid="ur5e",
            source=Path("trajectory.json"),
        )
    except ValueError as error:
        assert "zero padding" in str(error)
    else:
        raise AssertionError("Expected nonzero UR5e padding to fail")


def test_task_directories_discovers_robot_task_hierarchy(tmp_path: Path) -> None:
    expected = []
    for robot_uid in converter.ROBOT_SPECS:
        task_dir = tmp_path / robot_uid / "blocks_blue_block_to_plate"
        task_dir.mkdir(parents=True)
        (task_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (task_dir / "task_metadata.json").write_text("{}", encoding="utf-8")
        expected.append(task_dir)
    failed_dir = tmp_path / "ur5e" / "blocks_blue_block_to_plate" / "failed_episodes"
    failed_dir.mkdir()
    (failed_dir / "manifest.json").write_text("{}", encoding="utf-8")

    assert converter._task_directories(tmp_path) == sorted(expected)


def test_validate_robot_accepts_three_robot_contracts() -> None:
    for robot_uid, spec in converter.ROBOT_SPECS.items():
        metadata = {
            "robot": {
                "uid": robot_uid,
                "arm_dof": spec["arm_dof"],
                "joint_names": [f"joint_{index}" for index in range(spec["arm_dof"])],
                "joint_state_dimension": 7,
            }
        }
        assert converter._validate_robot(metadata, source=Path("task_metadata.json")) == robot_uid


def test_validate_task_requires_preprocessed_224_square_video(tmp_path: Path) -> None:
    metadata = {
        "schema_version": "7.0",
        "dataset_release_version": "4.2",
        "frequency_hz": 10,
        "robot": {
            "uid": "panda",
            "arm_dof": 7,
            "joint_names": [f"joint_{index}" for index in range(7)],
            "joint_state_dimension": 7,
        },
        "action_format": {"dimension": 7},
        "phase_labels": converter.PHASE_LABELS,
        "video": {"resolution": [640, 360], "fps": 10},
        "annotation": {
            "rotation_tolerance_profiles_rad": converter.DEFAULT_ROTATION_TOLERANCE_PROFILES_RAD,
            "rotation_tolerance_profile_order": list(converter.ROTATION_TOLERANCE_PROFILE_ORDER),
        },
    }
    manifest = {"schema_version": "7.0"}
    (tmp_path / "task_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    try:
        converter._validate_task(tmp_path)
    except ValueError as error:
        assert "resize-with-pad processed to 224x224" in str(error)
    else:
        raise AssertionError("Expected non-224 video metadata to fail")


def test_validate_task_warns_and_uses_defaults_without_metadata(tmp_path: Path) -> None:
    task_dir = tmp_path / "panda" / "scene" / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "manifest.json").write_text(json.dumps({"schema_version": "7.0"}), encoding="utf-8")

    terminal = io.StringIO()
    with redirect_stdout(terminal):
        metadata, _, robot_uid, width, height = converter._validate_task(task_dir)

    assert "task_metadata.json is missing" in terminal.getvalue()
    assert metadata["annotation"]["rotation_tolerance_profiles_rad"] == (
        converter.DEFAULT_ROTATION_TOLERANCE_PROFILES_RAD
    )
    assert (robot_uid, width, height) == ("panda", 224, 224)


def test_require_vector_rejects_wrong_action_dimension() -> None:
    try:
        converter._require_vector([0.0] * 6, 7, field="action", source=Path("trajectory.json"))
    except ValueError as error:
        assert "shape (7,)" in str(error)
    else:
        raise AssertionError("Expected an invalid action dimension to fail")


def test_phase_from_frame_uses_schema_7_four_stage_ids() -> None:
    cases = (
        ("pregrasp", 0),
        ("grasp", 1),
        ("postgrasp", 2),
        ("release", 3),
    )
    for phase, expected in cases:
        value = converter.phase_from_frame(
            {"phase": phase},
            source=Path("trajectory.json"),
        )
        np.testing.assert_array_equal(value, [expected])
        assert value.dtype == np.int64


def test_phase_from_frame_rejects_legacy_integer_phase() -> None:
    try:
        converter.phase_from_frame(
            {"phase": 0},
            source=Path("trajectory.json"),
        )
    except ValueError as error:
        assert "phase must be one of" in str(error)
    else:
        raise AssertionError("Expected a legacy integer phase to fail")


def test_tolerance_targets_follow_the_frame_annotation_reference() -> None:
    targets = converter.tolerance_targets_from_frame(
        {"phase": "grasp", "stage_annotation_id": 7},
        annotations={
            7: {
                "id": 7,
                "stage_target_pose": {
                    "position_m": [0.4, 0.1, 0.5],
                    "rotation_vector_base_rad": [0.0, 0.0, 0.0],
                },
                "tolerance_frame_rotation_vector_base_rad": [0.0, 0.0, np.pi / 2],
                "rotation_tolerance_profile": "pregrasp",
            }
        },
        tolerance_profiles={"pregrasp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        source=Path("trajectory.json"),
    )

    assert set(targets) == {"stage_target_pose", "rotation_tolerance"}
    np.testing.assert_allclose(targets["stage_target_pose"], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(targets["rotation_tolerance"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert all(value.dtype == np.float32 for value in targets.values())


def test_load_tolerance_annotation_uses_task_bounds_and_ignores_episode_refinements(tmp_path: Path) -> None:
    annotation = {
        "rotation_tolerance_bounds_order": list(converter.ROTATION_TOLERANCE_PROFILE_ORDER),
        "rotation_tolerance_bounds_rad": {
            "pregrasp": [0.0, 0.0, 0.1, 0.2, 0.0, 0.0],
            "postgrasp": [0.0, 0.0, 0.0, 0.0, 0.7, 0.8],
        },
        "episode_refinements": [
            {
                "episode_index": 3,
                "seed": 1320234444,
                "refined_bounds_deg": {
                    "pregrasp": {
                        "x": {"negative": 0.0, "positive": 0.0},
                        "y": {"negative": 0.0, "positive": 10.0},
                        "z": {"negative": 0.0, "positive": 0.0},
                    },
                    "postgrasp": {
                        "x": {"negative": 0.0, "positive": 0.0},
                        "y": {"negative": 0.0, "positive": 0.0},
                        "z": {"negative": 45.0, "positive": 45.0},
                    },
                },
            }
        ],
    }
    (tmp_path / "tolerance_annotation.json").write_text(json.dumps(annotation), encoding="utf-8")

    task_profiles = converter._load_tolerance_annotation(
        tmp_path,
        converter._default_tolerance_profiles(),
    )

    np.testing.assert_allclose(task_profiles["pregrasp"], [0.0, 0.0, 0.1, 0.2, 0.0, 0.0])
    np.testing.assert_allclose(task_profiles["postgrasp"], [0.0, 0.0, 0.0, 0.0, 0.7, 0.8])


def test_load_tolerance_annotation_falls_back_without_file(tmp_path: Path) -> None:
    expected = converter._default_tolerance_profiles()

    task_profiles = converter._load_tolerance_annotation(tmp_path, expected)

    assert task_profiles == expected


def test_load_tolerance_summary_converts_degrees_to_directional_radians(tmp_path: Path) -> None:
    summary_path = tmp_path / "tolerance_summary.csv"
    summary_path.write_text(
        "robot_uid,training_scene_id,training_task_id,"
        "pre_rx_neg,pre_rx_pos,pre_ry_neg,pre_ry_pos,pre_rz_neg,pre_rz_pos,"
        "post_rx_neg,post_rx_pos,post_ry_neg,post_ry_pos,post_rz_neg,post_rz_pos\n"
        "panda,scene,task,30,20,10,0,45,5,1,2,3,4,5,6\n",
        encoding="utf-8-sig",
    )

    profiles = converter._load_tolerance_summary(summary_path)[("panda", "scene", "task")]

    np.testing.assert_allclose(profiles["pregrasp"], np.deg2rad([30, 20, 10, 0, 45, 5]))
    np.testing.assert_allclose(profiles["postgrasp"], np.deg2rad([1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(profiles["grasp"], np.zeros(6))
    np.testing.assert_array_equal(profiles["release"], np.zeros(6))


def test_tolerance_summary_allows_rows_for_tasks_not_present_in_raw_data() -> None:
    summary_profiles = {
        ("panda", "bell", "click_bell"): converter._default_tolerance_profiles(),
        ("panda", "blocks", "stack_blocks"): converter._default_tolerance_profiles(),
    }
    terminal = io.StringIO()

    with redirect_stdout(terminal):
        converter._report_unused_tolerance_summary_rows(
            summary_profiles,
            {("panda", "bell", "click_bell")},
            raw_dir=Path("/raw"),
        )

    assert "Ignoring 1 tolerance row(s)" in terminal.getvalue()


def test_load_single_axis_searches_overrides_corresponding_axes(tmp_path: Path) -> None:
    (tmp_path / "single_axis_rx.json").write_text(
        json.dumps(
            {
                "method": "single_axis_only_search",
                "search": {
                    "single_axis_only": ["x"],
                },
                "rotation_tolerance_bounds_rad": {
                    "pregrasp": [0.1745, 0.1745, 0.0, 0.0, 0.0, 0.0],
                    "postgrasp": [0.5236, 0.5236, 0.5236, 0.5236, 0.7854, 0.7854],
                },
            }
        ),
        encoding="utf-8",
    )

    base = converter._default_tolerance_profiles()
    merged = converter._load_single_axis_searches(tmp_path, base)

    np.testing.assert_allclose(merged["pregrasp"][0], 0.1745)
    np.testing.assert_allclose(merged["pregrasp"][1], 0.1745)
    np.testing.assert_allclose(merged["pregrasp"][2], base["pregrasp"][2])
    np.testing.assert_allclose(merged["pregrasp"][3], base["pregrasp"][3])
    np.testing.assert_allclose(merged["pregrasp"][4], base["pregrasp"][4])
    np.testing.assert_allclose(merged["pregrasp"][5], base["pregrasp"][5])


def test_load_single_axis_searches_merges_multiple_axes(tmp_path: Path) -> None:
    (tmp_path / "single_axis_rx.json").write_text(
        json.dumps(
            {
                "method": "single_axis_only_search",
                "search": {
                    "single_axis_only": ["x"],
                },
                "rotation_tolerance_bounds_rad": {
                    "pregrasp": [10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "single_axis_ry.json").write_text(
        json.dumps(
            {
                "method": "single_axis_only_search",
                "search": {
                    "single_axis_only": ["y"],
                },
                "rotation_tolerance_bounds_rad": {
                    "pregrasp": [0.0, 0.0, 5.0, 5.0, 0.0, 0.0],
                },
            }
        ),
        encoding="utf-8",
    )

    base = converter._default_tolerance_profiles()
    merged = converter._load_single_axis_searches(tmp_path, base)

    np.testing.assert_allclose(merged["pregrasp"][0], 10.0)
    np.testing.assert_allclose(merged["pregrasp"][1], 10.0)
    np.testing.assert_allclose(merged["pregrasp"][2], 5.0)
    np.testing.assert_allclose(merged["pregrasp"][3], 5.0)
    np.testing.assert_allclose(merged["pregrasp"][4], base["pregrasp"][4])
    np.testing.assert_allclose(merged["pregrasp"][5], base["pregrasp"][5])


def test_load_tolerance_annotation_with_single_axis_search(tmp_path: Path) -> None:
    (tmp_path / "tolerance_annotation.json").write_text(
        json.dumps(
            {
                "rotation_tolerance_bounds_order": list(converter.ROTATION_TOLERANCE_PROFILE_ORDER),
                "rotation_tolerance_bounds_rad": {
                    "pregrasp": [0.0, 0.0, 0.1, 0.2, 0.0, 0.0],
                    "postgrasp": [0.0, 0.0, 0.0, 0.0, 0.7, 0.8],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "single_axis_rx.json").write_text(
        json.dumps(
            {
                "method": "single_axis_only_search",
                "search": {
                    "single_axis_only": ["x"],
                },
                "rotation_tolerance_bounds_rad": {
                    "pregrasp": [0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
                },
            }
        ),
        encoding="utf-8",
    )

    task_profiles = converter._load_tolerance_annotation(
        tmp_path,
        converter._default_tolerance_profiles(),
    )

    # rx values from single-axis search override tolerance_annotation
    np.testing.assert_allclose(task_profiles["pregrasp"][0], 0.3)
    np.testing.assert_allclose(task_profiles["pregrasp"][1], 0.3)
    # ry values from tolerance_annotation remain
    np.testing.assert_allclose(task_profiles["pregrasp"][2], 0.1)
    np.testing.assert_allclose(task_profiles["pregrasp"][3], 0.2)
    # postgrasp unchanged
    np.testing.assert_allclose(task_profiles["postgrasp"][4], 0.7)
    np.testing.assert_allclose(task_profiles["postgrasp"][5], 0.8)


def test_repair_partial_dataset_rolls_back_uncommitted_tail(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset"
    meta_path = output_path / "meta"
    data_path = output_path / "data/chunk-000"
    image_path = output_path / "images/image/episode_000002"
    meta_path.mkdir(parents=True)
    data_path.mkdir(parents=True)
    image_path.mkdir(parents=True)
    (meta_path / "info.json").write_text(
        json.dumps({
            "total_episodes": 3,
            "total_frames": 999,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "splits": {"train": "0:3"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {"state": {"dtype": "float32"}},
        }),
        encoding="utf-8",
    )
    (meta_path / "episodes.jsonl").write_text(
        '\n'.join([
            json.dumps({"episode_index": 0, "tasks": ["a"], "length": 5}),
            json.dumps({"episode_index": 1, "tasks": ["a"], "length": 6}),
        ]) + '\n',
        encoding="utf-8",
    )
    (meta_path / "episodes_stats.jsonl").write_text(
        '\n'.join([
            json.dumps({"episode_index": 0, "stats": {}}),
            json.dumps({"episode_index": 1, "stats": {}}),
        ]) + '\n',
        encoding="utf-8",
    )
    (meta_path / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "a"}) + '\n', encoding="utf-8"
    )
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


def test_episode_fingerprint_is_stable_and_sensitive() -> None:
    states = np.asarray([[1.0] * 7, [2.0] * 7], dtype=np.float32)
    actions = np.asarray([[3.0] * 7, [4.0] * 7], dtype=np.float32)
    robot_ids = np.asarray([[0], [0]], dtype=np.int64)

    fingerprint = converter._fingerprint_arrays(states, actions, robot_ids)

    assert fingerprint == converter._fingerprint_arrays(states.copy(), actions.copy(), robot_ids.copy())
    changed_actions = actions.copy()
    changed_actions[-1, -1] += 1.0
    assert fingerprint != converter._fingerprint_arrays(states, changed_actions, robot_ids)


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
