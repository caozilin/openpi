import hashlib
import json

import numpy as np
import pytest

from openpi.shared import franka_mujoco_dataset as dataset_format


@pytest.fixture
def episode(tmp_path):
    frames = [
        {
            "index": i,
            "video_frame_index": i,
            "time_s": i / 10,
            "action": [0.0] * 7,
            "subtask_index": i // 4,
            "phase": phase,
        }
        for i, phase in enumerate(["pregrasp", "grasp", "postgrasp", "release"] * 2)
    ]
    trajectory = {"schema_version": "8.0", "episode": {"frame_count": 8}, "trajectory": frames}
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(trajectory))
    labels = {
        **dataset_format.tolerance_label_metadata(),
        "frame_count": 8,
        "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": {"method": "bayesian_search"},
        "subtasks": [
            {"subtask_index": i, "pregrasp": [0.1 + i / 10] * 6, "postgrasp": [0.3 + i / 10] * 6} for i in range(2)
        ],
    }
    (tmp_path / dataset_format.TOLERANCE_LABEL_FILE).write_text(json.dumps(labels))
    return tmp_path


def load(episode):
    document, digest = dataset_format.read_trajectory(episode)
    return dataset_format.load_rotation_tolerance(episode, document, digest)


def test_labels_expand_by_subtask_and_phase_without_changing_sources(episode):
    before = {p.name: p.read_bytes() for p in episode.iterdir()}
    values = load(episode)
    assert values.shape == (8, 6)
    assert values.dtype == np.float32
    np.testing.assert_allclose(values[:, 0], [0.1, 0, 0.3, 0, 0.2, 0, 0.4, 0])
    assert {p.name: p.read_bytes() for p in episode.iterdir()} == before


def test_unlabeled_episodes_are_rejected(episode):
    (episode / dataset_format.TOLERANCE_LABEL_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="missing tolerance labels"):
        load(episode)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trajectory_sha256", "0" * 64),
        ("frame_count", 7),
        ("source", {}),
        ("reference", "stage_endpoint"),
        ("units", "deg"),
        ("order", ["x"] * 6),
        ("coordinate_convention", "rotation_vector"),
        ("subtasks", None),
    ],
)
def test_corrupt_or_stale_label_contract_is_rejected(episode, field, value):
    path = episode / dataset_format.TOLERANCE_LABEL_FILE
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=".+"):
        load(episode)


@pytest.mark.parametrize("values", [[0.0] * 5, [0.0] * 7, [-0.1] * 6, [float("nan")] * 6, [float("inf")] * 6])
def test_invalid_six_dimensional_bounds_are_rejected(episode, values):
    path = episode / dataset_format.TOLERANCE_LABEL_FILE
    document = json.loads(path.read_text())
    document["subtasks"][0]["pregrasp"] = values
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="six finite nonnegative"):
        load(episode)


@pytest.mark.parametrize("variant", ["missing", "duplicate", "unknown", "strict", "bool_index"])
def test_labels_require_complete_unambiguous_subtasks(episode, variant):
    path = episode / dataset_format.TOLERANCE_LABEL_FILE
    document = json.loads(path.read_text())
    subtasks = document["subtasks"]
    if variant == "missing":
        del subtasks[1]["postgrasp"]
    elif variant == "duplicate":
        subtasks.append(subtasks[0])
    elif variant == "unknown":
        subtasks[1]["subtask_index"] = 9
    elif variant == "bool_index":
        subtasks[0]["subtask_index"] = False
    else:
        subtasks[0]["grasp"] = [0.0] * 6
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=".+"):
        load(episode)


@pytest.mark.parametrize("variant", ["schema", "index", "time", "phase", "subtask", "action", "stage"])
def test_invalid_base_trajectory_is_rejected(episode, variant):
    path = episode / "trajectory.json"
    document = json.loads(path.read_text())
    frame = document["trajectory"][0]
    if variant == "schema":
        document["schema_version"] = "7.0"
    elif variant == "index":
        frame["video_frame_index"] = 2
    elif variant == "time":
        frame["time_s"] = 0.1
    elif variant == "phase":
        frame["phase"] = 0
    elif variant == "subtask":
        frame["subtask_index"] = -1
    elif variant == "action":
        frame["action"] = [0.0] * 13
    else:
        document["stage_annotations"] = []
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=".+"):
        load(episode)


def test_changed_trajectory_invalidates_previous_labels(episode):
    path = episode / "trajectory.json"
    document = json.loads(path.read_text())
    document["trajectory"][0]["action"][0] = 0.01
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="SHA-256"):
        load(episode)


def test_export_contract_rejects_old_metadata_and_unlabeled_joint_training():
    with pytest.raises(ValueError, match="reconvert"):
        dataset_format.validate_export_info({})
    info = dataset_format.export_metadata(joint_task_tolerance=False)
    dataset_format.validate_export_info(info, joint_task_tolerance=False)
    with pytest.raises(ValueError, match="labeled"):
        dataset_format.validate_export_info(info, joint_task_tolerance=True)
    dataset_format.validate_export_info(dataset_format.export_metadata(joint_task_tolerance=True))
