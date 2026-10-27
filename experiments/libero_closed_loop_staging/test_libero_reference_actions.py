import hashlib
import json

import h5py
import numpy as np
import pytest

from tasktol.benchmark.libero_reference_actions import load_reference_actions, verify_reference_anchor
from tasktol.geometry import rotation_matrix


@pytest.fixture
def reference_file(tmp_path):
    path = tmp_path / "labels.hdf5"
    source = tmp_path / "source.hdf5"
    states = np.zeros((2, 8))
    source_actions = np.zeros((2, 7))
    source_actions[:, 6] = [-1, 1]
    actions = np.array([[0.01, 0, 0, 0, 0.03, 0, -1], [0, 0.01, 0, 0.02, 0, 0, 1]])
    anchor_p = np.array([0.3, -0.1, 0.8])
    anchor_r = rotation_matrix(np.array([0.1, 0, 0.2]))
    rotations = [rotation_matrix(actions[0, 3:6]) @ anchor_r]
    rotations.append(rotation_matrix(actions[1, 3:6]) @ rotations[0])
    with h5py.File(path, "w") as file:
        file.attrs["schema"] = "libero_osc_independent_reference_actions_v1"
        file.attrs["source_hdf5"] = str(source)
        file.attrs["action_contract"] = json.dumps(dict(position="metres", rotation="radians"))
        demo = file.create_group("data/demo_0")
        demo.attrs["input_state_action_sha256"] = hashlib.sha256(states.tobytes() + source_actions.tobytes()).hexdigest()
        for key, value in dict(actions=actions, anchor_position_m=anchor_p, anchor_tool_rotation=anchor_r,
                               goal_positions_m=anchor_p + np.cumsum(actions[:, :3], axis=0),
                               goal_tool_rotations=np.array(rotations)).items():
            demo.create_dataset(key, data=value)
    return path, dict(source_path=source, source_states=states, source_actions=source_actions), actions, anchor_p, anchor_r


def test_reference_actions_preserve_physical_commands_and_explicit_anchor(reference_file):
    path, inputs, expected, position, rotation = reference_file
    result = load_reference_actions(path, "demo_0", **inputs)
    np.testing.assert_array_equal(result.actions, expected)
    verify_reference_anchor(result, position, rotation)
    with pytest.raises(ValueError, match="anchor differs"):
        verify_reference_anchor(result, position + [0.001, 0, 0], rotation)


@pytest.mark.parametrize("change, message", [
    ("source_path", "different source"), ("state", "hash does not match"),
    ("gripper", "gripper actions differ"), ("pose", "do not reconstruct"),
    ("rotation", "belong to SO"),
])
def test_reference_actions_reject_stale_or_corrupt_annotations(reference_file, change, message):
    path, inputs, _, _, _ = reference_file
    if change == "source_path":
        inputs["source_path"] = path.with_name("other.hdf5")
    elif change == "state":
        inputs["source_states"][0, 0] = 1
    else:
        with h5py.File(path, "r+") as file:
            group = file["data/demo_0"]
            if change == "gripper":
                group["actions"][0, 6] = 1
            elif change == "pose":
                group["actions"][0, 0] += 0.001
            elif change == "rotation":
                group["anchor_tool_rotation"][0, 0] = 2
    with pytest.raises(ValueError, match=message):
        load_reference_actions(path, "demo_0", **inputs)
