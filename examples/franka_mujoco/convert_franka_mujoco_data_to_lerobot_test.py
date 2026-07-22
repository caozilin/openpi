from pathlib import Path

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


def test_require_vector_rejects_wrong_action_dimension() -> None:
    try:
        converter._require_vector([0.0] * 6, 7, field="action", source=Path("trajectory.json"))
    except ValueError as error:
        assert "shape (7,)" in str(error)
    else:
        raise AssertionError("Expected an invalid action dimension to fail")


def test_phase_from_frame_uses_schema_5_four_stage_ids() -> None:
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
                    "rotation_vector_base_rad": [3.1, 0.2, -0.1],
                },
                "tolerance_frame_rotation_vector_base_rad": [3.14, 0.0, 0.3],
                "rotation_tolerance_profile": "pregrasp",
            }
        },
        tolerance_profiles={"pregrasp": [0.0, 0.5, 0.0]},
        source=Path("trajectory.json"),
    )

    assert set(targets) == {"stage_target_pose", "tolerance_frame", "rotation_tolerance"}
    np.testing.assert_allclose(targets["stage_target_pose"], [3.1, 0.2, -0.1])
    np.testing.assert_allclose(targets["tolerance_frame"], [3.14, 0.0, 0.3])
    np.testing.assert_allclose(targets["rotation_tolerance"], [0.0, 0.5, 0.0])
    assert all(value.dtype == np.float32 for value in targets.values())
