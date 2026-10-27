import numpy as np

from tasktol.geometry import rotation_matrix

from experiments.libero_tasktol_compat.osc_reference import (
    decode_source_goals,
    difference_reference_goals,
    integrate_reference_actions,
)


def test_feedback_relative_goals_and_zero_orientation_hold():
    positions = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.003, 0.0, 0.0]])
    rotations = np.array([rotation_matrix(np.array([0.1 * i, 0.0, 0.0])) for i in range(3)])
    held = rotation_matrix(np.array([0.0, 0.3, 0.0]))
    actions = np.zeros((3, 6))
    actions[:, 0] = 0.01
    actions[1, 5] = 0.2
    p, r = decode_source_goals(positions, rotations, actions, held)
    np.testing.assert_allclose(p[:, 0], [0.01, 0.012, 0.013], atol=1e-15)
    np.testing.assert_allclose(r[0], held, atol=1e-15)
    np.testing.assert_allclose(r[1], rotation_matrix(actions[1, 3:]) @ rotations[1], atol=1e-15)
    np.testing.assert_array_equal(r[2], r[1])


def test_relabelled_actions_reconstruct_noncommuting_goals_from_initial_anchor():
    anchor_p = np.array([0.3, -0.1, 0.9])
    anchor_r = rotation_matrix(np.array([0.4, -0.2, 0.1]))
    positions = np.array([[0.32, -0.1, 0.9], [0.34, -0.09, 0.91], [0.34, -0.09, 0.91]])
    rotations = np.array([
        rotation_matrix(np.array([0.0, 0.3, 0.0])) @ anchor_r,
        rotation_matrix(np.array([0.2, 0.0, -0.1])) @ anchor_r,
        rotation_matrix(np.array([0.2, 0.0, -0.1])) @ anchor_r,
    ])
    gripper = np.array([-1.0, 1.0, 1.0])
    actions = difference_reference_goals(positions, rotations, gripper, anchor_p, anchor_r)
    reconstructed_p, reconstructed_r = integrate_reference_actions(actions, anchor_p, anchor_r)
    np.testing.assert_allclose(actions[0, :3], positions[0] - anchor_p, atol=1e-15)
    np.testing.assert_allclose(actions[0, 3:6], [0.0, 0.3, 0.0], atol=1e-14)
    np.testing.assert_array_equal(actions[:, 6], gripper)
    np.testing.assert_allclose(reconstructed_p, positions, atol=1e-14)
    np.testing.assert_allclose(reconstructed_r, rotations, atol=1e-13)
    np.testing.assert_allclose(actions[2, :6], 0.0, atol=1e-14)
