"""Convert native OSC feedback-relative goals to independent reference increments.

Actions here use metres and world-frame rotation vectors in radians. Gripper
values retain the source contract. These are not normalized native OSC deltas.
"""

import math

import numpy as np

from tasktol.geometry import rotation_matrix, rotation_vector
from tasktol.tolerance.runtime import advance_reference_pose


def decode_source_goals(positions, rotations, scaled_actions, initial_held_rotation):
    """Decode OSC_POSE with no pose limits, including its zero-rotation hold."""
    held = np.asarray(initial_held_rotation, dtype=float).copy()
    goals_p, goals_r = [], []
    for position, rotation, action in zip(positions, rotations, scaled_actions, strict=True):
        if any(not math.isclose(float(x), 0.0) for x in action[3:6]):
            held = rotation_matrix(action[3:6]) @ rotation
        goals_p.append(position + action[:3])
        goals_r.append(held.copy())
    return np.asarray(goals_p), np.asarray(goals_r)


def difference_reference_goals(positions, rotations, gripper, anchor_position, anchor_rotation):
    """Difference absolute goals on R^3 x SO(3), including the first anchor."""
    previous_p = np.asarray(anchor_position, dtype=float)
    previous_r = np.asarray(anchor_rotation, dtype=float)
    actions = []
    for position, rotation, grip in zip(positions, rotations, gripper, strict=True):
        actions.append(np.r_[position - previous_p, rotation_vector(rotation @ previous_r.T), grip])
        previous_p, previous_r = position, rotation
    return np.asarray(actions)


def integrate_reference_actions(actions, anchor_position, anchor_rotation):
    """Use the maintained TaskTol nominal-reference integrator, with no IK."""
    position = np.asarray(anchor_position, dtype=float).copy()
    rotation = np.asarray(anchor_rotation, dtype=float).copy()
    positions, rotations = [], []
    for action in actions:
        position, rotation = advance_reference_pose(position, rotation, action)
        positions.append(position.copy())
        rotations.append(rotation.copy())
    return np.asarray(positions), np.asarray(rotations)
