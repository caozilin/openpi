import numpy as np
import pytest

from examples.libero.tasktol_actions import decode_tasktol_action


def test_noisy_hold_does_not_become_full_speed_motion_or_change_pose_units():
    actions = np.asarray([[.012, -.003, .004, .01, -.02, .03, g] for g in (-1.1, -.8, -.2, .2, .8, 1.1)])
    decoded = decode_tasktol_action(actions)
    np.testing.assert_array_equal(decoded[:, :6], actions[:, :6])
    np.testing.assert_array_equal(decoded[:, 6], [-1, -1, 0, 0, 1, 1])


def test_saved_ternary_labels_are_exactly_preserved():
    for grip in (-1, 0, 1):
        action = np.asarray([.01, -.02, .03, .04, -.05, .06, grip])
        np.testing.assert_array_equal(decode_tasktol_action(action), action)


@pytest.mark.parametrize('action', [np.zeros(6), np.zeros((2, 8)), [0, 0, 0, 0, 0, 0, np.nan]])
def test_invalid_action_is_rejected(action):
    with pytest.raises(ValueError):
        decode_tasktol_action(action)
