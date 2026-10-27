"""Decode physical TaskTol policy outputs, including native gripper holds."""

import numpy as np


def decode_tasktol_action(action, *, gripper_deadband=.5):
    """After unnormalization, preserve m/rad motion and decode three grip intents.

    Native Panda applies sign(g), so a near-zero policy output would otherwise
    become a full-speed opening/closing command instead of the learned hold.
    This decoder changes no simulator or joint-reference control behavior.
    """
    values = np.asarray(action, dtype=float)
    if values.ndim not in (1, 2) or values.shape[-1] != 7 or not np.all(np.isfinite(values)):
        raise ValueError('Expected finite physical action(s) with final dimension seven')
    if not np.isfinite(gripper_deadband) or not 0 <= gripper_deadband < 1:
        raise ValueError('Gripper deadband must be in [0, 1)')
    decoded = values.copy()
    grip = values[..., 6]
    decoded[..., 6] = np.where(grip > gripper_deadband, 1, np.where(grip < -gripper_deadband, -1, 0))
    return decoded
