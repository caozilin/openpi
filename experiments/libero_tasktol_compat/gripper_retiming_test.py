"""Check pulse timing against the actual pinned Panda format_action source."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE = Path('/media/czl/sata/franka_my_code/franka_mujoco/experiments/libero_closed_loop/.venv/lib/python3.12/site-packages/robosuite/models/grippers/panda_gripper.py')


def format_action():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PandaGripper')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'format_action')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['format_action']


def rollout(commands):
    fn = format_action()
    gripper = SimpleNamespace(dof=1, speed=.01, current_action=np.zeros(1))
    results = []
    for command in commands:
        for _ in range(25):
            fn(gripper, np.asarray([command]))
        results.append(gripper.current_action.copy())
    return np.asarray(results)


def test_pulses_preserve_integrator_phase_through_saturation_and_reversal():
    original = np.asarray([-1] * 8 + [1] * 8 + [-1] * 8)
    expected = rollout(original)
    for scale in (2, 3, 4):
        pulses = np.zeros(len(original) * scale)
        pulses[::scale] = original
        np.testing.assert_array_equal(rollout(pulses)[scale - 1::scale], expected)
        assert not np.array_equal(rollout(np.repeat(original, scale))[scale - 1::scale], expected)


def test_scaling_command_magnitude_cannot_slow_native_sign_integrator():
    commands = np.asarray([-1] * 3 + [1] * 3)
    np.testing.assert_array_equal(rollout(commands / 2), rollout(commands))
