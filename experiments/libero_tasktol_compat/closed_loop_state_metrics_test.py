"""Physical metrics must handle quaternion equivalence and real time alignment."""

import unittest

import mujoco
import numpy as np

from experiments.libero_tasktol_compat.closed_loop_state_metrics import compare_states


def model_and_state():
    arm = ''
    for index in reversed(range(7)):
        arm = f'<body pos="0 0 .1"><joint name="robot0_joint{index+1}"/><geom type="sphere" size=".03"/>{arm}</body>'
    fingers = ''.join(f'<body pos="{i*.1} 0 0"><joint name="gripper0_finger{i}" type="slide"/><geom type="sphere" size=".02"/></body>' for i in range(2))
    model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{arm}{fingers}<body pos="1 0 0"><freejoint name="object_free"/><geom type="sphere" size=".03"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    state = np.r_[0., data.qpos, data.qvel]
    return model, np.vstack([state, state + np.r_[.05, np.zeros(len(state)-1)]])


class StateMetricsTest(unittest.TestCase):
    def test_quaternion_sign_does_not_change_physical_state_error(self):
        model, source = model_and_state()
        actual = source.copy()
        start = model.jnt_qposadr[model.joint('object_free').id]
        actual[:, 1 + start + 3:1 + start + 7] *= -1
        result, _, _ = compare_states(model, actual, source)
        self.assertAlmostEqual(result['objective'], 0.)

    def test_translation_and_joint_error_have_separate_units(self):
        model, source = model_and_state()
        actual = source.copy()
        start = model.jnt_qposadr[model.joint('object_free').id]
        actual[:, 1 + start] += .01
        actual[:, 1] += .1
        result, _, _ = compare_states(model, actual, source)
        self.assertAlmostEqual(result['objects'][0]['translation']['mean'], .01)
        self.assertAlmostEqual(result['groups']['arm_position_rad']['rmse'], .1 / np.sqrt(7))
        self.assertAlmostEqual(result['groups']['object_translation_m']['normalized_weighted_cost'], 2 / 3)

    def test_velocity_is_evaluated_even_at_identical_pose(self):
        model, source = model_and_state()
        actual = source.copy()
        actual[:, 1 + model.nq] += 1
        result, _, _ = compare_states(model, actual, source)
        self.assertAlmostEqual(result['groups']['arm_position_rad']['rmse'], 0.)
        self.assertGreater(result['objective'], 0.)

    def test_time_shift_is_not_hidden_by_alignment(self):
        model, source = model_and_state()
        actual = source.copy()
        actual[:, 0] += .05
        with self.assertRaises(AssertionError):
            compare_states(model, actual, source)


if __name__ == '__main__':
    unittest.main()
