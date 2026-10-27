"""The reconstruction objective must not constrain original joint trajectories."""

import unittest
import numpy as np

from experiments.libero_tasktol_compat.closed_loop_state_metrics_test import model_and_state
from experiments.libero_tasktol_compat.task_pose_object_metrics import task_cost


class TaskOnlyCostTest(unittest.TestCase):
    def test_robot_and_finger_joint_errors_are_not_penalties(self):
        model,source=model_and_state();actual=source.copy()
        actual[:,1:10]+=.05
        actual[:,1+model.nq:1+model.nq+9]+=.5
        positions=np.zeros((2,3));rotations=np.tile(np.eye(3),(2,1,1))
        result=task_cost(model,actual,source,positions,rotations,positions,rotations)
        self.assertEqual(result['objective'],0.)
        self.assertFalse(result['robot_joint_tracking_penalty'])
        self.assertFalse(result['finger_joint_tracking_penalty'])

    def test_end_effector_pose_error_is_penalized(self):
        model,source=model_and_state();positions=np.zeros((2,3));actual_p=positions.copy();actual_p[:,0]=.01
        rotations=np.tile(np.eye(3),(2,1,1))
        result=task_cost(model,source,source,actual_p,rotations,positions,rotations)
        self.assertGreater(result['objective'],0.)


if __name__=='__main__':unittest.main()
