"""Pure FK/reference checks for strict joint13 rollouts; no solve or stepping."""
import numpy as np


class RecordedNominalReplay:
    def __init__(self, env, initial):
        from tasktol.simulation.libero import LiberoTaskTolSession
        env.set_state(initial)
        env.sim.forward()
        self.session = LiberoTaskTolSession(env, joint_control_mode='native')

    def verify(self, action, selection, row):
        from tasktol.tolerance.runtime import advance_reference_pose
        from tasktol.geometry import rotation_matrix, rotation_vector
        s = self.session
        old_position, old_rotation = s.target.position.copy(), s.target.rotation.copy()
        position, rotation = advance_reference_pose(old_position, old_rotation, action)
        assert row['bounds_rad'] is None
        assert row['semantic_mask'] == row['effective_mask'] == '000'
        assert row['phase'] == 'APP' and row['subtask_index'] == 0
        if row['optimizer']['feasible']:
            fk = s.ik.kinematics.evaluate(np.asarray(row['joint_target_rad']),
                                        include_manipulability=False, include_link_points=False)
            error = s.ik.kinematics.pose_error(fk, position, rotation, None)
            # Match canonical IPOPT componentwise residuals (L-infinity),
            # not an extra L2 requirement with the same scalar tolerance.
            assert np.max(np.abs(error[:3])) <= s.ik.settings.position_tolerance + 1e-9
            assert np.max(np.abs(error[3:])) <= s.ik.settings.rotation_tolerance + 1e-9
            np.testing.assert_allclose(np.max(np.abs(error[:3])), row['optimizer']['position_residual_m'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(np.max(np.abs(error[3:])), row['optimizer']['strict_rotation_residual_rad'], atol=1e-10, rtol=0)
            s.target.position, s.target.rotation = position, rotation
        np.testing.assert_allclose(s.target.position, row['nominal_target']['position_m'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(s.target.rotation, rotation_matrix(np.asarray(row['nominal_target']['rotation_vector_rad'])), atol=1e-10, rtol=0)
