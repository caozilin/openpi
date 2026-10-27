import numpy as np
import pytest
from tasktol.control.joint_tracker import JointReferenceTracker
from tasktol.simulation.preset_runtime import PresetSimulationBackend


@pytest.mark.parametrize('interval,dt',[(.1,.001),(.05,.002)])
def test_shared_calculation_matches_original_backend_arithmetic(interval,dt):
    legacy=JointReferenceTracker(7);shared=JointReferenceTracker(7)
    start=np.linspace(-.2,.2,7);goal=start+np.linspace(.01,.03,7)
    actual=start.copy();velocity=np.zeros(7);elapsed=0.
    for _ in range(round(interval/dt)+3):
        expected_elapsed=min(interval,elapsed+dt);alpha=expected_elapsed/interval
        expected=(1-alpha)*start+alpha*goal
        expected_command,expected_diagnostic=legacy.update(expected,actual,dt,velocity)
        reference,command,diagnostic,new_elapsed,new_alpha=PresetSimulationBackend.track_joint_reference(
            tracker=shared,interpolation_start=start,joint_target=goal,interpolation_elapsed=elapsed,
            interpolation_duration_s=interval,physics_dt=dt,actual_q=actual,actual_velocity_q=velocity)
        np.testing.assert_array_equal(reference,expected)
        np.testing.assert_array_equal(command,expected_command)
        assert diagnostic==expected_diagnostic and new_elapsed==expected_elapsed and new_alpha==alpha
        elapsed=new_elapsed;velocity=(reference-actual)/dt;actual=reference.copy()
