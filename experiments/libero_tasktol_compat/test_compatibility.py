"""Validate the mass-matrix API translation against native MuJoCo physics."""
import mujoco
import numpy as np
import pytest

from compatibility import ControllerMuJoCoAPI


@pytest.fixture
def arm():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint axis="0 0 1"/><geom size=".1" mass="1"/>
        <body pos=".3 0 0"><joint axis="0 1 0"/><geom size=".08" mass=".5"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    return model, data


@pytest.mark.parametrize('q', [[0, 0], [.4, -.6], [-1., .8]])
def test_mass_matrix_matches_native_at_multiple_configurations(arm, q):
    model, data = arm
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    expected = np.empty((model.nv, model.nv))
    actual = np.empty_like(expected)
    mujoco.mj_fullM(model, data, expected)
    api = ControllerMuJoCoAPI(mujoco)
    token = api.current_data.set(data)
    try:
        api.mj_fullM(model, actual, data.qM)
    finally:
        api.current_data.reset(token)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    assert np.linalg.eigvalsh(actual).min() > 0


def test_foreign_data_and_missing_context_are_rejected(arm):
    model, data = arm
    other = mujoco.MjData(model)
    api = ControllerMuJoCoAPI(mujoco)
    destination = np.empty((model.nv, model.nv))
    with pytest.raises(LookupError):
        api.mj_fullM(model, destination, data.qM)
    token = api.current_data.set(data)
    try:
        with pytest.raises(ValueError, match='current controller'):
            api.mj_fullM(model, destination, other.qM)
    finally:
        api.current_data.reset(token)
