"""Opt-in API bridge for robosuite 1.4.1 and MuJoCo 3.10.

Never changes the installed packages or the global mujoco module. The original
robosuite controller update still runs; only its mj_fullM call is translated to
the new binding signature using that controller's actual MjData.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps


class ControllerMuJoCoAPI:
    def __init__(self, native):
        self.native = native
        self.current_data = ContextVar('robosuite_controller_mjdata')

    def __getattr__(self, name):
        return getattr(self.native, name)

    def mj_fullM(self, model, destination, packed_mass):
        import numpy as np
        data = self.current_data.get()
        if not np.shares_memory(packed_mass, data.qM):
            raise ValueError('Mass matrix must belong to the current controller MjData')
        return self.native.mj_fullM(model, data, destination)


def install_mass_matrix_bridge():
    import mujoco
    import robosuite
    from robosuite.controllers import base_controller

    if robosuite.__version__ != '1.4.1':
        raise ValueError('This bridge is limited to inspected robosuite 1.4.1')
    if 'd: mujoco._structs.MjData' not in (mujoco.mj_fullM.__doc__ or ''):
        raise ValueError('Expected the MuJoCo binding mj_fullM(model, data, dst)')
    if isinstance(base_controller.mujoco, ControllerMuJoCoAPI):
        return ['robosuite_1_4_1_mj_fullM_signature_bridge']
    api = ControllerMuJoCoAPI(mujoco)
    original_update = base_controller.Controller.update

    @wraps(original_update)
    def update(self, *args, **kwargs):
        token = api.current_data.set(self.sim.data._data)
        try:
            return original_update(self, *args, **kwargs)
        finally:
            api.current_data.reset(token)

    base_controller.mujoco = api
    base_controller.Controller.update = update
    return ['robosuite_1_4_1_mj_fullM_signature_bridge']
