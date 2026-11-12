"""LIBERO joint supervision through the pretrained 32D projection interface."""
import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from openpi.policies.libero_policy import LiberoInputs
from openpi.shared.libero_joint_dataset import validate_tolerance

@dataclasses.dataclass(frozen=True)
class LiberoJointInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data['observation/state'], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f'LIBERO requires finite preaction state8, got {state.shape}')
        result = LiberoInputs(self.model_type)(data)
        result['state'] = state
        if 'actions' in data:
            actions = np.asarray(data['actions'], dtype=np.float32)
            if actions.shape[-1] != 7 or not np.isfinite(actions).all():
                raise ValueError('Require finite physical LIBERO actions7')
            flag = np.asarray(data['trajectory_is_tolerance']).reshape(-1)
            if flag.shape != (1,) or flag[0] not in (0, 1):
                raise ValueError('trajectory_is_tolerance must contain one binary value')
            tolerance = validate_tolerance(data['rotation_tolerance'], leading_shape=actions.shape[:-1])
            result['actions'] = np.concatenate([actions, tolerance], axis=-1)
        return result

@dataclasses.dataclass(frozen=True)
class LiberoJointOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data['actions'])
        if actions.ndim != 2 or actions.shape[-1] != 32:
            raise ValueError('Require pretrained 32D interface before extracting effective joint13')
        return {'actions': actions[:, :7], 'rotation_tolerance': actions[:, 7:13]}
