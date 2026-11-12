"""Dedicated 20Hz LIBERO joint13 config; keeps the original state8 contract."""
import dataclasses
from pathlib import Path
from openpi import transforms
from openpi.models import model
from openpi.policies import libero_joint_policy
from openpi.shared import libero_joint_dataset
from openpi.training import config

@dataclasses.dataclass(frozen=True)
class LeRobotLiberoJointDataConfig(config.DataConfigFactory):
    tolerance_trajectory_action_loss_weight: float = 1.0

    def create(self, assets_dirs: Path, model_config: model.BaseModelConfig) -> config.DataConfig:
        if model_config.action_dim != 32:
            raise ValueError('LIBERO joint13 requires the pretrained 32-D interface; normalize13 then pad32')
        base = self.create_base_config(assets_dirs, model_config)
        if base.norm_stats is not None:
            for field in ('mean', 'std', 'q01', 'q99'):
                values = getattr(base.norm_stats['actions'], field)
                if values is not None and values.shape != (13,):
                    raise ValueError(f'Recompute joint13 normalization: actions.{field} must be shape13')
        return dataclasses.replace(base,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                'observation/image':'image', 'observation/wrist_image':'wrist_image',
                'observation/state':'state', 'actions':'actions', 'prompt':'prompt',
                'rotation_tolerance':'rotation_tolerance', 'trajectory_is_tolerance':'trajectory_is_tolerance',
            })]),
            data_transforms=transforms.Group(inputs=[libero_joint_policy.LiberoJointInputs(model_config.model_type, self.tolerance_trajectory_action_loss_weight)],
                                            outputs=[libero_joint_policy.LiberoJointOutputs()]),
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=('actions', 'rotation_tolerance'), source_schema=libero_joint_dataset.SOURCE_SCHEMA)
