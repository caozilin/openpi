import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


STATE_DIM = 7
ACTION_DIM = 7
ROTATION_6D_DIM = 6
ROTATION_TOLERANCE_DIM = 6
JOINT_ACTION_DIM = ACTION_DIM + ROTATION_6D_DIM + ROTATION_TOLERANCE_DIM
TARGET_ROTATION_SLICE = slice(ACTION_DIM, ACTION_DIM + ROTATION_6D_DIM)
ROTATION_TOLERANCE_SLICE = slice(TARGET_ROTATION_SLICE.stop, JOINT_ACTION_DIM)
TASK_TOLERANCE_DIMS = {
    "stage_target_pose": ROTATION_6D_DIM,
    "rotation_tolerance": ROTATION_TOLERANCE_DIM,
}


def make_franka_mujoco_example() -> dict:
    """Create a random observation in the Franka MuJoCo runtime format."""
    return {
        "observation/state": np.random.rand(STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a rank-3 image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaMujocoInputs(transforms.DataTransformFn):
    """Convert Franka MuJoCo observations to the common Pi model input format."""

    model_type: _model.ModelType
    joint_task_tolerance: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"Expected Franka state shape ({STATE_DIM},), got {state.shape}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"Expected {ACTION_DIM}-D Franka actions, got shape {actions.shape}")
            if self.joint_task_tolerance:
                targets = []
                for key, dimension in TASK_TOLERANCE_DIMS.items():
                    value = np.asarray(data[key], dtype=np.float32)
                    if value.shape[:-1] != actions.shape[:-1] or value.shape[-1] != dimension:
                        raise ValueError(
                            f"Expected {key} shape {(*actions.shape[:-1], dimension)}, got {value.shape}"
                        )
                    targets.append(value)
                actions = np.concatenate((actions, *targets), axis=-1)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaMujocoOutputs(transforms.DataTransformFn):
    """Unpack physical Franka actions and optional joint tolerance targets."""

    joint_task_tolerance: bool = False

    def __call__(self, data: dict) -> dict:
        predictions = np.asarray(data["actions"])
        outputs = {"actions": predictions[:, :ACTION_DIM]}
        if self.joint_task_tolerance:
            if predictions.shape[-1] < JOINT_ACTION_DIM:
                raise ValueError(f"Expected at least {JOINT_ACTION_DIM} joint output dimensions, got {predictions.shape}")
            outputs.update(
                {
                    "stage_target_pose": predictions[:, TARGET_ROTATION_SLICE],
                    "rotation_tolerance": predictions[:, ROTATION_TOLERANCE_SLICE],
                }
            )
        return outputs
